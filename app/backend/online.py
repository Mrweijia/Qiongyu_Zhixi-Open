"""Resource-bounded online training endpoints.

This module deliberately trains only a small, deterministic sklearn Ridge
baseline.  It accepts CSV data, never evaluates user-provided code, and keeps
one background worker so that the web process remains usable on a small CPU
server.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from threading import Lock
import time
from typing import Any
import uuid
import zipfile

from flask import Blueprint, jsonify, request, send_file
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_FOLDER = ROOT_DIR / "outputs" / "runtime" / "online-training"
BUNDLE_DIR = ROOT_DIR / "models" / "checkpoints" / "multistep_v2"
MODEL_DIR = ROOT_DIR / "models" / "checkpoints"
MODEL_REGISTRY_PATH = ROOT_DIR / "configs" / "model_registry.yaml"

# Limits are intentionally conservative: requests are untrusted and this runs
# in the same process as the prediction API.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_ROWS = 5_000
MAX_COLUMNS = 33  # target plus at most 32 numeric predictors
MAX_FEATURES = 32
MAX_HORIZONS = 3
MAX_HORIZON = 24
MIN_SAMPLES = 60
MAX_TRAIN_SECONDS = 30.0
JOB_TTL_SECONDS = 60 * 60
# Anonymous submitants must not be able to exhaust the single-worker queue,
# the request budget or the runtime disk of a small server.
MAX_QUEUED_JOBS = 8
MAX_SUBMITS_PER_MINUTE = 5
MAX_RUNTIME_DISK_BYTES = 256 * 1024 * 1024
JOB_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

TRAINING_FOLDER.mkdir(parents=True, exist_ok=True)
online_bp = Blueprint("online", __name__)

_submission_times = deque()
_submission_lock = Lock()


class TrainingQueueFull(ValueError):
    """Raised when the bounded training queue is already saturated."""


def _rate_limit_exceeded() -> bool:
    now = time.monotonic()
    with _submission_lock:
        while _submission_times and now - _submission_times[0] > 60:
            _submission_times.popleft()
        if len(_submission_times) >= MAX_SUBMITS_PER_MINUTE:
            return True
        _submission_times.append(now)
        return False


def _runtime_disk_bytes() -> int:
    from app.backend import predict as predict_mod

    total = 0
    for folder in (TRAINING_FOLDER, predict_mod.UPLOAD_FOLDER, predict_mod.RESULTS_FOLDER):
        if not folder.is_dir():
            continue
        for item in folder.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    return total


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_horizons(value: Any) -> list[int]:
    """Parse a small JSON list or comma-separated list without eval()."""
    if value is None or value == "":
        return [1]
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text) if text.startswith("[") else text.split(",")
        except json.JSONDecodeError as exc:
            raise ValueError("horizons 必须是如 1,2,3 或 [1,2,3] 的整数列表") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError("horizons 必须是整数列表")
    try:
        horizons = [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("horizons 只能包含整数") from exc
    if not horizons or len(horizons) > MAX_HORIZONS:
        raise ValueError(f"horizons 需要 1 至 {MAX_HORIZONS} 个值")
    if any(item < 1 or item > MAX_HORIZON for item in horizons):
        raise ValueError(f"horizons 必须在 1 至 {MAX_HORIZON} 小时之间")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons 不能重复")
    return sorted(horizons)


# FLASH-05: online training must be anchored to explicit timestamps. A row
# index alone cannot prove t -> t+h order, so one of these column names must
# exist and parse cleanly (F-020).
TIME_COLUMN_CANDIDATES = ("time", "timestamp", "datetime", "date_time", "ts")
ALLOWED_TIME_FORMATS = ("iso8601", "%Y%m%d%H", "%Y-%m-%d %H:%M:%S")
MAX_TIME_ERRORS_REPORTED = 20
TRAINING_TIMEZONE = timezone.utc


def _parse_time_column(series: "pd.Series") -> "pd.Series":
    """Parse the time column to timezone-naive UTC datetimes, or raise.

    Mixed timezones, unparseable values and naive stamps in foreign offsets are
    rejected with a per-row reason so users can fix their export.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if parsed.isna().all():
        raise ValueError(
            f"时间列 {'/'.join(TIME_COLUMN_CANDIDATES)} 无法解析为时间戳；"
            f"支持的格式：ISO-8601 或 YYYY-MM-DD HH:MM:SS")
    bad_index = parsed.index[parsed.isna()].tolist()
    if bad_index:
        preview = ", ".join(str(index + 2) for index in bad_index[:MAX_TIME_ERRORS_REPORTED])
        suffix = f" 等 {len(bad_index)} 处" if len(bad_index) > MAX_TIME_ERRORS_REPORTED else ""
        raise ValueError(f"以下 CSV 行的时间无法解析（行号含表头偏移）：{preview}{suffix}")
    return parsed.dt.tz_convert("UTC").dt.tz_localize(None)


def _enforce_time_contract(numeric: "pd.DataFrame", raw: "pd.DataFrame") -> tuple["pd.DataFrame", dict[str, Any]]:
    """Validate the time column and rebuild frame ordered by it.

    Returns (ordered_frame, contract_info). Rejects duplicates, reordered rows
    are accepted only after an explicit sort (row order is not trusted), and
    hour gaps are reported. The manifest records everything needed to audit
    the training window order. ``raw`` keeps the original strings because the
    numeric projection cannot hold timestamps.
    """
    column = next((name for name in raw.columns if name.lower() in TIME_COLUMN_CANDIDATES), None)
    if column is None:
        raise ValueError(
            "CSV 必须包含一个时间列（" + "/".join(TIME_COLUMN_CANDIDATES) +
            "），行号顺序不能证明 t→t+h 的样本时序")
    moments = _parse_time_column(raw[column])

    duplicated = int(moments.duplicated().sum())
    if duplicated:
        raise ValueError(f"时间列存在 {duplicated} 个重复时间戳，无法构造可靠的 t→t+h 样本")

    ordered = numeric.assign(_moment=moments).sort_values("_moment", kind="stable")
    gaps = ordered["_moment"].diff().dropna()
    gap_hours = gaps.dt.total_seconds() / 3600.0
    gap_stats = {
        "max_gap_hours": float(gap_hours.max()) if len(gap_hours) else 0.0,
        "non_hourly_steps": int((gap_hours != 1.0).sum()),
    }
    contract_info = {
        "time_column": column,
        "timezone": "UTC",
        "start": ordered["_moment"].iloc[0].isoformat(),
        "end": ordered["_moment"].iloc[-1].isoformat(),
        "rows": int(len(ordered)),
        "duplicates_removed": 0,
        "row_order_changed": bool((ordered.index.to_numpy() != np.arange(len(ordered))).any()),
        "gap_stats": gap_stats,
    }
    return ordered.drop(columns=["_moment"]), contract_info


def _read_and_validate_csv(path: Path, target: str, horizons: list[int],
                           enforce_time: bool = True) -> tuple[pd.DataFrame, list[str], dict[str, Any] | None]:
    try:
        frame = pd.read_csv(path, nrows=MAX_ROWS + 1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("CSV 无法解析或为空") from exc
    if frame.empty:
        raise ValueError("CSV 不能为空")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"CSV 最多允许 {MAX_ROWS} 行")
    if frame.shape[1] > MAX_COLUMNS:
        raise ValueError(f"CSV 最多允许 {MAX_COLUMNS} 列")
    if not target or target not in frame.columns:
        raise ValueError("target 必须是 CSV 中存在的列名")
    if frame.columns.duplicated().any():
        raise ValueError("CSV 列名不能重复")

    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric[target].notna().sum() < MIN_SAMPLES + max(horizons):
        raise ValueError(f"target 至少需要 {MIN_SAMPLES + max(horizons)} 个有限数值")
    features = [name for name in numeric.columns if numeric[name].notna().any()]
    if target not in features:
        raise ValueError("target 必须是数值列")
    # The target itself is retained as a lag-0 predictor for the persistence
    # comparison, so it does not consume one of the user feature slots.
    if not features or len(features) - 1 > MAX_FEATURES:
        raise ValueError(f"除 target 外的数值特征数最多为 {MAX_FEATURES}")
    time_contract: dict[str, Any] | None = None
    if enforce_time:
        numeric, time_contract = _enforce_time_contract(numeric, frame)
    return numeric, features, time_contract


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    r2: float | None
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        r2 = None
    else:
        r2 = float(r2_score(y_true, y_pred))
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": r2,
    }


class OnlineTrainingManager:
    """In-memory job registry with a single FIFO background worker."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="online-train")

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired: list[tuple[str, Path]] = []
        for job_id, job in self._jobs.items():
            if job["status"] in {"queued", "running"}:
                continue
            if now - job.get("finished_epoch", now) > JOB_TTL_SECONDS:
                expired.append((job_id, Path(job["work_dir"])))
        for job_id, folder in expired:
            self._jobs.pop(job_id, None)
            shutil.rmtree(folder, ignore_errors=True)

    def submit(self, upload, target: str, horizons: list[int]) -> dict[str, Any]:
        self._cleanup_for_request()
        with self._lock:
            queued = sum(1 for item in self._jobs.values() if item["status"] == "queued")
            if queued >= MAX_QUEUED_JOBS:
                raise TrainingQueueFull("训练队列已满，请稍后再试")
        job_id = str(uuid.uuid4())
        work_dir = TRAINING_FOLDER / job_id
        work_dir.mkdir(parents=True, exist_ok=False)
        source_path = work_dir / "input.csv"
        try:
            total = 0
            with source_path.open("wb") as output:
                while True:
                    chunk = upload.stream.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError(f"CSV 文件不能超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                    output.write(chunk)
            if total == 0:
                raise ValueError("CSV 不能为空")
            _, features, _ = _read_and_validate_csv(source_path, target, horizons)
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        job = {
            "job_id": job_id,
            "status": "queued",
            "deployment": "download-only",
            "target": target,
            "horizons": horizons,
            "feature_count": len(features) - 1,
            "created_at": utc_now(),
            "work_dir": str(work_dir),
            "source_path": str(source_path),
        }
        with self._lock:
            self._jobs[job_id] = job
            job["queue_position"] = sum(1 for item in self._jobs.values() if item["status"] == "queued")
        self._executor.submit(self._run, job_id)
        return self.public_job(job_id) or {"job_id": job_id, "status": "queued"}

    def _cleanup_for_request(self) -> None:
        with self._lock:
            self._cleanup_expired_locked()

    def public_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup_expired_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            hidden = {"work_dir", "source_path", "package_path", "finished_epoch"}
            return {key: value for key, value in job.items() if key not in hidden}

    def package_path(self, job_id: str) -> Path | None:
        with self._lock:
            self._cleanup_expired_locked()
            job = self._jobs.get(job_id)
            if not job or job["status"] != "completed":
                return None
            path = Path(job["package_path"])
            return path if path.is_file() and path.parent == Path(job["work_dir"]) else None

    def _set(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)

    def _run(self, job_id: str) -> None:
        started = time.monotonic()
        self._set(job_id, status="running", started_at=utc_now())
        try:
            with self._lock:
                job = self._jobs[job_id].copy()
            frame, features, time_contract = _read_and_validate_csv(
                Path(job["source_path"]), job["target"], job["horizons"])
            package_path, metrics = self._train(frame, features, job, started, time_contract)
            elapsed = round(time.monotonic() - started, 3)
            self._set(job_id, status="completed", metrics=metrics,
                      elapsed_seconds=elapsed, package_path=str(package_path),
                      completed_at=utc_now(), finished_epoch=time.time())
        except ValueError as exc:
            self._set(job_id, status="failed", error=str(exc)[:300],
                      finished_at=utc_now(), finished_epoch=time.time())
        except Exception:
            # Avoid exposing implementation paths or tracebacks through a public API.
            self._set(job_id, status="failed", error="训练任务执行失败", finished_at=utc_now(),
                      finished_epoch=time.time())
        finally:
            with self._lock:
                job = self._jobs.get(job_id)
                source = Path(job["source_path"]) if job else None
            if source:
                source.unlink(missing_ok=True)

    def _check_deadline(self, started: float) -> None:
        if time.monotonic() - started > MAX_TRAIN_SECONDS:
            raise ValueError(f"训练超过 {int(MAX_TRAIN_SECONDS)} 秒资源限制")

    def _train(self, frame: pd.DataFrame, features: list[str], job: dict[str, Any],
               started: float, time_contract: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
        target = job["target"]
        models: dict[str, Pipeline] = {}
        horizon_metrics: dict[str, Any] = {}
        for horizon in job["horizons"]:
            self._check_deadline(started)
            x = frame.loc[:, features].iloc[:-horizon].copy()
            y = frame[target].iloc[horizon:].to_numpy(dtype=float)
            persistence = frame[target].iloc[:-horizon].to_numpy(dtype=float)
            valid = np.isfinite(y) & np.isfinite(persistence)
            x, y, persistence = x.loc[valid], y[valid], persistence[valid]
            if len(y) < MIN_SAMPLES:
                raise ValueError(f"horizon={horizon} 的有效样本不足 {MIN_SAMPLES} 行")
            split = int(len(y) * 0.8)
            if split < 40 or len(y) - split < 10:
                raise ValueError(f"horizon={horizon} 无足够的时序验证样本")
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
            model.fit(x.iloc[:split], y[:split])
            predicted = model.predict(x.iloc[split:])
            baseline = persistence[split:]
            model_metrics = _metrics(y[split:], predicted)
            persistence_metrics = _metrics(y[split:], baseline)
            horizon_metrics[str(horizon)] = {
                "model": model_metrics,
                "persistence": persistence_metrics,
                "rmse_improvement": float(persistence_metrics["rmse"] - model_metrics["rmse"]),
            }
            models[str(horizon)] = model

        self._check_deadline(started)
        work_dir = Path(job["work_dir"])
        model_path = work_dir / "model.joblib"
        manifest_path = work_dir / "manifest.json"
        package_path = work_dir / "online_model.zip"
        package_manifest = {
            "format_version": 1,
            "model_type": "sklearn_ridge_direct_multihorizon",
            "deployment": "download-only",
            "target": target,
            "features": features,
            "horizons": job["horizons"],
            "trained_at": utc_now(),
            "limits": {"max_rows": MAX_ROWS, "max_train_seconds": MAX_TRAIN_SECONDS},
            "metrics": horizon_metrics,
            "time_contract": time_contract,
        }
        joblib.dump({"models": models, "manifest": package_manifest}, model_path)
        manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_path, arcname="model.joblib")
            archive.write(manifest_path, arcname="manifest.json")
        model_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        return package_path, {"validation": horizon_metrics, "model_type": package_manifest["model_type"]}


manager = OnlineTrainingManager()


LIFECYCLE_LABELS = {
    "deployed": "当前部署",
    "approved_candidate": "批准候选",
    "research_only": "研究中",
    "superseded": "历史作废",
    "invalid": "包不完整",
}
_DEFAULT_LIFECYCLE = "research_only"
# Smoke bundles are never listed, whatever the registry says.
_NEVER_LISTED_SUFFIX = "_smoke"


def _load_lifecycle_registry() -> dict[str, dict[str, Any]]:
    """Parse configs/model_registry.yaml without requiring PyYAML.

    FLASH-06: the lifecycle comes from this explicit registry, never from the
    directory being complete; an unlisted model defaults to research_only so
    a complete file set can no longer be advertised as available (F-003).
    """
    entries: dict[str, dict[str, Any]] = {}
    if not MODEL_REGISTRY_PATH.is_file():
        return entries
    try:
        text = MODEL_REGISTRY_PATH.read_text(encoding="utf-8")
    except OSError:
        return entries
    in_models = False
    current_model: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indented = raw[0] in (" ", "\t")
        stripped = raw.strip()
        if not indented:
            in_models = stripped == "models:"
            current_model = None
            continue
        if not in_models:
            continue
        if stripped.endswith(":") and ":" not in stripped[:-1]:
            current_model = stripped[:-1]
            entries[current_model] = {}
            continue
        if current_model is not None and ":" in stripped:
            key, _, value = stripped.partition(":")
            entries[current_model][key.strip()] = value.strip().strip('"').strip("'")
    return entries


def _registry_entry(name: str) -> dict[str, Any]:
    registry = _load_lifecycle_registry()
    entry = registry.get(name)
    if entry:
        return entry
    return {"lifecycle": _DEFAULT_LIFECYCLE, "note": "未在注册表中登记，默认按研究产物处理"}


def _scan_manifest(path: Path) -> dict[str, Any] | None:
    """Return a checked-in manifest only when the bundle is complete and real."""
    folder = path.parent
    required = ("manifest.json", "model.pt", "scaler.npz")
    if not all((folder / name).is_file() for name in required):
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("smoke") is True or folder.name.endswith(_NEVER_LISTED_SUFFIX):
        return None
    return manifest


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _model_metadata(manifest: dict[str, Any], folder: Path) -> dict[str, Any] | None:
    """Map supported manifest shapes to the public model-registry contract."""
    architecture = manifest.get("architecture")
    if not isinstance(architecture, dict):
        architecture = {}
    horizon = _safe_int(architecture.get("horizon") or manifest.get("horizon"))
    input_steps = _safe_int(architecture.get("input_steps") or manifest.get("input_steps"))
    outputs = manifest.get("pred_names")
    stations = manifest.get("stations")
    inputs = manifest.get("feature_order")
    if not isinstance(outputs, list) or not outputs or not isinstance(stations, list) or not stations:
        return None
    if not isinstance(inputs, list) or not inputs:
        inputs = list(outputs)
    train_range = manifest.get("train_range") or manifest.get("train_time_range")
    metrics = manifest.get("evaluation_metrics") or manifest.get("test_metrics")
    # FLASH-06: technical completeness and approval state are separate fields;
    # the deprecated blanket "available" status is no longer emitted.
    entry = _registry_entry(folder.name)
    lifecycle = entry.get("lifecycle", _DEFAULT_LIFECYCLE)
    if lifecycle not in LIFECYCLE_LABELS:
        lifecycle = _DEFAULT_LIFECYCLE
    return {
        "id": folder.name,
        "kind": "pretrained",
        "model_type": manifest.get("model_type") or manifest.get("run") or folder.name,
        "lifecycle": lifecycle,
        "lifecycle_label": LIFECYCLE_LABELS[lifecycle],
        "lifecycle_note": entry.get("note", ""),
        "deployed": lifecycle == "deployed",
        "downloadable": False,
        "version": str(manifest.get("model_sha256") or "")[:12] or None,
        "name": manifest.get("name") or manifest.get("run") or folder.name,
        "train_range": train_range,
        "stations": stations,
        "inputs": inputs,
        "outputs": outputs,
        "input_steps": input_steps,
        "horizons": list(range(1, horizon + 1)) if horizon else None,
        "evaluation_metrics": metrics,
        "created_utc": manifest.get("created_utc"),
        "model_sha256": manifest.get("model_sha256"),
    }


@online_bp.get("/models")
def models():
    """List checked-in pretrained bundles with their approval lifecycle.

    FLASH-06: bundles missing model.pt/scaler.npz are listed as invalid (not
    silently hidden) when the registry knows them, and no bundle is ever
    reported with the old blanket `available` status.
    """
    items: list[dict[str, Any]] = []
    if not MODEL_DIR.is_dir():
        return jsonify({"models": items, "timestamp": utc_now()})
    for folder in sorted(MODEL_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.endswith(_NEVER_LISTED_SUFFIX):
            continue
        manifest = _scan_manifest(folder / "manifest.json")
        if manifest is None:
            items.append({
                "id": folder.name,
                "kind": "pretrained",
                "lifecycle": "invalid",
                "lifecycle_label": LIFECYCLE_LABELS["invalid"],
                "lifecycle_note": "模型包不完整或清单不可解析，未通过健康校验",
                "deployed": False,
                "downloadable": False,
                "name": folder.name,
            })
            continue
        try:
            item = _model_metadata(manifest, folder)
        except (OSError, ValueError, TypeError):
            item = None
        if item is None:
            # A malformed bundle is not advertised as usable.
            continue
        items.append(item)
    return jsonify({"models": items, "timestamp": utc_now()})


@online_bp.post("/train")
def submit_training():
    if request.content_length and request.content_length > MAX_UPLOAD_BYTES + 16 * 1024:
        return jsonify({"error": f"请求不能超过约 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}), 413
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "请以 multipart/form-data 的 file 字段上传 CSV"}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"error": "只支持 .csv 文件"}), 400
    target = (request.form.get("target") or "").strip()
    try:
        horizons = _parse_horizons(request.form.get("horizons"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if _rate_limit_exceeded():
        return jsonify({"error": f"训练提交频率超过每分钟 {MAX_SUBMITS_PER_MINUTE} 次，请稍后再试"}), 429
    if _runtime_disk_bytes() > MAX_RUNTIME_DISK_BYTES:
        return jsonify({"error": "服务器训练磁盘配额已满，请清理后重试"}), 507
    try:
        job = manager.submit(upload, target, horizons)
    except TrainingQueueFull as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError:
        return jsonify({"error": "训练临时目录不可用"}), 503
    return jsonify({
        "job_id": job["job_id"], "status": job["status"],
        "deployment": job.get("deployment", "download-only"),
        "status_url": f"/api/train/{job['job_id']}",
        "download_url": f"/api/train/{job['job_id']}/download",
        "limits": {"max_rows": MAX_ROWS, "max_features": MAX_FEATURES,
                   "max_horizons": MAX_HORIZONS, "max_train_seconds": MAX_TRAIN_SECONDS},
    }), 202


def _valid_job_id(job_id: str) -> bool:
    return bool(JOB_ID_RE.fullmatch(job_id))


@online_bp.get("/train/<job_id>")
def training_status(job_id: str):
    if not _valid_job_id(job_id):
        return jsonify({"error": "任务不存在"}), 404
    job = manager.public_job(job_id)
    if job is None:
        return jsonify({"error": "任务不存在或已过期"}), 404
    return jsonify(job)


@online_bp.get("/train/<job_id>/download")
def download_training(job_id: str):
    if not _valid_job_id(job_id):
        return jsonify({"error": "任务不存在"}), 404
    job = manager.public_job(job_id)
    if job is None:
        return jsonify({"error": "任务不存在或已过期"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "训练尚未完成"}), 409
    package = manager.package_path(job_id)
    if package is None:
        return jsonify({"error": "模型包不存在或已过期"}), 404
    return send_file(package, as_attachment=True,
                     download_name=f"online_model_{job_id}.zip", mimetype="application/zip")
