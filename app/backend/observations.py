"""Authenticated observation push into a bounded rolling window.

The product receives data the user already owns instead of scraping everything
itself (docs/archive/2026-09-03_项目现状与协作路线.md §五).  The server keeps only the hours needed
to drive the pretrained model and refuses anything that would silently change
the model's scope: unknown stations, unknown fields, malformed timestamps or
out-of-range values are reported per record rather than dropped quietly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import wraps
import csv
import hmac
import json
import logging
import math
import os
import tempfile
import uuid
from pathlib import Path
import threading
from typing import Any

from flask import Blueprint, g, jsonify, request

logger = logging.getLogger(__name__)

# FLASH-04: per-request request_id for error correlation without leaking paths.
def _request_id() -> str:
    existing = getattr(g, "qiongyu_request_id", None)
    if existing:
        return existing
    value = uuid.uuid4().hex[:12]
    g.qiongyu_request_id = value
    return value


def error_response(status: int, code: str, message: str, **extra: Any):
    """Stable JSON error envelope; never carries raw exceptions or paths."""
    body: dict[str, Any] = {
        "error": {"code": code, "message": message, "request_id": _request_id()},
    }
    body.update(extra)
    return jsonify(body), status


ROOT_DIR = Path(__file__).resolve().parents[2]
WINDOW_DIR = ROOT_DIR / "outputs" / "runtime" / "observations"
WINDOW_PATH = WINDOW_DIR / "window.json"
BUNDLE_DIR = ROOT_DIR / "models" / "checkpoints" / "multistep_v2"

# The model consumes a 12-hour input window; the buffer keeps enough history
# that a pushed hour can actually complete a window without storing full
# history, which the product explicitly promises not to do.
INPUT_STEPS = 12
MAX_WINDOW_HOURS = 72
MAX_BATCH_RECORDS = 240
MAX_BODY_BYTES = 1 * 1024 * 1024
ALLOWED_SKEW = timedelta(hours=1)
# Readiness is not just "12 keys exist": the newest complete hour must be this
# old at most, otherwise a client could satisfy the window with hours that are
# days apart and the forecast would silently run on stale input.
MAX_WINDOW_AGE = timedelta(hours=1)

# Plausible physical ceilings; anything above is a sensor/mapping error, not a
# reading we are willing to feed into a forecast.
POLLUTANT_MAX = 2000.0
WEATHER_BOUNDS = {
    "tmp_C": (-60.0, 60.0),
    "wind_dir": (0.0, 360.0),
    "wind_spd": (0.0, 75.0),
    "rh": (0.0, 100.0),
}

WEATHER_FIELDS = tuple(WEATHER_BOUNDS)
_write_lock = Lock = threading.Lock()
observations_bp = Blueprint("observations", __name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Contract:
    """What the deployed model is actually allowed to receive."""

    def __init__(self, stations: list[str], pollutants: list[str], weather: list[str]) -> None:
        self.stations = list(stations)
        self.pollutants = list(pollutants)
        self.weather = list(weather)

    @property
    def ready(self) -> bool:
        return bool(self.stations and self.pollutants)

    def describe(self) -> dict[str, Any]:
        return {
            "stations": self.stations,
            "pollutants": self.pollutants,
            "weather": self.weather,
            "input_steps": INPUT_STEPS,
        }


def load_contract() -> Contract:
    """Derive the input contract from the bundle manifest, never from the client."""
    manifest_path = BUNDLE_DIR / "manifest.json"
    if not manifest_path.is_file():
        return Contract([], [], [])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Contract([], [], [])
    features = list(manifest.get("feature_order", []))
    weather = [name for name in features if name in WEATHER_BOUNDS]
    pollutants = [name for name in features if name not in WEATHER_BOUNDS]
    return Contract(list(manifest.get("stations", [])), pollutants, weather)


def _load_window() -> dict[str, Any]:
    if not WINDOW_PATH.is_file():
        return {"updated_utc": None, "hours": {}}
    try:
        data = json.loads(WINDOW_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"updated_utc": None, "hours": {}}
    if not isinstance(data, dict) or not isinstance(data.get("hours"), dict):
        return {"updated_utc": None, "hours": {}}
    return data


def _save_window(window: dict[str, Any]) -> None:
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = WINDOW_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(window, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, WINDOW_PATH)


def _trim(window: dict[str, Any]) -> int:
    hours = window["hours"]
    if len(hours) <= MAX_WINDOW_HOURS:
        return 0
    ordered = sorted(hours)
    dropped = ordered[: len(ordered) - MAX_WINDOW_HOURS]
    for key in dropped:
        hours.pop(key, None)
    return len(dropped)


def _parse_hour(value: Any) -> datetime | None:
    """Accept an ISO-8601 stamp and snap it to a whole, timezone-aware hour."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        # The training grid is Beijing time; an unqualified stamp is ambiguous,
        # so we interpret it as Beijing instead of silently assuming UTC.
        moment = moment.replace(tzinfo=timezone(timedelta(hours=8)))
    if (moment.minute, moment.second, moment.microsecond) != (0, 0, 0):
        return None
    return moment


def _check_number(value: Any, low: float, high: float) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < low or number > high:
        return None
    return round(number, 4)


def _validate(record: Any, contract: Contract, now: datetime) -> tuple[str | None, dict | None, str | None]:
    """Return (rejection_reason, normalised_record, hour_key) for one payload item."""
    if not isinstance(record, dict):
        return "record_not_object", None, None

    moment = _parse_hour(record.get("time"))
    if moment is None:
        return "invalid_time", None, None
    if moment > now + ALLOWED_SKEW:
        return "future_time", None, None

    station = record.get("station")
    if not isinstance(station, str) or not station.strip():
        return "missing_station", None, None
    station = station.strip()
    if station not in contract.stations:
        # A new station cannot reuse a station-specific model: it needs
        # adaptation, transfer or retraining, so pushing it in is refused.
        return "unknown_station", None, None

    pollutants = record.get("pollutants", {})
    weather = record.get("weather", {})
    if not isinstance(pollutants, dict) or not isinstance(weather, dict):
        return "measurements_not_object", None, None

    unknown = [name for name in list(pollutants) + list(weather) if name not in contract.pollutants + contract.weather]
    if unknown:
        # Extra columns are refused too: the deployed model has a fixed feature
        # order, so accepting them would imply a capability it does not have.
        return "unknown_field", None, None

    clean_pollutants: dict[str, float] = {}
    for name, value in pollutants.items():
        if name not in contract.pollutants:
            continue
        number = _check_number(value, 0.0, POLLUTANT_MAX)
        if number is None:
            return f"invalid_value:{name}", None, None
        clean_pollutants[name] = number
    clean_weather: dict[str, float] = {}
    for name, value in weather.items():
        if name not in contract.weather:
            continue
        low, high = WEATHER_BOUNDS[name]
        number = _check_number(value, low, high)
        if number is None:
            return f"invalid_value:{name}", None, None
        clean_weather[name] = number

    if not clean_pollutants and not clean_weather:
        return "empty_record", None, None

    normalised = {"station": station, "pollutants": clean_pollutants, "weather": clean_weather}
    return None, normalised, moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def _parse_hour_key(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _required_fields(contract: Contract) -> list[str]:
    """All fields a record must carry for the model to be usable."""
    return list(contract.pollutants) + list(contract.weather)


def _hour_is_complete(payload: Any, contract: Contract) -> tuple[bool, list[str]]:
    """A stored hour is complete when EVERY station carries EVERY required field.

    Previously readiness only checked that all station keys existed, so a
    12h x 10-station window with only PM2.5 per record was reported ready and
    then failed inside the predictor with a 500 (F-015).
    """
    if not isinstance(payload, dict):
        return False, ["hour_payload_not_object"]
    missing: list[str] = []
    for station in contract.stations:
        entry = payload.get(station)
        if not isinstance(entry, dict):
            missing.append(f"missing_station:{station}")
            continue
        for field in _required_fields(contract):
            group = "pollutants" if field in contract.pollutants else "weather"
            values = entry.get(group)
            if not isinstance(values, dict) or field not in values:
                missing.append(f"missing_field:{station}:{field}")
    return not missing, missing[:20]


def _complete_keys(hours: dict[str, Any], contract: Contract) -> list[str]:
    ordered = sorted(hours)
    return [key for key in ordered if _hour_is_complete(hours[key], contract)[0]]


def _coverage(window: dict[str, Any], contract: Contract) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    hours = window["hours"]
    ordered = sorted(hours)
    ordered_complete = _complete_keys(hours, contract)

    consecutive = 0
    previous = None
    latest = None
    ready = False
    reasons: list[str] = []
    for key in ordered_complete:
        moment = _parse_hour_key(key)
        if previous is None or moment == previous + timedelta(hours=1):
            consecutive += 1
        else:
            consecutive = 1
        previous = moment
        latest = moment
        if consecutive >= INPUT_STEPS and now - moment <= MAX_WINDOW_AGE:
            ready = True

    latest_age_minutes = None
    if latest is not None:
        latest_age_minutes = max(0, int((now - latest).total_seconds() // 60))

    if len(ordered) < INPUT_STEPS:
        reasons.append(f"missing_hours:{len(ordered)}_of_{INPUT_STEPS}")
    if len(ordered_complete) < INPUT_STEPS:
        reasons.append(f"incomplete_hours:{len(ordered_complete)}_of_{INPUT_STEPS}")
    if not ordered_complete:
        reasons.append("no_complete_hours")
    elif latest is not None and now - latest > MAX_WINDOW_AGE:
        reasons.append("window_stale")

    # Collect concrete missing-station/field samples (max 8) for the reason.
    sample_issues: list[str] = []
    for key in ordered[-3:]:
        _, issues = _hour_is_complete(hours[key], contract)
        sample_issues.extend(issues)
        if len(sample_issues) >= 8:
            break
    if sample_issues and not ready:
        reasons.append("field_issues:" + ";".join(sample_issues))

    return {
        "hours_stored": len(hours),
        "complete_hours": len(ordered_complete),
        "consecutive_complete_hours": consecutive,
        "hours_needed": INPUT_STEPS,
        "earliest": ordered[0] if ordered else None,
        "latest": ordered[-1] if ordered else None,
        "latest_age_minutes": latest_age_minutes,
        "ready_for_prediction": bool(contract.ready and ready),
        "reasons": reasons,
    }


# The training grid is Beijing time (UTC+8, no DST), so a fixed offset is the
# faithful conversion when the window re-emits its UTC hour keys.
BEIJING = timezone(timedelta(hours=8))
WEATHER_CSV_STATIONS = ("57687099999", "59287199999")


def _latest_prediction_run(hours: dict[str, Any], contract: Contract | None = None) -> list[str] | None:
    """Hour keys of the newest >= INPUT_STEPS consecutive complete run.

    Uses the SAME per-field completeness rule as _coverage (FLASH-01): a run
    only counts when every station carries every required pollutant+weather
    field, so readiness and prediction can never disagree.
    """
    contract = contract or load_contract()
    expected = set(contract.stations)
    if not expected:
        return None
    complete = _complete_keys(hours, contract)
    run: list[tuple[str, datetime]] = []
    previous: datetime | None = None
    for key in complete:
        moment = _parse_hour_key(key)
        if previous is not None and moment == previous + timedelta(hours=1):
            run.append((key, moment))
        else:
            run = [(key, moment)]
        previous = moment
    if len(run) < INPUT_STEPS:
        return None
    if datetime.now(timezone.utc) - run[-1][1] > MAX_WINDOW_AGE:
        return None
    return [key for key, _ in run[-INPUT_STEPS:]]


def _dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Invert the same Magnus relation load_weather_long applies to TMP/DEWP."""
    a, b = 17.27, 237.7
    rh = min(100.0, max(1.0, rh_pct))
    g = math.log(rh / 100.0) + a * temp_c / (b + temp_c)
    return b * g / (a - g)


def _mean_field(payload: dict[str, Any], group: str, field: str) -> float | None:
    values = [station[group][field] for station in payload.values()
              if isinstance(station.get(group), dict) and field in station[group]]
    return sum(values) / len(values) if values else None


def _write_prediction_inputs(window: dict[str, Any], contract: Contract,
                             out_dir: Path) -> tuple[Path, Path]:
    """Materialise the recent complete run as the two upload-format CSVs."""
    keys = _latest_prediction_run(window["hours"], contract)
    if keys is None:
        raise ValueError("窗口未形成最近连续完整的输入小时")
    out_dir.mkdir(parents=True, exist_ok=True)
    pollution_path = out_dir / "window_pollution.csv"
    weather_path = out_dir / "window_weather.csv"

    hours = window["hours"]
    with pollution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "hour", "type", *contract.stations])
        for key in keys:
            beijing = _parse_hour_key(key).astimezone(BEIJING)
            payload = hours[key]
            for pollutant in contract.pollutants:
                row = [beijing.strftime("%Y%m%d"), beijing.hour, pollutant]
                row.extend(
                    payload[station]["pollutants"].get(pollutant, "")
                    for station in contract.stations
                )
                writer.writerow(row)

    with weather_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["STATION", "DATE", "TMP", "DEWP", "WND"])
        for key in keys:
            beijing = _parse_hour_key(key).astimezone(BEIJING)
            payload = hours[key]
            tmp = _mean_field(payload, "weather", "tmp_C")
            rh = _mean_field(payload, "weather", "rh")
            wind_dir = _mean_field(payload, "weather", "wind_dir")
            wind_spd = _mean_field(payload, "weather", "wind_spd")
            stamp = beijing.strftime("%Y-%m-%dT%H:%M:%S")
            tmp_cell = f"{tmp * 10:+.0f},1" if tmp is not None else ""
            dew_cell = ""
            if tmp is not None and rh is not None:
                dew_cell = f"{_dew_point_c(tmp, rh) * 10:+.0f},1"
            wnd_cell = ""
            if wind_dir is not None and wind_spd is not None:
                wnd_cell = f"{wind_dir:.0f},1,N,{wind_spd * 10:.0f},1"
            # Prototype approximation: one averaged surface reading is copied to
            # both configured stations; it is not two independent observations.
            for station in WEATHER_CSV_STATIONS:
                writer.writerow([station, stamp, tmp_cell, dew_cell, wnd_cell])
    return pollution_path, weather_path


def _authenticated() -> bool:
    token = os.environ.get("QIONGYU_API_TOKEN", "").strip()
    if not token:
        return False
    supplied = request.headers.get("Authorization", "")
    if not supplied.startswith("Bearer "):
        return False
    return hmac.compare_digest(supplied[len("Bearer "):].strip(), token)


def require_token(handler):
    """Push endpoints fail closed: no configured token means no ingestion."""

    @wraps(handler)
    def wrapper(*args, **kwargs):
        if not os.environ.get("QIONGYU_API_TOKEN", "").strip():
            return error_response(503, "token_not_configured",
                                  "服务未配置 QIONGYU_API_TOKEN，数据接收接口已关闭")
        if not _authenticated():
            return error_response(401, "unauthorized", "缺少或错误的 Bearer 令牌")
        return handler(*args, **kwargs)

    return wrapper


@observations_bp.post("/v1/observations")
@require_token
def push_observations():
    """Append one or more hourly readings to the rolling prediction window."""
    if request.content_length and request.content_length > MAX_BODY_BYTES:
        return error_response(413, "body_too_large",
                              f"请求体不能超过 {MAX_BODY_BYTES // 1024} KB")

    # A chunked request has no Content-Length, so the header check alone can be
    # bypassed; read the actual body (cached, so request.get_json() below still
    # works) and enforce the same ceiling on the bytes we really received.
    raw_body = request.get_data(cache=True)
    if len(raw_body) > MAX_BODY_BYTES:
        return error_response(413, "body_too_large",
                              f"请求体不能超过 {MAX_BODY_BYTES // 1024} KB")

    contract = load_contract()
    if not contract.ready:
        return error_response(503, "bundle_unavailable", "模型包不可用，无法确定输入契约")

    body = request.get_json(silent=True)
    if isinstance(body, dict) and "observations" in body:
        records = body.get("observations")
    elif isinstance(body, list):
        records = body
    else:
        records = [body] if isinstance(body, dict) else None
    if not isinstance(records, list) or not records:
        return error_response(400, "invalid_payload",
                              "请提交 observations 列表或单个观测对象")
    if len(records) > MAX_BATCH_RECORDS:
        return error_response(413, "batch_too_large",
                              f"单次最多推送 {MAX_BATCH_RECORDS} 条记录")

    now = datetime.now(timezone.utc)
    accepted, rejected, superseded, dropped = 0, [], 0, 0
    with Lock:
        # FLASH-03: the read happens INSIDE the same lock as the merge/trim/save,
        # so two concurrent pushes cannot both read the pre-write window and then
        # clobber each other's accepted records (F-016).
        window = _load_window()
        for index, record in enumerate(records):
            reason, normalised, hour_key = _validate(record, contract, now)
            if reason is not None:
                station = record.get("station") if isinstance(record, dict) else None
                rejected.append({"index": index, "station": station, "reason": reason})
                continue
            existing = window["hours"].setdefault(hour_key, {})
            if normalised["station"] in existing:
                superseded += 1
            existing[normalised["station"]] = normalised
            accepted += 1
        dropped = _trim(window)
        window["updated_utc"] = utc_now()
        if accepted:
            _save_window(window)

    return jsonify({
        "accepted": accepted,
        "rejected": rejected,
        "superseded": superseded,
        "dropped_outside_window": dropped,
        "contract": contract.describe(),
        "window": _coverage(window, contract),
        "timestamp": utc_now(),
    }), (200 if accepted else 422)


@observations_bp.get("/v1/observations/window")
def get_window():
    """Public, read-only coverage view so users can see what is missing."""
    contract = load_contract()
    window = _load_window()
    return jsonify({
        "coverage": _coverage(window, contract),
        "contract": contract.describe(),
        "limits": {"max_window_hours": MAX_WINDOW_HOURS,
                   "max_batch_records": MAX_BATCH_RECORDS},
        "updated_utc": window.get("updated_utc"),
        "timestamp": utc_now(),
    }), 200


@observations_bp.post("/v1/observations/predict")
@require_token
def predict_from_window():
    """Forecast from the rolling window once it holds recent, complete input."""
    contract = load_contract()
    if not contract.ready:
        return error_response(503, "bundle_unavailable", "模型包不可用，无法确定输入契约")

    window = _load_window()
    coverage = _coverage(window, contract)
    if not coverage["ready_for_prediction"]:
        return error_response(409, "window_not_ready",
                              "滚动窗口尚未形成最近 12 小时连续完整输入",
                              coverage=coverage)

    # FLASH-04: every request gets its own TemporaryDirectory, so concurrent
    # window predictions can never share or delete each other's input files
    # (F-017). The directory cleans itself up on both success and failure.
    with tempfile.TemporaryDirectory(prefix="qiongyu-window-", dir=str(WINDOW_DIR)) as tmp:
        tmp_dir = Path(tmp)
        try:
            pollution_path, weather_path = _write_prediction_inputs(
                window, contract, tmp_dir)
            from app.backend.predict import build_prediction_payload, get_predictor
            predictor = get_predictor()
            result = predictor.predict_with_timeline(pollution_path, weather_path)
            payload = build_prediction_payload(predictor, result, session_id="observations",
                                               write_result=False)
            payload["download_url"] = None
            payload["source"] = "observation_window"
            return jsonify(payload), 200
        except Exception:
            logger.exception("window prediction failed request_id=%s", _request_id())
            return error_response(500, "window_prediction_failed", "窗口预测执行失败，请稍后重试")


@observations_bp.delete("/v1/observations/window")
@require_token
def clear_window():
    """Drop the stored window; pushed data is ephemeral by design."""
    with Lock:
        _save_window({"updated_utc": utc_now(), "hours": {}})
    return jsonify({"cleared": True, "timestamp": utc_now()}), 200
