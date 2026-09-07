"""Low-cost, leakage-aware LightGBM development experiment.

The chronological validation period is split again: its first half is used
only for early stopping and its second half is the development score set.
The test period is never loaded by this script.  It compares the existing
same-station 12-hour representation with a feature-engineered variant that
adds only information available at forecast issue time.

Usage:
    py -3.13 src/eval/feature_lgbm.py
    py -3.13 src/eval/feature_lgbm.py --only engineered --n-estimators 800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import pipeline as pl  # noqa: E402


def add_lgbm_tuning_args(parser: argparse.ArgumentParser) -> None:
    """Expose one shared, auditable LightGBM search surface."""
    parser.add_argument("--objective", choices=("regression", "huber", "regression_l1"),
                        default="regression")
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.1)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--huber-alpha", type=float, default=0.9)


def lgbm_params_from_args(args, seed: int) -> dict:
    """Build deterministic LightGBM parameters shared by CV and dev runs."""
    if not 0 < args.learning_rate <= 1:
        raise ValueError("learning_rate must be in (0, 1]")
    if args.num_leaves < 2:
        raise ValueError("num_leaves must be at least 2")
    if args.min_child_samples < 1:
        raise ValueError("min_child_samples must be positive")
    if not 0 < args.subsample <= 1 or not 0 < args.colsample_bytree <= 1:
        raise ValueError("subsample and colsample_bytree must be in (0, 1]")
    if args.reg_alpha < 0 or args.reg_lambda < 0:
        raise ValueError("regularization values must be non-negative")
    if args.max_bin < 2:
        raise ValueError("max_bin must be at least 2")
    params = {
        "objective": args.objective,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1 if args.subsample < 1 else 0,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "n_jobs": -1,
        "random_state": seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if args.objective == "huber":
        if not 0 < args.huber_alpha < 1:
            raise ValueError("huber_alpha must be in (0, 1)")
        params["alpha"] = args.huber_alpha
    return params


def parse_daily_lags(value: str) -> tuple[int, ...]:
    """Parse causal target-clock lag hours such as ``24,168``."""
    if not value.strip():
        return ()
    lags = tuple(sorted({int(item.strip()) for item in value.split(",")
                         if item.strip()}))
    if not lags or any(lag < pl.HORIZON for lag in lags):
        raise ValueError(
            f"daily lags must be at least the {pl.HORIZON}-hour forecast horizon")
    return lags


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def setup(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pl.validate_window_config(cfg)
    tl = pl.build_timeline_arrays(
        ROOT / cfg["data"]["pollution"], ROOT / cfg["data"]["weather"],
        max_gap_hours=cfg["data"]["max_gap_hours"],
        weather_timezone=cfg["data"]["weather_timezone"])
    ws = pl.enumerate_windows(tl, cfg["data"]["max_missing_frac"])
    split = pl.split_from_config(tl, ws, cfg["data"])
    scaler = pl.Scaler.fit(tl.x_raw, pl.train_time_mask(tl, split))
    # A nested chronological split keeps the validation tail independent of
    # LightGBM early stopping and feature/hyperparameter selection.
    nested_cut = (split.i_tr + split.i_va) // 2
    val_lo = ws.starts[split.val]
    val_hi = val_lo + pl.WINDOW_LEN - 1
    tune_ids = split.val[val_hi <= nested_cut - 1]
    score_ids = split.val[val_lo > nested_cut - 1]
    if not len(tune_ids) or not len(score_ids):
        raise ValueError("validation split is too short for tune/score nesting")
    return cfg, tl, ws, split, scaler, tune_ids, score_ids


def _calendar(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    doy = timestamps.dayofyear.to_numpy()
    month = timestamps.month.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
    ]).astype(np.float32)


def build_features(tl, ws, ids, scaler, engineered: bool, horizon: int,
                   station_indices: np.ndarray, station_names: list[str],
                   daily_lags: tuple[int, ...] = ()):
    starts = ws.starts[ids]
    n, f, length = len(station_indices), len(pl.FEATURE_ORDER), pl.INPUT_STEPS
    idx = starts[:, None] + np.arange(length)[None, :]
    raw = tl.x_filled[idx][:, :, station_indices, :]  # [W,L,N,F], causal values
    norm_nan = ((raw - scaler.mu[station_indices]) /
                scaler.sd[station_indices]).astype(np.float32)
    norm = np.nan_to_num(norm_nan, nan=0.0, posinf=0.0, neginf=0.0)
    own = norm.transpose(0, 2, 1, 3).reshape(len(starts) * n, length * f)
    names = [f"own_lag{length-lag}_{name}" for lag in range(length)
             for name in pl.FEATURE_ORDER]
    if not engineered:
        return own.astype(np.float32), names

    blocks = [own]

    # Explicit observation state: model can distinguish a real train-mean
    # value from an imputed neutral zero in normalised space.
    obs = np.isfinite(tl.x_raw[idx][:, :, station_indices, :len(pl.POLL_TYPES)])
    obs_flat = obs.transpose(0, 2, 1, 3).reshape(len(starts) * n, -1).astype(np.float32)
    blocks.append(obs_flat)
    names += [f"observed_lag{length-lag}_{name}" for lag in range(length)
              for name in pl.POLL_TYPES]

    # Last-hour city field (all stations, six pollutants) supplies a cheap
    # spatial context without constructing a learned graph.
    spatial_last = norm[:, -1, :, :len(pl.POLL_TYPES)].reshape(len(starts), -1)
    blocks.append(np.repeat(spatial_last, n, axis=0))
    names += [f"last_{station}_{poll}" for station in station_names
              for poll in pl.POLL_TYPES]

    # Causal rolling summaries of each target station.
    for width in (3, 6, 12):
        segment = norm_nan[:, -width:]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(segment, axis=1)
            std = np.nanstd(segment, axis=1)
        slope = segment[:, -1] - segment[:, 0]
        for label, values in (("mean", mean), ("std", std), ("slope", slope)):
            blocks.append(np.nan_to_num(values, nan=0.0).reshape(len(starts) * n, f))
            names += [f"own_{label}{width}_{feature}" for feature in pl.FEATURE_ORDER]

    # City-wide concentration summaries are robust to a single station outage.
    for width in (3, 6, 12):
        segment = norm_nan[:, -width:, :, :len(pl.POLL_TYPES)]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(segment, axis=(1, 2))
            std = np.nanstd(segment, axis=(1, 2))
        summary = np.concatenate([mean, std], axis=1)
        blocks.append(np.repeat(np.nan_to_num(summary, nan=0.0), n, axis=0))
        names += ([f"city_mean{width}_{p}" for p in pl.POLL_TYPES] +
                  [f"city_std{width}_{p}" for p in pl.POLL_TYPES])

    # Wind direction is circular; u/v avoids the 0/360-degree discontinuity.
    wd = raw[..., pl.FEATURE_ORDER.index("wind_dir")]
    speed = raw[..., pl.FEATURE_ORDER.index("wind_spd")]
    rad = np.deg2rad(wd)
    uv = np.stack([-speed * np.sin(rad), -speed * np.cos(rad)], axis=-1)
    blocks.append(np.nan_to_num(uv, nan=0.0).transpose(0, 2, 1, 3).reshape(len(starts) * n, -1))
    names += [f"wind_{axis}_lag{length-lag}" for lag in range(length) for axis in ("u", "v")]

    station_ids = np.tile(np.arange(n), len(starts))
    blocks.append(np.eye(n, dtype=np.float32)[station_ids])
    names += [f"station_{s}" for s in station_names]
    coords = np.array([pl.STATION_COORDS[s] for s in station_names], dtype=np.float32)
    blocks.append(np.tile(coords, (len(starts), 1)))
    names += ["station_lon", "station_lat"]

    target_times = tl.times[starts + pl.INPUT_STEPS + horizon]
    blocks.append(np.repeat(_calendar(target_times), n, axis=0))
    names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
              "doy_sin", "doy_cos", "month_sin", "month_cos"]

    # Same target-clock observations from prior days/weeks. For every horizon,
    # target_row-lag is no later than the forecast issue row because accepted
    # lags are at least HORIZON. Missing early-history rows stay neutral and
    # are explicitly identified by the observation mask.
    poll_count = len(pl.POLL_TYPES)
    target_rows = starts + pl.INPUT_STEPS + horizon
    for lag in daily_lags:
        lag_rows = target_rows - lag
        valid_rows = lag_rows >= 0
        lag_raw = np.full((len(starts), n, poll_count), np.nan, dtype=np.float32)
        lag_observed = np.zeros_like(lag_raw, dtype=np.float32)
        if valid_rows.any():
            source_rows = lag_rows[valid_rows]
            lag_raw[valid_rows] = tl.x_filled[source_rows][:, station_indices, :poll_count]
            lag_observed[valid_rows] = np.isfinite(
                tl.x_raw[source_rows][:, station_indices, :poll_count])
        lag_norm = ((lag_raw - scaler.mu[station_indices, :poll_count]) /
                    scaler.sd[station_indices, :poll_count])
        blocks.append(np.nan_to_num(lag_norm, nan=0.0).reshape(len(starts) * n,
                                                               poll_count))
        blocks.append(lag_observed.reshape(len(starts) * n, poll_count))
        names += [f"own_target_clock_lag{lag}_{poll}" for poll in pl.POLL_TYPES]
        names += [f"observed_target_clock_lag{lag}_{poll}" for poll in pl.POLL_TYPES]

        city_lag = np.nan_to_num(lag_norm, nan=0.0).reshape(len(starts), n * poll_count)
        blocks.append(np.repeat(city_lag, n, axis=0))
        names += [f"city_target_clock_lag{lag}_{station}_{poll}"
                  for station in station_names for poll in pl.POLL_TYPES]
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def targets(tl, ws, ids, horizon, pollutant_slot, station_indices):
    rows = ws.starts[ids] + pl.INPUT_STEPS + horizon
    values = tl.x_raw[rows][:, station_indices, pl.PRED_IDX[pollutant_slot]].reshape(-1)
    return values, np.isfinite(values)


def metrics(pred, true, mask, persistence, station_names):
    result = {}
    for h in range(pl.HORIZON):
        step = {}
        pollutant_r2 = []
        pollutant_mae = []
        for k, name in enumerate(pl.PRED_NAMES):
            valid = mask[:, h, :, k] & np.isfinite(pred[:, h, :, k])
            yt, yp = true[:, h, :, k][valid], pred[:, h, :, k][valid]
            pooled = {
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "r2": float(r2_score(yt, yp)), "n": int(valid.sum()),
            }
            station_rows = []
            for station in range(len(station_names)):
                sv = valid[:, station]
                if sv.sum() < 24 or np.nanvar(true[:, h, station, k][sv]) < 1e-12:
                    continue
                sy = true[:, h, station, k][sv]
                sp = pred[:, h, station, k][sv]
                station_rows.append((mean_absolute_error(sy, sp),
                                     np.sqrt(mean_squared_error(sy, sp)),
                                     r2_score(sy, sp)))
            if station_rows:
                arr = np.asarray(station_rows)
                pooled.update(macro_station_mae=float(arr[:, 0].mean()),
                              macro_station_rmse=float(arr[:, 1].mean()),
                              macro_station_r2=float(arr[:, 2].mean()),
                              stations_scored=int(len(arr)))
            pv = valid & np.isfinite(persistence[:, h, :, k])
            base_mae = mean_absolute_error(true[:, h, :, k][pv], persistence[:, h, :, k][pv])
            model_mae = mean_absolute_error(true[:, h, :, k][pv], pred[:, h, :, k][pv])
            pooled["mae_skill_vs_persistence"] = float(1.0 - model_mae / base_mae)
            step[name] = pooled
            pollutant_r2.append(pooled["r2"])
            pollutant_mae.append(pooled["mae"])
        step["macro_pollutant_r2"] = float(np.mean(pollutant_r2))
        step["macro_pollutant_mae"] = float(np.mean(pollutant_mae))
        # Kept only for backward comparison; heterogeneous pollutant units make
        # this less interpretable than macro_pollutant_r2.
        joint = mask[:, h] & np.isfinite(pred[:, h])
        step["legacy_joint_r2"] = float(r2_score(true[:, h][joint], pred[:, h][joint]))
        result[f"T+{h + 1}"] = step
    return result


def block_bootstrap_delta(candidate, control, true, mask, reps=300, block=24, seed=53):
    rng = np.random.default_rng(seed)
    w = len(true)
    out = {}
    if w < block:
        return out
    for h in range(pl.HORIZON):
        out[f"T+{h + 1}"] = {}
        for k, name in enumerate(pl.PRED_NAMES):
            mae_delta, r2_delta = [], []
            valid = mask[:, h, :, k]
            for _ in range(reps):
                starts = rng.integers(0, w - block + 1, size=int(np.ceil(w / block)))
                ids = np.concatenate([np.arange(s, s + block) for s in starts])[:w]
                vv = valid[ids]
                yt = true[ids, h, :, k][vv]
                pc = candidate[ids, h, :, k][vv]
                pb = control[ids, h, :, k][vv]
                if len(yt) < 24:
                    continue
                mae_delta.append(mean_absolute_error(yt, pc) - mean_absolute_error(yt, pb))
                r2_delta.append(r2_score(yt, pc) - r2_score(yt, pb))
            out[f"T+{h + 1}"][name] = {
                "mae_delta_engineered_minus_plain_ci95": np.quantile(mae_delta, [.025, .975]).tolist(),
                "r2_delta_engineered_minus_plain_ci95": np.quantile(r2_delta, [.025, .975]).tolist(),
                "block_hours": block, "repetitions": reps,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multistep_2022_2026.yaml")
    ap.add_argument("--only", default="plain,engineered")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    add_lgbm_tuning_args(ap)
    ap.add_argument("--daily-lags", default="",
                    help="causal target-clock lag hours, e.g. 24,168")
    ap.add_argument("--output-dir", default="outputs/experiments/feature_lgbm_dev_purged")
    ap.add_argument("--drop-station", default="",
                    help="comma-separated station ablation, e.g. 1344A")
    args = ap.parse_args()
    daily_lags = parse_daily_lags(args.daily_lags)
    cfg_path = ROOT / args.config
    cfg, tl, ws, split, scaler, tune_ids, score_ids = setup(cfg_path)
    train_ids = split.train
    dropped = {x.strip() for x in args.drop_station.split(",") if x.strip()}
    unknown_stations = dropped - set(pl.SEL_STATIONS)
    if unknown_stations:
        raise ValueError(f"unknown stations to drop: {sorted(unknown_stations)}")
    station_indices = np.array(
        [i for i, name in enumerate(pl.SEL_STATIONS) if name not in dropped], dtype=np.int64)
    station_names = [pl.SEL_STATIONS[i] for i in station_indices]
    n_stations = len(station_names)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "models"
    model_dir.mkdir(exist_ok=True)
    variants = [x.strip() for x in args.only.split(",")]
    unknown = set(variants) - {"plain", "engineered"}
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    true = np.empty((len(score_ids), pl.HORIZON, n_stations, len(pl.PRED_IDX)))
    mask = np.empty_like(true, dtype=bool)
    last = tl.x_filled[ws.starts[score_ids] + pl.INPUT_STEPS - 1][:, station_indices][:, :, pl.PRED_IDX]
    persistence = np.repeat(last[:, None], pl.HORIZON, axis=1)
    predictions = {name: np.empty_like(true) for name in variants}
    best_iterations = {name: {} for name in variants}
    feature_counts = {}

    for h in range(pl.HORIZON):
        matrices = {}
        for variant in variants:
            engineered = variant == "engineered"
            x_train, feature_names = build_features(
                tl, ws, train_ids, scaler, engineered, h, station_indices, station_names,
                daily_lags=daily_lags)
            x_tune, _ = build_features(
                tl, ws, tune_ids, scaler, engineered, h, station_indices, station_names,
                daily_lags=daily_lags)
            x_score, _ = build_features(
                tl, ws, score_ids, scaler, engineered, h, station_indices, station_names,
                daily_lags=daily_lags)
            matrices[variant] = (x_train, x_tune, x_score, feature_names)
            feature_counts[variant] = len(feature_names)
        for k, pollutant in enumerate(pl.PRED_NAMES):
            y_train, ok_train = targets(tl, ws, train_ids, h, k, station_indices)
            y_tune, ok_tune = targets(tl, ws, tune_ids, h, k, station_indices)
            y_score, ok_score = targets(tl, ws, score_ids, h, k, station_indices)
            true[:, h, :, k] = y_score.reshape(len(score_ids), -1)
            mask[:, h, :, k] = ok_score.reshape(len(score_ids), -1)
            for variant in variants:
                x_train, x_tune, x_score, feature_names = matrices[variant]
                model = lgb.LGBMRegressor(
                    **lgbm_params_from_args(args, int(cfg["seed"])))
                model.fit(
                    x_train[ok_train], y_train[ok_train],
                    eval_set=[(x_tune[ok_tune], y_tune[ok_tune])],
                    eval_metric="l1",
                    callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)])
                pred = model.predict(x_score, num_iteration=model.best_iteration_)
                predictions[variant][:, h, :, k] = pred.reshape(len(score_ids), n_stations)
                key = f"T+{h + 1}/{pollutant}"
                best_iterations[variant][key] = int(model.best_iteration_)
                joblib.dump(model, model_dir / f"{variant}_h{h + 1}_{pollutant}.joblib")
        del matrices

    metric_predictions = dict(predictions)
    hybrid_rule = None
    if {"plain", "engineered"}.issubset(predictions):
        hybrid = predictions["engineered"].copy()
        hybrid[..., pl.PRED_NAMES.index("PM2.5")] = predictions["plain"][..., pl.PRED_NAMES.index("PM2.5")]
        metric_predictions["hybrid"] = hybrid
        hybrid_rule = "plain for PM2.5; engineered for PM10/NO2/O3 (fixed before any test evaluation)"

    result = {
        "run": "feature_lgbm_dev", "reportable_as_final_test": False,
        "pipeline_revision": pl.PIPELINE_REVISION,
        "selection_protocol": "train fit; validation first half early-stop; 15-hour straddle purge; validation second half score",
        "data_ranges": {
            "train_end": str(split.cut_train),
            "tune": [str(tl.times[ws.starts[tune_ids[0]]]),
                     str(tl.times[ws.starts[tune_ids[-1]] + pl.WINDOW_LEN - 1])],
            "score": [str(tl.times[ws.starts[score_ids[0]]]),
                      str(tl.times[ws.starts[score_ids[-1]] + pl.WINDOW_LEN - 1])],
            "test_windows_accessed": False,
            "nested_straddle_windows_purged": int(
                len(split.val) - len(tune_ids) - len(score_ids)),
        },
        "feature_counts": feature_counts, "best_iterations": best_iterations,
        "daily_lags": list(daily_lags),
        "lgbm_params": lgbm_params_from_args(args, int(cfg["seed"])),
        "stations": station_names, "dropped_stations": sorted(dropped),
        "hybrid_rule": hybrid_rule,
        "metrics": {name: metrics(pred, true, mask, persistence, station_names)
                    for name, pred in metric_predictions.items()},
        "data_sha256": {
            "pollution": sha256(ROOT / cfg["data"]["pollution"]),
            "weather": sha256(ROOT / cfg["data"]["weather"]),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if {"plain", "engineered"}.issubset(predictions):
        result["paired_block_bootstrap"] = block_bootstrap_delta(
            predictions["engineered"], predictions["plain"], true, mask)
    (out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.snapshot.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
