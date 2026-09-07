"""Historical four-fold rolling backtest for plain/engineered LightGBM.

The fixed hybrid uses plain PM2.5 and engineered PM10/NO2/O3.  Its rule is
defined before this backtest and is never changed per fold.  All score periods
end by 2024-12-31, before the project's observed 2025-2026 development test.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import pipeline as pl  # noqa: E402
from src.eval.feature_lgbm import (  # noqa: E402
    add_lgbm_tuning_args,
    block_bootstrap_delta,
    build_features,
    lgbm_params_from_args,
    metrics,
    parse_daily_lags,
    sha256,
    targets,
)
from src.eval.rolling_pm25_residual_lgbm import (  # noqa: E402
    FOLDS,
    select_fold_windows,
)


def fit_lgbm(x_train, y_train, ok_train, x_tune, y_tune, ok_tune,
             args, seed):
    model = lgb.LGBMRegressor(**lgbm_params_from_args(args, seed))
    model.fit(
        x_train[ok_train], y_train[ok_train],
        eval_set=[(x_tune[ok_tune], y_tune[ok_tune])], eval_metric="l1",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)])
    return model


def aggregate_results(folds):
    output = {}
    models = tuple(folds[0]["metrics"])
    for h in range(pl.HORIZON):
        horizon = f"T+{h + 1}"
        output[horizon] = {"models": {}, "pollutant_decisions": {}}
        for model in models:
            model_rows = [fold["metrics"][model][horizon] for fold in folds]
            output[horizon]["models"][model] = {}
            for metric in ("macro_pollutant_mae", "macro_pollutant_r2",
                           "legacy_joint_r2"):
                values = np.asarray([row[metric] for row in model_rows], dtype=float)
                output[horizon]["models"][model][f"{metric}_mean"] = float(values.mean())
                output[horizon]["models"][model][f"{metric}_std"] = float(
                    values.std(ddof=1))

        if {"plain", "engineered"}.issubset(models):
            for pollutant in pl.PRED_NAMES:
                plain_mae = np.asarray([
                    fold["metrics"]["plain"][horizon][pollutant]["mae"]
                    for fold in folds])
                eng_mae = np.asarray([
                    fold["metrics"]["engineered"][horizon][pollutant]["mae"]
                    for fold in folds])
                plain_r2 = np.asarray([
                    fold["metrics"]["plain"][horizon][pollutant]["r2"]
                    for fold in folds])
                eng_r2 = np.asarray([
                    fold["metrics"]["engineered"][horizon][pollutant]["r2"]
                    for fold in folds])
                output[horizon]["pollutant_decisions"][pollutant] = {
                    "engineered_mae_better_folds": int(np.sum(eng_mae < plain_mae)),
                    "engineered_r2_better_folds": int(np.sum(eng_r2 > plain_r2)),
                    "mean_mae_delta_engineered_minus_plain": float(
                        np.mean(eng_mae - plain_mae)),
                    "mean_r2_delta_engineered_minus_plain": float(
                        np.mean(eng_r2 - plain_r2)),
                }

            plain_macro = np.asarray([
                fold["metrics"]["plain"][horizon]["macro_pollutant_r2"]
                for fold in folds])
            hybrid_macro = np.asarray([
                fold["metrics"]["hybrid"][horizon]["macro_pollutant_r2"]
                for fold in folds])
            output[horizon]["hybrid_decision"] = {
                "macro_r2_better_than_plain_folds": int(
                    np.sum(hybrid_macro > plain_macro)),
                "mean_macro_r2_delta_hybrid_minus_plain": float(
                    np.mean(hybrid_macro - plain_macro)),
                "advance_rule_3_of_4": bool(np.sum(hybrid_macro > plain_macro) >= 3),
            }
    return output


def flatten_fold_metrics(fold_name, all_metrics):
    rows = []
    for model, horizons in all_metrics.items():
        for horizon, result in horizons.items():
            for pollutant in pl.PRED_NAMES:
                rows.append({
                    "fold": fold_name,
                    "model": model,
                    "horizon": horizon,
                    "pollutant": pollutant,
                    **result[pollutant],
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multistep_2022_2026.yaml")
    ap.add_argument("--drop-station", default="")
    ap.add_argument("--only", default="plain,engineered",
                    help="comma-separated variants: plain,engineered")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    add_lgbm_tuning_args(ap)
    ap.add_argument("--daily-lags", default="",
                    help="causal target-clock lag hours, e.g. 24,168")
    ap.add_argument(
        "--output-dir",
        default="outputs/experiments/feature_lgbm_rolling4_historical")
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()
    daily_lags = parse_daily_lags(args.daily_lags)
    variants = tuple(x.strip() for x in args.only.split(",") if x.strip())
    if not variants or set(variants) - {"plain", "engineered"}:
        raise ValueError("--only must contain plain and/or engineered")

    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    pl.validate_window_config(cfg)
    tl = pl.build_timeline_arrays(
        ROOT / cfg["data"]["pollution"], ROOT / cfg["data"]["weather"],
        max_gap_hours=cfg["data"]["max_gap_hours"],
        weather_timezone=cfg["data"]["weather_timezone"])
    ws = pl.enumerate_windows(tl, cfg["data"]["max_missing_frac"])

    dropped = {x.strip() for x in args.drop_station.split(",") if x.strip()}
    unknown = dropped - set(pl.SEL_STATIONS)
    if unknown:
        raise ValueError(f"unknown stations: {sorted(unknown)}")
    station_indices = np.asarray(
        [i for i, station in enumerate(pl.SEL_STATIONS) if station not in dropped],
        dtype=np.int64)
    station_names = [pl.SEL_STATIONS[i] for i in station_indices]
    n_stations = len(station_names)

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        (out / "models").mkdir(exist_ok=True)

    folds_payload = []
    flat_rows = []
    for fold_number, fold in enumerate(FOLDS):
        train_ids, tune_ids, score_ids, i_train, split_audit = select_fold_windows(
            tl, ws, fold)
        scaler = pl.Scaler.fit(tl.x_raw, np.arange(len(tl.times)) <= i_train)
        true = np.empty(
            (len(score_ids), pl.HORIZON, n_stations, len(pl.PRED_IDX)),
            dtype=np.float64)
        mask = np.empty_like(true, dtype=bool)
        last = tl.x_filled[
            ws.starts[score_ids] + pl.INPUT_STEPS - 1
        ][:, station_indices][:, :, pl.PRED_IDX]
        persistence = np.repeat(last[:, None], pl.HORIZON, axis=1)
        predictions = {variant: np.empty_like(true) for variant in variants}
        best_iterations = {variant: {} for variant in variants}
        feature_counts = {}

        for h in range(pl.HORIZON):
            matrices = {}
            for variant in variants:
                engineered = variant == "engineered"
                x_train, feature_names = build_features(
                    tl, ws, train_ids, scaler, engineered, h,
                    station_indices, station_names, daily_lags=daily_lags)
                x_tune, _ = build_features(
                    tl, ws, tune_ids, scaler, engineered, h,
                    station_indices, station_names, daily_lags=daily_lags)
                x_score, _ = build_features(
                    tl, ws, score_ids, scaler, engineered, h,
                    station_indices, station_names, daily_lags=daily_lags)
                matrices[variant] = x_train, x_tune, x_score
                feature_counts[variant] = len(feature_names)

            for pollutant_index, pollutant in enumerate(pl.PRED_NAMES):
                y_train, ok_train = targets(
                    tl, ws, train_ids, h, pollutant_index, station_indices)
                y_tune, ok_tune = targets(
                    tl, ws, tune_ids, h, pollutant_index, station_indices)
                y_score, ok_score = targets(
                    tl, ws, score_ids, h, pollutant_index, station_indices)
                true[:, h, :, pollutant_index] = y_score.reshape(
                    len(score_ids), n_stations)
                mask[:, h, :, pollutant_index] = ok_score.reshape(
                    len(score_ids), n_stations)

                for variant in variants:
                    x_train, x_tune, x_score = matrices[variant]
                    seed = int(cfg["seed"]) + fold_number * 100 + h * 10 + pollutant_index
                    model = fit_lgbm(
                        x_train, y_train, ok_train, x_tune, y_tune, ok_tune,
                        args, seed)
                    pred = model.predict(
                        x_score, num_iteration=model.best_iteration_)
                    predictions[variant][:, h, :, pollutant_index] = pred.reshape(
                        len(score_ids), n_stations)
                    key = f"T+{h + 1}/{pollutant}"
                    best_iterations[variant][key] = int(model.best_iteration_)
                    if args.save_models:
                        joblib.dump(
                            model,
                            out / "models" /
                            f"fold_{fold['name']}_{variant}_h{h + 1}_{pollutant}.joblib")
            del matrices

        all_predictions = dict(predictions)
        if {"plain", "engineered"}.issubset(predictions):
            hybrid = predictions["engineered"].copy()
            pm25 = pl.PRED_NAMES.index("PM2.5")
            hybrid[..., pm25] = predictions["plain"][..., pm25]
            all_predictions["hybrid"] = hybrid
        all_metrics = {
            name: metrics(pred, true, mask, persistence, station_names)
            for name, pred in all_predictions.items()
        }
        fold_payload = {
            **fold,
            "split_audit": split_audit,
            "scaler_fit_end": str(tl.times[i_train]),
            "feature_counts": feature_counts,
            "best_iterations": best_iterations,
            "metrics": all_metrics,
        }
        if {"plain", "engineered"}.issubset(predictions):
            fold_payload["paired_block_bootstrap_engineered_vs_plain"] = (
                block_bootstrap_delta(
                    predictions["engineered"], predictions["plain"], true, mask,
                    seed=int(cfg["seed"]) + fold_number * 1000))
        folds_payload.append(fold_payload)
        flat_rows.extend(flatten_fold_metrics(fold["name"], all_metrics))

    payload = {
        "run": "feature_lgbm_rolling4_historical",
        "pipeline_revision": pl.PIPELINE_REVISION,
        "reportable_as_final_test": False,
        "protocol": (
            "four fixed rolling-origin folds; expanding train; later early-stop; "
            "full-window boundary purge; still-later outer score"),
        "hybrid_rule": (
            "plain PM2.5; engineered PM10/NO2/O3; fixed before rolling evaluation"),
        "source_files_contain_later_rows": True,
        "current_2025_2026_dev_test_windows_used": False,
        "rows_after_2024_12_31_used_for_fit_or_score": False,
        "stations": station_names,
        "dropped_stations": sorted(dropped),
        "variants": list(variants),
        "daily_lags": list(daily_lags),
        "lgbm_params": lgbm_params_from_args(args, int(cfg["seed"])),
        "seed_policy": "config seed + fold*100 + horizon*10 + pollutant",
        "folds": folds_payload,
        "aggregate": aggregate_results(folds_payload),
        "data_sha256": {
            "pollution": sha256(ROOT / cfg["data"]["pollution"]),
            "weather": sha256(ROOT / cfg["data"]["weather"]),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(flat_rows).to_csv(out / "fold_metrics.csv", index=False)
    (out / "config.snapshot.yaml").write_text(
        cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
