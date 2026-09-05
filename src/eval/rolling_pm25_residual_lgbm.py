"""Four-fold rolling-origin backtest for the PM2.5 residual LightGBM.

Every fold uses an expanding training range, a later early-stopping range,
and a still later outer score range.  Windows that straddle either boundary
are discarded.  All score ranges end in 2024, so the already-observed
2025-2026 development test period is not used by this backtest.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import pipeline as pl  # noqa: E402
from src.eval.feature_lgbm import build_features, sha256, targets  # noqa: E402
from src.eval.pm25_residual_lgbm import (  # noqa: E402
    anchor,
    fit_model,
    paired_bootstrap,
    score,
)


FOLDS = (
    {
        "name": "A",
        "train_end": "2023-09-30 23:00:00",
        "tune_end": "2023-12-31 23:00:00",
        "score_end": "2024-03-31 23:00:00",
    },
    {
        "name": "B",
        "train_end": "2023-12-31 23:00:00",
        "tune_end": "2024-03-31 23:00:00",
        "score_end": "2024-06-30 23:00:00",
    },
    {
        "name": "C",
        "train_end": "2024-03-31 23:00:00",
        "tune_end": "2024-06-30 23:00:00",
        "score_end": "2024-09-30 23:00:00",
    },
    {
        "name": "D",
        "train_end": "2024-06-30 23:00:00",
        "tune_end": "2024-09-30 23:00:00",
        "score_end": "2024-12-31 23:00:00",
    },
)


def _time_index(times: pd.DatetimeIndex, stamp: str) -> int:
    target = pd.Timestamp(stamp)
    location = times.get_indexer([target])[0]
    if location < 0:
        raise ValueError(f"fold boundary is absent from hourly timeline: {stamp}")
    return int(location)


def select_fold_windows(tl, ws, fold):
    """Return non-overlapping full-window train/tune/score indices."""
    i_train = _time_index(tl.times, fold["train_end"])
    i_tune = _time_index(tl.times, fold["tune_end"])
    i_score = _time_index(tl.times, fold["score_end"])
    if not i_train < i_tune < i_score:
        raise ValueError(f"invalid chronological fold: {fold}")

    starts = ws.starts
    ends = starts + pl.WINDOW_LEN - 1
    train = np.flatnonzero(ends <= i_train)
    tune = np.flatnonzero((starts > i_train) & (ends <= i_tune))
    score_ids = np.flatnonzero((starts > i_tune) & (ends <= i_score))
    if not len(train) or not len(tune) or not len(score_ids):
        raise ValueError(f"empty fold partition: {fold['name']}")

    assert ends[train].max() < starts[tune].min()
    assert ends[tune].max() < starts[score_ids].min()
    used = np.concatenate([train, tune, score_ids])
    assert len(np.unique(used)) == len(used)

    audit = {
        "train_windows": int(len(train)),
        "tune_windows": int(len(tune)),
        "score_windows": int(len(score_ids)),
        "train_range": [str(tl.times[starts[train[0]]]),
                        str(tl.times[ends[train[-1]]])],
        "tune_range": [str(tl.times[starts[tune[0]]]),
                       str(tl.times[ends[tune[-1]]])],
        "score_range": [str(tl.times[starts[score_ids[0]]]),
                        str(tl.times[ends[score_ids[-1]]])],
        "train_tune_straddle_windows_purged": int(
            np.sum((starts <= i_train) & (ends > i_train))),
        "tune_score_straddle_windows_purged": int(
            np.sum((starts <= i_tune) & (ends > i_tune))),
        "window_sets_disjoint": True,
        "maximum_referenced_time": str(tl.times[ends[score_ids].max()]),
    }
    return train, tune, score_ids, i_train, audit


def aggregate_results(fold_results):
    output = {}
    for h in range(pl.HORIZON):
        key = f"T+{h + 1}"
        output[key] = {}
        for model in ("plain_absolute", "engineered_residual_huber", "persistence"):
            output[key][model] = {}
            for metric in ("mae", "rmse", "r2", "mae_skill_vs_persistence",
                           "macro_station_mae", "macro_station_r2"):
                values = np.asarray(
                    [row[key][model][metric] for row in fold_results], dtype=float)
                output[key][model][f"{metric}_mean"] = float(values.mean())
                output[key][model][f"{metric}_std"] = float(values.std(ddof=1))

        residual_mae = np.asarray([
            row[key]["engineered_residual_huber"]["mae"] for row in fold_results])
        plain_mae = np.asarray([
            row[key]["plain_absolute"]["mae"] for row in fold_results])
        persistence_mae = np.asarray([
            row[key]["persistence"]["mae"] for row in fold_results])
        residual_r2 = np.asarray([
            row[key]["engineered_residual_huber"]["r2"] for row in fold_results])
        plain_r2 = np.asarray([
            row[key]["plain_absolute"]["r2"] for row in fold_results])
        output[key]["decision"] = {
            "residual_mae_better_than_plain_folds": int(np.sum(residual_mae < plain_mae)),
            "residual_mae_better_than_persistence_folds": int(
                np.sum(residual_mae < persistence_mae)),
            "residual_r2_better_than_plain_folds": int(np.sum(residual_r2 > plain_r2)),
            "mean_mae_delta_residual_minus_plain": float(
                np.mean(residual_mae - plain_mae)),
            "advance_mae_rule_3_of_4": bool(np.sum(residual_mae < plain_mae) >= 3),
        }
    return output


def flatten_fold_metrics(fold_name, results):
    rows = []
    for horizon, models in results.items():
        for model, values in models.items():
            if not isinstance(values, dict):
                continue
            rows.append({"fold": fold_name, "horizon": horizon, "model": model, **values})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multistep_2022_2026.yaml")
    ap.add_argument("--drop-station", default="")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    ap.add_argument(
        "--output-dir",
        default="outputs/experiments/pm25_residual_lgbm_rolling4_historical")
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()

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
        [i for i, name in enumerate(pl.SEL_STATIONS) if name not in dropped],
        dtype=np.int64)
    station_names = [pl.SEL_STATIONS[i] for i in station_indices]

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        (out / "models").mkdir(exist_ok=True)

    folds_payload = []
    flat_rows = []
    for fold_number, fold in enumerate(FOLDS):
        train_ids, tune_ids, score_ids, i_train, split_audit = select_fold_windows(
            tl, ws, fold)
        scaler_mask = np.arange(len(tl.times)) <= i_train
        scaler = pl.Scaler.fit(tl.x_raw, scaler_mask)
        results, bootstrap, best_iterations = {}, {}, {}

        for h in range(pl.HORIZON):
            x_plain_train, _ = build_features(
                tl, ws, train_ids, scaler, False, h, station_indices, station_names)
            x_plain_tune, _ = build_features(
                tl, ws, tune_ids, scaler, False, h, station_indices, station_names)
            x_plain_score, _ = build_features(
                tl, ws, score_ids, scaler, False, h, station_indices, station_names)
            x_eng_train, _ = build_features(
                tl, ws, train_ids, scaler, True, h, station_indices, station_names)
            x_eng_tune, _ = build_features(
                tl, ws, tune_ids, scaler, True, h, station_indices, station_names)
            x_eng_score, _ = build_features(
                tl, ws, score_ids, scaler, True, h, station_indices, station_names)

            y_train, ok_train = targets(tl, ws, train_ids, h, 0, station_indices)
            y_tune, ok_tune = targets(tl, ws, tune_ids, h, 0, station_indices)
            y_score, ok_score = targets(tl, ws, score_ids, h, 0, station_indices)
            a_train = anchor(tl, ws, train_ids, station_indices)
            a_tune = anchor(tl, ws, tune_ids, station_indices)
            a_score = anchor(tl, ws, score_ids, station_indices)
            seed = int(cfg["seed"]) + fold_number * 100 + h

            plain = fit_model(
                x_plain_train, y_train, ok_train, x_plain_tune, y_tune, ok_tune,
                "regression", args, seed)
            ok_delta_train = ok_train & np.isfinite(a_train)
            ok_delta_tune = ok_tune & np.isfinite(a_tune)
            residual = fit_model(
                x_eng_train, y_train - a_train, ok_delta_train,
                x_eng_tune, y_tune - a_tune, ok_delta_tune,
                "huber", args, seed)

            plain_pred = plain.predict(
                x_plain_score, num_iteration=plain.best_iteration_)
            residual_pred = a_score + residual.predict(
                x_eng_score, num_iteration=residual.best_iteration_)
            fallback = ~np.isfinite(residual_pred)
            residual_pred[fallback] = plain_pred[fallback]
            persistence = a_score.copy()
            valid = ok_score & np.isfinite(plain_pred) & np.isfinite(residual_pred)

            key = f"T+{h + 1}"
            results[key] = {
                "plain_absolute": score(
                    plain_pred, y_score, valid, persistence, station_names),
                "engineered_residual_huber": score(
                    residual_pred, y_score, valid, persistence, station_names),
                "persistence": score(
                    persistence, y_score, valid & np.isfinite(persistence),
                    persistence, station_names),
                "residual_fallback_cells": int(fallback.sum()),
            }
            bootstrap[key] = paired_bootstrap(
                residual_pred, plain_pred, y_score, valid, len(station_names),
                seed=seed)
            best_iterations[key] = {
                "plain_absolute": int(plain.best_iteration_),
                "engineered_residual_huber": int(residual.best_iteration_),
            }
            if args.save_models:
                joblib.dump(plain, out / "models" / f"fold_{fold['name']}_plain_h{h + 1}.joblib")
                joblib.dump(
                    residual,
                    out / "models" / f"fold_{fold['name']}_residual_h{h + 1}.joblib")

        fold_payload = {
            **fold,
            "split_audit": split_audit,
            "scaler_fit_end": str(tl.times[i_train]),
            "results": results,
            "paired_block_bootstrap": bootstrap,
            "best_iterations": best_iterations,
        }
        folds_payload.append(fold_payload)
        flat_rows.extend(flatten_fold_metrics(fold["name"], results))

    aggregate = aggregate_results([fold["results"] for fold in folds_payload])
    payload = {
        "run": "pm25_residual_lgbm_rolling4_historical",
        "pipeline_revision": pl.PIPELINE_REVISION,
        "reportable_as_final_test": False,
        "protocol": (
            "four fixed rolling-origin folds; expanding train; later early-stop; "
            "15-hour full-window boundary purge; still-later outer score"),
        "source_files_contain_later_rows": True,
        "current_2025_2026_dev_test_windows_used": False,
        "rows_after_2024_12_31_used_for_fit_or_score": False,
        "stations": station_names,
        "dropped_stations": sorted(dropped),
        "folds": folds_payload,
        "aggregate": aggregate,
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
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
