"""Cheap PM2.5-specific residual LightGBM experiment on nested validation.

The candidate predicts a robust Huber residual over the last observed PM2.5
value.  It is compared with an absolute-target plain LightGBM and Persistence;
no test windows are constructed or evaluated.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import pipeline as pl
from src.eval.feature_lgbm import build_features, setup, targets


def anchor(tl, ws, ids, station_indices):
    rows = ws.starts[ids] + pl.INPUT_STEPS - 1
    return tl.x_filled[rows][:, station_indices, pl.PRED_IDX[0]].reshape(-1)


def score(pred, true, valid, persistence, station_names):
    yt, yp = true[valid], pred[valid]
    pvalid = valid & np.isfinite(persistence)
    base_mae = mean_absolute_error(true[pvalid], persistence[pvalid])
    station_rows = []
    n = len(station_names)
    y2, p2, v2 = true.reshape(-1, n), pred.reshape(-1, n), valid.reshape(-1, n)
    for i in range(n):
        ok = v2[:, i]
        if ok.sum() >= 24 and np.var(y2[:, i][ok]) > 1e-12:
            station_rows.append((mean_absolute_error(y2[:, i][ok], p2[:, i][ok]),
                                 r2_score(y2[:, i][ok], p2[:, i][ok])))
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)), "n": int(valid.sum()),
        "mae_skill_vs_persistence": float(
            1.0 - mean_absolute_error(true[pvalid], pred[pvalid]) / base_mae),
        "macro_station_mae": float(np.mean([x[0] for x in station_rows])),
        "macro_station_r2": float(np.mean([x[1] for x in station_rows])),
    }


def paired_bootstrap(candidate, control, true, valid, n_stations,
                     repetitions=300, block=24, seed=53):
    rng = np.random.default_rng(seed)
    windows = len(true) // n_stations
    y = true.reshape(windows, n_stations)
    c = candidate.reshape(windows, n_stations)
    b = control.reshape(windows, n_stations)
    m = valid.reshape(windows, n_stations)
    mae_delta, r2_delta = [], []
    for _ in range(repetitions):
        starts = rng.integers(0, windows - block + 1,
                              size=int(np.ceil(windows / block)))
        ids = np.concatenate([np.arange(s, s + block) for s in starts])[:windows]
        ok = m[ids]
        yt, yc, yb = y[ids][ok], c[ids][ok], b[ids][ok]
        mae_delta.append(mean_absolute_error(yt, yc) - mean_absolute_error(yt, yb))
        r2_delta.append(r2_score(yt, yc) - r2_score(yt, yb))
    return {
        "mae_delta_candidate_minus_control_ci95": np.quantile(mae_delta, [.025, .975]).tolist(),
        "r2_delta_candidate_minus_control_ci95": np.quantile(r2_delta, [.025, .975]).tolist(),
        "block_hours": block, "repetitions": repetitions,
    }


def fit_model(x_train, y_train, ok_train, x_tune, y_tune, ok_tune,
              objective, args, seed):
    model = lgb.LGBMRegressor(
        objective=objective, alpha=0.9, n_estimators=args.n_estimators,
        learning_rate=0.035, num_leaves=63, min_child_samples=40,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.75,
        reg_lambda=0.1, n_jobs=-1, random_state=seed, verbosity=-1)
    model.fit(
        x_train[ok_train], y_train[ok_train],
        eval_set=[(x_tune[ok_tune], y_tune[ok_tune])], eval_metric="l1",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multistep_2022_2026.yaml")
    ap.add_argument("--drop-station", default="")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    ap.add_argument("--output-dir", default="outputs/experiments/pm25_residual_lgbm_dev_purged")
    args = ap.parse_args()

    cfg, tl, ws, split, scaler, tune_ids, score_ids = setup(ROOT / args.config)
    dropped = {x.strip() for x in args.drop_station.split(",") if x.strip()}
    if dropped - set(pl.SEL_STATIONS):
        raise ValueError(f"unknown stations: {sorted(dropped - set(pl.SEL_STATIONS))}")
    station_indices = np.array(
        [i for i, name in enumerate(pl.SEL_STATIONS) if name not in dropped], dtype=np.int64)
    station_names = [pl.SEL_STATIONS[i] for i in station_indices]
    out = ROOT / args.output_dir
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    results, boot, best_iterations = {}, {}, {}
    for h in range(pl.HORIZON):
        x_plain_train, _ = build_features(
            tl, ws, split.train, scaler, False, h, station_indices, station_names)
        x_plain_tune, _ = build_features(
            tl, ws, tune_ids, scaler, False, h, station_indices, station_names)
        x_plain_score, _ = build_features(
            tl, ws, score_ids, scaler, False, h, station_indices, station_names)
        x_eng_train, _ = build_features(
            tl, ws, split.train, scaler, True, h, station_indices, station_names)
        x_eng_tune, _ = build_features(
            tl, ws, tune_ids, scaler, True, h, station_indices, station_names)
        x_eng_score, _ = build_features(
            tl, ws, score_ids, scaler, True, h, station_indices, station_names)

        y_train, ok_train = targets(tl, ws, split.train, h, 0, station_indices)
        y_tune, ok_tune = targets(tl, ws, tune_ids, h, 0, station_indices)
        y_score, ok_score = targets(tl, ws, score_ids, h, 0, station_indices)
        a_train = anchor(tl, ws, split.train, station_indices)
        a_tune = anchor(tl, ws, tune_ids, station_indices)
        a_score = anchor(tl, ws, score_ids, station_indices)

        plain = fit_model(x_plain_train, y_train, ok_train, x_plain_tune,
                          y_tune, ok_tune, "regression", args, cfg["seed"])
        ok_delta_train = ok_train & np.isfinite(a_train)
        ok_delta_tune = ok_tune & np.isfinite(a_tune)
        residual = fit_model(
            x_eng_train, y_train - a_train, ok_delta_train,
            x_eng_tune, y_tune - a_tune, ok_delta_tune,
            "huber", args, cfg["seed"])
        plain_pred = plain.predict(x_plain_score, num_iteration=plain.best_iteration_)
        residual_pred = a_score + residual.predict(
            x_eng_score, num_iteration=residual.best_iteration_)
        fallback = ~np.isfinite(residual_pred)
        residual_pred[fallback] = plain_pred[fallback]
        persistence = a_score.copy()
        valid = ok_score & np.isfinite(plain_pred) & np.isfinite(residual_pred)

        key = f"T+{h + 1}"
        results[key] = {
            "plain_absolute": score(plain_pred, y_score, valid, persistence, station_names),
            "engineered_residual_huber": score(
                residual_pred, y_score, valid, persistence, station_names),
            "persistence": score(persistence, y_score, valid & np.isfinite(persistence),
                                 persistence, station_names),
            "residual_fallback_cells": int(fallback.sum()),
        }
        boot[key] = paired_bootstrap(
            residual_pred, plain_pred, y_score, valid, len(station_names), seed=cfg["seed"] + h)
        best_iterations[key] = {
            "plain_absolute": int(plain.best_iteration_),
            "engineered_residual_huber": int(residual.best_iteration_),
        }
        joblib.dump(plain, model_dir / f"plain_h{h + 1}.joblib")
        joblib.dump(residual, model_dir / f"residual_huber_h{h + 1}.joblib")

    payload = {
        "run": "pm25_residual_lgbm_dev", "pipeline_revision": pl.PIPELINE_REVISION,
        "reportable_as_final_test": False, "test_windows_accessed": False,
        "protocol": "train fit; validation first half early-stop; 15-hour purge; validation second half score",
        "stations": station_names, "dropped_stations": sorted(dropped),
        "results": results, "paired_block_bootstrap": boot,
        "best_iterations": best_iterations,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
