"""Validation-fitted convex blend of complementary formal models.

The blend never learns from the test targets.  Per horizon/pollutant weights
are fitted on the chronological validation split using non-negative least
squares and normalised to sum to one.  Candidates:

* LightGBM (retrained on the train split)
* three Wind-Gated TCN seeds (53/42/123)
* Causal GAT seed 53
* persistence

Usage:
    py -3.13 src/eval/blend_ensemble.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import joblib
import numpy as np
import torch
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import pipeline as pl  # noqa: E402
from src.models.innovations import (  # noqa: E402
    CausalTGCN, WindGatedTCN, row_normalize_with_self, sparsify_in_topk,
)
from src.training.train_multistep import build_prediction_rows, set_seed  # noqa: E402

CFG_PATH = ROOT / "configs" / "multistep_2022_2026.yaml"
WIND_CFG_PATH = ROOT / "configs" / "innovations" / "wind_gated_tcn.yaml"
CAUSAL_CFG_PATH = ROOT / "configs" / "innovations" / "causal_gat.yaml"
OUT = ROOT / "outputs" / "experiments" / "validation_blend"
BUNDLE = ROOT / "models" / "checkpoints" / "validation_blend"
WIND_BUNDLES = {
    "wind_seed53": ROOT / "models" / "checkpoints" / "wind_gated_tcn",
    "wind_seed42": ROOT / "models" / "checkpoints" / "wind_gated_tcn_seed42",
    "wind_seed123": ROOT / "models" / "checkpoints" / "wind_gated_tcn_seed123",
}
CAUSAL_BUNDLE = ROOT / "models" / "checkpoints" / "causal_gat"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def setup():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    pl.validate_window_config(cfg)
    set_seed(cfg["seed"])
    tl = pl.build_timeline_arrays(
        ROOT / cfg["data"]["pollution"], ROOT / cfg["data"]["weather"],
        max_gap_hours=cfg["data"]["max_gap_hours"],
        weather_timezone=cfg["data"]["weather_timezone"])
    ws = pl.enumerate_windows(tl, cfg["data"]["max_missing_frac"])
    split = pl.split_from_config(tl, ws, cfg["data"])
    scaler = pl.Scaler.fit(tl.x_raw, pl.train_time_mask(tl, split))
    x_norm = scaler.transform_filled(tl.x_filled)
    static = pl.build_static_geo_adj()
    a_seq = pl.build_dynamic_adj_seq(static, tl)
    return cfg, tl, ws, split, scaler, x_norm, static, a_seq


def loader(tl, ws, ids, scaler, x_norm, static, a_seq, batch_size=256):
    ds = pl.make_dataset(tl, ws, x_norm, scaler, a_seq, static, ids,
                         use_dynamic_adj=True)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def predict_torch(model, data_loader, device, scaler):
    model.eval()
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for x, y, mask, adj, season in data_loader:
            pred = model(x.to(device), adj.to(device))
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
            masks.append(mask.numpy())
    pred_norm = np.concatenate(preds)
    true_norm = np.concatenate(trues)
    mask = np.concatenate(masks)
    pred_raw = np.stack([scaler.inverse_pollution(pred_norm[:, h])
                         for h in range(pl.HORIZON)], axis=1)
    true_raw = np.stack([scaler.inverse_pollution(true_norm[:, h])
                         for h in range(pl.HORIZON)], axis=1)
    return pred_raw, true_raw, mask


def wind_model(cfg, bundle, device):
    model = WindGatedTCN(
        in_f=len(pl.FEATURE_ORDER), hidden=cfg["model"].get("hidden", 64),
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
        dilations=tuple(cfg["model"].get("dilations", [1, 2, 4])),
        kernel_size=cfg["model"].get("kernel_size", 3),
        dropout=cfg["model"].get("dropout", 0.1),
        pred_channels=tuple(pl.PRED_IDX))
    model.load_state_dict(torch.load(bundle / "model.pt", map_location="cpu"))
    return model.to(device)


def causal_model(cfg, device):
    graph = np.load(ROOT / "outputs" / "experiments" / "causal_gat" / "causal_graph.npz")
    adj = row_normalize_with_self(
        sparsify_in_topk(graph["weights"], cfg["model"].get("causal_topk", 4)))
    model = CausalTGCN(
        in_f=len(pl.FEATURE_ORDER), causal_weights=torch.from_numpy(adj),
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
        g_h=cfg["model"]["g_h"], gru_h=cfg["model"]["gru_h"],
        gat_heads=cfg["model"].get("gat_heads", 4),
        gat_layers=cfg["model"].get("gat_layers", 2),
        dropout=cfg["model"].get("dropout", 0.1),
        pred_channels=tuple(pl.PRED_IDX),
        use_geo_fallback=cfg["model"].get("use_geo_fallback", False))
    model.load_state_dict(torch.load(CAUSAL_BUNDLE / "model.pt", map_location="cpu"))
    return model.to(device)


def lightgbm_predictions(tl, ws, split, x_norm, bundle):
    train_starts = ws.starts[split.train]
    val_starts = ws.starts[split.val]
    test_starts = ws.starts[split.test]
    y_raw = tl.x_raw[:, :, :6][:, :, pl.PRED_IDX]
    ok_all = ~np.isnan(y_raw)

    def matrix(starts):
        idx = starts[:, None] + np.arange(pl.INPUT_STEPS)[None, :]
        xw = x_norm[idx]
        return xw.transpose(0, 2, 1, 3).reshape(len(starts) * len(pl.SEL_STATIONS), -1)

    x_train, x_val, x_test = matrix(train_starts), matrix(val_starts), matrix(test_starts)
    val_pred = np.empty((len(val_starts), pl.HORIZON, len(pl.SEL_STATIONS), len(pl.PRED_IDX)))
    test_pred = np.empty((len(test_starts), pl.HORIZON, len(pl.SEL_STATIONS), len(pl.PRED_IDX)))
    for h in range(pl.HORIZON):
        target_rows = train_starts + pl.INPUT_STEPS + h
        for k in range(len(pl.PRED_IDX)):
            y = y_raw[target_rows, :, k].reshape(-1)
            valid = ok_all[target_rows, :, k].reshape(-1)
            model = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
                n_jobs=-1, random_state=53)
            model.fit(x_train[valid], y[valid])
            joblib.dump(model, bundle / f"lightgbm_h{h + 1}_{pl.PRED_NAMES[k]}.joblib")
            val_pred[:, h, :, k] = model.predict(x_val).reshape(len(val_starts), -1)
            test_pred[:, h, :, k] = model.predict(x_test).reshape(len(test_starts), -1)
    return val_pred, test_pred


def persistence(tl, ws, ids):
    last = ws.starts[ids] + pl.INPUT_STEPS - 1
    value = tl.x_filled[last][:, :, pl.PRED_IDX]
    return np.repeat(value[:, None], pl.HORIZON, axis=1)


def targets(tl, ws, ids):
    values, masks = [], []
    for h in range(pl.HORIZON):
        row = ws.starts[ids] + pl.INPUT_STEPS + h
        block = tl.x_raw[row][:, :, pl.PRED_IDX]
        values.append(block)
        masks.append(np.isfinite(block))
    return np.stack(values, axis=1), np.stack(masks, axis=1)


def fit_convex(val_candidates, y_val, mask_val):
    names = list(val_candidates)
    weights = np.zeros((pl.HORIZON, len(pl.PRED_IDX), len(names)), dtype=np.float64)
    val_blend = np.zeros_like(y_val)
    for h in range(pl.HORIZON):
        for k in range(len(pl.PRED_IDX)):
            valid = mask_val[:, h, :, k]
            x = np.column_stack([val_candidates[name][:, h, :, k][valid] for name in names])
            y = y_val[:, h, :, k][valid]
            reg = LinearRegression(positive=True, fit_intercept=False).fit(x, y)
            coef = np.maximum(reg.coef_, 0.0)
            coef = coef / coef.sum() if coef.sum() > 1e-12 else np.full(len(names), 1 / len(names))
            weights[h, k] = coef
            val_blend[:, h, :, k] = sum(
                coef[i] * val_candidates[name][:, h, :, k]
                for i, name in enumerate(names))
    return names, weights, val_blend


def apply_weights(candidates, names, weights):
    first = candidates[names[0]]
    out = np.zeros_like(first)
    for h in range(pl.HORIZON):
        for k in range(len(pl.PRED_IDX)):
            out[:, h, :, k] = sum(
                weights[h, k, i] * candidates[name][:, h, :, k]
                for i, name in enumerate(names))
    return out


def finite_fallback(candidates):
    """Replace a candidate's rare non-finite cells with LightGBM predictions."""
    fallback = candidates["lightgbm"]
    if not np.isfinite(fallback).all():
        raise ValueError("LightGBM fallback contains non-finite predictions")
    counts = {}
    for name, values in candidates.items():
        bad = ~np.isfinite(values)
        counts[name] = int(bad.sum())
        if bad.any():
            candidates[name] = np.where(bad, fallback, values)
    return counts


def metrics(pred, true, mask):
    result = {}
    for h in range(pl.HORIZON):
        step = {}
        for k, name in enumerate(pl.PRED_NAMES):
            valid = mask[:, h, :, k]
            yt, yp = true[:, h, :, k][valid], pred[:, h, :, k][valid]
            step[name] = {
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "r2": float(r2_score(yt, yp)), "n": int(valid.sum())}
        step["joint_r2"] = float(r2_score(true[:, h][mask[:, h]], pred[:, h][mask[:, h]]))
        result[f"T+{h + 1}"] = step
    return result


def main():
    cfg, tl, ws, split, scaler, x_norm, static, a_seq = setup()
    missing = [str(path / "model.pt") for path in WIND_BUNDLES.values()
               if not (path / "model.pt").exists()]
    if missing:
        raise FileNotFoundError("missing seed checkpoints: " + ", ".join(missing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = loader(tl, ws, split.val, scaler, x_norm, static, a_seq)
    test_loader = loader(tl, ws, split.test, scaler, x_norm, static, a_seq)

    wind_cfg = yaml.safe_load(WIND_CFG_PATH.read_text(encoding="utf-8"))
    causal_cfg = yaml.safe_load(CAUSAL_CFG_PATH.read_text(encoding="utf-8"))
    val_candidates, test_candidates = {}, {}
    y_val, mask_val = targets(tl, ws, split.val)
    y_test, mask_test = targets(tl, ws, split.test)

    BUNDLE.mkdir(parents=True, exist_ok=True)
    print("training LightGBM candidates...", flush=True)
    val_candidates["lightgbm"], test_candidates["lightgbm"] = lightgbm_predictions(
        tl, ws, split, x_norm, BUNDLE)
    for name, bundle in WIND_BUNDLES.items():
        print(f"predicting {name}...", flush=True)
        model = wind_model(wind_cfg, bundle, device)
        val_candidates[name], _, _ = predict_torch(model, val_loader, device, scaler)
        test_candidates[name], _, _ = predict_torch(model, test_loader, device, scaler)
        del model
    print("predicting causal_gat...", flush=True)
    model = causal_model(causal_cfg, device)
    val_candidates["causal_gat"], _, _ = predict_torch(model, val_loader, device, scaler)
    test_candidates["causal_gat"], _, _ = predict_torch(model, test_loader, device, scaler)
    val_candidates["persistence"] = persistence(tl, ws, split.val)
    test_candidates["persistence"] = persistence(tl, ws, split.test)

    fallback_counts = {
        "validation": finite_fallback(val_candidates),
        "test": finite_fallback(test_candidates),
    }

    names, weight_array, val_blend = fit_convex(val_candidates, y_val, mask_val)
    test_blend = apply_weights(test_candidates, names, weight_array)
    weights = {
        f"T+{h + 1}": {
            pl.PRED_NAMES[k]: {names[i]: float(weight_array[h, k, i])
                               for i in range(len(names))}
            for k in range(len(pl.PRED_IDX))}
        for h in range(pl.HORIZON)}
    hashes = {name: sha256(bundle / "model.pt") for name, bundle in WIND_BUNDLES.items()}
    hashes["causal_gat"] = sha256(CAUSAL_BUNDLE / "model.pt")
    payload = {
        "run": "validation_blend", "model": "validation_convex_blend",
        "pipeline_revision": pl.PIPELINE_REVISION,
        "fit_split": "validation", "test_targets_used_for_fit": False,
        "candidates": names, "weights": weights,
        "nonfinite_fallback_to_lightgbm": fallback_counts,
        "validation": metrics(val_blend, y_val, mask_val),
        "test": metrics(test_blend, y_test, mask_test),
        "checkpoint_sha256": hashes, "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "smoke": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    shutil.copy(CFG_PATH, OUT / "config.snapshot.yaml")
    p_real = test_blend.transpose(1, 0, 2, 3)
    t_real = y_test.transpose(1, 0, 2, 3)
    rows = build_prediction_rows(p_real, t_real, mask_test, tl.times, ws.starts, split.test)
    rows.to_csv(OUT / "predictions_test.csv", index=False)
    (BUNDLE / "weights.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    scaler.save(BUNDLE / "scaler.npz")
    lgb_hashes = {p.name: sha256(p) for p in sorted(BUNDLE.glob("lightgbm_*.joblib"))}
    manifest = {
        "model_type": "validation_convex_blend",
        "pipeline_revision": pl.PIPELINE_REVISION,
        "fit_split": "validation", "test_targets_used_for_fit": False,
        "data": cfg["data"], "feature_order": pl.FEATURE_ORDER,
        "stations": pl.SEL_STATIONS, "pred_names": pl.PRED_NAMES,
        "input_steps": pl.INPUT_STEPS, "horizon": pl.HORIZON,
        "candidate_checkpoint_sha256": hashes,
        "lightgbm_model_sha256": lgb_hashes,
        "weights_sha256": sha256(BUNDLE / "weights.json"),
        "scaler_sha256": sha256(BUNDLE / "scaler.npz"),
        "metrics_path": str((OUT / "metrics.json").relative_to(ROOT)).replace("\\", "/"),
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "smoke": False,
    }
    (BUNDLE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["test"], ensure_ascii=False, indent=2))
    print(f"saved: {OUT}")
    print(f"bundle: {BUNDLE}")


if __name__ == "__main__":
    main()
