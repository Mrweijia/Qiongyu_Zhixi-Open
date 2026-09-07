"""Baselines on identical leak-free windows.

Implements the complete paper-plan chain: Persistence, Climatology, LightGBM,
GRU, LSTM, CNN-LSTM and Transformer.

Usage:
    py -3.13 src/eval/baselines.py            # all baselines + comparison table
    py -3.13 src/eval/baselines.py --config configs/multistep_2022_2026.yaml
    py -3.13 src/eval/baselines.py --only cnn_lstm,transformer --epochs 30
Writes one isolated output directory per data configuration so legacy metrics
cannot be silently mixed with the corrected 2022-2026 data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data import pipeline as pl  # noqa: E402
from src.training.train_multistep import (  # noqa: E402
    evaluate, masked_mse, per_cell_metrics, set_seed, build_prediction_rows,
)

EXP = ROOT_DIR / "outputs/experiments"
DEFAULT_CONFIG = "configs/multistep_2022_2026.yaml"


def shared_setup(cfg):
    set_seed(cfg["seed"])
    pl.validate_window_config(cfg)
    tl = pl.build_timeline_arrays(ROOT_DIR / cfg["data"]["pollution"],
                                  ROOT_DIR / cfg["data"]["weather"],
                                  max_gap_hours=cfg["data"]["max_gap_hours"],
                                  weather_timezone=cfg["data"]["weather_timezone"])
    ws = pl.enumerate_windows(tl, cfg["data"]["max_missing_frac"])
    split = pl.split_from_config(tl, ws, cfg["data"])
    scaler = pl.Scaler.fit(tl.x_raw, pl.train_time_mask(tl, split))
    x_norm = scaler.transform_filled(tl.x_filled)
    static = pl.build_static_geo_adj()
    a_seq = pl.build_dynamic_adj_seq(static, tl)
    return tl, ws, split, scaler, x_norm, static, a_seq


def raw_metrics(pred_real: np.ndarray, tl, ws, split):
    """pred_real [W,H,N,K] raw units vs x_raw targets [W,H,N,K]; per-cell metrics."""
    h_dim = pl.HORIZON
    met = {}
    t_real = np.stack([tl.x_raw[ws.starts[split.test] + pl.INPUT_STEPS + h][:, :, pl.PRED_IDX]
                       for h in range(h_dim)], axis=1)
    ok = ~np.isnan(t_real)
    for h in range(h_dim):
        step = {}
        vis = ok[:, h] & ~np.isnan(pred_real[:, h])
        for k, name in enumerate(pl.PRED_NAMES):
            sel = vis[:, :, k]
            if sel.sum() < 10:
                continue
            yt, yp = t_real[:, h, :, k][sel], pred_real[:, h, :, k][sel]
            step[name] = {"mae": float(mean_absolute_error(yt, yp)),
                          "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                          "r2": float(r2_score(yt, yp)), "n": int(sel.sum())}
        sel = vis
        step["joint_r2"] = float(r2_score(t_real[:, h][sel], pred_real[:, h][sel]))
        met[f"T+{h+1}"] = step
    return met, t_real, ok


def save(base_dir: Path, name: str, met: dict, config_name: str,
         extra: dict | None = None):
    d = base_dir / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {"run": name, "config": config_name,
               "pipeline_revision": pl.PIPELINE_REVISION, "test": met}
    if extra:
        payload.update(extra)
    (d / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"== {name} ==")
    for step, m in met.items():
        print(f"  {step}: joint R2={m['joint_r2']:.3f} " +
              " ".join(f"{p}:{m[p]['mae']:.1f}/{m[p]['r2']:.2f}" for p in pl.PRED_NAMES if p in m))


def run_persistence(tl, ws, split, base_dir, config_name):
    x_last = tl.x_filled[:, :, :6][:, :, pl.PRED_IDX]
    ends = ws.starts[split.test] + pl.INPUT_STEPS - 1
    pred = np.repeat(x_last[ends][:, None], pl.HORIZON, axis=1)  # [W,H,N,K]
    met, *_ = raw_metrics(pred, tl, ws, split)
    save(base_dir, "persistence", met, config_name)


def run_climatology(tl, ws, split, scaler, base_dir, config_name):
    train_t = tl.times[: split.i_tr]
    raw = tl.x_raw[: split.i_tr][:, :, pl.PRED_IDX]
    mon = train_t.month.to_numpy()
    hour = train_t.hour.to_numpy()
    clim = np.full((12, 24) + raw.shape[1:], np.nan)
    for m in range(1, 13):
        for h in range(24):
            sel = (mon == m) & (hour == h)
            if sel.sum() >= 5:
                with np.errstate(invalid="ignore"):
                    clim[m - 1, h] = np.nanmean(raw[sel], axis=0)
    glob = np.nanmean(np.where(np.isnan(clim), np.nan, clim), axis=(0, 1))
    fallback = np.where(np.isnan(glob), 0.0, np.nanmean(raw, axis=(0, 1, 2)))
    for m in range(12):
        for h in range(24):
            clim[m, h] = np.where(np.isnan(clim[m, h]), fallback, clim[m, h])
    # Targets begin after the full input window, not one hour after its start.
    tgt_idx = (ws.starts[split.test][:, None] + pl.INPUT_STEPS
               + np.arange(pl.HORIZON)[None, :])
    tt = tl.times[np.ravel(np.clip(tgt_idx, 0, len(tl.times) - 1))]
    mon = tt.month.to_numpy().reshape(tgt_idx.shape)
    hour = tt.hour.to_numpy().reshape(tgt_idx.shape)
    pred = clim[mon - 1, hour]  # [W,H,N,K]
    met, *_ = raw_metrics(pred, tl, ws, split)
    save(base_dir, "climatology", met, config_name)


def run_lightgbm(tl, ws, split, scaler, x_norm, base_dir, config_name):
    import lightgbm as lgb
    ends = ws.starts[split.test]
    tr_ids = ws.starts[split.train]
    y_raw = tl.x_raw[:, :, :6][:, :, pl.PRED_IDX]
    ok_all = ~np.isnan(y_raw)
    out_pred = np.full((len(ends), pl.HORIZON, len(pl.SEL_STATIONS), 4), np.nan)
    feats = x_norm  # [T,N,F]

    def matrix(starts):
        idx = starts[:, None] + np.arange(pl.INPUT_STEPS)[None, :]
        xw = feats[idx]                        # [W, L, N, F]
        return xw.transpose(0, 2, 1, 3).reshape(len(starts) * len(pl.SEL_STATIONS), -1)
    Xtr = matrix(tr_ids)
    Xte = matrix(ends)
    for h in range(pl.HORIZON):
        tgt_rows = tr_ids + pl.INPUT_STEPS + h
        for k in range(4):
            yv = y_raw[tgt_rows, :, k].reshape(-1)
            ov = ok_all[tgt_rows, :, k].reshape(-1)
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
            m.fit(Xtr[ov], yv[ov])
            out_pred[:, h, :, k] = m.predict(Xte).reshape(len(ends), len(pl.SEL_STATIONS))
    met, *_ = raw_metrics(out_pred, tl, ws, split)
    save(base_dir, "lightgbm", met, config_name)


class SeqBaseline(nn.Module):
    def __init__(self, in_f, hidden=32, horizon=3, n_out=4, kind="gru"):
        super().__init__()
        net = nn.GRU(in_f, hidden, batch_first=True) if kind == "gru" else nn.LSTM(in_f, hidden, batch_first=True)
        self.net = net
        self.heads = nn.ModuleList([nn.Linear(hidden, n_out) for _ in range(horizon)])
        self.horizon = horizon

    def forward(self, x, a, s):
        b, t, n, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.net(seq)
        h = out[:, -1].reshape(b, n, -1)
        return torch.stack([head(h) for head in self.heads], dim=2)


class CNNLSTMBaseline(nn.Module):
    """Per-station temporal CNN followed by LSTM; no graph information."""

    def __init__(self, in_f, hidden=32, horizon=3, n_out=4):
        super().__init__()
        self.conv = nn.Conv1d(in_f, hidden, kernel_size=3, padding=1)
        self.net = nn.LSTM(hidden, hidden, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, n_out) for _ in range(horizon)])

    def forward(self, x, a, s):
        b, t, n, f = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        seq = torch.relu(self.conv(seq)).transpose(1, 2)
        out, _ = self.net(seq)
        h = out[:, -1].reshape(b, n, -1)
        return torch.stack([head(h) for head in self.heads], dim=2)


class TransformerBaseline(nn.Module):
    """Per-station supervised temporal Transformer; no spatial graph."""

    def __init__(self, in_f, hidden=32, horizon=3, n_out=4, max_steps=48):
        super().__init__()
        self.proj = nn.Linear(in_f, hidden)
        self.pos = nn.Parameter(torch.randn(max_steps, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, nhead=4,
                                           dim_feedforward=4 * hidden,
                                           dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.net = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, n_out) for _ in range(horizon)])

    def forward(self, x, a, s):
        b, t, n, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        z = self.proj(seq) + self.pos[:t]
        h = self.norm(self.net(z))[:, -1].reshape(b, n, -1)
        return torch.stack([head(h) for head in self.heads], dim=2)


def run_seq(kind: str, tl, ws, split, scaler, x_norm, static, a_seq,
            cfg, base_dir, config_name, epochs=30, patience=5,
            max_train_windows=None, max_eval_windows=None):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kind in ("gru", "lstm"):
        model = SeqBaseline(len(pl.FEATURE_ORDER), kind=kind).to(device)
    elif kind == "cnn_lstm":
        model = CNNLSTMBaseline(len(pl.FEATURE_ORDER)).to(device)
    elif kind == "transformer":
        model = TransformerBaseline(len(pl.FEATURE_ORDER)).to(device)
    else:
        raise ValueError(f"unknown sequence baseline: {kind}")
    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                      weight_decay=cfg["train"]["weight_decay"])

    def loader(ids, shuffle):
        ds = pl.make_dataset(tl, ws, x_norm, scaler, a_seq, static, ids, use_dynamic_adj=False)
        return torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle)

    train_ids = split.train if max_train_windows is None else split.train[:max_train_windows]
    val_ids = split.val if max_eval_windows is None else split.val[:max_eval_windows]
    test_ids = split.test if max_eval_windows is None else split.test[:max_eval_windows]
    tr, va, te = loader(train_ids, True), loader(val_ids, False), loader(test_ids, False)
    best = {"loss": float("inf"), "state": None}
    best_epoch = 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y, m, a, s in tr:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = masked_mse(model(x, a, s).permute(0, 2, 1, 3), y, m)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        vl = evaluate(model, va, device, tl, scaler)["loss"]
        if vl < best["loss"]:
            best = {"loss": vl, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            best_epoch = ep
        if ep - best_epoch >= patience:
            break
    model.load_state_dict(best["state"])
    model.to(device)
    r = evaluate(model, te, device, tl, scaler, collect=True)
    met, p_real, t_real, mask = per_cell_metrics(r["pred"], r["true"], r["mask"], scaler)
    save(base_dir, kind, met, config_name,
         {"note": "same windows/split/scaler, no graph", "best_epoch": best_epoch,
          "smoke": max_train_windows is not None or max_eval_windows is not None})
    rows = build_prediction_rows(p_real, t_real, mask, tl.times, ws.starts, test_ids)
    rows.to_csv(base_dir / kind / "predictions_test.csv", index=False)


def comparison(base_dir: Path):
    rows = []
    sources = {}
    for p in sorted(base_dir.glob("*/metrics.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = j.get("run", p.parent.name)
        sources[name] = j.get("test", {})
    for name, met in sources.items():
        for step, m in met.items():
            for pol in pl.PRED_NAMES:
                if pol in m:
                    rows.append({"model": name, "step": step, "pollutant": pol,
                                 **{k: round(v, 4) for k, v in m[pol].items() if k in ("mae", "rmse", "r2", "n")}})
    df = pd.DataFrame(rows)
    df.to_csv(base_dir / "comparison_test.csv", index=False)
    print("\n== COMPARISON (test R2) ==")
    print(df.pivot_table(index="model", columns=["step", "pollutant"], values="r2").round(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--only", default="persistence,climatology,lightgbm,gru,lstm,cnn_lstm,transformer")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-train-windows", type=int, default=None,
                    help="smoke/debug only; omitted for reportable experiments")
    ap.add_argument("--max-eval-windows", type=int, default=None,
                    help="smoke/debug only; caps both validation and test windows")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg_path = ROOT_DIR / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config_name = cfg.get("name", cfg_path.stem)
    base_dir = (ROOT_DIR / args.output_dir if args.output_dir else
                EXP / f"baselines_{config_name}")
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "config.snapshot.yaml").write_text(
        cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    tl, ws, split, scaler, x_norm, static, a_seq = shared_setup(cfg)
    seq_args = (tl, ws, split, scaler, x_norm, static, a_seq, cfg,
                base_dir, config_name, args.epochs, args.patience,
                args.max_train_windows, args.max_eval_windows)
    fns = {"persistence": lambda: run_persistence(tl, ws, split, base_dir, config_name),
           "climatology": lambda: run_climatology(tl, ws, split, scaler, base_dir, config_name),
           "lightgbm": lambda: run_lightgbm(tl, ws, split, scaler, x_norm, base_dir, config_name),
           "gru": lambda: run_seq("gru", *seq_args),
           "lstm": lambda: run_seq("lstm", *seq_args),
           "cnn_lstm": lambda: run_seq("cnn_lstm", *seq_args),
           "transformer": lambda: run_seq("transformer", *seq_args)}
    for name in args.only.split(","):
        key = name.strip()
        if key not in fns:
            raise ValueError(f"unknown baseline {key}; choose from {sorted(fns)}")
        fns[key]()
    comparison(base_dir)


if __name__ == "__main__":
    main()
