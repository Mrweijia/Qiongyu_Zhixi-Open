"""One-command reproducible trainer for the multi-step (T+1..T+3) model v2.

Usage:
    py -3.13 src/training/train_multistep.py --config configs/multistep_v2.yaml [--tag NAME]

Produces outputs/experiments/<name|tag>/{config.snapshot.yaml, metrics.json,
predictions_test.csv, loss_curve.png, train_log.txt} and a deployable bundle
models/checkpoints/<name|tag>/{model.pt, scaler.npz, manifest.json}.
The test split is evaluated exactly once, after early stopping on validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import pipeline as pl  # noqa: E402
from src.models.wu_v2 import HardMOESeasonV2  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT_DIR, check=True).stdout.strip()
    except Exception:
        return "unknown"


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - y) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1.0)


def evaluate(model: nn.Module, loader, device: torch.device, tl, scaler,
             collect: bool = False):
    model.eval()
    preds, trues, masks = [], [], []
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for x, y, m, a, s in loader:
            x, y, m, a, s = x.to(device), y.to(device), m.to(device), a.to(device), s.to(device)
            p = model(x, a, s).permute(0, 2, 1, 3)  # [B,N,H,K] -> [B,H,N,K]
            l = masked_mse(p, y, m)
            tot += float(l) * x.shape[0]
            cnt += x.shape[0]
            if collect:
                preds.append(p.cpu().numpy())
                trues.append(y.cpu().numpy())
                masks.append(m.cpu().numpy())
    if not collect:
        return {"loss": tot / max(cnt, 1)}
    return {"loss": tot / max(cnt, 1),
            "pred": np.concatenate(preds), "true": np.concatenate(trues),
            "mask": np.concatenate(masks)}


def per_cell_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray,
                     scaler: pl.Scaler):
    """pred/true [W,H,N,K] normalized, mask [W,H,N,K]. Returns metrics + denorm arrays."""
    h_dim = pred.shape[1]
    met: dict = {}
    p_real = np.stack([scaler.inverse_pollution(pred[:, h]) for h in range(h_dim)])
    t_real = np.stack([scaler.inverse_pollution(true[:, h]) for h in range(h_dim)])
    for h in range(h_dim):
        step_metrics = {}
        ph, th, mh = p_real[h], t_real[h], mask[:, h]
        for k, name in enumerate(pl.PRED_NAMES):
            sel = mh[:, :, k] > 0.5
            if not sel.any():
                continue
            yt = th[..., k][sel]
            yp = ph[..., k][sel]
            step_metrics[name] = {
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "r2": float(r2_score(yt, yp)),
                "n": int(len(yt)),
            }
        sel_all = mh > 0.5
        step_metrics["joint_r2"] = float(r2_score(t_real[h][sel_all], p_real[h][sel_all]))
        met[f"T+{h + 1}"] = step_metrics
    return met, p_real, t_real, mask


def build_prediction_rows(p_real, t_real, m_all, times, starts, window_ids) -> pd.DataFrame:
    w = len(window_ids)
    rows = []
    for h in range(p_real.shape[0]):
        for k, name in enumerate(pl.PRED_NAMES):
            sel = m_all[:, h, :, k] > 0.5
            wi, si = np.where(sel)
            for j in range(len(wi)):
                win = int(window_ids[wi[j]])
                s0 = int(starts[win])
                t = times[s0 + pl.INPUT_STEPS + h]
                rows.append((t, pl.SEL_STATIONS[si[j]], h + 1, name,
                             float(t_real[h][wi[j], si[j], k]),
                             float(p_real[h][wi[j], si[j], k])))
    return pd.DataFrame(rows, columns=["prediction_time", "station", "step",
                                       "pollutant", "y_true", "y_pred"])


def season_breakdown(p_real, t_real, m_all, times, starts, window_ids, seasons):
    out: dict = {}
    for s, sname in enumerate(["winter", "spring", "summer", "autumn"]):
        sel_w = seasons[window_ids] == s
        if sel_w.sum() < 10:
            continue
        block = {}
        for h in range(p_real.shape[0]):
            mh = m_all[sel_w, h]
            th = t_real[h][sel_w]
            ph = p_real[h][sel_w]
            sel = mh > 0.5
            if sel.sum() < 10:
                continue
            block[f"T+{h + 1}"] = {
                "mae": float(mean_absolute_error(th[sel], ph[sel])),
                "r2": float(r2_score(th[sel], ph[sel])),
            }
        out[sname] = block
    return out


def build_all(cfg: dict, run_name: str, log_lines: list[str]):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poll_path = ROOT_DIR / cfg["data"]["pollution"]
    wx_path = ROOT_DIR / cfg["data"]["weather"]

    t0 = time.time()
    tl = pl.build_timeline_arrays(poll_path, wx_path,
                                  max_gap_hours=cfg["data"]["max_gap_hours"],
                                  weather_timezone=cfg["data"]["weather_timezone"])
    ws = pl.enumerate_windows(tl, max_missing_frac=cfg["data"]["max_missing_frac"])
    split = pl.split_from_config(tl, ws, cfg["data"])
    scaler = pl.Scaler.fit(tl.x_raw, pl.train_time_mask(tl, split))
    x_norm = scaler.transform_filled(tl.x_filled)
    static_a = pl.build_static_geo_adj()
    adj_seq = pl.build_dynamic_adj_seq(static_a, tl)
    log_lines.append(f"[data] T={len(tl.times)} broken_hours={tl.broken_hours} "
                     f"windows={len(ws.starts)} train={len(split.train)} val={len(split.val)} "
                     f"test={len(split.test)} build={time.time() - t0:.1f}s")
    print(log_lines[-1], flush=True)

    def make_loader(wids, shuffle):
        ds = pl.make_dataset(tl, ws, x_norm, scaler, adj_seq if cfg["model"]["use_dynamic_adj"] else np.repeat(static_a[None], len(tl.times), 0),
                             static_a, wids, use_dynamic_adj=cfg["model"]["use_dynamic_adj"])
        return torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle), ds

    tr_ld, _ = make_loader(split.train, True)
    va_ld, _ = make_loader(split.val, False)
    te_ld, _ = make_loader(split.test, False)

    m = cfg["model"]
    model = HardMOESeasonV2(
        in_f=len(pl.FEATURE_ORDER), g_h=m["g_h"], gru_h=m["gru_h"], attn_dim=m["attn_dim"],
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX), n_experts=m["n_experts"],
        use_attention=m["use_attention"], use_seasonal_experts=m["use_seasonal_experts"],
        predict_delta=m.get("predict_delta", False), pred_channels=tuple(pl.PRED_IDX),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log_lines.append(f"[model] params={n_params} device={device}")
    print(log_lines[-1], flush=True)
    return (device, tl, ws, split, scaler, tr_ld, va_ld, te_ld, model,
            sha256_file(poll_path), sha256_file(wx_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval-split", choices=("validation", "test"), default="validation",
                    help="default prevents accidental repeated test inspection")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT_DIR / args.config).read_text(encoding="utf-8"))
    pl.validate_window_config(cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_name = args.tag or cfg["name"]
    exp_dir = ROOT_DIR / cfg["output"]["experiment_dir"] if not args.tag else ROOT_DIR / "outputs/experiments" / f"{cfg['name']}_{args.tag}"
    bundle_dir = ROOT_DIR / cfg["output"]["bundle_dir"] if not args.tag else ROOT_DIR / "models/checkpoints" / f"multistep_v2_{args.tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT_DIR / args.config, exp_dir / "config.snapshot.yaml")

    log_lines: list[str] = []
    device, tl, ws, split, scaler, tr_ld, va_ld, te_ld, model, h_poll, h_wx = build_all(cfg, run_name, log_lines)

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                     weight_decay=cfg["train"]["weight_decay"])
    tr_ct, va_ct, te_ct = split.train, split.val, split.test
    history, best = [], {"loss": float("inf"), "epoch": 0,
                         "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    t_start = time.time()
    for ep in range(1, cfg["train"]["epochs_max"] + 1):
        model.train()
        tot, cnt = 0.0, 0
        for x, y, m, a, s in tr_ld:
            x, y, m, a, s = x.to(device), y.to(device), m.to(device), a.to(device), s.to(device)
            opt.zero_grad()
            loss = masked_mse(model(x, a, s).permute(0, 2, 1, 3), y, m)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            tot += loss.detach().item() * x.shape[0]
            cnt += x.shape[0]
        tr_loss = tot / cnt
        va = evaluate(model, va_ld, device, tl, scaler)["loss"]
        history.append({"epoch": ep, "train": tr_loss, "val": va})
        line = f"epoch {ep:02d} train {tr_loss:.4f} val {va:.4f} ({time.time() - t_start:.0f}s)"
        print(line, flush=True)
        if va < best["loss"]:
            best = {"loss": va, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if ep - best["epoch"] >= cfg["train"]["patience"]:
            print(f"early stop at epoch {ep}", flush=True)
            break

    model.load_state_dict(best["state"])
    if device.type == "cuda":
        model.to(device)

    eval_ld = va_ld if args.eval_split == "validation" else te_ld
    eval_ids = va_ct if args.eval_split == "validation" else te_ct
    if args.eval_split == "test":
        print("[warning] evaluating the locked test split; freeze all choices first", flush=True)
    evaluated = evaluate(model, eval_ld, device, tl, scaler, collect=True)
    met_eval, p_t, t_t, m_t = per_cell_metrics(
        evaluated["pred"], evaluated["true"], evaluated["mask"], scaler)
    breakdown = season_breakdown(p_t, t_t, m_t, tl.times, ws.starts, eval_ids, ws.seasons)

    pred_rows = build_prediction_rows(p_t, t_t, m_t, tl.times, ws.starts, eval_ids)
    pred_rows.to_csv(exp_dir / f"predictions_{args.eval_split}.csv", index=False)

    metrics = {
        "run": run_name, "git_commit": git_commit(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_revision": pl.PIPELINE_REVISION,
        "wall_time_s": round(time.time() - t_start, 1), "best_epoch": best["epoch"],
        "data": {"pollution_sha256": h_poll, "weather_sha256": h_wx,
                 "grid_hours": int(len(tl.times)), "broken_hours": tl.broken_hours,
                 "train_range": [str(tl.times[0]), str(split.cut_train)],
                 "val_range": [str(split.cut_train), str(split.cut_val)],
                 "test_range": [str(split.cut_val), str(tl.times[-1])]},
        "windows": {"train": int(len(tr_ct)), "val": int(len(va_ct)), "test": int(len(te_ct))},
        "evaluation_split": args.eval_split,
        args.eval_split: met_eval, f"{args.eval_split}_season_breakdown": breakdown,
        "epochs": history,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["epoch"] for h in history], [h["train"] for h in history], label="train")
    ax.plot([h["epoch"] for h in history], [h["val"] for h in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE (normalized)")
    ax.legend()
    fig.savefig(exp_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    torch.save(model.state_dict(), bundle_dir / "model.pt")
    scaler.save(bundle_dir / "scaler.npz")
    manifest = {
        "name": run_name,
        "pipeline_revision": pl.PIPELINE_REVISION,
        "architecture": {**cfg["model"], "in_f": len(pl.FEATURE_ORDER),
                         "horizon": cfg["horizon"], "input_steps": cfg["input_steps"],
                         "n_out": len(pl.PRED_IDX), "params": sum(p.numel() for p in model.parameters())},
        "stations": pl.SEL_STATIONS,
        "feature_order": pl.FEATURE_ORDER,
        "pred_names": pl.PRED_NAMES,
        "weather_timezone": cfg["data"]["weather_timezone"],
        "scaler": "scaler.npz (mu/sd per station-feature, TRAIN-only)",
        "train_time_range": metrics["data"]["train_range"],
        "evaluation_split": args.eval_split,
        "evaluation_metrics": met_eval,
        "git_commit": metrics["git_commit"],
        "data_sha256": {"pollution": h_poll, "weather": h_wx},
        "created_utc": metrics["created_utc"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_sha256": sha256_file(bundle_dir / "model.pt"),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    (exp_dir / "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(json.dumps(met_eval, indent=2))
    print(f"bundle: {bundle_dir}")
    print(f"artifacts: {exp_dir}")


if __name__ == "__main__":
    main()
