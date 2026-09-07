"""Unified training entry-point for the six Stage-3 innovation models.

Usage:
    py -3.13 src/training/train_innovation.py --config configs/innovations/causal_gat.yaml

The model type is read from ``cfg["model"]["type"]``. Each type wires its own
pre-processing (causal discovery, PDE forcing, multimodal tensors, pre-trained
ckpt) and evaluation path (DDPM sampling, exceedance stats).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from src.models.innovations import (  # noqa: E402
    CausalTGCN, AdvectionDiffusionTGCN, ConditionalPM25Diffusion,
    STMaskFormer, GraphNeuralODE, MultimodalTGCN, WindGatedTCN,
    AdaptiveGraphTransformer,
    causal_graph_from_timeline, row_normalize_with_self,
    save_causal_graph_artifacts, sparsify_in_topk,
    wind_uv_from_dir_speed, PDEResidualLoss,
    exceedance_stats, crps_ensemble,
    load_foundation_ckpt,
)
from src.data.mm_adapter import build_mm_tensors  # noqa: E402


# ─── helpers ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
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


# ─── data prep ───────────────────────────────────────────────────────────────

def prep_data(cfg: dict):
    set_seed(cfg["seed"])
    pl.validate_window_config(cfg)
    poll_path = ROOT_DIR / cfg["data"]["pollution"]
    wx_path = ROOT_DIR / cfg["data"]["weather"]
    tl = pl.build_timeline_arrays(
        poll_path,
        wx_path,
        max_gap_hours=cfg["data"]["max_gap_hours"],
        weather_timezone=cfg["data"]["weather_timezone"],
    )
    ws = pl.enumerate_windows(tl, max_missing_frac=cfg["data"]["max_missing_frac"])
    split = pl.split_from_config(tl, ws, cfg["data"])
    scaler = pl.Scaler.fit(tl.x_raw, pl.train_time_mask(tl, split))
    x_norm = scaler.transform_filled(tl.x_filled)
    static_a = pl.build_static_geo_adj()
    return tl, ws, split, scaler, x_norm, static_a


def make_loaders(tl, ws, split, scaler, x_norm, adj_seq, static_a, cfg):
    class IndexedDataset(torch.utils.data.Dataset):
        """Attach timeline start indices required by time/multimodal models."""

        def __init__(self, base, window_ids):
            self.base = base
            self.starts = torch.from_numpy(ws.starts[window_ids].astype(np.int64))

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            return (*self.base[i], self.starts[i])

    limits = cfg.get("_limits", {})

    def _loader(wids, shuffle, key):
        limit = limits.get(key)
        if limit:
            wids = wids[:int(limit)]
        a = adj_seq if cfg["model"].get("use_dynamic_adj", True) else np.repeat(static_a[None], len(tl.times), 0)
        ds = pl.make_dataset(tl, ws, x_norm, scaler, a, static_a, wids,
                             use_dynamic_adj=cfg["model"].get("use_dynamic_adj", True))
        wrapped = IndexedDataset(ds, wids)
        return torch.utils.data.DataLoader(
            wrapped, batch_size=cfg["train"]["batch_size"], shuffle=shuffle), wrapped
    tr_ld, _ = _loader(split.train, True, "train")
    va_ld, _ = _loader(split.val, False, "val")
    te_ld, _ = _loader(split.test, False, "test")
    return tr_ld, va_ld, te_ld


# ─── model builders ──────────────────────────────────────────────────────────

def build_causal_gat(cfg, tl, static_a, exp_dir):
    # causal discovery on TRAIN only
    if cfg["data"].get("train_end"):
        train_end = int(tl.times.searchsorted(
            pd.Timestamp(cfg["data"]["train_end"]), side="right"))
    else:
        train_end = int(len(tl.times) * cfg["data"]["train_ratio"])
    if cfg.get("_smoke"):
        train_end = min(train_end, 2000)
    train_mask = np.arange(len(tl.times)) < train_end
    g = causal_graph_from_timeline(tl, train_mask, channel="PM2.5",
                                   lag=cfg["model"].get("causal_lag", 12),
                                   alpha=cfg["model"].get("causal_alpha", 0.01),
                                   method=cfg["model"].get("causal_method", "conditional"),
                                   top_k=cfg["model"].get("causal_topk", 4))
    causal_adj = row_normalize_with_self(
        sparsify_in_topk(g.weights, cfg["model"].get("causal_topk", 4)))
    save_causal_graph_artifacts(g, exp_dir)
    cw = torch.from_numpy(causal_adj)
    geo = torch.from_numpy(static_a) if cfg["model"].get("use_geo_fallback", False) else None
    return CausalTGCN(
        in_f=len(pl.FEATURE_ORDER), causal_weights=cw,
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
        g_h=cfg["model"]["g_h"], gru_h=cfg["model"]["gru_h"],
        gat_heads=cfg["model"].get("gat_heads", 4),
        gat_layers=cfg["model"].get("gat_layers", 2),
        dropout=cfg["model"].get("dropout", 0.1),
        pred_channels=tuple(pl.PRED_IDX),
        use_geo_fallback=cfg["model"].get("use_geo_fallback", False),
        geo_adj=geo,
    ), causal_adj


def build_pde_gat(cfg, tl, static_a, exp_dir):
    coords = torch.tensor([[pl.STATION_COORDS[s][0], pl.STATION_COORDS[s][1]]
                           for s in pl.SEL_STATIONS])
    m = AdvectionDiffusionTGCN(
        in_f=len(pl.FEATURE_ORDER), coords=coords,
        g_h=cfg["model"]["g_h"], gru_h=cfg["model"]["gru_h"],
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
        length_scale_km=cfg["model"].get("length_scale_km", 25.0),
        blend_init=cfg["model"].get("blend_init", 0.5),
        diff_init=cfg["model"].get("diff_init", 0.3),
        pred_channels=tuple(pl.PRED_IDX),
    )
    pde_loss = PDEResidualLoss(
        coords, length_scale_km=cfg["model"].get("length_scale_km", 25.0),
        dt_h=1.0, k_channel=0, weight=cfg["model"].get("pde_weight", 0.1),
    )
    return m, pde_loss


def build_ddpm(cfg, tl, static_a, exp_dir):
    return ConditionalPM25Diffusion(
        in_f=len(pl.FEATURE_ORDER), n_nodes=len(pl.SEL_STATIONS),
        horizon=cfg["horizon"], g_h=cfg["model"]["g_h"], gru_h=cfg["model"]["gru_h"],
        cond_dim=cfg["model"].get("cond_dim", 64),
        hidden=cfg["model"].get("ddpm_hidden", 128),
        T_sched=cfg["model"].get("T_sched", 100),
        pred_channels=tuple(pl.PRED_IDX),
        pm25_slot=cfg["model"].get("pm25_slot", 0),
    ), None


def build_foundation(cfg, tl, static_a, exp_dir):
    ckpt = ROOT_DIR / cfg["model"]["pretrained_ckpt"]
    if ckpt.with_suffix(".pt").exists() and ckpt.with_suffix(".json").exists():
        m, stats = load_foundation_ckpt(
            ckpt, in_f=len(pl.FEATURE_ORDER), n_stations=len(pl.SEL_STATIONS),
            horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
            target_feature_order=list(pl.FEATURE_ORDER),
            target_station_names=list(pl.SEL_STATIONS))
    elif cfg["model"].get("require_pretrained", True):
        raise FileNotFoundError(
            f"预训练权重不存在: {ckpt.with_suffix('.pt')}；先运行 "
            "py -3.13 src/training/pretrain_foundation.py")
    else:
        m = STMaskFormer(
            n_stations=len(pl.SEL_STATIONS), in_f=len(pl.FEATURE_ORDER),
            d=cfg["model"].get("d", 128), layers=cfg["model"].get("layers", 4),
            heads=cfg["model"].get("heads", 4),
            hour_slots=cfg["model"].get("hour_slots", 24),
            out_poll=cfg["model"].get("out_poll", 6), horizon=cfg["horizon"],
            n_out=len(pl.PRED_IDX), dropout=cfg["model"].get("dropout", 0.1),
            pred_channels=tuple(pl.PRED_IDX))
        stats = {"initialisation": "random"}
    return m, stats


def build_gnode(cfg, tl, static_a, exp_dir):
    return GraphNeuralODE(
        in_f=len(pl.FEATURE_ORDER), n_nodes=len(pl.SEL_STATIONS),
        g_h=cfg["model"]["g_h"], gru_h=cfg["model"]["gru_h"],
        ode_dim=cfg["model"].get("ode_dim", 64),
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
        n_substeps=cfg["model"].get("ode_substeps", 4),
        dropout=cfg["model"].get("dropout", 0.1),
        pred_channels=tuple(pl.PRED_IDX),
    ), None


def build_multimodal(cfg, tl, static_a, exp_dir):
    mm = build_mm_tensors(tl, cfg.get("mm", {}))
    return MultimodalTGCN(
        in_f=len(pl.FEATURE_ORDER),
        n_extra=cfg["model"]["n_extra"],
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
        g_h=cfg["model"]["g_h"], gru_h=cfg["model"]["gru_h"],
        extra_h=cfg["model"].get("extra_h", 32),
        p_drop_modality=cfg["model"].get("p_drop_modality", 0.2),
        pred_channels=tuple(pl.PRED_IDX),
    ), mm


def build_wind_gated_tcn(cfg, tl, static_a, exp_dir):
    return WindGatedTCN(
        in_f=len(pl.FEATURE_ORDER), hidden=cfg["model"].get("hidden", 64),
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
        dilations=tuple(cfg["model"].get("dilations", [1, 2, 4])),
        kernel_size=cfg["model"].get("kernel_size", 3),
        dropout=cfg["model"].get("dropout", 0.1),
        pred_channels=tuple(pl.PRED_IDX)), None


def build_adaptive_graph_transformer(cfg, tl, static_a, exp_dir):
    return AdaptiveGraphTransformer(
        in_f=len(pl.FEATURE_ORDER), n_nodes=len(pl.SEL_STATIONS),
        d_model=cfg["model"].get("d_model", 64),
        layers=cfg["model"].get("layers", 2),
        heads=cfg["model"].get("heads", 4),
        graph_rank=cfg["model"].get("graph_rank", 16),
        max_steps=max(cfg.get("input_steps", pl.INPUT_STEPS), pl.INPUT_STEPS),
        horizon=cfg["horizon"], n_out=len(pl.PRED_IDX),
        dropout=cfg["model"].get("dropout", 0.1),
        pred_channels=tuple(pl.PRED_IDX)), None


BUILDERS = {
    "causal_gat": build_causal_gat,
    "pde_gat": build_pde_gat,
    "ddpm": build_ddpm,
    "foundation": build_foundation,
    "gnode": build_gnode,
    "multimodal": build_multimodal,
    "wind_gated_tcn": build_wind_gated_tcn,
    "adaptive_graph_transformer": build_adaptive_graph_transformer,
}


# ─── model-aware batches and evaluation ─────────────────────────────────────

def _unpack_batch(batch, device):
    x, y, mask, adj, season, starts = batch
    return (x.to(device), y.to(device), mask.to(device), adj.to(device),
            season.to(device), starts.to(device))


def _wind_uv(x, scaler):
    wd_idx = pl.FEATURE_ORDER.index("wind_dir")
    ws_idx = pl.FEATURE_ORDER.index("wind_spd")
    wd = x[..., wd_idx] * float(scaler.sd[0, wd_idx]) + float(scaler.mu[0, wd_idx])
    ws_ms = x[..., ws_idx] * float(scaler.sd[0, ws_idx]) + float(scaler.mu[0, ws_idx])
    u, v = wind_uv_from_dir_speed(wd, ws_ms * 3.6)  # source is m/s; PDE uses km/h
    return torch.stack([u, v], dim=-1)


def _time_ids(starts, tl, device):
    starts_np = starts.detach().cpu().numpy().astype(np.int64)
    idx = starts_np[:, None] + np.arange(pl.INPUT_STEPS)[None, :]
    flat = pd.DatetimeIndex(tl.times.to_numpy()[idx.reshape(-1)])
    shape = idx.shape
    hours = torch.from_numpy(flat.hour.to_numpy().reshape(shape).astype(np.int64)).to(device)
    wdays = torch.from_numpy(flat.dayofweek.to_numpy().reshape(shape).astype(np.int64)).to(device)
    return hours, wdays


def _prepare_mm_tensor(mm_tensors, split, cfg, exp_dir):
    parts = [mm_tensors["era5"]]
    names = list(mm_tensors.get("era5_cols", []))
    if "aod" in mm_tensors:
        parts.append(mm_tensors["aod"])
        names.append("aod")
    raw = np.concatenate(parts, axis=-1).astype(np.float64)
    expected = int(cfg["model"]["n_extra"])
    if raw.shape[-1] != expected:
        raise ValueError(f"multimodal n_extra={expected}, but adapter produced {raw.shape[-1]}")
    train = raw[:split.i_tr]
    mu = np.nanmean(train, axis=(0, 1))
    sd = np.nanstd(train, axis=(0, 1))
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    norm = np.nan_to_num((raw - mu) / sd, nan=0.0).astype(np.float32)
    (exp_dir / "multimodal_scaler.json").write_text(json.dumps({
        "features": names, "mu": mu.tolist(), "sd": sd.tolist(),
        "fit_end_index_exclusive": int(split.i_tr),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return torch.from_numpy(norm)


def _slice_extra(starts, mm_tensor, device):
    starts_cpu = starts.detach().cpu().long()
    offsets = torch.arange(pl.INPUT_STEPS, dtype=torch.long)
    idx = starts_cpu[:, None] + offsets[None, :]
    return mm_tensor[idx].to(device)


def _predict_batch(model, mtype, x, a, starts, context, training=False):
    if mtype == "pde_gat":
        uv = _wind_uv(x, context["scaler"])
        return model(x, a, uv=uv), {"uv": uv}
    if mtype == "foundation":
        hours, wdays = _time_ids(starts, context["tl"], x.device)
        station_ids = torch.arange(x.shape[2], device=x.device)
        return model(x, a, station_ids=station_ids,
                     hour_ids=hours, wday_ids=wdays), {}
    if mtype == "multimodal":
        extra = _slice_extra(starts, context["mm_tensor"], x.device)
        return model(x, a, x_extra=extra, training=training), {}
    return model(x, a), {}


def _batch_loss(model, mtype, batch, device, context, training=False):
    x, y, mask, a, season, starts = _unpack_batch(batch, device)
    if mtype == "ddpm":
        return model.training_loss(x, a, y, mask), None, (x, y, mask, a, starts)
    pred, aux = _predict_batch(model, mtype, x, a, starts, context, training=training)
    loss = masked_mse(pred, y, mask)
    if training and mtype == "pde_gat" and context.get("pde_loss") is not None:
        last_obs = x[:, -1][:, :, pl.PRED_IDX]
        # Future observed wind is unavailable at issue time.  Use the last
        # available forcing for every forecast step (a transparent persistence
        # assumption), rather than incorrectly mapping t-2,t-1,t to t+1..t+3.
        future_uv = aux["uv"][:, -1:].expand(-1, context["horizon"], -1, -1)
        loss = loss + context["pde_loss"](pred, last_obs, future_uv)
    return loss, pred, (x, y, mask, a, starts)


def evaluate_model(model, mtype, loader, device, context, collect=False):
    model.eval()
    preds, trues, masks = [], [], []
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss, pred, packed = _batch_loss(
                model, mtype, batch, device, context, training=False)
            x, y, mask, _, _ = packed
            tot += float(loss) * x.shape[0]
            cnt += x.shape[0]
            if collect:
                if pred is None:
                    raise ValueError("collect=True is unsupported for DDPM; use _eval_ddpm")
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())
                masks.append(mask.cpu().numpy())
    result = {"loss": tot / max(cnt, 1)}
    if collect:
        result.update(pred=np.concatenate(preds), true=np.concatenate(trues),
                      mask=np.concatenate(masks))
    return result


def per_cell_metrics(pred, true, mask, scaler):
    h_dim = pred.shape[1]
    met = {}
    p_real = np.stack([scaler.inverse_pollution(pred[:, h]) for h in range(h_dim)])
    t_real = np.stack([scaler.inverse_pollution(true[:, h]) for h in range(h_dim)])
    for h in range(h_dim):
        sm = {}
        ph, th, mh = p_real[h], t_real[h], mask[:, h]
        for k, name in enumerate(pl.PRED_NAMES):
            sel = mh[:, :, k] > 0.5
            if not sel.any():
                continue
            yt, yp = th[..., k][sel], ph[..., k][sel]
            sm[name] = {"mae": float(mean_absolute_error(yt, yp)),
                        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                        "r2": float(r2_score(yt, yp)), "n": int(len(yt))}
        sm["joint_r2"] = float(r2_score(t_real[h][mh > 0.5], p_real[h][mh > 0.5]))
        met[f"T+{h + 1}"] = sm
    return met, p_real, t_real, mask


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--pretrained-ckpt", default=None,
                    help="override foundation model checkpoint base path")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on 128/64/64 windows; results are marked non-reportable")
    ap.add_argument("--eval-split", choices=("validation", "test"), default="validation",
                    help="default avoids accidental repeated test inspection; use test only after freeze")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT_DIR / args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs_max"] = args.epochs
    if args.pretrained_ckpt is not None:
        cfg["model"]["pretrained_ckpt"] = args.pretrained_ckpt
    if args.smoke:
        cfg["_smoke"] = True
        cfg["_limits"] = {"train": 128, "val": 64, "test": 64}
        cfg["train"]["epochs_max"] = 1
        cfg["train"]["patience"] = 1
        if cfg["model"]["type"] == "ddpm":
            cfg["model"]["T_sched"] = min(10, cfg["model"].get("T_sched", 10))
            cfg.setdefault("eval", {})["ddpm_samples"] = 4
    mtype = cfg["model"]["type"]
    if mtype not in BUILDERS:
        raise ValueError(f"未知模型类型: {mtype}，可选: {list(BUILDERS)}")
    tag = args.tag or ("smoke" if args.smoke else None)
    run_name = f"{cfg['name']}_{tag}" if tag else cfg["name"]
    exp_dir = ROOT_DIR / cfg["output"]["experiment_dir"] if not tag \
        else ROOT_DIR / "outputs/experiments" / run_name
    bundle_dir = ROOT_DIR / cfg["output"]["bundle_dir"] if not tag \
        else ROOT_DIR / "models/checkpoints" / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT_DIR / args.config, exp_dir / "config.snapshot.yaml")

    requested = cfg.get("device", "auto")
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else ("cpu" if requested == "auto" else requested))
    tl, ws, split, scaler, x_norm, static_a = prep_data(cfg)
    adj_seq = pl.build_dynamic_adj_seq(static_a, tl) if cfg["model"].get("use_dynamic_adj", True) else static_a
    tr_ld, va_ld, te_ld = make_loaders(tl, ws, split, scaler, x_norm, adj_seq, static_a, cfg)

    # build model
    model, extra = BUILDERS[mtype](cfg, tl, static_a, exp_dir)
    model = model.to(device)
    pde_loss_fn = None
    if mtype == "pde_gat":
        pde_loss_fn = extra.to(device)
        extra = None
    mm_tensor = (_prepare_mm_tensor(extra, split, cfg, exp_dir)
                 if mtype == "multimodal" else None)
    if mtype == "foundation":
        (exp_dir / "pretraining_stats.json").write_text(
            json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")
    context = {"tl": tl, "scaler": scaler, "horizon": cfg["horizon"],
               "pde_loss": pde_loss_fn, "mm_tensor": mm_tensor}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {mtype} params={n_params:,} device={device}", flush=True)

    opt_params = list(model.parameters())
    if pde_loss_fn is not None:
        opt_params += list(pde_loss_fn.parameters())
    opt = optim.AdamW(opt_params, lr=cfg["train"]["lr"],
                      weight_decay=cfg["train"]["weight_decay"])
    history, best = [], {"loss": float("inf"), "epoch": 0,
                         "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    t_start = time.time()

    # --- training loop ---
    for ep in range(1, cfg["train"]["epochs_max"] + 1):
        model.train()
        if pde_loss_fn is not None:
            pde_loss_fn.train()
        tot, cnt = 0.0, 0
        for batch in tr_ld:
            opt.zero_grad()
            loss, _, packed = _batch_loss(
                model, mtype, batch, device, context, training=True)
            x = packed[0]
            loss.backward()
            nn.utils.clip_grad_norm_(opt_params, cfg["train"]["grad_clip"])
            opt.step()
            tot += loss.detach().item() * x.shape[0]
            cnt += x.shape[0]
        tr_loss = tot / cnt
        va = evaluate_model(model, mtype, va_ld, device, context)["loss"]
        history.append({"epoch": ep, "train": tr_loss, "val": va})
        print(f"epoch {ep:02d} train {tr_loss:.4f} val {va:.4f} ({time.time() - t_start:.0f}s)", flush=True)
        if va < best["loss"]:
            best = {"loss": va, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if ep - best["epoch"] >= cfg["train"]["patience"]:
            break

    model.load_state_dict(best["state"])
    model.eval()

    # --- evaluation ---
    eval_loader = va_ld if args.eval_split == "validation" else te_ld
    if args.eval_split == "test":
        print("[warning] evaluating the locked test split; do not tune after reading it", flush=True)
    if mtype == "ddpm":
        _eval_ddpm(model, eval_loader, device, scaler, exp_dir, cfg,
                   run_name, best, history, t_start, args.eval_split)
    else:
        evaluated = evaluate_model(model, mtype, eval_loader, device, context, collect=True)
        met_eval, p_t, t_t, m_t = per_cell_metrics(
            evaluated["pred"], evaluated["true"], evaluated["mask"], scaler)
        (exp_dir / "metrics.json").write_text(json.dumps({
            "run": run_name, "model": mtype, "best_epoch": best["epoch"],
            "pipeline_revision": pl.PIPELINE_REVISION,
            "wall_time_s": round(time.time() - t_start, 1),
            "evaluation_split": args.eval_split,
            args.eval_split: met_eval, "epochs": history,
            "git_commit": git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "smoke": bool(cfg.get("_smoke", False)),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(met_eval, indent=2))

    # save bundle
    torch.save(model.state_dict(), bundle_dir / "model.pt")
    scaler.save(bundle_dir / "scaler.npz")
    manifest = {
        "run": run_name, "model_type": mtype, "feature_order": pl.FEATURE_ORDER,
        "pipeline_revision": pl.PIPELINE_REVISION,
        "stations": pl.SEL_STATIONS, "pred_names": pl.PRED_NAMES,
        "weather_timezone": cfg["data"]["weather_timezone"],
        "input_steps": pl.INPUT_STEPS, "horizon": cfg["horizon"],
        "config": cfg, "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": args.eval_split,
        "smoke": bool(cfg.get("_smoke", False)),
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["model_sha256"] = sha256_file(bundle_dir / "model.pt")
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bundle: {bundle_dir}")


def _eval_ddpm(model, te_ld, device, scaler, exp_dir, cfg,
               run_name, best, history, t_start, split_name="validation"):
    """DDPM evaluation: sample S=64 trajectories, report median MAE + exceedance."""
    model.eval()
    S = cfg["eval"].get("ddpm_samples", 64)
    thr = cfg["eval"].get("exceedance_threshold_ugm3", 75.0)
    all_med, all_true, all_exceed, all_crps = [], [], [], []
    with torch.no_grad():
        for batch in te_ld:
            x, y, m, a, _, _ = _unpack_batch(batch, device)
            delta = model.sample_delta(x, a, n_samples=S, chunk=min(S, 32))
            # delta [S,B,N,H] normalised -> add last obs + denormalise
            base = x[:, -1][:, :, model.pred_channels[model.pm25_slot]]  # [B,N]
            anchor = base.unsqueeze(0).unsqueeze(-1)  # [1,B,N,1]
            samp_norm = delta + anchor  # [S,B,N,H]
            # denorm
            mu0 = torch.as_tensor(scaler.mu[:, pl.PRED_IDX[model.pm25_slot]],
                                  dtype=x.dtype, device=device)
            sd0 = torch.as_tensor(scaler.sd[:, pl.PRED_IDX[model.pm25_slot]],
                                  dtype=x.dtype, device=device)
            samp_raw = samp_norm * sd0[None, None, :, None] + mu0[None, None, :, None]  # [S,B,N,H]
            y_raw = scaler.inverse_pollution(y.cpu().numpy())[..., model.pm25_slot]
            y_mask = m[..., model.pm25_slot].cpu().numpy() > 0.5
            samp = samp_raw.cpu().numpy()
            for h in range(y_raw.shape[1]):
                for b in range(y_raw.shape[0]):
                    s_h = samp[:, b, :, h]  # [S,N]
                    oh = y_raw[b, h, :]  # [N]
                    valid = np.isfinite(oh) & y_mask[b, h]
                    if not valid.any():
                        continue
                    med = np.median(s_h, axis=0)[valid]
                    all_med.append(med)
                    all_true.append(oh[valid])
                    all_exceed.append(exceedance_stats(s_h[:, valid], oh[valid], thr))
                    all_crps.append(crps_ensemble(s_h[:, valid], oh[valid]))
    if not all_true:
        raise RuntimeError("DDPM test split contains no valid PM2.5 targets")
    med = np.concatenate(all_med)
    tv = np.concatenate(all_true)
    # Aggregate by the number of valid station-target cells.  A plain mean of
    # per-sample metrics overweights hours with fewer reporting stations and
    # even turns the count n into a non-integer average.
    total_n = int(sum(e.get("n", 0) for e in all_exceed))
    exc_agg = {
        k: float(sum(e[k] * e.get("n", 0) for e in all_exceed if k in e) / max(total_n, 1))
        for k in all_exceed[0] if k not in {"n", "crps", "median_mae"}
    }
    exc_agg["n"] = total_n
    crps_weighted = float(sum(c * e.get("n", 0) for c, e in zip(all_crps, all_exceed, strict=True))
                          / max(total_n, 1))
    metrics = {"run": run_name, "model": "ddpm", "best_epoch": best["epoch"],
               "pipeline_revision": pl.PIPELINE_REVISION,
               "wall_time_s": round(time.time() - t_start, 1),
               "epochs": history, "git_commit": git_commit(),
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "smoke": bool(cfg.get("_smoke", False)),
               "evaluation_split": split_name, split_name: {
        "median_mae": float(mean_absolute_error(tv, med)),
        "median_rmse": float(np.sqrt(mean_squared_error(tv, med))),
        "crps": crps_weighted,
        **exc_agg,
    }}
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics[split_name], indent=2))


if __name__ == "__main__":
    main()
