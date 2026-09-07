"""Pre-train the STMaskFormer on multi-city data (UCI Beijing + Changsha
historical + CNEMC snapshot).

Usage:
    py -3.13 src/training/pretrain_foundation.py [--epochs 20]

Produces ``models/checkpoints/foundation_pretrained/model.pt`` + ``.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DEFAULT_PT_DIR = ROOT_DIR / "models" / "checkpoints" / "foundation_pretrained"

# ─── data readers ────────────────────────────────────────────────────────────

COLS_6 = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]


def _dt(hour: pd.Series) -> pd.Series:
    return pd.to_datetime(hour.astype(int).astype(str), format="%Y%m%d%H", errors="coerce")


def read_uci_beijing(root: Path) -> dict:
    """12 stations, hourly, 2013.3–2017.2 -> {city_name: {stations, times, values, obs}}."""
    root = Path(root)
    dfs = []
    for f in sorted(root.glob("PRSA_Data_*.csv")):
        df = pd.read_csv(f, dtype=str)
        df["dt"] = df["year"].str.cat([df["month"], df["day"], df["hour"]], sep="")
        df["dt"] = _dt(df["dt"])
        df = df.dropna(subset=["dt"]).set_index("dt")
        site = f.stem.replace("PRSA_Data_", "").rsplit("_", 1)[0]
        for c in COLS_6:
            if c in df.columns:
                suf = df[[c]].rename(columns={c: "val"}).assign(site=site, poll=c)
                suf["val"] = pd.to_numeric(suf["val"], errors="coerce")
                dfs.append(suf)
    if not dfs:
        raise FileNotFoundError(f"UCI 北京站 CSV 未找到于 {root}")
    long = pd.concat(dfs)
    pivot = long.pivot_table(index="dt", columns=["site", "poll"], values="val")
    stations = sorted(long["site"].unique())
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([stations, COLS_6]))
    vals = pivot.to_numpy(dtype=np.float64).reshape(pivot.shape[0], len(stations), len(COLS_6))
    obs = np.isfinite(vals).all(axis=-1)
    return {"beijing_12": {"stations": stations,
                           "times": pivot.index,
                           "values": vals, "obs": obs}}


def read_changsha_historical(path: Path) -> dict:
    """Read the wide Changsha archive: date/hour/type + one column per site."""
    df = pd.read_csv(path, dtype=str)
    df["dt"] = _dt(df["date"].str.cat(df["hour"], sep=""))
    df = df.dropna(subset=["dt"])
    id_cols = {"date", "hour", "type", "dt"}
    station_cols = [c for c in df.columns if c not in id_cols]
    long = []
    for c in COLS_6:
        sub = df[df["type"] == c].copy()
        for site in station_cols:
            data = sub[["dt", site]].rename(columns={site: "val"})
            data["val"] = pd.to_numeric(data["val"], errors="coerce")
            data["site"] = site
            data["poll"] = c
            long.append(data)
    if not long:
        return {}
    all_df = pd.concat([d for d in long if not d.empty], ignore_index=True)
    pivot = all_df.pivot_table(index="dt", columns=["site", "poll"], values="val")
    stations = sorted(all_df["site"].unique())
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([stations, COLS_6]))
    vals = pivot.to_numpy(dtype=np.float64).reshape(pivot.shape[0], len(stations), len(COLS_6))
    return {"changsha": {"stations": stations,
                         "times": pivot.index, "values": vals,
                         "obs": np.isfinite(vals).all(axis=-1)}}


def read_cnemc_snapshot(path: Path) -> dict:
    """Group by area, keep cities with >= 6 stations, >= 96 hours."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["dt"] = pd.to_datetime(df["timepoint"], errors="coerce")
    df = df.dropna(subset=["dt"])
    source_cols = {"PM2.5": "pm2_5", "PM10": "pm10", "SO2": "so2",
                   "NO2": "no2", "CO": "co", "O3": "o3"}
    for target, source in source_cols.items():
        df[target] = pd.to_numeric(df[source], errors="coerce")
    cities = {}
    for area, grp in df.groupby("area"):
        n_st = grp["stationcode"].nunique()
        if n_st < 6:
            continue
        sts = grp["stationcode"].unique()[:12]  # cap
        pivots = {}
        for s in sts:
            sub = grp[grp["stationcode"] == s].groupby("dt")[COLS_6].mean()
            pivots[s] = sub
        if len(pivots) < 6:
            continue
        common_idx = None
        for s_df in pivots.values():
            common_idx = s_df.index if common_idx is None else common_idx.intersection(s_df.index)
        if common_idx is None or len(common_idx) < 96:
            continue
        arr = np.stack([pivots[s].reindex(common_idx).to_numpy(dtype=np.float64)
                        for s in pivots], axis=1)
        mask = np.isfinite(arr).all(axis=-1)
        cities[area] = {"stations": list(pivots.keys()), "times": common_idx,
                        "values": arr, "obs": mask}
    return cities


def truncate_before(cities: dict[str, dict], cutoff: pd.Timestamp) -> dict[str, dict]:
    """Enforce an as-of-date for every pretraining source.

    Self-supervised reconstruction still observes the downstream period's
    measurements and distribution.  A foundation checkpoint used for a
    historical forecast experiment therefore must not include timestamps at
    or after the downstream training boundary.
    """
    kept = {}
    for name, city in cities.items():
        times = pd.DatetimeIndex(city["times"])
        mask = times < cutoff
        if int(mask.sum()) < 12:
            continue
        kept[name] = {**city, "times": times[mask],
                      "values": city["values"][mask], "obs": city["obs"][mask]}
    return kept


# ─── dataset ─────────────────────────────────────────────────────────────────

class SingleCityDataset(torch.utils.data.Dataset):
    """Iterable over windows of length L for one city."""

    def __init__(self, city: dict, L: int = 12, stride: int = 1,
                 global_stats: tuple | None = None, station_offset: int = 0):
        vals = city["values"]  # [T, N, P]
        T, N, P = vals.shape
        # normalise
        mu, sd = global_stats if global_stats else (np.nanmean(vals, axis=(0, 1)),
                                                    np.nanstd(vals, axis=(0, 1)) + 1e-8)
        # Missing tokens are represented by ``obs`` and a learned mask token;
        # keep the numeric tensor finite because NaN * 0 is still NaN.
        self.vals = np.nan_to_num((vals - mu) / sd, nan=0.0,
                                  posinf=0.0, neginf=0.0)
        self.obs = city["obs"]  # [T, N]
        self.times = city["times"]
        self.station_offset = station_offset
        self.L = L
        self.N = N
        self.P = P
        self.mu, self.sd = mu, sd
        self.windows = np.arange(T - L + 1, step=stride)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s = self.windows[idx]
        x = self.vals[s:s + self.L].astype(np.float32)  # [L,N,P]
        obs = self.obs[s:s + self.L].astype(np.float32)  # [L,N]
        ts = self.times[s:s + self.L]
        hour = np.array([t.hour for t in ts], dtype=np.int64)
        wday = np.array([t.weekday() for t in ts], dtype=np.int64)
        # build input: 6 pollutants + 1 obs mask channel
        x_full = np.concatenate([x, obs[..., None]], axis=-1)  # [L,N,7]
        # random mask for pretraining: 15% of valid tokens
        tr_mask = obs.copy()
        valid = np.where(obs > 0.5)
        nv = len(valid[0])
        if nv > 0:
            nm = max(1, int(0.15 * nv))
            sel = np.random.choice(nv, nm, replace=False)
            tr_mask[valid[0][sel], valid[1][sel]] = 0.0
        x_masked = x_full * tr_mask[..., None]  # mask_token added in tokenise
        return (torch.from_numpy(x_masked), torch.from_numpy(tr_mask),
                torch.from_numpy(x_full[..., :self.P]), torch.from_numpy(obs),
                torch.from_numpy(hour), torch.from_numpy(wday))


# ─── training loop ───────────────────────────────────────────────────────────

def pretrain_main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 1. load cities
    cities: dict[str, dict] = {}
    uci = read_uci_beijing(ROOT_DIR / "data/source/external/uci_beijing_pm25/PRSA_Data_20130301-20170228")
    cs = read_changsha_historical(
        ROOT_DIR / "data/cleaned/historical/changsha_sites_2014_2025.csv")
    cnemc = read_cnemc_snapshot(ROOT_DIR / "data/source/external/cnemc_multicity/envdata_sync/cnemc_hourly_snapshots_merged.csv")
    cities.update(uci); cities.update(cs); cities.update(cnemc)
    cutoff = pd.Timestamp(args.cutoff)
    cities = truncate_before(cities, cutoff)
    if args.max_cities:
        cities = dict(list(cities.items())[:args.max_cities])
    print(f"cities: {len(cities)} total: {sum(len(c['times']) for c in cities.values())} hours")

    # 2. global stats for pollutants
    all_vals = np.concatenate([c["values"].reshape(-1, c["values"].shape[-1]) for c in cities.values()])
    mu = np.nanmean(all_vals, axis=0)
    sd = np.nanstd(all_vals, axis=0) + 1e-8
    stats = {"mu": mu.tolist(), "sd": sd.tolist(), "pollutants": COLS_6}

    # 3. build datasets & station embedding offset
    total_stations = 0
    datasets = []
    station_registry = []
    for name, city in cities.items():
        stride = max(1, len(city["times"]) // args.max_windows_per_city) \
            if args.max_windows_per_city else 1
        ds = SingleCityDataset(city, L=12, stride=stride, global_stats=(mu, sd),
                               station_offset=total_stations)
        datasets.append((name, ds))
        station_registry.extend(
            {"city": name, "station": str(station),
             "embedding_index": total_stations + i}
            for i, station in enumerate(city["stations"]))
        total_stations += ds.N

    # 4. model
    cfg = dict(d=128, layers=4, heads=4, hour_slots=24, out_poll=len(COLS_6),
               in_f=len(COLS_6) + 1, horizon=3, n_out=4, dropout=0.1)
    from src.models.innovations.foundation import STMaskFormer
    model = STMaskFormer(n_stations=total_stations, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0.0, 0
        t0 = time.time()
        # shuffle cities each epoch
        order = list(range(len(datasets)))
        np.random.default_rng(ep).shuffle(order)
        for idx in order:
            name, ds = datasets[idx]
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
            stat_ids = torch.arange(ds.station_offset, ds.station_offset + ds.N, device=device)
            for batch in loader:
                xm, tmask, target, obs, hr, wd = [t.to(device) for t in batch]
                pred = model.pretrain_forward(xm, tmask, stat_ids, hr, wd)
                # loss on masked positions only
                masked = (obs - tmask) > 0.5  # [B,L,N] originally valid but now masked
                loss = F.huber_loss(pred[masked], target[masked], reduction="mean")
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total_loss += loss.item()
                n_steps += 1
        scheduler.step()
        avg = total_loss / max(n_steps, 1)
        print(f"epoch {ep:02d} loss {avg:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # 5. save
    pt_dir = ROOT_DIR / args.output_dir if args.output_dir else DEFAULT_PT_DIR
    pt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pt_dir / "model.pt")
    save_meta = {**cfg, "n_stations": total_stations, "stats": stats,
                 "epochs": args.epochs, "cities": list(cities.keys()),
                 "feature_order": COLS_6 + ["obs_mask"],
                 "station_registry": station_registry,
                 "pretrain_cutoff_exclusive": str(cutoff)}
    save_meta["smoke"] = bool(args.max_windows_per_city or args.max_cities)
    (pt_dir / "model.json").write_text(json.dumps(save_meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"saved to {pt_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-windows-per-city", type=int, default=None,
                    help="optional compute cap; windows are evenly strided")
    ap.add_argument("--max-cities", type=int, default=None)
    ap.add_argument("--cutoff", default="2025-02-16T08:00:00",
                    help="exclusive as-of timestamp; defaults to downstream train boundary")
    ap.add_argument("--output-dir", default=None,
                    help="relative to project root; use a *_smoke path for capped runs")
    args = ap.parse_args()
    pretrain_main(args)
