"""Multimodal data adapter: ERA5 reanalysis grid (2 points) + optional AOD.

Reads the Open-Meteo ERA5 grid CSV (``data/cleaned/training/open_meteo_2022_2026.csv``
or the source-external equivalent) and aligns the per-grid-point features to the
pipeline grid via IDW interpolation to each monitoring station.

Optional AOD CSV: ``datetime, aod`` (derived from CAMS or FY-4A, one scalar per
hour). If absent, the multimodal model falls back to a learned zero embedding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data import pipeline as pl

# Columns present in the Open-Meteo CSV (west/east points)
ERA5_COLS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
             "wind_direction_10m", "pressure_msl", "precipitation", "dew_point_2m"]
# Grid point coordinates (matching the fetch script constants)
GRID_COORDS = {
    "era5_point_west": (112.783333, 28.116666),
    "era5_point_east": (113.219633, 28.189158),
}


def load_era5_grid(path: Path) -> tuple[np.ndarray, pd.DatetimeIndex, list[str], list[str]]:
    """Read Open-Meteo CSV -> x [T, G, F] (physical units), times, station_ids."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    present = [c for c in ERA5_COLS if c in df.columns]
    points = df["station_id"].unique().tolist()
    pivots = []
    for st in points:
        sub = df[df["station_id"] == st].set_index("datetime")[present]
        sub = sub.sort_index()
        pivots.append(sub)
    common = pivots[0].index
    for p in pivots[1:]:
        common = common.intersection(p.index)
    arrays = []
    for p in pivots:
        # Strictly causal alignment.  Linear interpolation and backward fill
        # use future reanalysis values to fill an earlier model input.
        arr = p.reindex(common).ffill(limit=3)
        arrays.append(arr.to_numpy(dtype=np.float64))
    return np.stack(arrays, axis=1), common, present, points  # [T, G, F]


def grid_to_station_idw(grid: np.ndarray, grid_ids: list[str],
                        grid_times: pd.DatetimeIndex,
                        timeline_times: pd.DatetimeIndex,
                        fill_limit: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """IDW-interpolate ERA5 grid [T_g, G, F] to the 10 station positions,
    reindexed to ``timeline_times``. Returns (station_features [T,N,F_wx],
    obs_mask [T,N] bool)."""
    g = len(grid_ids)
    n = len(pl.SEL_STATIONS)
    f = grid.shape[2]
    # grid coords
    gc = np.array([[GRID_COORDS[st][1], GRID_COORDS[st][0]] for st in grid_ids])  # [G,2]
    sc = np.array([[pl.STATION_COORDS[s][1], pl.STATION_COORDS[s][0]] for s in pl.SEL_STATIONS])  # [N,2]
    # inverse distance matrix
    dmat = np.zeros((n, g))
    for i in range(n):
        for j in range(g):
            dmat[i, j] = pl.haversine(sc[i, 0], sc[i, 1], gc[j, 0], gc[j, 1])
    inv = 1.0 / (dmat + 1e-6)
    w = inv / inv.sum(axis=1, keepdims=True)  # [N,G]
    # station-level features on the grid's own time index
    grid_feat = np.einsum("tgf,ng->tnf", grid, w)  # [T_g, N, F]
    df_grid = pd.DataFrame(grid_feat.reshape(grid_feat.shape[0], -1), index=grid_times)
    df_grid = df_grid.reindex(timeline_times).ffill(limit=fill_limit)
    obs = df_grid.notna().to_numpy()
    filled = df_grid.fillna(0.0).to_numpy(dtype=np.float64)
    obs_reshaped = obs.reshape(len(timeline_times), n, f).all(axis=-1)
    return filled.reshape(len(timeline_times), n, f), obs_reshaped


def load_aod(path: Optional[Path], timeline_times: pd.DatetimeIndex,
             fill_limit: int = 6) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Optional AOD: ``datetime, aod`` -> hourly [T, 1] and obs mask."""
    if path is None or not path.exists():
        return None, None
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    aod_col = next((c for c in df.columns if "aod" in c.lower() or "aerosol" in c.lower()), None)
    if aod_col is None:
        return None, None
    ser = df[aod_col].reindex(timeline_times).ffill(limit=fill_limit)
    obs = ser.notna().to_numpy()
    return ser.fillna(0.0).to_numpy(dtype=np.float64).reshape(-1, 1), obs


def build_mm_tensors(tl: pl.TimelineArrays, cfg: dict | None = None) -> dict:
    """Builds era5 and optional aod tensors aligned to the timeline grid
    [T, N, F_extra] for the trainer to concatenate with x_norm channels.

    cfg keys: ``era5_path``, ``aod_path`` (optional)."""
    cfg = cfg or {}
    era5_path = Path(cfg.get("era5_path", "data/cleaned/training/open_meteo_2022_2026.csv"))
    aod_path = Path(cfg.get("aod_path", "")) if cfg.get("aod_path") else None
    if not era5_path.exists():
        raise FileNotFoundError(f"ERA5 open-meteo CSV not found: {era5_path}")
    grid, grid_times, cols, grid_ids = load_era5_grid(era5_path)
    feats, obs = grid_to_station_idw(grid, grid_ids, grid_times, tl.times)
    aod_val, aod_obs = load_aod(aod_path, tl.times)
    result = {"era5": feats, "era5_obs": obs, "era5_cols": cols}
    if aod_val is not None:
        aod_broadcast = np.repeat(aod_val[:, None, :], len(pl.SEL_STATIONS), axis=1)  # [T,N,1]
        result["aod"] = aod_broadcast
        result["aod_obs"] = np.repeat(aod_obs[:, None], len(pl.SEL_STATIONS), axis=1)
    return result
