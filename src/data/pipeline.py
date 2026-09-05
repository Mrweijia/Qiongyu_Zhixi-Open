"""Leak-free preprocessing pipeline for the Changsha multi-station forecast model.

Fixes over ``src/models/wu.py`` (see docs/project/data.md):

* 5.4 weather station id ``59287199999`` (old config had ``592871999999``) + hard
  data-contract check that every configured weather station exists in the input.
* 5.2 chronological 70/15/15 split with straddle-purge (>= full window length);
  NO random splitting, scaler fitted on TRAIN rows only.
* 5.7 full hourly reindex, CAUSAL filling only (ffill limit), windows spanning a
  raw gap longer than ``max_gap_hours`` are dropped (never zero-filled silently).
* 5.6 the dynamic adjacency sequence is sliced PER SAMPLE (window-aligned).
* 5.3 scaler (mean/std per station-feature) is part of the model bundle, exposed
  via :class:`Scaler` so inference reuses the training statistics.
* Weather timestamps have an explicit source-timezone contract.  UTC inputs are
  converted to Asia/Shanghai exactly once; missing/unknown declarations fail.

Shapes convention
-----------------
X      : float64 [T, N, F]  F = 6 pollutants + 4 weather features (raw units)
A_dyn  : float32 [T, N, N]  wind-aligned row-normalised adjacency per timestamp
times  : list[pd.Timestamp] full hourly grid
Windows: input = times[i : i+L], targets = times[i+L : i+L+H]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Constants (single source of truth for v2 code paths) ────────────────────
SEL_STATIONS = [
    "1335A", "1336A", "1337A", "1338A", "1339A",
    "1340A", "1341A", "1342A", "1343A", "1344A",
]
POLL_TYPES = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
# Model predicts these 4 (indices into POLL_TYPES)
PRED_IDX = [0, 1, 3, 4]
PRED_NAMES = [POLL_TYPES[i] for i in PRED_IDX]

INPUT_STEPS = 12
HORIZON = 3          # T+1, T+2, T+3
WINDOW_LEN = INPUT_STEPS + HORIZON  # 15 timestamps per sample

# FIX (5.4): real data station id is 59287199999 (was mis-spelled in wu.py).
WEATHER_STATIONS: dict[str, tuple[float, float]] = {
    "57687099999": (28.116666, 112.783333),
    "59287199999": (28.189158, 113.219633),
}

STATION_COORDS = {
    "1335A": (113.0833, 28.2325), "1336A": (112.8872, 28.2189),
    "1337A": (113.0792, 28.2053), "1338A": (112.9394, 28.1900),
    "1339A": (113.0178, 28.1322), "1340A": (112.9792, 28.2597),
    "1341A": (113.0014, 28.1944), "1342A": (112.9840, 28.1178),
    "1343A": (112.8908, 28.1308), "1344A": (112.9581, 28.3611),
}

DIST_THRESHOLD_KM = 50.0
GAUSS_SIGMA_KM = 20.0
WX_FEATURES = ["tmp_C", "wind_dir", "wind_spd", "rh"]
FEATURE_ORDER = POLL_TYPES + WX_FEATURES  # F = 10
PIPELINE_REVISION = "2026-09-03-scientific-audit-v1"
WEATHER_TIMEZONES = {"UTC", "Asia/Shanghai"}


class DataContractError(ValueError):
    """Raised when input data violates the agreed schema (station ids etc.)."""


def validate_weather_timezone(source_timezone: str | None) -> str:
    """Return a supported weather time basis or reject an ambiguous input."""
    if source_timezone not in WEATHER_TIMEZONES:
        allowed = ", ".join(sorted(WEATHER_TIMEZONES))
        raise DataContractError(
            "气象数据必须显式声明 weather_timezone，"
            f"可选值为 {allowed}；不能猜测 DATE 是 UTC 还是北京时间。"
        )
    return source_timezone


def validate_window_config(cfg: dict) -> None:
    """Reject configs whose advertised window differs from pipeline reality."""
    got_input = int(cfg.get("input_steps", INPUT_STEPS))
    got_horizon = int(cfg.get("horizon", HORIZON))
    if got_input != INPUT_STEPS or got_horizon != HORIZON:
        raise DataContractError(
            "当前 pipeline 的窗口由常量固定为 "
            f"input_steps={INPUT_STEPS}, horizon={HORIZON}，但配置声明 "
            f"input_steps={got_input}, horizon={got_horizon}。"
            "请先参数化 pipeline，不能静默按另一组窗口训练。")
    validate_weather_timezone(cfg.get("data", {}).get("weather_timezone"))


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def get_season_index(ts: pd.Timestamp) -> int:
    m = ts.month
    if m in (12, 1, 2):
        return 0
    if m in (3, 4, 5):
        return 1
    if m in (6, 7, 8):
        return 2
    return 3


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_pollution_long(path: Path | str) -> pd.DataFrame:
    """Long tidy frame: datetime, site, type, value (NaN preserved)."""
    df = pd.read_csv(path, dtype=str)
    date_num = pd.to_numeric(df["date"], errors="coerce")
    hour_num = pd.to_numeric(df["hour"], errors="coerce")
    ok = date_num.notna() & hour_num.notna()
    df = df.loc[ok].copy()
    df["datetime"] = (
        pd.to_datetime(date_num[ok].astype(int).astype(str), format="%Y%m%d", errors="coerce")
        + pd.to_timedelta(hour_num[ok].astype(int), unit="h")
    )
    df = df[df["datetime"].notna()]
    val_cols = [c for c in df.columns if c not in {"date", "hour", "type", "datetime"}]
    long = df.melt(id_vars=["datetime", "type"], value_vars=val_cols,
                   var_name="site", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long[long["site"].isin(SEL_STATIONS) & long["type"].isin(POLL_TYPES)]
    dup = long.duplicated(subset=["datetime", "site", "type"], keep="first")
    if dup.any():
        long = long.loc[~dup]
    return long.dropna(subset=["value"])


def _parse_wnd(cell: object) -> tuple[float, float]:
    parts = str(cell).split(",")
    try:
        d = float(parts[0])
        if not 0.0 <= d <= 360.0 or d == 999.0:
            d = np.nan
    except (ValueError, IndexError):
        d = np.nan
    try:
        s = float(parts[3]) / 10.0
        if s < 0.0 or s >= 999.0:
            s = np.nan
    except (ValueError, IndexError):
        s = np.nan
    return d, s


def _signed_num(cell: object) -> float:
    try:
        value = float(str(cell).split(",")[0])
        # NOAA ISD uses all-9 fields for missing observations.  Letting these
        # through creates temperatures/dew points near 1,000 C after scaling.
        return np.nan if abs(value) >= 9999 else value
    except (ValueError, IndexError):
        return np.nan


def load_weather_long(
    path: Path | str,
    *,
    source_timezone: str | None,
) -> pd.DataFrame:
    """Tidy weather frame: datetime, STATION, tmp_C, wind_dir, wind_spd, rh.

    ``source_timezone`` is mandatory.  ``UTC`` timestamps are converted to
    timezone-naive Asia/Shanghai values before alignment; ``Asia/Shanghai``
    values are already local and are not shifted.  This prevents the historical
    silent 8-hour feature leakage from returning when an old NOAA file is used.
    Contract check: every station in WEATHER_STATIONS must be present (fix 5.4).
    Accepts both DEWP and DEW humidity columns (repo data uses DEW).
    """
    source_timezone = validate_weather_timezone(source_timezone)
    df = pd.read_csv(path, dtype=str)
    df = df.loc[:, ~df.columns.duplicated()]  # header repeats some columns
    if source_timezone == "UTC":
        df["datetime"] = (
            pd.to_datetime(df["DATE"], errors="coerce", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    else:
        parsed = pd.to_datetime(df["DATE"], errors="coerce")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        df["datetime"] = parsed
    df = df[df["datetime"].notna() & df["STATION"].notna()]
    df["tmp_C"] = df["TMP"].map(_signed_num) / 10.0
    dew_col = "DEWP" if "DEWP" in df.columns else ("DEW" if "DEW" in df.columns else None)
    df["dew_C"] = df[dew_col].map(_signed_num) / 10.0 if dew_col else np.nan
    parsed = df["WND"].map(lambda x: _parse_wnd(x) if isinstance(x, str) else (np.nan, np.nan))
    df["wind_dir"] = [p[0] for p in parsed]
    df["wind_spd"] = [p[1] for p in parsed]

    def rh(t: float, td: float) -> float:
        if np.isnan(t) or np.isnan(td):
            return np.nan
        a, b = 17.27, 237.7
        return float(min(100.0, max(0.0, 100 * math.exp(a * td / (b + td) - a * t / (b + t)))))

    df["rh"] = [rh(t, d) for t, d in zip(df["tmp_C"], df["dew_C"], strict=True)]

    present = set(df["STATION"].unique())
    missing = set(WEATHER_STATIONS) - present
    if missing:
        raise DataContractError(
            f"配置的气象站未出现在输入数据中: {sorted(missing)}；数据包含 {sorted(present)}"
        )
    out = df[["datetime", "STATION", "tmp_C", "wind_dir", "wind_spd", "rh"]]
    return out.sort_values(["STATION", "datetime"])


# ─── Tensor construction on a gap-aware hourly grid ─────────────────────────

@dataclass
class TimelineArrays:
    times: pd.DatetimeIndex        # full hourly grid [T]
    exists: np.ndarray             # bool [T], hour observed in pollution pivot
    x_raw: np.ndarray              # float64 [T, N, F], NaN = unknown
    x_filled: np.ndarray           # float64 [T, N, F], causal-ffilled; NaN where unfilled
    break_mask: np.ndarray         # bool [T], part of a raw hole longer than max_gap
    wx_raw: np.ndarray             # float64 [T, M, 4] station-level observed
    broken_hours: int


def build_timeline_arrays(
    pollution_path: Path | str,
    weather_path: Path | str,
    max_gap_hours: int = 3,
    *,
    weather_timezone: str | None = None,
) -> TimelineArrays:
    poll = load_pollution_long(pollution_path)
    wx = load_weather_long(weather_path, source_timezone=weather_timezone)

    pivot = poll.pivot_table(index="datetime", columns=["site", "type"], values="value")
    cols = pd.MultiIndex.from_product([SEL_STATIONS, POLL_TYPES], names=["site", "type"])
    pivot = pivot.reindex(columns=cols)
    t0, t1 = pivot.index.min(), pivot.index.max()
    grid = pd.date_range(t0, t1, freq="h")
    pivot = pivot.reindex(grid)

    exists = pivot.notna().any(axis=1).to_numpy()
    # mark holes longer than max_gap_hours as breaks (windows must not cross them)
    break_mask = np.zeros(len(grid), dtype=bool)
    run_start = None
    for i in range(len(grid) + 1):
        hole = (i == len(grid)) or (not exists[i])
        if hole and run_start is None:
            run_start = i
        elif not hole and run_start is not None:
            if i - run_start > max_gap_hours:
                break_mask[run_start:i] = True
            run_start = None

    x_poll = pivot.to_numpy(dtype=np.float64).reshape(len(grid), len(SEL_STATIONS), len(POLL_TYPES))

    # causal filling only: forward fill with limit == max_gap_hours
    x_poll_filled = pd.DataFrame(x_poll.reshape(len(grid), -1)).ffill(limit=max_gap_hours).to_numpy()
    x_poll_filled = x_poll_filled.reshape(len(grid), len(SEL_STATIONS), len(POLL_TYPES))

    # weather: grid per observation station, causal ffill, then inverse-distance to monitors
    m = len(WEATHER_STATIONS)
    wx_stations = list(WEATHER_STATIONS)
    wx_grid = np.full((len(grid), m, len(WX_FEATURES)), np.nan)
    for k, st in enumerate(wx_stations):
        sub = wx[wx["STATION"] == st].set_index("datetime")[WX_FEATURES]
        sub = sub[~sub.index.duplicated(keep="first")].reindex(grid)
        wx_grid[:, k, :] = sub.ffill(limit=max_gap_hours).to_numpy()

    dmat = np.zeros((len(SEL_STATIONS), m))
    for i, (lon, lat) in enumerate((STATION_COORDS[s] for s in SEL_STATIONS)):
        ds = np.array([haversine(lat, lon, *WEATHER_STATIONS[st][::-1]) for st in wx_stations])
        inv = 1.0 / (ds + 1e-6)
        dmat[i] = inv / inv.sum()
    # Missing-aware IDW: renormalise over the weather stations that actually
    # observed each feature.  Treating a missing station as a literal zero
    # biases temperature, humidity and wind toward zero.  Wind direction is a
    # circular variable, so 350 and 10 degrees must average to 0, not 180.
    valid = np.isfinite(wx_grid)
    weights = dmat[None, :, :, None] * valid[:, None, :, :]
    denom = weights.sum(axis=2)
    numer = (np.nan_to_num(wx_grid, nan=0.0)[:, None, :, :] * weights).sum(axis=2)
    x_wx = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    wd_idx = WX_FEATURES.index("wind_dir")
    wd_rad = np.deg2rad(wx_grid[:, :, wd_idx])
    wd_w = weights[:, :, :, wd_idx]
    sin_sum = (np.nan_to_num(np.sin(wd_rad), nan=0.0)[:, None, :] * wd_w).sum(axis=2)
    cos_sum = (np.nan_to_num(np.cos(wd_rad), nan=0.0)[:, None, :] * wd_w).sum(axis=2)
    wd = (np.rad2deg(np.arctan2(sin_sum, cos_sum)) + 360.0) % 360.0
    x_wx[:, :, wd_idx] = np.where(denom[:, :, wd_idx] > 0, wd, np.nan)

    x_raw = np.concatenate([x_poll, x_wx], axis=2)
    x_filled = np.concatenate([x_poll_filled, x_wx], axis=2)

    return TimelineArrays(
        times=grid,
        exists=exists,
        x_raw=x_raw,
        x_filled=x_filled,
        break_mask=break_mask,
        wx_raw=wx_grid,
        broken_hours=int(break_mask.sum()),
    )


# ─── Adjacency ───────────────────────────────────────────────────────────────

def build_static_geo_adj() -> np.ndarray:
    n = len(SEL_STATIONS)
    a = np.zeros((n, n), dtype=np.float32)
    coords = [STATION_COORDS[s] for s in SEL_STATIONS]
    for i, (lon1, lat1) in enumerate(coords):
        for j, (lon2, lat2) in enumerate(coords):
            d = haversine(lat1, lon1, lat2, lon2)
            if d <= DIST_THRESHOLD_KM:
                a[i, j] = math.exp(-d * d / (2 * GAUSS_SIGMA_KM ** 2))
    np.fill_diagonal(a, 1.0)
    d_inv = np.diag(1 / np.sqrt(a.sum(axis=1) + 1e-6))
    return (d_inv @ a @ d_inv).astype(np.float32)


def build_dynamic_adj_seq(static_a: np.ndarray, tl: TimelineArrays) -> np.ndarray:
    """Wind-aligned dynamic adjacency per TIMESTAMP of the grid [T, N, N].

    Uses observed (un-normalised) wind direction; for missing hours falls back to
    the previous timestamp. Window datasets slice their own segment (fix 5.6).
    """
    n = len(SEL_STATIONS)
    coords = [STATION_COORDS[s] for s in SEL_STATIONS]
    transport_ang = np.zeros((n, n), dtype=np.float32)
    for i, (lon_i, lat_i) in enumerate(coords):
        for j, (lon_j, lat_j) in enumerate(coords):
            # A[i,j] sends the feature at j into receiver i, hence j -> i.
            transport_ang[i, j] = math.atan2(lat_i - lat_j, lon_i - lon_j)

    wx_stations = list(WEATHER_STATIONS)
    dmat = np.zeros((n, len(wx_stations)))
    for i, (lon, lat) in enumerate(coords):
        ds = np.array([haversine(lat, lon, *WEATHER_STATIONS[st][::-1]) for st in wx_stations])
        inv = 1.0 / (ds + 1e-6)
        dmat[i] = inv / inv.sum()
    # monitor-level wind direction (channel 6+1 of filled features, same v1 convention);
    # timestamps where weather is fully missing take a neutral 0 deg so A_t stays finite.
    wd = tl.x_filled[:, :, len(POLL_TYPES) + 1]

    a_seq = np.zeros((len(tl.times), n, n), dtype=np.float32)
    for t in range(len(tl.times)):
        valid_wind = np.isfinite(wd[t])
        # NOAA direction is meteorological FROM, clockwise from north; graph
        # bearings are mathematical TO, counter-clockwise from east.
        to_bearing = (np.nan_to_num(wd[t], nan=0.0) + 180.0) % 360.0
        flow = np.deg2rad((90.0 - to_bearing) % 360.0)
        # Wind at the sender j determines transport along j -> i.
        align = np.maximum(0, np.cos(flow[None, :] - transport_ang))
        # Unknown wind direction means "no directional evidence", therefore
        # retain the geographic graph instead of inventing a north wind.
        align[:, ~valid_wind] = 1.0
        a_t = static_a * align
        np.fill_diagonal(a_t, 1.0)
        a_seq[t] = a_t / (a_t.sum(axis=1, keepdims=True) + 1e-6)
    return a_seq


# ─── Scaler (train-only, persisted with the bundle) ─────────────────────────

@dataclass
class Scaler:
    mu: np.ndarray  # [N, F]
    sd: np.ndarray  # [N, F]

    @staticmethod
    def fit(x_raw: np.ndarray, train_mask: np.ndarray) -> "Scaler":
        sub = np.where(train_mask[:, None, None], x_raw, np.nan)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(sub, axis=0)
            sd = np.nanstd(sub, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where(np.isnan(sd) | (sd < 1e-6), 1.0, sd)
        return Scaler(mu.astype(np.float64), sd.astype(np.float64))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mu) / self.sd).astype(np.float32)

    def transform_filled(self, x: np.ndarray) -> np.ndarray:
        """Scale first, then encode residual missing values as the train mean.

        Filling raw missing values with zero before standardisation encoded
        them as extreme low pollution/weather observations.  Zero in the
        normalised space is the train-only mean and is the neutral value used
        by the existing models until explicit observation-mask channels are
        introduced.
        """
        return np.nan_to_num(self.transform(x), nan=0.0, posinf=0.0,
                             neginf=0.0).astype(np.float32)

    def inverse_pollution(self, y: np.ndarray) -> np.ndarray:
        """y [..., N, K] in normalized pollution space -> raw units."""
        mu_k = self.mu[:, PRED_IDX]
        sd_k = self.sd[:, PRED_IDX]
        return (y * sd_k + mu_k).astype(np.float64)

    def transform_pollution(self, y: np.ndarray) -> np.ndarray:
        mu_k = self.mu[:, PRED_IDX]
        sd_k = self.sd[:, PRED_IDX]
        return ((y - mu_k) / sd_k).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, mu=self.mu, sd=self.sd)

    @staticmethod
    def load(path: Path) -> "Scaler":
        z = np.load(path)
        return Scaler(z["mu"], z["sd"])


# ─── Window building & chronological split ──────────────────────────────────

@dataclass
class WindowSet:
    starts: np.ndarray            # int [W], grid index of window input start
    seasons: np.ndarray           # int8 [W], routed by FIRST target timestamp


def enumerate_windows(tl: TimelineArrays, max_missing_frac: float = 0.25) -> WindowSet:
    n_valid = []
    l = INPUT_STEPS
    h = HORIZON
    t = len(tl.times)
    for i in range(t - (l + h) + 1):
        rng = slice(i, i + l + h)
        if tl.break_mask[rng].any():
            continue
        xw = tl.x_filled[i:i + l]
        nan_frac = np.isnan(xw).mean()
        if nan_frac > max_missing_frac:
            continue
        n_valid.append(i)
    starts = np.array(n_valid, dtype=np.int64)
    seasons = np.array([get_season_index(tl.times[s + INPUT_STEPS]) for s in starts], dtype=np.int8)
    return WindowSet(starts=starts, seasons=seasons)


def last_valid_window(tl: TimelineArrays, max_missing_frac: float = 0.25) -> int:
    """Index of the last INPUT_STEPS-only window usable for inference.

    Inference needs only the 12h input (targets are unknown), so unlike
    enumerate_windows it must NOT require L+H grid rows at the tail.
    """
    l = INPUT_STEPS
    t = len(tl.times)
    for s in range(t - l, -1, -1):
        if tl.break_mask[s:s + l].any():
            continue
        if np.isnan(tl.x_filled[s:s + l]).mean() > max_missing_frac:
            continue
        return s
    raise DataContractError(
        f"找不到有效的连续 {l} 小时输入窗口（数据存在过长断档或大面积缺失）")


@dataclass
class SplitIndex:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    cut_train: pd.Timestamp
    cut_val: pd.Timestamp
    i_tr: int
    i_va: int


def chronological_split(tl: TimelineArrays, ws: WindowSet,
                        train_ratio: float = 0.70, val_ratio: float = 0.15,
                        cut_train: str | pd.Timestamp | None = None,
                        cut_val: str | pd.Timestamp | None = None) -> SplitIndex:
    """Split on window ranges; windows straddling a cut are dropped (auto-purge)."""
    t = len(tl.times)
    if (cut_train is None) != (cut_val is None):
        raise DataContractError("cut_train 和 cut_val 必须同时提供")
    if cut_train is not None:
        cut_tr = pd.Timestamp(cut_train)
        cut_va = pd.Timestamp(cut_val)
        if not tl.times[0] <= cut_tr < cut_va < tl.times[-1]:
            raise DataContractError(
                f"非法显式切分边界: {cut_tr=} {cut_va=}，数据范围 "
                f"{tl.times[0]}..{tl.times[-1]}")
        i_tr = int(tl.times.searchsorted(cut_tr, side="right"))
        i_va = int(tl.times.searchsorted(cut_va, side="right"))
        cut_tr, cut_va = tl.times[i_tr - 1], tl.times[i_va - 1]
    else:
        i_tr = int(t * train_ratio)
        i_va = int(t * (train_ratio + val_ratio))
        cut_tr, cut_va = tl.times[i_tr - 1], tl.times[i_va - 1]
    lo = ws.starts
    hi = ws.starts + WINDOW_LEN - 1
    train = hi <= i_tr - 1
    val = (lo > i_tr - 1) & (hi <= i_va - 1)
    test = lo > i_va - 1
    return SplitIndex(np.where(train)[0], np.where(val)[0], np.where(test)[0],
                      cut_tr, cut_va, i_tr, i_va)


def split_from_config(tl: TimelineArrays, ws: WindowSet, data_cfg: dict) -> SplitIndex:
    """Use stable date cutoffs when configured; ratios remain a legacy fallback."""
    return chronological_split(
        tl, ws,
        float(data_cfg.get("train_ratio", 0.70)),
        float(data_cfg.get("val_ratio", 0.15)),
        cut_train=data_cfg.get("train_end"),
        cut_val=data_cfg.get("validation_end"))


def train_time_mask(tl: TimelineArrays, split: SplitIndex) -> np.ndarray:
    return np.arange(len(tl.times)) <= split.i_tr - 1


# ─── Dataset (torch) ─────────────────────────────────────────────────────────

def make_dataset(tl: TimelineArrays, ws: WindowSet, x_norm: np.ndarray,
                 scaler: Scaler, adj_seq: np.ndarray, static_adj: np.ndarray,
                 window_ids: np.ndarray, use_dynamic_adj: bool = True):
    import torch

    class MultiStepWindowDataset(torch.utils.data.Dataset):
        """item = (x [L,N,F], y [H,N,K], y_mask [H,N,K], A [L,N,N], season int)."""

        def __init__(self) -> None:
            self.x = torch.from_numpy(np.nan_to_num(x_norm, nan=0.0))
            self.tl = tl
            self.ws = ws
            self.ids = torch.from_numpy(window_ids.astype(np.int64))
            a = adj_seq if use_dynamic_adj else np.repeat(static_adj[None], len(tl.times), axis=0)
            self.a = torch.from_numpy(a.astype(np.float32))
            raw_poll = torch.from_numpy(tl.x_raw[:, :, :len(POLL_TYPES)].astype(np.float64))
            mu_k = torch.from_numpy(scaler.mu[:, PRED_IDX])
            sd_k = torch.from_numpy(scaler.sd[:, PRED_IDX])
            ynorm = (raw_poll[:, :, PRED_IDX] - mu_k) / sd_k
            self.y = ynorm
            self.y_ok = ~torch.isnan(ynorm)

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, j: int):
            w = int(self.ids[j])
            s = int(self.ws.starts[w])
            x = self.x[s:s + INPUT_STEPS]
            tg = slice(s + INPUT_STEPS, s + WINDOW_LEN)
            y = self.y[tg].clone()
            y_ok = self.y_ok[tg]
            y = torch.where(y_ok, y, torch.zeros_like(y))
            a = self.a[s:s + INPUT_STEPS]
            return x, y, y_ok.float(), a, int(self.ws.seasons[w])

    return MultiStepWindowDataset()
