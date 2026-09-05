"""Unified experiment protocol: schema, validation, sample index, receipt.

SCI-01: two protocols must never share a leaderboard unless their stations,
targets, window, horizon, mask and split are identical.  This module gives
every run a stable ``protocol_id``, a deterministic sample index, a leakage
audit and a machine-readable receipt without rewriting the hard-coded
``src.data.pipeline`` (which keeps serving the deployed multistep_v2 bundle).

Protocols differ only in declarative fields; the same data timeline is built
once (10 stations) and each protocol selects its own station subset.  This is
the same subset semantics ``rolling_feature_lgbm --drop-station 1344A`` already
uses, so it cannot silently break the existing evaluation layer.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import pipeline as pl  # noqa: E402


# ─── Schema ────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "2026-09-05-sci01-v1"

# Fields a protocol YAML must declare before a receipt may be produced.
REQUIRED_PROTOCOL_FIELDS = (
    "protocol_id",
    "schema_version",
    "description",
    "data",
    "stations",
    "pollutants",
    "input_steps",
    "horizons",
    "split",
    "seed",
    "eval_access_type",
)

REQUIRED_DATA_FIELDS = ("pollution", "weather", "weather_timezone",
                        "max_gap_hours", "max_missing_frac")
REQUIRED_SPLIT_FIELDS = ("train_end", "validation_end")


class ProtocolError(ValueError):
    """Raised when a protocol spec is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    schema_version: str
    description: str
    data: dict[str, Any]
    stations: tuple[str, ...]
    pollutants: tuple[str, ...]
    input_steps: int
    horizons: tuple[int, ...]
    split: dict[str, Any]
    seed: int
    eval_access_type: str
    raw: dict[str, Any]

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def window_len(self) -> int:
        return self.input_steps + self.max_horizon

    @property
    def pollution_path(self) -> str:
        return self.data["pollution"]

    @property
    def weather_path(self) -> str:
        return self.data["weather"]

    @property
    def weather_timezone(self) -> str:
        return self.data["weather_timezone"]

    @property
    def max_gap_hours(self) -> int:
        return int(self.data["max_gap_hours"])

    @property
    def max_missing_frac(self) -> float:
        return float(self.data["max_missing_frac"])

    @property
    def train_end(self) -> str:
        return self.split["train_end"]

    @property
    def validation_end(self) -> str:
        return self.split["validation_end"]


def _require(mapping: dict, key: str, scope: str) -> Any:
    if key not in mapping:
        raise ProtocolError(f"protocol {scope} missing required field: {key}")
    return mapping[key]


def load_protocol(path: Path | str) -> ProtocolSpec:
    """Load and validate a protocol YAML into a frozen spec."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError(f"cannot read protocol file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProtocolError(f"protocol file {path} must hold a mapping")

    for key in REQUIRED_PROTOCOL_FIELDS:
        _require(raw, key, "")

    data = _require(raw, "data", "")
    for key in REQUIRED_DATA_FIELDS:
        _require(data, key, "data")

    split = _require(raw, "split", "")
    for key in REQUIRED_SPLIT_FIELDS:
        _require(split, key, "split")

    stations = tuple(raw["stations"])
    pollutants = tuple(raw["pollutants"])
    if not stations or not pollutants:
        raise ProtocolError("protocol stations and pollutants must be non-empty")
    if len(set(stations)) != len(stations):
        raise ProtocolError("protocol stations must not repeat")

    unknown_stations = set(stations) - set(pl.SEL_STATIONS)
    if unknown_stations:
        raise ProtocolError(
            f"protocol references stations absent from the timeline: "
            f"{sorted(unknown_stations)}")

    input_steps = int(raw["input_steps"])
    horizons = tuple(sorted(int(h) for h in raw["horizons"]))
    if input_steps < 1 or not horizons or any(h < 1 for h in horizons):
        raise ProtocolError("input_steps and horizons must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ProtocolError("horizons must not repeat")

    seed = int(raw.get("seed", 0))
    eval_access_type = str(raw.get("eval_access_type", "development_test"))

    return ProtocolSpec(
        protocol_id=str(raw["protocol_id"]),
        schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
        description=str(raw.get("description", "")),
        data=dict(data),
        stations=stations,
        pollutants=pollutants,
        input_steps=input_steps,
        horizons=horizons,
        split=dict(split),
        seed=seed,
        eval_access_type=eval_access_type,
        raw=dict(raw),
    )


# ─── Sample index / windows ────────────────────────────────────────────────

@dataclass
class ProtocolWindows:
    starts: np.ndarray          # [W] grid index of window input start
    station_index: np.ndarray   # [W] index into full 10-station timeline


def enumerate_protocol_windows(tl: pl.TimelineArrays,
                               spec: ProtocolSpec,
                               station_index: int) -> np.ndarray:
    """Deterministic window starts for one station under the protocol window.

    A window is usable when its input+max_horizon grid slice is free of
    ``break_mask`` holes and its *input* missing fraction stays under
    ``max_missing_frac``.  This mirrors ``pl.enumerate_windows`` but honours
    the protocol's own input_steps / max_horizon instead of the global
    12h/3h constants.
    """
    l = spec.input_steps
    h = spec.max_horizon
    t = len(tl.times)
    n_valid = []
    for i in range(t - (l + h) + 1):
        rng = slice(i, i + l + h)
        if tl.break_mask[rng].any():
            continue
        xw = tl.x_filled[i:i + l, station_index, :]
        nan_frac = np.isnan(xw).mean()
        if nan_frac > spec.max_missing_frac:
            continue
        n_valid.append(i)
    return np.asarray(n_valid, dtype=np.int64)


def split_protocol_windows(starts: np.ndarray, tl: pl.TimelineArrays,
                           spec: ProtocolSpec) -> dict[str, np.ndarray]:
    """Chronological split with purge of windows straddling a cut.

    A window belongs to a partition only if its whole input+target span lies
    inside the partition's grid range; windows crossing a boundary are dropped
    (same auto-purge semantics as ``pl.chronological_split``).
    """
    if spec.train_end >= spec.validation_end:
        raise ProtocolError("split.train_end must precede split.validation_end")
    i_tr = int(tl.times.searchsorted(pd.Timestamp(spec.train_end), side="right"))
    i_va = int(tl.times.searchsorted(pd.Timestamp(spec.validation_end), side="right"))
    lo = starts
    hi = starts + spec.window_len - 1
    train = np.flatnonzero(hi <= i_tr - 1)
    val = np.flatnonzero((lo > i_tr - 1) & (hi <= i_va - 1))
    test = np.flatnonzero(lo > i_va - 1)
    return {"train": train, "validation": val, "test": test}


def station_indices(spec: ProtocolSpec) -> np.ndarray:
    """Index into the full 10-station timeline for this protocol's stations."""
    return np.asarray(
        [i for i, station in enumerate(pl.SEL_STATIONS) if station in spec.stations],
        dtype=np.int64,
    )


def target_mask(tl: pl.TimelineArrays, starts: np.ndarray, spec: ProtocolSpec,
                station_index: int, pollutant: str) -> np.ndarray:
    """[W, H] boolean mask of observed (non-filled) target values.

    The mask uses the RAW observation (x_raw) so a causally filled value is
    never scored as a true observation.  Columns align with sorted horizons.
    """
    k = pl.POLL_TYPES.index(pollutant)
    h = len(spec.horizons)
    mask = np.zeros((len(starts), h), dtype=bool)
    for j, horizon in enumerate(spec.horizons):
        target_rows = starts + spec.input_steps + horizon - 1
        mask[:, j] = ~np.isnan(
            tl.x_raw[target_rows, station_index, k])
    return mask


# ─── Receipt ───────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_index_digest(starts_by_station: dict[int, np.ndarray]) -> str:
    h = hashlib.sha256()
    for station_index in sorted(starts_by_station):
        arr = np.asarray(starts_by_station[station_index], dtype=np.int64)
        h.update(arr.tobytes())
        h.update(b"\x00")
    return h.hexdigest()


def build_receipt(tl: pl.TimelineArrays,
                  spec: ProtocolSpec,
                  union_starts: np.ndarray,
                  starts_by_station: dict[int, np.ndarray],
                  split: dict[str, np.ndarray],
                  poll_path: Path,
                  weather_path: Path) -> dict[str, Any]:
    """Machine-readable receipt for one protocol run."""
    split_ranges = {}
    for name, ids in split.items():
        if len(ids) == 0:
            split_ranges[name] = {"windows": 0}
            continue
        sel = union_starts[ids]
        split_ranges[name] = {
            "windows": int(len(ids)),
            "input_start": str(tl.times[sel.min()]),
            "target_end": str(tl.times[(sel + spec.window_len - 1).max()]),
        }
    no_overlap = (
        split_ranges["train"]["windows"] and split_ranges["validation"]["windows"]
        and split_ranges["test"]["windows"]
        and split_ranges["train"]["target_end"] < split_ranges["validation"]["input_start"]
        and split_ranges["validation"]["target_end"] < split_ranges["test"]["input_start"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": spec.protocol_id,
        "description": spec.description,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": {"pollution": sha256(poll_path), "weather": sha256(weather_path)},
        "stations": list(spec.stations),
        "pollutants": list(spec.pollutants),
        "input_steps": spec.input_steps,
        "horizons": list(spec.horizons),
        "split": split_ranges,
        "no_window_overlap": bool(no_overlap),
        "sample_index_sha256": sample_index_digest(starts_by_station),
        "seed": spec.seed,
        "eval_access_type": spec.eval_access_type,
        "pipeline_revision": pl.PIPELINE_REVISION,
        "protocol_schema_version": SCHEMA_VERSION,
        "full_receipt_hash": "",
    }


def finalize_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Canonicalise field order and add the self-referential receipt hash."""
    canonical = json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    receipt["full_receipt_hash"] = digest
    return receipt


# ─── Deterministic single-entry runner ─────────────────────────────────────

def dry_run_protocol(protocol_path: Path | str,
                     output_dir: Path | str) -> dict[str, Any]:
    """Load a protocol, build its sample index and write a receipt JSON.

    This is the minimal closed loop SCI-01 requires: it proves protocol
    loading, sample construction and receipt generation without training any
    model.  Station-level windows are merged per protocol; the split partition
    is station-invariant (all stations share the same chronological grid).
    """
    protocol_path = Path(protocol_path)
    spec = load_protocol(protocol_path)

    tl = pl.build_timeline_arrays(
        ROOT / spec.pollution_path,
        ROOT / spec.weather_path,
        max_gap_hours=spec.max_gap_hours,
        weather_timezone=spec.weather_timezone,
    )
    starts_by_station = {}
    for si in station_indices(spec):
        starts_by_station[int(si)] = enumerate_protocol_windows(tl, spec, int(si))

    # The partition is computed on the union of window starts.  All stations
    # share the same timeline so any station's starts are representative;
    # merge via set union to stay safe even if stations differ in missingness.
    union = np.sort(np.unique(np.concatenate(list(starts_by_station.values()))))
    split = split_protocol_windows(union, tl, spec)

    receipt = build_receipt(tl, spec, union, starts_by_station, split,
                            ROOT / spec.pollution_path, ROOT / spec.weather_path)
    receipt["straddle_windows_purged"] = int(
        len(union) - sum(len(v) for v in split.values()))
    receipt["split_windows_per_station"] = {
        str(pl.SEL_STATIONS[si]): int(len(v))
        for si, v in starts_by_station.items()
    }
    receipt = finalize_receipt(receipt)

    out = ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{spec.protocol_id}_receipt.json"
    out_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return receipt
