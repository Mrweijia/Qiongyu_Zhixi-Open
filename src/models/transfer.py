"""Portable input contract for the SCI-05 flexible and strict miao models.

Unlike the deployed Changsha bundle, these helpers never encode station IDs or
the number of nodes into the scaler.  A caller supplies its station list and
row-normalised adjacency; the model is therefore portable at the interface
level, while its accuracy still needs local validation before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class TransferInputError(ValueError):
    """Raised when a request is outside a model package's declared boundary."""


@dataclass(frozen=True)
class InputCapabilities:
    """Machine-readable safety boundary stored in each transfer manifest."""

    model_id: str
    feature_order: tuple[str, ...]
    input_steps: int
    horizons: tuple[int, ...]
    max_missing_fraction: float
    max_consecutive_missing_hours: int
    allow_missing_station: bool
    allow_missing_feature: bool

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "feature_order": list(self.feature_order),
            "input_steps": self.input_steps,
            "horizons": list(self.horizons),
            "max_missing_fraction": self.max_missing_fraction,
            "max_consecutive_missing_hours": self.max_consecutive_missing_hours,
            "allow_missing_station": self.allow_missing_station,
            "allow_missing_feature": self.allow_missing_feature,
        }


@dataclass(frozen=True)
class TransferScaler:
    """Feature-wise train-only scaler with shape ``[F]``, never ``[N,F]``."""

    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, train_rows: np.ndarray) -> "TransferScaler":
        if values.ndim != 3:
            raise TransferInputError("values must have shape [time, station, feature]")
        if train_rows.shape != (values.shape[0],) or not train_rows.any():
            raise TransferInputError("train_rows must be a nonempty [time] mask")
        train = values[train_rows]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(train, axis=(0, 1))
            sd = np.nanstd(train, axis=(0, 1))
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
        return cls(mu.astype(np.float64), sd.astype(np.float64))

    def transform_filled(self, values: np.ndarray) -> np.ndarray:
        if values.ndim != 3 or values.shape[-1] != len(self.mu):
            raise TransferInputError("values feature dimension does not match scaler")
        scaled = (values - self.mu[None, None, :]) / self.sd[None, None, :]
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def inverse_targets(self, values: np.ndarray, target_indices: Iterable[int]) -> np.ndarray:
        idx = np.asarray(tuple(target_indices), dtype=np.int64)
        return values * self.sd[idx][None, None, :] + self.mu[idx][None, None, :]


def _longest_nan_run(mask: np.ndarray) -> int:
    """Largest consecutive run in a boolean 1-D mask."""
    best = current = 0
    for missing in mask:
        current = current + 1 if missing else 0
        best = max(best, current)
    return best


def validate_transfer_window(
    values: np.ndarray,
    stations: list[str] | tuple[str, ...],
    capabilities: InputCapabilities,
) -> dict[str, object]:
    """Validate a raw ``[L,N,F]`` window and return an auditable summary."""
    if values.ndim != 3:
        raise TransferInputError("input must have shape [time, station, feature]")
    steps, nodes, features = values.shape
    if steps != capabilities.input_steps:
        raise TransferInputError(f"requires {capabilities.input_steps} input hours, got {steps}")
    if nodes != len(stations) or not stations or len(set(stations)) != len(stations):
        raise TransferInputError("stations must be a nonempty unique list matching input nodes")
    if features != len(capabilities.feature_order):
        raise TransferInputError("feature dimension does not match model capability")

    missing = ~np.isfinite(values)
    missing_fraction = float(missing.mean())
    station_missing = missing.all(axis=(0, 2))
    feature_missing = missing.all(axis=(0, 1))
    max_gap = max(_longest_nan_run(missing[:, node, :].all(axis=1))
                  for node in range(nodes))
    if station_missing.any() and not capabilities.allow_missing_station:
        raise TransferInputError("entire missing station is not supported by this model")
    if feature_missing.any() and not capabilities.allow_missing_feature:
        names = [capabilities.feature_order[i] for i in np.flatnonzero(feature_missing)]
        raise TransferInputError("entire missing feature is not supported: " + ", ".join(names))
    if max_gap > capabilities.max_consecutive_missing_hours:
        raise TransferInputError("continuous station gap exceeds model capability")
    if missing_fraction > capabilities.max_missing_fraction:
        raise TransferInputError("input missing fraction exceeds model capability")
    return {
        "stations": list(stations),
        "input_missing_fraction": missing_fraction,
        "max_consecutive_station_gap_hours": int(max_gap),
        "missing_station_count": int(station_missing.sum()),
        "missing_feature_count": int(feature_missing.sum()),
    }


def validate_adjacency(adjacency: np.ndarray, stations: list[str] | tuple[str, ...]) -> np.ndarray:
    """Require a finite row-normalised graph supplied for the caller's nodes."""
    nodes = len(stations)
    if adjacency.shape != (nodes, nodes) or not np.isfinite(adjacency).all():
        raise TransferInputError("adjacency must be finite with shape [stations, stations]")
    if (adjacency < 0).any() or not np.allclose(adjacency.sum(axis=1), 1.0, atol=1e-5):
        raise TransferInputError("adjacency rows must be nonnegative and sum to one")
    return adjacency.astype(np.float32, copy=False)
