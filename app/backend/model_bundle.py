"""Shared bundle health checker - single source for predict/online/models."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_KEYS = ("stations", "feature_order", "pred_names", "architecture", "model_sha256")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def check_bundle(bundle_dir: Path) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Return (ok, reason_code, manifest).

    Reason codes are stable machine-readable strings for 503 JSON.
    ok==True means manifest.json + model.pt + scaler.npz all present and coherent.
    No GPU model load is performed here (health must stay lightweight).
    """
    manifest_path = bundle_dir / "manifest.json"
    model_path = bundle_dir / "model.pt"
    scaler_path = bundle_dir / "scaler.npz"

    if not manifest_path.is_file():
        return False, "missing_manifest", None
    if not model_path.is_file():
        return False, "missing_model_pt", None
    if not scaler_path.is_file():
        return False, "missing_scaler", None

    manifest = _load_json(manifest_path)
    if manifest is None:
        return False, "manifest_unparseable", None

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            return False, f"manifest_missing_field:{key}", None

    stations = manifest.get("stations")
    feature_order = manifest.get("feature_order")
    pred_names = manifest.get("pred_names")
    arch = manifest.get("architecture")
    if not isinstance(stations, list) or not stations:
        return False, "manifest_bad_stations", manifest
    if not isinstance(feature_order, list) or not feature_order:
        return False, "manifest_bad_feature_order", manifest
    if not isinstance(pred_names, list) or not pred_names:
        return False, "manifest_bad_pred_names", manifest
    if not isinstance(arch, dict):
        return False, "manifest_bad_architecture", manifest

    # scaler must be loadable and shape must match per-station x feature
    try:
        import numpy as np  # lazy - absent in very minimal env returns invalid
        data = np.load(str(scaler_path))
        # accept both NpzFile and dict-like
        mu = data["mu"] if "mu" in data else data.get("mu") if hasattr(data, "get") else None
        sd = data["sd"] if "sd" in data else data.get("sd") if hasattr(data, "get") else None
        # fallback: check at least some array exists
        if mu is None and sd is None:
            # try list keys
            keys = list(data.files) if hasattr(data, "files") else []
            if not keys:
                return False, "scaler_empty", manifest
        else:
            # shape check: stations x features
            if mu is not None and hasattr(mu, "shape"):
                exp = (len(stations), len(feature_order))
                if mu.shape != exp:
                    return False, f"scaler_shape_mismatch:got_{mu.shape}_exp_{exp}", manifest
    except Exception:
        return False, "scaler_unreadable", manifest

    # optional hash check if file small enough; not required but cheap
    # do not fail on hash mismatch alone -> caller decides

    return True, None, manifest

