"""Bundle-backed multi-step predictor shared by training eval and the web backend.

Public API (stable contract, see docs in bundle manifest):
    predictor = load_bundle(Path("models/checkpoints/multistep_v2"), device)
    result    = predictor.predict("pollution.csv", "weather.csv")  # np.ndarray [H, N, K]
    result    = predictor.predict_with_timeline(...)                # adds times/stations
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import pipeline as pl
from src.models.wu_v2 import HardMOESeasonV2


class InputValidationError(ValueError):
    pass


@dataclass
class PredictionResult:
    values: np.ndarray                 # [H, N, K] denormalized µg/m³
    times: pd.DatetimeIndex            # [H] forecast timestamps
    origin: pd.Timestamp               # last observed input hour
    stations: list
    pollutants: list
    # Completeness of the input window, surfaced so that a displayed grade can
    # be traced back to how much real observation actually supported it.
    quality: dict | None = None


class MultiStepPredictor:
    def __init__(self, model: torch.nn.Module, scaler: pl.Scaler,
                 manifest: dict, device: torch.device) -> None:
        self.model = model
        self.scaler = scaler
        self.manifest = manifest
        self.device = device
        self.use_dynamic_adj = bool(manifest["architecture"]["use_dynamic_adj"])
        self.weather_timezone = pl.validate_weather_timezone(
            manifest.get("weather_timezone")
        )

    @torch.no_grad()
    def predict(self, pollution_csv: Path | str, weather_csv: Path | str) -> np.ndarray:
        return self.predict_with_timeline(pollution_csv, weather_csv).values

    @torch.no_grad()
    def predict_with_timeline(self, pollution_csv: Path | str,
                              weather_csv: Path | str) -> PredictionResult:
        tl = pl.build_timeline_arrays(
            pollution_csv,
            weather_csv,
            weather_timezone=self.weather_timezone,
        )
        if len(tl.times) < pl.INPUT_STEPS:
            raise InputValidationError(
                f"输入不足：整点网格仅 {len(tl.times)} 小时，至少需要 {pl.INPUT_STEPS} 小时")
        try:
            start = pl.last_valid_window(tl)
        except pl.DataContractError as exc:
            raise InputValidationError(str(exc)) from exc
        x_norm = self.scaler.transform_filled(tl.x_filled)
        start = int(start)
        x = torch.from_numpy(x_norm[start:start + pl.INPUT_STEPS]).unsqueeze(0).to(self.device)
        static_a = pl.build_static_geo_adj()
        adj = pl.build_dynamic_adj_seq(static_a, tl) if self.use_dynamic_adj else \
            np.repeat(static_a[None], len(tl.times), axis=0)
        a = torch.from_numpy(adj[start:start + pl.INPUT_STEPS].astype(np.float32)).unsqueeze(0).to(self.device)
        origin = tl.times[start + pl.INPUT_STEPS - 1]
        season = pl.get_season_index(origin + pd.Timedelta(hours=1))
        out = self.model(x, a, torch.tensor([season], device=self.device))  # [1,N,H,K]
        out = out[0].permute(1, 0, 2).cpu().numpy()                        # [H,N,K]
        values = self.scaler.inverse_pollution(out.astype(np.float64))
        window = tl.x_raw[start:start + pl.INPUT_STEPS]
        return PredictionResult(
            values=values,
            times=pd.date_range(origin + pd.Timedelta(hours=1), periods=pl.HORIZON, freq="h"),
            origin=origin,
            stations=list(pl.SEL_STATIONS),
            pollutants=list(pl.PRED_NAMES),
            quality={
                "window_hours": int(window.shape[0]),
                "required_hours": int(pl.INPUT_STEPS),
                "grid_hours": int(len(tl.times)),
                "broken_hours": int(tl.broken_hours),
                "input_missing_frac": round(float(np.isnan(window).mean()), 6),
                "output_missing_cells": int(np.count_nonzero(np.isnan(values))),
            },
        )


def load_bundle(bundle_dir: Path | str, device: torch.device | str = "auto") -> MultiStepPredictor:
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        pl.validate_weather_timezone(manifest.get("weather_timezone"))
    except pl.DataContractError as exc:
        raise InputValidationError(f"模型包的气象时区契约无效：{exc}") from exc
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = manifest["architecture"]
    model = HardMOESeasonV2(
        in_f=arch["in_f"], g_h=arch["g_h"], gru_h=arch["gru_h"], attn_dim=arch["attn_dim"],
        horizon=arch["horizon"], n_out=arch["n_out"], n_experts=arch["n_experts"],
        use_attention=arch["use_attention"], use_seasonal_experts=arch["use_seasonal_experts"],
        predict_delta=arch.get("predict_delta", False), pred_channels=tuple(pl.PRED_IDX),
    )
    model.load_state_dict(torch.load(bundle_dir / "model.pt", map_location=device))
    model.to(device).eval()
    scaler = pl.Scaler.load(bundle_dir / "scaler.npz")
    return MultiStepPredictor(model, scaler, manifest, device)
