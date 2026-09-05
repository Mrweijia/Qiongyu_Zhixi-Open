"""Direction 4 — spatiotemporal foundation model (masked pre-training → fine-tune).

Architecture: each `(station, timestep)` pair is a token. A standard
TransformerEncoder (pre-LN) with full attention over L*N tokens (up to
12*10=120 tokens — tiny, no flash attention needed) is pre-trained on
multi-city data via random masking of pollutant values, then fine-tuned on
the Changsha multi-step forecast task.

Pretraining city list (see ``src/training/pretrain_foundation.py``):
* UCI Beijing 12 stations (2013–2017 hourly)
* Changsha 10 stations (2014–2025 historical archive)
* CNEMC multi-city snapshot (latest 3 days, 300+ stations, capped per city)

Fine-tuning input projection: the pre-trained model expects 6 pollutants +
1 observed-mask channel = 7 input dims; the fine-tune data has 6 pollutants
+ 4 weather features = 10 dims. The adapter (:func:`adapt_input_proj`)
transfers the first 6 dims and initialises the weather channels from zero,
so the weather features are learned from scratch — a standard domain-shift
mitigation for weather-vs-no-weather pretraining.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.models.innovations.common import DeltaStepHead


# ─── positional encoding ────────────────────────────────────────────────────

class SinePosition(nn.Module):
    def __init__(self, d: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, ::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.pe[idx]


# ─── core model ─────────────────────────────────────────────────────────────

class STMaskFormer(nn.Module):
    def __init__(self, n_stations: int, in_f: int = 7, d: int = 128,
                 layers: int = 4, heads: int = 4, hour_slots: int = 24,
                 out_poll: int = 6, horizon: int = 3, n_out: int = 4,
                 dropout: float = 0.1,
                 pred_channels: tuple = (0, 1, 3, 4)) -> None:
        super().__init__()
        self.d = d
        self.in_f = in_f
        self.out_poll = out_poll
        self.register_buffer("pred_channels",
                             torch.tensor(list(pred_channels), dtype=torch.long),
                             persistent=False)
        # input projection
        self.val_proj = nn.Linear(in_f, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.station_emb = nn.Embedding(n_stations, d)
        self.hour_emb = nn.Embedding(hour_slots, d)
        self.wday_emb = nn.Embedding(7, d)  # 0=Monday
        # transformer
        enc_layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                               dropout=dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.out_norm = nn.LayerNorm(d)
        # pre-training head
        self.pretrain_head = nn.Linear(d, out_poll)
        # fine-tune head
        self.ft_head = DeltaStepHead(d, horizon, n_out)

    # -- tokenisation ---------------------------------------------------------
    def tokenise(self, x: torch.Tensor, mask: torch.Tensor | None,
                 station_ids: torch.Tensor, hour_ids: torch.Tensor,
                 wday_ids: torch.Tensor) -> torch.Tensor:
        """x [B,L,N,F]; mask [B,L,N] (bool, 1=observed, 0=masked) or None.
        station_ids [N] long, hour_ids [B,L], wday_ids [B,L] long.
        Returns z [B,L,N,d]."""
        b, l, n, _ = x.shape
        z = self.val_proj(x)  # [B,L,N,d]
        st = self.station_emb(station_ids)  # [N,d] -> 1,1,N,d
        hr = self.hour_emb(hour_ids)  # [B,L] -> unsqueeze -> B,L,1,d
        wd = self.wday_emb(wday_ids)
        z = z + st.view(1, 1, n, self.d) + hr.view(b, l, 1, self.d) + wd.view(b, l, 1, self.d)
        if mask is not None:
            z = z * mask.float().unsqueeze(-1) + self.mask_token.view(1, 1, 1, self.d) * (1 - mask.float().unsqueeze(-1))
        return z

    # -- pretraining forward --------------------------------------------------
    def pretrain_forward(self, x: torch.Tensor, mask: torch.Tensor,
                         station_ids: torch.Tensor, hour_ids: torch.Tensor,
                         wday_ids: torch.Tensor) -> torch.Tensor:
        """x [B,L,N,F] raw values (normalised), mask [B,L,N] bool obs indicator.
        -> pred [B,L,N,out_poll] at ALL positions (loss computed on masked only)."""
        z = self.tokenise(x, mask, station_ids, hour_ids, wday_ids)
        b, l, n = x.shape[:3]
        z = z.reshape(b, l * n, self.d)
        out = self.out_norm(self.encoder(z))
        return self.pretrain_head(out).reshape(b, l, n, self.out_poll)

    # -- fine-tuning forward --------------------------------------------------
    def forecast(self, x: torch.Tensor, station_ids: torch.Tensor,
                 hour_ids: torch.Tensor, wday_ids: torch.Tensor) -> torch.Tensor:
        """x [B,L,N,F] normalised full input (fine-tune: F=10, weather included).
        -> [B,H,N,K] delta prediction."""
        z = self.tokenise(x, None, station_ids, hour_ids, wday_ids)
        b, l, n = x.shape[:3]
        z = z.reshape(b, l * n, self.d)
        h_tokens = self.out_norm(self.encoder(z)).reshape(b, l, n, self.d)
        # per-station pooling: mean across time
        h_st = h_tokens.mean(dim=1)  # [B,N,d]
        base = x[:, -1][:, :, self.pred_channels]  # [B,N,n_out]
        return self.ft_head(h_st, base)

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None,
                station_ids: torch.Tensor | None = None,
                hour_ids: torch.Tensor | None = None,
                wday_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Trainer-friendly forecast dispatch; ``a`` is intentionally unused."""
        b, l, n, _ = x.shape
        if station_ids is None:
            station_ids = torch.arange(n, device=x.device)
        if hour_ids is None:
            hour_ids = torch.arange(l, device=x.device).view(1, l).expand(b, -1) % 24
        if wday_ids is None:
            wday_ids = torch.zeros((b, l), dtype=torch.long, device=x.device)
        return self.forecast(x, station_ids, hour_ids, wday_ids)


# ─── checkpoint helpers ─────────────────────────────────────────────────────

def adapt_input_proj(state_dict: dict, from_f: int, to_f: int,
                     d: int, source_features: list[str] | None = None,
                     target_features: list[str] | None = None) -> dict:
    """Transfer the pre-trained val_proj weight from ``from_f`` to ``to_f``
    input channels: copy the first ``from_f`` cols, zero-init the rest."""
    w = state_dict["val_proj.weight"]
    new_w = torch.zeros(d, to_f)
    if source_features and target_features:
        source_pos = {name: i for i, name in enumerate(source_features)}
        for target_i, name in enumerate(target_features):
            source_i = source_pos.get(name)
            if source_i is not None and source_i < w.shape[1]:
                new_w[:, target_i] = w[:, source_i]
    else:
        new_w[:, :min(from_f, to_f)] = w[:, :min(from_f, to_f)]
    state_dict["val_proj.weight"] = new_w
    if "val_proj.bias" in state_dict:
        state_dict["val_proj.bias"] = state_dict["val_proj.bias"].clone()
    return state_dict


def save_foundation_ckpt(model: STMaskFormer, path: Path, stats: dict,
                          config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path.with_suffix(".pt"))
    meta = {**config, "n_stations": model.station_emb.num_embeddings,
            "stats": stats}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_foundation_ckpt(path: Path, **overrides) -> tuple[STMaskFormer, dict]:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    cfg = {**meta, **overrides}
    m = STMaskFormer(n_stations=cfg["n_stations"], in_f=cfg.get("in_f", 7),
                     d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"],
                     hour_slots=cfg.get("hour_slots", 24), out_poll=cfg["out_poll"],
                     horizon=cfg.get("horizon", 3), n_out=cfg.get("n_out", 4),
                     dropout=cfg.get("dropout", 0.1))
    sd = torch.load(path.with_suffix(".pt"), map_location="cpu")
    source_in_f = meta.get("in_f", sd["val_proj.weight"].shape[1])
    if "in_f" in overrides and overrides["in_f"] != source_in_f:
        source_features = meta.get("feature_order")
        if source_features is None and source_in_f == 7:
            source_features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "obs_mask"]
        sd = adapt_input_proj(
            sd, source_in_f, overrides["in_f"], cfg["d"],
            source_features=source_features,
            target_features=overrides.get("target_feature_order"))
    # Map named target stations when the pretraining manifest contains them.
    # Otherwise start every target from the source mean; the first rows often
    # belong to another city and are not semantically interchangeable.
    target_n = cfg["n_stations"]
    old_emb = sd.get("station_emb.weight")
    target_names = overrides.get("target_station_names")
    registry = meta.get("station_registry", [])
    if old_emb is not None and target_names:
        lookup = {str(row["station"]): int(row["embedding_index"])
                  for row in registry if "station" in row and "embedding_index" in row}
        mean_emb = old_emb.mean(0)
        rows = [old_emb[lookup[name]] if name in lookup and lookup[name] < len(old_emb)
                else mean_emb for name in target_names]
        sd["station_emb.weight"] = torch.stack(rows)
    elif old_emb is not None and old_emb.shape[0] != target_n:
        sd["station_emb.weight"] = old_emb.mean(0, keepdim=True).expand(target_n, -1).clone()
    m.load_state_dict(sd, strict=False)
    stats = cfg.get("stats", {})
    return m, stats
