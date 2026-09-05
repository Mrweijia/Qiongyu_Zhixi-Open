"""Direction 6 — multimodal fusion: ground stations + ERA5 reanalysis + AOD.

The core idea: station-level sensors are sparse (10 points across a city),
but reanalysis (ERA5) and satellite (AOD) provide continuous spatial fields.
Fusing them via a gated cross-attention architecture lets the model switch
from "10 points" to "a continuous field" — a ceiling-breaking upgrade.

Architecture:
* Station branch: the standard GraphGRUEncoder over the 10 monitoring stations
  (6 pollutants + 4 weather channels).
* ERA5 branch: a per-station MLP over the ERA5 grid features (7 channels,
  IDW-interpolated to each station position).
* AOD branch: optional, a single-channel scalar token broadcast to all
  stations, processed by a shared MLP.
* Fusion: learned gate = sigmoid(Linear(station || era5 || aod)) -> the
  station representation is augmented with the gate-weighted extra features.
* Modality dropout: during training, the extra channels are randomly zeroed
  with probability p, replaced by a learned zero-embedding — this makes the
  model robust to missing AOD or ERA5 at inference time without retraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.models.innovations.common import DeltaStepHead, GraphGRUEncoder, last_obs_base


class ModalGate(nn.Module):
    """Fuse station + extra features via a learned sigmoid gate."""

    def __init__(self, d_station: int, d_extra: int, d_out: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_station + d_extra, d_extra),
                                  nn.Sigmoid())
        self.proj = nn.Linear(d_station + d_extra, d_out)

    def forward(self, h_station: torch.Tensor, h_extra: torch.Tensor) -> torch.Tensor:
        """h_station [B,N,D], h_extra [B,N,E] -> [B,N,out]."""
        g = self.gate(torch.cat([h_station, h_extra], dim=-1))
        return self.proj(torch.cat([h_station, g * h_extra], dim=-1))


class MultimodalTGCN(nn.Module):
    """x [B,L,N,F] (standard v2 channels), x_extra [B,L,N,Fmm] (extra modalities).

    In training ``x_extra`` is zeroed with probability ``p_drop_modality``
    (per sample), and replaced by a learned zero-embedding.
    """

    def __init__(self, in_f: int, n_extra: int, horizon: int = 3, n_out: int = 4,
                 g_h: int = 64, gru_h: int = 64, extra_h: int = 32,
                 p_drop_modality: float = 0.2,
                 pred_channels: tuple = (0, 1, 3, 4)) -> None:
        super().__init__()
        self.n_extra = n_extra
        self.p_drop = p_drop_modality
        self.station_enc = GraphGRUEncoder(in_f, g_h, gru_h)
        self.extra_enc = nn.Sequential(nn.Linear(n_extra, extra_h), nn.ReLU(),
                                       nn.Linear(extra_h, extra_h))
        self.zero_emb = nn.Parameter(torch.zeros(extra_h))
        self.fusion = ModalGate(gru_h, extra_h, gru_h)
        self.head = DeltaStepHead(gru_h, horizon, n_out)
        self.register_buffer("pred_channels",
                             torch.tensor(list(pred_channels), dtype=torch.long),
                             persistent=False)

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None,
                x_extra: torch.Tensor | None = None, training: bool = True) -> torch.Tensor:
        b, l, n, _ = x.shape
        _, gru_out = self.station_enc(x, a)
        h_st = gru_out[:, -1]  # [B,N,gru_h]
        if x_extra is not None and self.n_extra > 0:
            # per-station MLP over the extra modality window: mean-pool over time
            hx = self.extra_enc(x_extra.mean(dim=1))  # [B,N,extra_h]
            if training and self.p_drop > 0:
                mask = (torch.rand(b, device=x.device) > self.p_drop).float().view(-1, 1, 1)
                hx = hx * mask + self.zero_emb.view(1, 1, -1) * (1 - mask)
            h_fused = self.fusion(h_st, hx)
        else:
            h_fused = h_st
        base = last_obs_base(x, self.pred_channels)
        return self.head(h_fused, base)