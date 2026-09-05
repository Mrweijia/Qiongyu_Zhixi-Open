"""Shared building blocks for the six stage-3 innovation models.

Conventions (identical across all innovation models):
* input  x : float32 [B, L, N, F]  normalised (pipeline ``Scaler`` units)
* adj      : [N, N] or [T, N, N] or [B, T, N, N] row-normalised message matrices
             (see :func:`src.models.wu_v2.graph_aggregate`)
* output   : float32 [B, H, N, K]  — matches the dataset target layout
  ``y [B, H, N, K]`` directly (unlike v2 which returns [B, N, H, K]).
* delta heads predict increments over the last normalised observation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.wu_v2 import GraphConv, graph_aggregate  # noqa: F401  (re-export)

__all__ = ["GraphConv", "graph_aggregate", "GraphGRUEncoder", "DeltaStepHead"]


class GraphGRUEncoder(nn.Module):
    """GNN (spatial) -> GRU (temporal) per station, the T-GCN workhorse.

    forward(x [B,T,N,F], a) -> (z [B,T,N,g_h] post-GCN, h [B,T,N,gru_h] gru states).
    """

    def __init__(self, in_f: int, g_h: int, gru_h: int, layers: int = 1) -> None:
        super().__init__()
        dims = [in_f] + [g_h] * max(layers, 1)
        self.gcn = nn.ModuleList(
            [GraphConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.gru = nn.GRU(g_h, gru_h, batch_first=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        z = x
        for blk in self.gcn:
            z = blk(graph_aggregate(z, a))
        b, t, n, _ = z.shape
        seq = z.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        gru_out, _ = self.gru(seq)
        gru_out = gru_out.reshape(b, n, t, -1).permute(0, 2, 1, 3)  # [B,T,N,gru_h]
        return z, gru_out


class DeltaStepHead(nn.Module):
    """Direct multi-step heads anchored at the last observed value.

    forward(h_last [B,N,D], base [B,N,K]) -> [B,H,N,K] (delta + anchor).
    """

    def __init__(self, in_dim: int, horizon: int, n_out: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_out = n_out
        self.heads = nn.ModuleList([nn.Linear(in_dim, n_out) for _ in range(horizon)])

    def forward(self, h_last: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        steps = torch.stack([head(h_last) for head in self.heads], dim=2)  # [B,N,H,K]
        steps = steps + base[:, :, None, :]
        return steps.permute(0, 2, 1, 3)  # [B,H,N,K]


def last_obs_base(x: torch.Tensor, pred_channels: torch.Tensor) -> torch.Tensor:
    """x [B,L,N,F] -> [B,N,K] last-step values of the predicted channels."""
    return x[:, -1][:, :, pred_channels]
