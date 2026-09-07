"""Two practical model directions added for the Changsha PM2.5 project.

``wind_gated_tcn`` keeps the observed wind-driven adjacency from the data
pipeline and uses causal dilated convolutions at several temporal scales.  It
is deliberately cheaper and more stable than an ODE or diffusion model.

``adaptive_graph_transformer`` mixes the supplied physical graph with a
learned low-rank station graph, then models each station history with a small
Transformer.  The learned graph is bounded by a convex gate so it cannot
silently discard the physical prior.

Both models follow the innovation-suite contract:
``x [B,L,N,F] + a -> y [B,H,N,K]`` in normalised units.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.innovations.common import DeltaStepHead, last_obs_base
from src.models.wu_v2 import graph_aggregate


class CausalTemporalBlock(nn.Module):
    """Residual gated temporal convolution without future padding."""

    def __init__(self, channels: int, kernel: int, dilation: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.left_pad = dilation * (kernel - 1)
        self.filter = nn.Conv2d(channels, channels, (1, kernel),
                                dilation=(1, dilation))
        self.gate = nn.Conv2d(channels, channels, (1, kernel),
                              dilation=(1, dilation))
        self.out = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z [B,C,N,L]; pad only on the left of the time dimension.
        zp = F.pad(z, (self.left_pad, 0, 0, 0))
        h = torch.tanh(self.filter(zp)) * torch.sigmoid(self.gate(zp))
        return self.norm(z + self.dropout(self.out(h)))


class WindGatedTCN(nn.Module):
    """Wind-graph aggregation followed by multi-scale causal TCN blocks."""

    def __init__(self, in_f: int, hidden: int = 64, horizon: int = 3,
                 n_out: int = 4, dilations: tuple[int, ...] = (1, 2, 4),
                 kernel_size: int = 3, dropout: float = 0.1,
                 pred_channels: tuple[int, ...] = (0, 1, 3, 4)) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_f * 2, hidden)
        self.blocks = nn.ModuleList([
            CausalTemporalBlock(hidden, kernel_size, d, dropout)
            for d in dilations
        ])
        self.time_score = nn.Linear(hidden, 1)
        self.head = DeltaStepHead(hidden, horizon, n_out)
        self.register_buffer("pred_channels",
                             torch.tensor(pred_channels, dtype=torch.long),
                             persistent=False)

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        if a is None:
            raise ValueError("WindGatedTCN requires a physical/dynamic adjacency")
        spatial = graph_aggregate(x, a)
        z = self.input_proj(torch.cat([x, spatial], dim=-1))  # [B,L,N,C]
        z = z.permute(0, 3, 2, 1)                            # [B,C,N,L]
        for block in self.blocks:
            z = block(z)
        z = z.permute(0, 3, 2, 1)                            # [B,L,N,C]
        weights = torch.softmax(self.time_score(z), dim=1)
        pooled = (weights * z).sum(dim=1)
        return self.head(pooled, last_obs_base(x, self.pred_channels))


class AdaptiveGraphTransformer(nn.Module):
    """Physical/learned graph fusion plus a supervised temporal Transformer."""

    def __init__(self, in_f: int, n_nodes: int, d_model: int = 64,
                 layers: int = 2, heads: int = 4, graph_rank: int = 16,
                 max_steps: int = 48, horizon: int = 3, n_out: int = 4,
                 dropout: float = 0.1,
                 pred_channels: tuple[int, ...] = (0, 1, 3, 4)) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.node_src = nn.Parameter(torch.randn(n_nodes, graph_rank) * 0.05)
        self.node_dst = nn.Parameter(torch.randn(n_nodes, graph_rank) * 0.05)
        self.graph_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.input_proj = nn.Linear(in_f * 2, d_model)
        self.time_emb = nn.Parameter(torch.randn(max_steps, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = DeltaStepHead(d_model, horizon, n_out)
        self.register_buffer("pred_channels",
                             torch.tensor(pred_channels, dtype=torch.long),
                             persistent=False)

    def learned_adjacency(self) -> torch.Tensor:
        scores = torch.relu(self.node_src @ self.node_dst.T)
        scores = scores + torch.eye(self.n_nodes, device=scores.device)
        return torch.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        b, steps, n, _ = x.shape
        if n != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} nodes, got {n}")
        if steps > self.time_emb.shape[0]:
            raise ValueError(f"input steps {steps} exceed max_steps={self.time_emb.shape[0]}")
        learned = self.learned_adjacency()
        learned_x = graph_aggregate(x, learned)
        if a is None:
            physical_x = learned_x
        else:
            physical_x = graph_aggregate(x, a)
        gate = torch.sigmoid(self.graph_gate_logit)
        spatial = gate * physical_x + (1.0 - gate) * learned_x
        z = self.input_proj(torch.cat([x, spatial], dim=-1))
        z = z + self.time_emb[:steps].view(1, steps, 1, -1)
        seq = z.permute(0, 2, 1, 3).reshape(b * n, steps, -1)
        h = self.norm(self.encoder(seq))[:, -1].reshape(b, n, -1)
        return self.head(h, last_obs_base(x, self.pred_channels))


__all__ = ["CausalTemporalBlock", "WindGatedTCN", "AdaptiveGraphTransformer"]
