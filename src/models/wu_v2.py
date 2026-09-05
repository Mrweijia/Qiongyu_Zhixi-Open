"""Multi-step (T+1..T+3) T-GCN v2 with honest attention paths and ablation flags.

Output shape: [B, N_stations, H=3, K=4] where K = (PM2.5, PM10, NO2, O3).

Differences vs ``src/models/wu.py`` (audit 5.1 / 5.5 / 5.6):
* feature & temporal attention outputs FEED the downstream path (no dead branch);
* direct multi-step heads instead of a single 1-step head;
* vectorised batched GCN (einsum) and batched seasonal expert dispatch;
* ``use_attention`` / ``use_dynamic_adj`` / ``use_seasonal_experts`` flags build
  or skip the corresponding modules entirely (ablation-honest).
"""

from __future__ import annotations

import torch
import torch.nn as nn

POLL_DIM = 6
WX_DIM = 4
ATTN_DIM = 8


class FeatureAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha


class TemporalSelfAttention(nn.Module):
    def __init__(self, feature_dim: int, heads: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.attn(flat, flat, flat)
        return out.reshape(b, n, t, f).permute(0, 2, 1, 3)


class CrossAttention(nn.Module):
    """Pollution-query vs weather-key attention; output concat(poll, context)."""

    def __init__(self, poll_dim: int, wx_dim: int, attn_dim: int = ATTN_DIM) -> None:
        super().__init__()
        self.poll_proj = nn.Linear(poll_dim, attn_dim)
        self.wx_proj = nn.Linear(wx_dim, attn_dim)
        self.cross = nn.MultiheadAttention(attn_dim, 2, batch_first=True)
        self.fc = nn.Linear(poll_dim + attn_dim, poll_dim + attn_dim)
        self.attn_dim = attn_dim

    def forward(self, poll: torch.Tensor, wx: torch.Tensor) -> torch.Tensor:
        b, t, n, p = poll.shape
        _, _, _, w = wx.shape
        p_flat = poll.permute(0, 2, 1, 3).reshape(b * n, t, p)
        w_flat = wx.permute(0, 2, 1, 3).reshape(b * n, t, w)
        ctx, _ = self.cross(self.poll_proj(p_flat), self.wx_proj(w_flat), self.wx_proj(w_flat))
        ctx = ctx.reshape(b, n, t, self.attn_dim).permute(0, 2, 1, 3)
        return self.fc(torch.cat([poll, ctx], dim=-1))


class GraphConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)

    def forward(self, aggr: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(aggr))


def graph_aggregate(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """z [B,T,N,C], a [N,N] / [T,N,N] / [B,T,N,N] -> [B,T,N,C]."""
    if a.dim() == 2:
        return torch.einsum("ij,btjc->btic", a, z)
    if a.dim() == 3:
        return torch.einsum("tij,btjc->btic", a, z)
    return torch.einsum("btij,btjc->btic", a, z)


class TGCNwithAttnV2(nn.Module):
    def __init__(
        self,
        in_f: int = POLL_DIM + WX_DIM,
        g_h: int = 32,
        gru_h: int = 32,
        attn_dim: int = ATTN_DIM,
        horizon: int = 3,
        n_out: int = 4,
        use_attention: bool = True,
        predict_delta: bool = False,
        pred_channels: tuple = (0, 1, 3, 4),
    ) -> None:
        super().__init__()
        self.in_f = in_f
        self.poll_dim = POLL_DIM
        self.use_attention = use_attention
        self.horizon = horizon
        self.n_out = n_out
        self.predict_delta = predict_delta
        self.register_buffer("pred_channels", torch.tensor(list(pred_channels), dtype=torch.long), persistent=False)
        if use_attention:
            self.feature_attn = FeatureAttention(in_f)
            self.temporal_attn = TemporalSelfAttention(in_f)
            self.cross_attn = CrossAttention(in_f, in_f, attn_dim)
            gcn_in = in_f + attn_dim
        else:
            gcn_in = in_f
        self.gcn = GraphConv(gcn_in, g_h)
        self.gru = nn.GRU(g_h, gru_h, batch_first=True)
        self.step_heads = nn.ModuleList([nn.Linear(gru_h, n_out) for _ in range(horizon)])

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = x.shape
        if self.use_attention:
            xa = self.feature_attn(x)
            enhanced = xa + self.temporal_attn(xa)  # residual keeps the input signal intact
            z = self.cross_attn(enhanced, enhanced)
        else:
            z = x
        g = self.gcn(graph_aggregate(z, a))
        seq = g.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, h_n = self.gru(seq)
        h = h_n.squeeze(0).reshape(b, n, -1)
        steps = torch.stack([head(h) for head in self.step_heads], dim=2)  # [B,N,H,K]
        if self.predict_delta:
            base = x[:, -1][:, :, self.pred_channels]  # [B,N,K] normalized last observation
            steps = steps + base[:, :, None, :]
        return steps


class HardMOESeasonV2(nn.Module):
    def __init__(
        self,
        in_f: int = POLL_DIM + WX_DIM,
        g_h: int = 32,
        gru_h: int = 32,
        attn_dim: int = ATTN_DIM,
        horizon: int = 3,
        n_out: int = 4,
        n_experts: int = 4,
        use_attention: bool = True,
        use_seasonal_experts: bool = True,
        predict_delta: bool = False,
        pred_channels: tuple = (0, 1, 3, 4),
    ) -> None:
        super().__init__()
        self.n_experts = n_experts if use_seasonal_experts else 1
        self.use_seasonal_experts = use_seasonal_experts
        kw = dict(g_h=g_h, gru_h=gru_h, attn_dim=attn_dim, horizon=horizon,
                  n_out=n_out, use_attention=use_attention,
                  predict_delta=predict_delta, pred_channels=pred_channels)
        self.experts = nn.ModuleList(
            [TGCNwithAttnV2(in_f, **kw) for _ in range(self.n_experts)]
        )
        self.horizon = horizon
        self.n_out = n_out

    def forward(self, x: torch.Tensor, a: torch.Tensor, season_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_seasonal_experts:
            return self.experts[0](x, a)
        b = x.shape[0]
        out = None
        order = season_ids.argsort()
        s_sorted = season_ids[order]
        # one sub-batch per expert: scatter sorted indices, run 4 forwards total
        counts = torch.bincount(s_sorted.long(), minlength=self.n_experts)
        bounds = torch.cumsum(counts, dim=0)
        start = 0
        for e in range(self.n_experts):
            end = int(bounds[e].item())
            if end > start:
                idx = order[start:end]
                xe = x[idx]
                ae = a[idx] if a.dim() == 4 else a
                oe = self.experts[e](xe, ae)
                if out is None:
                    out = torch.zeros(b, *oe.shape[1:], device=x.device, dtype=oe.dtype)
                out[idx] = oe
            start = end
        assert out is not None
        return out
