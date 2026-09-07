"""Direction 1 — Causal GNN: GAT whose message passing is restricted to the
discovered transport-causal graph (``causal_discovery.py``).

vs the v2 baseline: the static geo-Gaussian / wind-heuristic adjacency is
replaced by directed causal edges (j -> i) found on TRAIN data only; edge
attention is initialised from the causality strength (-log10 adjusted p) so
"stronger evidence of transport" starts with a larger prior.
``use_geo_fallback`` lets the model ALSO admit geo edges with zero prior —
an ablation switch for "causal edges only vs causal + spatial".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.models.innovations.common import (DeltaStepHead, GraphGRUEncoder,
                                           graph_aggregate, last_obs_base)


class CausalGATLayer(nn.Module):
    """Multi-head masked attention over incoming causal edges (plus self-loop)."""

    def __init__(self, in_ch: int, out_ch: int, heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        self.out_ch = out_ch
        self.lin = nn.Linear(in_ch, out_ch * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_ch))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_ch))
        self.a_edge = nn.Parameter(torch.empty(1, heads, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.a_edge)
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, mask: torch.Tensor,
                prior: torch.Tensor) -> torch.Tensor:
        """z [B,T,N,C]; mask [N,N] bool allowed edges (i row, j col, incl diag);
        prior [N,N] float edge-strength feature (0 where absent)."""
        b, t, n, _ = z.shape
        h = self.lin(z).view(b, t, n, self.heads, self.out_ch)  # [B,T,N,Hd,O]
        # src[b,t,i,h] = <h[b,t,i,h,:], a_src[h]>  (query side), dst = key side
        src = (h * self.a_src).sum(-1)  # [B,T,N,Hd]
        dst = (h * self.a_dst).sum(-1)
        logits = src.unsqueeze(3) + dst.unsqueeze(2)  # [B,T,N,N,Hd] i,j,h
        logits = Fn.leaky_relu(
            logits + self.a_edge.view(1, 1, 1, 1, self.heads) * prior[None, None, :, :, None],
            0.2)
        dis = ~mask
        logits = logits.masked_fill(dis[None, None, :, :, None], float("-inf"))
        attn = self.drop(torch.softmax(logits, dim=3))
        out = torch.einsum("btijh,btjho->btiho", attn, h)
        return out.mean(dim=3)  # head-average -> [B,T,N,O]


class CausalTGCN(nn.Module):
    """x [B,L,N,F] -> [B,H,N,K]; constructor receives the causal graph arrays."""

    def __init__(self, in_f: int, causal_weights: torch.Tensor, horizon: int = 3,
                 n_out: int = 4, g_h: int = 64, gru_h: int = 64, gat_heads: int = 4,
                 gat_layers: int = 2, dropout: float = 0.1,
                 pred_channels: tuple = (0, 1, 3, 4), use_geo_fallback: bool = False,
                 geo_adj: torch.Tensor | None = None) -> None:
        super().__init__()
        n = causal_weights.shape[0]
        mask = causal_weights > 0
        if use_geo_fallback:
            if geo_adj is None:
                raise ValueError("use_geo_fallback=True 需要 geo_adj")
            mask = mask | (geo_adj > 0)
        mask = mask | torch.eye(n, dtype=torch.bool)
        self.register_buffer("mask", mask)
        self.register_buffer("prior", causal_weights.clamp(min=0))
        self.in_proj = nn.Linear(in_f, g_h)
        self.layers = nn.ModuleList(
            [CausalGATLayer(g_h, g_h, gat_heads, dropout) for _ in range(gat_layers - 1)]
            + [CausalGATLayer(g_h, g_h, 1, dropout)])
        self.gru = nn.GRU(g_h, gru_h, batch_first=True)
        self.head = DeltaStepHead(gru_h, horizon, n_out)
        self.register_buffer("pred_channels",
                             torch.tensor(list(pred_channels), dtype=torch.long),
                             persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        """a is accepted for trainer-signature uniformity; causal edges replace it."""
        z = self.in_proj(x)
        z = Fn.elu(self.layers[0](z, self.mask, self.prior))
        for blk in self.layers[1:]:
            z = z + self.drop(Fn.elu(blk(z, self.mask, self.prior)))  # residual stacks
        b, t, n, _ = z.shape
        seq = z.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, h_n = self.gru(seq)
        h = h_n.squeeze(0).reshape(b, n, -1)  # [B,N,gru_h]
        base = last_obs_base(x, self.pred_channels)
        return self.head(h, base)
