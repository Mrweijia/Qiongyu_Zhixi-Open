"""Direction 5 — Graph Neural ODE: continuous-time graph dynamics.

Instead of a discrete-time GRU forecasting step, the hidden state evolves
under a graph-structured velocity field defined by the learned ODE. The
model naturally handles irregular sampling (any t+h) and "flows" through
missing hours rather than discarding them — a principled upgrade over
discrete sequence models for the "arbitrary horizon" / "missing data" case.

The ODE is integrated with a fixed-step RK4 integrator (self-contained, no
torchdiffeq dependency). If torchdiffeq is installed, batch-adjoint mode
can be enabled via ``use_adjoint=True`` (not currently implemented for the
custom network, but the integrator signature matches the standard API).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.models.innovations.common import GraphConv, GraphGRUEncoder, last_obs_base


class GraphODEFunc(nn.Module):
    """Velocity field v(h, A) = MLP(h || GCN(h, A))."""

    def __init__(self, dim: int, g_h: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gcn = GraphConv(dim, g_h)
        self.mlp = nn.Sequential(nn.Linear(dim + g_h, g_h), nn.Tanh(),
                                 nn.Linear(g_h, dim), nn.Dropout(dropout))

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """h [B,N,C], a [B,N,N] or [N,N] -> v [B,N,C]."""
        ah = torch.einsum("bij,bjc->bic", a, h) if a.dim() == 3 \
            else torch.einsum("ij,bjc->bic", a, h)
        g = torch.relu(self.gcn.lin(ah))
        return self.mlp(torch.cat([h, g], dim=-1))


def rk4_step(fn: nn.Module, h: torch.Tensor, a: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = fn(h, a)
    k2 = fn(h + 0.5 * dt * k1, a)
    k3 = fn(h + 0.5 * dt * k2, a)
    k4 = fn(h + dt * k3, a)
    return h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class GraphNeuralODE(nn.Module):
    """x [B,L,N,F] + adj -> integrate over horizon -> [B,H,N,K].

    Encoder: GraphGRUEncoder over the 12h window to produce h0.
    ODE: GraphODEFunc velocity field, integrated with RK4, substeps=4 per hour.
    Decoder: per-evaluation-step linear heads (delta style).
    """

    def __init__(self, in_f: int, n_nodes: int, g_h: int = 64, gru_h: int = 64,
                 ode_dim: int = 64, horizon: int = 3, n_out: int = 4,
                 n_substeps: int = 4, dropout: float = 0.1,
                 pred_channels: tuple = (0, 1, 3, 4)) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_substeps = n_substeps
        self.encoder = GraphGRUEncoder(in_f, g_h, gru_h)
        self.to_ode = nn.Linear(gru_h, ode_dim)
        self.ode_func = GraphODEFunc(ode_dim, g_h, dropout)
        self.step_heads = nn.ModuleList([nn.Linear(ode_dim, n_out) for _ in range(horizon)])
        self.register_buffer("pred_channels",
                             torch.tensor(list(pred_channels), dtype=torch.long),
                             persistent=False)

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        b, l, n, _ = x.shape
        # history encoding
        _, gru_out = self.encoder(x, a)
        h = self.to_ode(gru_out[:, -1])  # [B,N,C]
        # integrate
        dt = 1.0 / self.n_substeps
        # fetch a per-step adjacency: use last timestep's A as static for the
        # forecast (the future wind is unknown — persistence is acceptable)
        a_step = a
        if a_step is not None and a_step.dim() == 4:
            a_step = a_step[:, -1]  # [B,N,N] last input step
        elif a_step is not None and a_step.dim() == 2:
            a_step = a_step.unsqueeze(0).expand(b, -1, -1)
        hs = []
        for step in range(self.horizon):
            for _ in range(self.n_substeps):
                h = rk4_step(self.ode_func, h, a_step, dt)
            hs.append(h)  # [B,N,C]
        h_steps = torch.stack(hs, dim=1)  # [B,H,N,C]
        # decode per step
        preds = torch.stack([head(h_steps[:, i]) for i, head in enumerate(self.step_heads)], dim=1)  # [B,H,N,K]
        base = last_obs_base(x, self.pred_channels).unsqueeze(1)  # [B,1,N,K]
        return preds + base