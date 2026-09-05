"""Direction 2 — physics-informed graph network (advection–diffusion operator).

Motivation (docs §4 方向2): the spatial spread of PM2.5 is essentially the
solution of the 2-D advection–diffusion equation

    ∂C/∂t + u·∇C = K ∇²C + S

so instead of the heuristic Gaussian-distance + wind-alignment adjacency in
``src/data/pipeline.py`` (the qualitative "Improved STGCN" version), the graph
operator here is a *discretisation of the PDE itself*:

* :func:`advection_operator` — upwind first-order discretisation of u·∇C:
  an ordered pair (j -> i) carries mass only when station j lies upwind of i,
  with strength proportional to the wind component along j->i / distance.
* :func:`diffusion_operator` — Gaussian-kernel graph approximation of K ∇²C
  on the station point cloud.
* :class:`AdvectionDiffusionTGCN` — T-GCN whose per-hour message matrix is the
  learned convex blend α·A_adv(t) + (1-α)·A_diff(t); α, the advection gain and
  the diffusivity are learned positive parameters (physics-structured graph,
  learnable blend).
* :class:`PDEResidualLoss` — soft PINN constraint: the discrete residual of
  the same operator evaluated on the predicted concentration field (weighted
  least-squares ∇, weighted-Laplacian ∇², learnable source term S).

Wind forcing (u, v in km/h, east/north-positive transport velocity) is passed
in as a separate physical [B,T,N,2] array — the model never reads normalised
weather, so it runs on observed station wind, NOAA ISD, or ERA5 alike.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.models.innovations.common import DeltaStepHead, GraphConv, last_obs_base


def wind_uv_from_dir_speed(wind_dir_deg: torch.Tensor,
                           wind_spd_kmh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Meteorological direction (FROM, clockwise from N) -> transport velocity
    (u east, v north) in km/h: mass moves where the wind blows TO."""
    to_rad = torch.deg2rad((wind_dir_deg + 180.0) % 360.0)
    return wind_spd_kmh * torch.sin(to_rad), wind_spd_kmh * torch.cos(to_rad)


def pair_geometry(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """coords [N,2] (lon,lat) -> local offsets dx [N,N,2] km with
    dx[i,j] = pos_i - pos_j (direction j -> i) and distances d [N,N] km."""
    kx = 111.320 * math.cos(math.radians(float(coords[:, 1].mean())))
    ky = 110.574
    dlon = coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0)  # [N,N] i,j
    dlat = coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0)
    dx = torch.stack([dlon * kx, dlat * ky], dim=-1)
    return dx, dx.norm(dim=-1)


def row_normalise(a: torch.Tensor) -> torch.Tensor:
    """Row-stochastic message matrix; the self-loop keeps >= 1/2 of each row
    (local persistence + unidentified sources/sinks)."""
    n = a.shape[-1]
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    self_w = a.amax(dim=-1, keepdim=True).clamp(min=0.5)
    a = a + eye * self_w
    return a / a.sum(dim=-1, keepdim=True)


def advection_operator(dx: torch.Tensor, d: torch.Tensor, u: torch.Tensor,
                       v: torch.Tensor) -> torch.Tensor:
    """Upwind discretisation of u·∇C as a row-normalised transport matrix.

    u, v [B,T,N] transport velocity AT THE UPWIND NODE (wind at j decides how
    much of j's concentration arrives at i). Output A [B,T,N,N], A[...,i,j] =
    mixing weight of neighbour j into node i; 0 where j is downwind of i.
    """
    dn = d.clamp(min=1e-6)
    rhat = dx / dn.unsqueeze(-1)                       # [N,N,2] unit j->i
    wj = torch.stack([u, v], dim=-1).unsqueeze(2)      # [B,T,1,N,2] wind at j
    proj = (wj * rhat.view(1, 1, *rhat.shape)).sum(-1)  # [B,T,N,N] i,j
    a = torch.relu(proj) / dn                           # strength ~ |u|/d
    a = a * (d > 1e-6)                                  # no self advection
    return row_normalise(a)


def diffusion_operator(d: torch.Tensor, diffusivity: torch.Tensor,
                       length_scale_km: float = 25.0, max_km: float = 50.0) -> torch.Tensor:
    """Gaussian-kernel graph approximation of ∇²C, row-normalised [N,N]."""
    keep = torch.exp(-(d / length_scale_km) ** 2) * (d <= max_km) * (d > 1e-6)
    return row_normalise(diffusivity * keep)


def _batched_aggregate(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """z [B,T,N,C] with a [B,T,N,N] -> einsum message passing."""
    return torch.einsum("btij,btjc->btic", a, z)


class AdvectionDiffusionTGCN(nn.Module):
    """T-GCN with PDE-structured message passing.

    forward(x [B,T,N,F], uv [B,T,N,2] km/h) -> [B,H,N,K].
    The blended per-hour operator is cached in ``last_ops`` for visualising
    the wind-aligned transport network each timestep.
    """

    def __init__(self, in_f: int, coords: torch.Tensor, g_h: int = 64,
                 gru_h: int = 64, horizon: int = 3, n_out: int = 4,
                 length_scale_km: float = 25.0, blend_init: float = 0.5,
                 diff_init: float = 0.3, pred_channels: tuple = (0, 1, 3, 4)) -> None:
        super().__init__()
        self.register_buffer("coords", coords)
        dx, d = pair_geometry(coords)
        self.register_buffer("dx", dx)
        self.register_buffer("d", d)
        self.length_scale_km = length_scale_km
        self.gcn = GraphConv(in_f, g_h)
        self.gru = nn.GRU(g_h, gru_h, batch_first=True)
        self.blend_logit = nn.Parameter(torch.log(torch.tensor(blend_init / (1 - blend_init))))
        self.diff_raw = nn.Parameter(torch.log(torch.tensor(diff_init)))
        self.adv_gain = nn.Parameter(torch.zeros(()))  # softplus gain on |u|/d
        self.head = DeltaStepHead(gru_h, horizon, n_out)
        self.register_buffer("pred_channels",
                             torch.tensor(list(pred_channels), dtype=torch.long),
                             persistent=False)
        self.last_ops: dict = {}

    def build_operator(self, uv: torch.Tensor) -> torch.Tensor:
        """uv [B,T,N,2] -> A [B,T,N,N]: learned blend of the two PDE terms."""
        alpha = torch.sigmoid(self.blend_logit)
        adv = advection_operator(self.dx, self.d,
                                 uv[..., 0] * Fn.softplus(self.adv_gain + 1.0), uv[..., 1])
        diff = diffusion_operator(self.d, Fn.softplus(self.diff_raw), self.length_scale_km)
        b, t = adv.shape[:2]
        diff = diff.view(1, 1, *diff.shape).expand(b, t, -1, -1)
        return alpha * adv + (1 - alpha) * diff

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None,
                uv: torch.Tensor | None = None) -> torch.Tensor:
        """a is accepted (and ignored) for trainer-signature uniformity."""
        if uv is None:
            raise ValueError("AdvectionDiffusionTGCN 需要物理强迫场 uv [B,T,N,2] (km/h)")
        a_phys = self.build_operator(uv)
        self.last_ops = {"A": a_phys.detach(),
                         "blend": self.blend_logit.detach().sigmoid().item()}
        g = self.gcn(_batched_aggregate(x, a_phys))
        b, t, n, _ = g.shape
        seq = g.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, h_n = self.gru(seq)
        h = h_n.squeeze(0).reshape(b, n, -1)  # [B,N,gru_h]
        base = last_obs_base(x, self.pred_channels)
        return self.head(h, base)


class PDEResidualLoss(nn.Module):
    """Soft PINN constraint on the predicted concentration field.

    For the predicted trajectory ``y [B,H,N,K]`` (normalised units, channel
    ``k_channel`` = PM2.5) anchored at ``last_obs [B,N,K]``, penalise

        R = (C_t - C_{t-1})/Δt + u·∇C - K ∇²C - S

    with ∇C by Gaussian-weighted least squares on the station cloud, ∇²C by
    the weighted graph Laplacian, learnable K>0 and per-station source S —
    the same operator family the model's message passing uses.
    """

    def __init__(self, coords: torch.Tensor, length_scale_km: float = 25.0,
                 dt_h: float = 1.0, k_channel: int = 0, weight: float = 0.1) -> None:
        super().__init__()
        self.length_scale = length_scale_km
        self.dt = dt_h
        self.k_channel = k_channel
        self.weight = weight
        self.register_buffer("coords", coords)
        self.log_K = nn.Parameter(torch.log(torch.tensor(0.2)))
        self.source = nn.Parameter(torch.zeros(coords.shape[0]))
        dx, d = pair_geometry(self.coords)
        w = torch.exp(-(d / self.length_scale) ** 2) * (d > 1e-6)
        self.register_buffer("gw", w)
        self.register_buffer("gdx", dx)

    def _forward_gradient(self, c: torch.Tensor) -> torch.Tensor:
        """∇C [B,S,N,2] via per-node weighted least squares over neighbours."""
        w, r = self.gw, self.gdx                     # w [N,N], r [N,N,2] i->j
        dc = c.unsqueeze(-1) - c.unsqueeze(-2)       # [B,S,N,N]: c_j - c_i
        wr = w.unsqueeze(-1) * r                     # [N,N,2]
        M = torch.einsum("ij,ija,ijb->iab", w, r, r)  # [N,2,2] normal matrix
        M = M + 1e-4 * torch.eye(2, device=c.device, dtype=c.dtype)  # ridge
        b = torch.einsum("bsnj,jna->bsna", dc, wr)   # [B,S,N,2]
        return torch.linalg.solve(M, b.unsqueeze(-1)).squeeze(-1)  # [B,S,N,2]

    def forward(self, y: torch.Tensor, last_obs: torch.Tensor,
                uv: torch.Tensor) -> torch.Tensor:
        """y [B,H,N,K], last_obs [B,N,K] (normalised), uv [B,H,N,2] km/h
        (persistence of the last observed wind is fine). -> scalar penalty."""
        k = self.k_channel
        anchor = last_obs[..., k:k + 1].unsqueeze(1)  # [B,1,N,1]
        c = torch.cat([anchor, y[..., k:k + 1]], dim=1).squeeze(-1)  # [B,S,N], S=H+1
        dcdt = (c[:, 1:] - c[:, :-1]) / self.dt       # [B,H,N]
        grad = self._forward_gradient(c[:, 1:])        # [B,H,N,2]
        adv = (uv * grad).sum(-1)                      # u·∇C
        dc = c[:, 1:].unsqueeze(-1) - c[:, 1:].unsqueeze(-2)  # [B,H,N,N] c_j - c_i
        lap = (self.gw * dc).sum(-1) / (self.gw.sum(-1) + 1e-6)  # [B,H,N] / [N]
        K = Fn.softplus(self.log_K)
        residual = dcdt + adv - K * lap - self.source.view(1, 1, -1)
        return self.weight * residual.pow(2).mean()
