"""Direction 3 — conditional DDPM probabilistic forecasting.

Upgrade over interval-valued / quantile baselines (docs §4 方向3): instead of
"predict 85", generate the FULL posterior of the future PM2.5 field and
answer "超标概率 78%".

Design (small-state diffusion on the forecast vector, not an image):
* Diffusion target y = future PM2.5 delta over the last observation,
  shape [N, H] (10 stations x 3 steps = 30 dims) — tiny state, so 100-step
  cosine-schedule DDPM trains in minutes on the RTX 4050 and samples fast
  enough for 100-sample probability fields at inference.
* Conditioning: a T-GCN encoder over the 12h history (+ graph adjacency)
  produces per-node context; the denoiser is node-wise (shared MLP) with one
  causal-graph-style message pass per step, so the generated trajectories
  respect station neighbourhoods instead of 30 independent scalars.
* eps-prediction parameterisation, cosine beta schedule (Ho et al. 2020 /
  Nichol & Dhariwal), standard ancestral sampling.

Probability products: from S sampled trajectories (denormalised) we report
exceedance probability P(PM2.5 > threshold), CRPS, and 90% prediction
intervals — see :func:`probability_metrics`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src.models.innovations.common import GraphGRUEncoder


# ─── schedules & utilities ───────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """alpha_bar_t of Nichol & Dhariwal 2021 -> betas clipped < 0.999."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    ac = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1 - ac[1:] / ac[:-1]
    return betas.clamp(1e-4, 0.999).float()


def sinusoidal_t_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class GaussianDiffusion(nn.Module):
    """Buffer-only math of the forward process + eps-prediction loss/sampling."""

    def __init__(self, T: int = 100) -> None:
        super().__init__()
        betas = cosine_beta_schedule(T)
        ab = torch.cumprod(1 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("ab", ab)
        self.register_buffer("ab_prev", torch.cat([torch.ones(1), ab[:-1]]))
        self.register_buffer("sqrt_ab", ab.sqrt())
        self.register_buffer("sqrt_1mab", (1 - ab).sqrt())
        # posterior variance beta_tilde_t = beta_t (1-ab_{t-1}) / (1-ab_t)
        self.register_buffer("post_var", betas * (1 - self.ab_prev) / (1 - ab))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """x0 [B,...] -> x_t; t [B] long."""
        noise = torch.randn_like(x0) if noise is None else noise
        s_ab = self.sqrt_ab[t].view(-1, *([1] * (x0.dim() - 1)))
        s_1 = self.sqrt_1mab[t].view(-1, *([1] * (x0.dim() - 1)))
        return s_ab * x0 + s_1 * noise

    @torch.no_grad()
    def sample(self, denoiser: nn.Module, shape: tuple, cond, adj,
               device: torch.device) -> torch.Tensor:
        """Ancestral DDPM sampling; returns x0 estimate [B,*shape]."""
        b = shape[0]
        x = torch.randn(*shape, device=device)
        for t in reversed(range(len(self.betas))):
            tt = torch.full((b,), t, device=device, dtype=torch.long)
            eps = denoiser(x, tt, cond, adj)
            # Ho et al. reverse step: mean = (x_t - sqrt(bt)/sqrt(1-ab_t) eps)/sqrt(a_t)
            coef = (self.post_var[t] / (1 - self.ab[t])).sqrt()
            mean = (x - coef * eps) / (1 - self.betas[t]).sqrt()
            x = mean + self.post_var[t].sqrt() * torch.randn_like(x) if t > 0 else mean
        return x


# ─── denoiser ────────────────────────────────────────────────────────────────

class NodeWiseDenoiser(nn.Module):
    """Shared-per-node eps predictor: [y_t (H,) per node, cond (C), t-emb] ->
    one GraphConv message pass over the station graph, then MLP -> eps (H,)."""

    def __init__(self, n_nodes: int, horizon: int, cond_dim: int, t_dim: int = 32,
                 hidden: int = 128) -> None:
        super().__init__()
        self.horizon = horizon
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(horizon + cond_dim, hidden)
        self.mix = nn.Linear(2 * hidden, hidden)  # concat own msg + neighbour msg
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, horizon))

    def forward(self, xt: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """xt [B,N,H] per-node flattened, t [B] long diffusion step (embedded
        here), cond [B,N,C], adj [N,N] / [B,N,N] / [B,T,N,N] (last step used)."""
        a = adj
        if a.dim() == 4:
            a = a[:, -1]
        elif a.dim() == 2:
            a = a.unsqueeze(0).expand(xt.shape[0], -1, -1)
        feat = torch.cat([xt, cond], dim=-1)           # [B,N,H+C]
        h = Fn.silu(self.inp(feat))                    # [B,N,Hid]
        te = self.t_mlp(sinusoidal_t_embed(t, 32)).unsqueeze(1)  # [B,1,Hid]
        msg = torch.einsum("bij,bjc->bic", a, h)       # neighbour aggregate
        z = self.mix(torch.cat([h + te, msg], dim=-1))
        return self.out(Fn.silu(z))


class ConditionalPM25Diffusion(nn.Module):
    """history [B,L,N,F] + graph -> distribution over future PM2.5 deltas
    y [B,N,H] (PM2.5 only, normalised delta space)."""

    def __init__(self, in_f: int, n_nodes: int, horizon: int = 3, g_h: int = 64,
                 gru_h: int = 64, cond_dim: int = 64, hidden: int = 128,
                 T_sched: int = 100, pred_channels: tuple = (0, 1, 3, 4),
                 pm25_slot: int = 0) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.pm25_slot = pm25_slot
        self.register_buffer("pred_channels",
                             torch.tensor(list(pred_channels), dtype=torch.long),
                             persistent=False)
        self.encoder = GraphGRUEncoder(in_f, g_h, gru_h)
        self.cond_proj = nn.Linear(gru_h, cond_dim)
        self.denoiser = NodeWiseDenoiser(n_nodes, horizon, cond_dim, hidden=hidden)
        self.diffusion = GaussianDiffusion(T_sched)

    # -- conditioning ---------------------------------------------------------
    def condition(self, x: torch.Tensor, a: torch.Tensor):
        _, gru_out = self.encoder(x, a)
        cond = self.cond_proj(gru_out[:, -1])          # [B,N,C]
        base = x[:, -1][:, :, self.pred_channels[self.pm25_slot]]  # [B,N]
        return cond, base

    # -- training -------------------------------------------------------------
    def training_loss(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor,
                      y_mask: torch.Tensor) -> torch.Tensor:
        """y [B,H,N,K] (full K channels), y_mask [B,H,N,K] — PM2.5 channel is
        extracted here; internal target is the delta y_pm25 - last_obs (keeps
        x0 near zero -> well-conditioned diffusion). Masked frames are replaced
        by the anchor before diffusion (the loss then re-applies the mask)."""
        cond, base = self.condition(x, a)
        k = self.pm25_slot
        # The pipeline keeps targets as float64 for metric fidelity; diffusion
        # network parameters follow the float32 input tensor.
        y_pm25 = y[..., k].to(dtype=x.dtype)            # [B,H,N]
        m = y_mask[..., k].to(dtype=x.dtype)            # [B,H,N]
        anchor = base[:, None, :]                       # [B,1,N]
        yh = torch.where(m > 0.5, y_pm25, anchor.expand_as(y_pm25))
        x0 = (yh - anchor).permute(0, 2, 1).contiguous()  # [B,N,H] delta
        b = x0.shape[0]
        t = torch.randint(0, len(self.diffusion.betas), (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.diffusion.q_sample(x0, t, noise)
        eps_hat = self.denoiser(xt, t, cond, a)
        m_d = m.permute(0, 2, 1)                         # [B,N,H]
        return ((eps_hat - noise) ** 2 * m_d).sum() / m_d.sum().clamp(min=1.0)

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def sample_delta(self, x: torch.Tensor, a: torch.Tensor, n_samples: int = 100,
                     chunk: int = 256) -> torch.Tensor:
        """-> samples of PM2.5 delta [S, B, N, H] (normalised units)."""
        cond, _ = self.condition(x, a)
        b = x.shape[0]
        out = []
        for s0 in range(0, n_samples, chunk):
            k = min(chunk, n_samples - s0)
            cond_k = cond.repeat_interleave(k, dim=0)   # [B*k,N,C]
            a_k = a.repeat_interleave(k, dim=0) if a.dim() == 4 else a
            shape = (b * k, self.n_nodes, self.horizon)
            d = self.diffusion.sample(self.denoiser, shape, cond_k, a_k, x.device)
            out.append(d.reshape(b, k, self.n_nodes, self.horizon))
        return torch.cat(out, dim=1).permute(1, 0, 2, 3).contiguous()  # [S,B,N,H]

    def forward(self, x, a, y=None, y_mask=None):  # dispatch helper
        if y is None:
            raise RuntimeError("diffusion model: use training_loss / sample_delta")
        return self.training_loss(x, a, y, y_mask)


# ─── probabilistic metrics ───────────────────────────────────────────────────

def crps_ensemble(samples: np.ndarray, obs: np.ndarray) -> float:
    """Continuous Ranked Probability Score, quantile (CWT 2006) form.

    samples [S, ...] ensemble, obs [...] (NaN allowed)."""
    ok = np.isfinite(obs)
    if not ok.any():
        return float("nan")
    s = np.sort(samples[:, ok], axis=0)                # [S, M]
    o = obs[ok][None, :]
    # CRPS = E|X-o| - 0.5 E|X-X'|
    e1 = np.abs(s - o).mean(axis=0)
    diffs = np.abs(s[:, None, :] - s[None, :, :]).mean(axis=(0, 1))
    return float((e1 - 0.5 * diffs).mean())


def exceedance_stats(samples_raw: np.ndarray, obs_raw: np.ndarray,
                     threshold: float = 75.0) -> dict:
    """samples_raw [S,...] µg/m³, obs_raw [...] one obs per entry (NaN ok).

    P(exceed) = fraction of samples > threshold; Brier = mean (p - 1{obs>thr})²;
    90% PI = central interval + coverage."""
    ok = np.isfinite(obs_raw)
    if not ok.any():
        return {}
    s = samples_raw[:, ok]
    o = obs_raw[ok]
    p_ex = (s > threshold).mean(axis=0)
    ev = (o > threshold).astype(np.float64)
    lo, hi = np.quantile(s, [0.05, 0.95], axis=0)
    inside = ((o >= lo) & (o <= hi)).mean()
    # PIT / reliability proxy: fraction of obs below its own 50% quantile
    med = np.quantile(s, 0.5, axis=0)
    return {
        "exceed_prob_mean": float(p_ex.mean()),
        "brier": float(((p_ex - ev) ** 2).mean()),
        "pi90_coverage": float(inside),
        "median_mae": float(np.abs(med - o).mean()),
        "crps": crps_ensemble(s, o),
        "n": int(ok.sum()),
    }
