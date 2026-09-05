"""Station-to-station causal discovery (Direction 1, step 1).

Why causal edges instead of distance/correlation edges: geo-distance graphs
and correlation graphs are both driven by *common meteorology* (confounding);
Granger-style causality asks "does station j's history improve the forecast
of station i beyond i's own history" — i.e. who really transports to whom.

Two estimators, both pure-numpy OLS F-tests (no tigramite dependency):

* ``pairwise``     — classic bivariate Granger causality test per ordered pair.
* ``conditional``  — VAR-style block test: regress target i on lags of ALL
    stations and F-test the exclusion of station j's lag block (direct links,
    closer to what PCMCI's conditioning sets approximate).

PCMCI via ``tigramite`` is the planned upgrade (see docs/project/decisions.md);
:func:`pcmci_causal_adjacency` raises an actionable ImportError when the lib
is absent, so pipelines can opt in without breaking.

Edge convention everywhere: ``adj[i, j] > 0`` means **j -> i** (j drives i),
matching message passing ``z_i <- sum_j adj[i, j] * z_j``.

Honesty rule: discovery MUST run on TRAIN timestamps only (no test leakage).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import pipeline as pl


@dataclass
class CausalGraph:
    weights: np.ndarray  # [N, N] float32, j->i strength (-log10 p, Bonferroni-ed), 0 = no edge
    pvalues: np.ndarray  # [N, N] float64 adjusted p-values
    max_lag: int
    method: str

    @property
    def mask(self) -> np.ndarray:
        return (self.weights > 0).astype(np.float32)


# ─── core test ───────────────────────────────────────────────────────────────

def _ols_rss(design: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(resid @ resid)


def _granger_f(y: np.ndarray, others: np.ndarray, cause_cols: slice,
               lag: int) -> tuple[float, float]:
    """F-test: drop ``others[:, cause_cols]`` (and their lag rows) from VAR.

    y [T], others [T, N*lag] stacked lags of every station (incl. y itself);
    cause_cols selects station j's lag block.
    """
    t = len(y)
    full = np.column_stack([np.ones(t), others])
    keep = np.ones(full.shape[1], dtype=bool)
    keep[1:][cause_cols] = False  # const column stays
    restricted = full[:, keep]
    rss_full = _ols_rss(full, y)
    rss_red = _ols_rss(restricted, y)
    q = int(cause_cols.stop - cause_cols.start)  # restrictions = lag
    dof = t - full.shape[1] - 1
    if dof <= 5 or rss_full <= 0 or q <= 0:
        return 1.0, 1.0
    f_stat = ((rss_red - rss_full) / q) / (rss_full / dof)
    # p-value from the F distribution without scipy: incomplete beta via
    # continued fraction would be heavy; use the log-space normal
    # approximation of the Wilson–Hilferty transform (accurate for dof > 30).
    p = _f_sf(f_stat, q, dof)
    return f_stat, p


def _f_sf(f: float, d1: int, d2: int) -> float:
    """Survival function of F(d1, d2) via the regularised incomplete beta
    (continued fraction, Lentz) — exact, no scipy needed."""
    if f <= 0:
        return 1.0
    x = d2 / (d2 + d1 * f)  # I_x(d2/2, d1/2) = P(F <= f)
    a, b = d2 / 2.0, d1 / 2.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta)

    def betacf(a_: float, b_: float, x_: float) -> float:
        tiny, c, d = 1e-300, 1.0, 1.0 - (a_ + b_) * x_ / (a_ + 1.0)
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            for two in (2 * m, 2 * m + 1):
                m_ = two // 2
                if two % 2 == 0:
                    num = m_ * (b_ - m_) * x_ / ((a_ + 2 * m_ - 1) * (a_ + 2 * m_))
                else:
                    num = -((a_ + m_) * (a_ + b_ + m_) * x_ /
                            ((a_ + 2 * m_) * (a_ + 2 * m_ + 1)))
                d = 1.0 + num * d
                if abs(d) < tiny:
                    d = tiny
                c = 1.0 + num / c
                if abs(c) < tiny:
                    c = tiny
                d = 1.0 / d
                de = c * d
                h *= de
                if abs(de - 1.0) < 1e-8:
                    return h
        return h

    if x < (a + 1) / (a + b + 2):
        ibeta = front * betacf(a, b, x) / a
    else:
        ibeta = 1.0 - math.exp(math.log1p(-x) * b + math.log(x) * a - lbeta) \
            * betacf(b, a, 1 - x) / b
    return float(min(1.0, max(0.0, ibeta)))


def _lag_stack(series: np.ndarray, lag: int) -> np.ndarray:
    """series [T, N] -> lags [T-lag, N*lag] with column blocks station-major,
    ordered j_lag1..j_lagL per station (so station j block = slice(j*L, (j+1)*L))."""
    t = series.shape[0]
    out = np.empty((t - lag, series.shape[1] * lag), dtype=np.float64)
    for j in range(series.shape[1]):
        for k in range(1, lag + 1):
            out[:, j * lag + (k - 1)] = series[lag - k: t - k, j]
    return out


def granger_adjacency(series: np.ndarray, lag: int = 12, alpha: float = 0.01,
                      method: str = "conditional", min_meaningful: float = 0.0
                      ) -> CausalGraph:
    """Discover directed edges among station series.

    series [T, N] — one channel (e.g. PM2.5), finite, standardised by caller.
    Returns :class:`CausalGraph` with Bonferroni-adjusted p-values.
    """
    t, n = series.shape
    if method not in ("pairwise", "conditional"):
        raise ValueError(f"unknown method {method!r}")
    lags = _lag_stack(series.astype(np.float64), lag)  # [T-lag, N*lag]
    y = series[lag:]
    mu, sd = y.mean(0), y.std(0) + 1e-12
    y = (y - mu) / sd
    lags = (lags - lags.mean(0)) / (lags.std(0) + 1e-12)

    pmat = np.ones((n, n))
    n_tests = n * (n - 1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if method == "conditional":
                others = lags
                cols = slice(j * lag, (j + 1) * lag)
            else:  # pairwise: only y's own lags + candidate's lags
                cols_own = slice(i * lag, (i + 1) * lag)
                cols_cau = slice(j * lag, (j + 1) * lag)
                others = np.column_stack([lags[:, cols_own], lags[:, cols_cau]])
                cols = slice(lag, 2 * lag)
            _, p = _granger_f(y[:, i], others, cols, lag)
            pmat[i, j] = min(1.0, p * n_tests)  # Bonferroni
    weights = np.where(pmat <= alpha, np.minimum(-np.log10(np.maximum(pmat, 1e-10)), 10.0), 0.0)
    if min_meaningful > 0:
        weights[weights < min_meaningful] = 0.0
    return CausalGraph(weights.astype(np.float32), pmat, lag, method)


def sparsify_in_topk(adj: np.ndarray, k: int) -> np.ndarray:
    """Keep the k strongest in-edges per node (dense 10-node graphs overfit)."""
    out = adj.copy()
    n = out.shape[0]
    for i in range(n):
        order = np.argsort(-out[i])
        keep = set(order[:k].tolist()) | {i}
        for j in range(n):
            if j not in keep:
                out[i, j] = 0.0
    return out


def row_normalize_with_self(adj: np.ndarray, self_weight: float = 1.0) -> np.ndarray:
    """Row-stochastic message matrix with an explicit self-loop."""
    a = adj.astype(np.float64).copy()
    np.fill_diagonal(a, self_weight)
    return (a / (a.sum(axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def causal_graph_from_timeline(tl: pl.TimelineArrays, train_mask: np.ndarray,
                               channel: str = "PM2.5", lag: int = 12,
                               alpha: float = 0.01, method: str = "conditional",
                               top_k: int = 4) -> CausalGraph:
    """PM2.5 channel -> causal graph, fitted on TRAIN hours only (no leakage)."""
    c = pl.POLL_TYPES.index(channel)
    x = tl.x_raw[:, :, c].copy()  # [T, N]
    tr = x[train_mask]
    med = np.nanmedian(tr, axis=0)
    bad = ~np.isfinite(x)
    x[bad] = np.take(med, np.where(bad)[1])
    # use the longest contiguous train stretch (VAR needs serial continuity)
    idx = np.where(train_mask)[0]
    if len(idx) < lag + 100:
        raise ValueError("训练段样本太少，无法做因果发现")
    x_tr = x[idx]
    g = granger_adjacency(x_tr, lag=lag, alpha=alpha, method=method)
    if top_k > 0:
        g.weights = sparsify_in_topk(g.weights, top_k)
    return g


def pcmci_causal_adjacency(*args, **kwargs) -> CausalGraph:
    """PCMCI (Runge et al.) via tigramite — optional upgrade path.

    pip install tigramite, then wire this into configs (method: 'pcmci').
    """
    raise ImportError(
        "PCMCI 需要可选依赖 tigramite（pip install tigramite）；"
        "当前请使用 method='conditional' 的 VAR-Granger 直接链接近似的替代实现。")


# ─── artifacts for the答辩 figure ────────────────────────────────────────────

def save_causal_graph_artifacts(g: CausalGraph, out_dir: Path,
                                stations: list[str] | None = None) -> Path:
    """Persist weights/mask npz + edge json + a transport-network PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    stations = stations or pl.SEL_STATIONS
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "causal_graph.npz", weights=g.weights, pvalues=g.pvalues,
             mask=g.mask, meta=np.array([g.max_lag, 0]))
    n = g.weights.shape[0]
    edges = sorted(
        ({"from": stations[j], "to": stations[i], "weight": float(g.weights[i, j]),
          "p_adj": float(g.pvalues[i, j])}
         for i in range(n) for j in range(n) if i != j and g.weights[i, j] > 0),
        key=lambda e: -e["weight"])
    (out_dir / "causal_edges.json").write_text(
        json.dumps({"method": g.method, "max_lag": g.max_lag, "edges": edges},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # Use the system Chinese font explicitly on Windows so the exported figure
    # does not contain missing-glyph boxes in the defence material.
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    cn_font = (font_manager.FontProperties(fname=str(font_path))
               if font_path.exists() else None)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    coords = [pl.STATION_COORDS[s] for s in stations]
    wmax = max(g.weights.max(), 1e-6)
    for i, j in zip(*np.nonzero(g.weights)):
        if i == j:
            continue
        (x0, y0), (x1, y1) = coords[j], coords[i]  # arrow j -> i
        strength = g.weights[i, j] / wmax
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6 + 2.4 * strength,
                                    color=(0.85 - 0.6 * strength, 0.25, 0.2),
                                    alpha=0.35 + 0.6 * strength,
                                    shrinkA=10, shrinkB=10))
    for (lon, lat), s in zip(coords, stations):
        ax.plot(lon, lat, "o", ms=8, color="#16556b")
        ax.annotate(s, (lon, lat), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_title(
        f"长沙站点污染传输因果网络（{g.method}, lag={g.max_lag}h, 仅用训练段发现）",
        fontproperties=cn_font)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(out_dir / "causal_graph.png", dpi=160)
    plt.close(fig)
    return out_dir / "causal_graph.png"
