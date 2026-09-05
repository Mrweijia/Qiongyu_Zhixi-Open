"""Stage-3 innovation models (see docs/project/decisions.md).

Each model is a self-contained module with a unified interface:

* ``causal_gat`` — causal discovery + masked GAT over transport edges
* ``pde_graph`` — advection-diffusion discretised adjacency + PINN residual
* ``ddpm`` — conditional diffusion probabilistic forecast (PM2.5 exceedance)
* ``foundation`` — masked ST-Transformer pretrain/fine-tune
* ``gnode`` — graph neural ODE, continuous-time dynamics
* ``multimodal`` — ERA5 + AOD gated fusion
* ``wind_gated_tcn`` — wind graph + multi-scale causal temporal convolutions
* ``adaptive_graph_transformer`` — physical/learned graph fusion + Transformer

All models read ``x [B,L,N,F]`` normalised and output ``[B,H,N,K]``.
"""

from .common import DeltaStepHead, GraphConv, GraphGRUEncoder, last_obs_base
from .codex_models import (AdaptiveGraphTransformer, CausalTemporalBlock,
                           WindGatedTCN)
from .causal_discovery import (CausalGraph, causal_graph_from_timeline,
                               granger_adjacency, row_normalize_with_self,
                               save_causal_graph_artifacts, sparsify_in_topk)
from .causal_gat import CausalGATLayer, CausalTGCN
from .ddpm import (ConditionalPM25Diffusion, GaussianDiffusion,
                   NodeWiseDenoiser, cosine_beta_schedule,
                   crps_ensemble, exceedance_stats)
from .foundation import (STMaskFormer, adapt_input_proj,
                         load_foundation_ckpt, save_foundation_ckpt)
from .gnode import GraphNeuralODE, GraphODEFunc, rk4_step
from .multimodal import ModalGate, MultimodalTGCN
from .pde_graph import (AdvectionDiffusionTGCN, PDEResidualLoss,
                        advection_operator, diffusion_operator,
                        pair_geometry, wind_uv_from_dir_speed)

__all__ = [
    "CausalGATLayer", "CausalTGCN",
    "CausalGraph", "causal_graph_from_timeline", "granger_adjacency",
    "row_normalize_with_self", "save_causal_graph_artifacts", "sparsify_in_topk",
    "AdvectionDiffusionTGCN", "PDEResidualLoss", "advection_operator",
    "diffusion_operator", "pair_geometry", "wind_uv_from_dir_speed",
    "ConditionalPM25Diffusion", "GaussianDiffusion", "NodeWiseDenoiser",
    "cosine_beta_schedule", "crps_ensemble", "exceedance_stats",
    "STMaskFormer", "adapt_input_proj", "load_foundation_ckpt", "save_foundation_ckpt",
    "GraphNeuralODE", "GraphODEFunc", "rk4_step",
    "ModalGate", "MultimodalTGCN",
    "DeltaStepHead", "GraphConv", "GraphGRUEncoder", "last_obs_base",
    "CausalTemporalBlock", "WindGatedTCN", "AdaptiveGraphTransformer",
]
