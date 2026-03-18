import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Union

from encoder import StructureAwareEncoder


class ConvexCombinationFusion(nn.Module):
    """
    Fuse the three branch outputs with a single learnable convex combination.

    The mixing weights are shared across all samples/tokens and constrained to
    the probability simplex via softmax.
    """

    def __init__(self, d_model: int, init_logits: Optional[Tensor] = None):
        super().__init__()
        if init_logits is None:
            init_logits = torch.zeros(3, dtype=torch.float32)
        else:
            init_logits = init_logits.detach().clone().float().view(3)
        self.branch_logits = nn.Parameter(init_logits)
        self.d_model = d_model

    def _weights(self, ref: Tensor) -> Tensor:
        return torch.softmax(self.branch_logits.to(dtype=ref.dtype, device=ref.device), dim=0)

    def forward(
        self,
        e_job: Tensor,
        e_mach: Tensor,
        e_global: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Dict[str, Tensor]]]:
        single = False
        if e_job.dim() == 2:
            e_job = e_job.unsqueeze(0)
            e_mach = e_mach.unsqueeze(0)
            e_global = e_global.unsqueeze(0)
            single = True

        weights = self._weights(e_job)
        w_job = weights[0].view(1, 1, 1)
        w_mach = weights[1].view(1, 1, 1)
        w_global = weights[2].view(1, 1, 1)

        contrib_job = w_job * e_job
        contrib_mach = w_mach * e_mach
        contrib_global = w_global * e_global
        fused = contrib_job + contrib_mach + contrib_global

        if not return_aux:
            return fused.squeeze(0) if single else fused

        B, N, _ = fused.shape
        gates = weights.view(1, 1, 3).expand(B, N, 3)
        contrib_norms = torch.cat(
            [
                torch.norm(contrib_job, dim=-1, keepdim=True),
                torch.norm(contrib_mach, dim=-1, keepdim=True),
                torch.norm(contrib_global, dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        aux = {
            "gates": gates,
            "contrib_norms": contrib_norms,
        }
        if single:
            aux = {k: v.squeeze(0) for k, v in aux.items()}
            return fused.squeeze(0), aux
        return fused, aux


class JSSPEncoderConvex(nn.Module):
    """
    Three-way structured encoder with a simple convex combination fusion.

    Unlike the original gated encoder, the branch weights are static learned
    parameters shared across all samples and tokens.
    """

    def __init__(
        self,
        input_dim: int = 9,
        d_model: int = 128,
        n_heads: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 2,
        use_global_encoder: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.job_encoder = StructureAwareEncoder(d_model, n_heads, dim_feedforward, dropout, num_layers)
        self.mach_encoder = StructureAwareEncoder(d_model, n_heads, dim_feedforward, dropout, num_layers)
        self.global_encoder = (
            StructureAwareEncoder(d_model, n_heads, dim_feedforward, dropout, num_layers)
            if use_global_encoder
            else nn.Identity()
        )
        self.fusion = ConvexCombinationFusion(d_model)

    def forward(
        self,
        x: Tensor,
        job_mask: Tensor,
        mach_mask: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Dict[str, Tensor]]]:
        h = F.relu(self.input_proj(x))
        h = self.input_norm(h)

        e_job = self.job_encoder(h, src_mask=job_mask)
        e_mach = self.mach_encoder(h, src_mask=mach_mask)
        e_global = self.global_encoder(h) if not isinstance(self.global_encoder, nn.Identity) else h
        return self.fusion(e_job, e_mach, e_global, return_aux=return_aux)


# Drop-in alias for consumers that want this module to mirror encoder.py.
JSSPEncoder = JSSPEncoderConvex
