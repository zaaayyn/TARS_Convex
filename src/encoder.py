import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Union

class GatedWeightedFusion(nn.Module):
    """Gated weighted fusion: Dynamically fuse job/machine/global representations by token"""
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        e_job: Tensor,
        e_mach: Tensor,
        e_global: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Dict[str, Tensor]]]:
        # Supports [N,d] or [B,N,d]
        single = False
        if e_job.dim() == 2:
            e_job = e_job.unsqueeze(0); e_mach = e_mach.unsqueeze(0); e_global = e_global.unsqueeze(0)
            single = True

        concat_all = torch.cat([e_job, e_mach, e_global], dim=-1)   # [B,N,3d]
        gates = self.gate(concat_all)                               # [B,N,3]
        w_job   = gates[..., 0:1]
        w_mach  = gates[..., 1:2]
        w_global= gates[..., 2:3]
        contrib_job = w_job * e_job
        contrib_mach = w_mach * e_mach
        contrib_global = w_global * e_global
        fused = contrib_job + contrib_mach + contrib_global  # [B,N,d]

        if not return_aux:
            return fused.squeeze(0) if single else fused

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


class StructureAwareEncoder(nn.Module):
    """
    A single-pass structured Transformer Encoder.
    Supports "pairwise masked" attention masks for jobs/machines, etc.:
    - x: [B,N,D] or [N,D]
    - src_mask: [B,N,N] or [N,N] (True if mask is used)
    """
    def __init__(self, d_model: int, n_heads: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1, n_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads

    def _fast_path_possible(self, x: Tensor, src_mask: Optional[Tensor]) -> bool:
        return (src_mask is not None and x.dim() == 3 and src_mask.dim() == 3
                and src_mask.shape[0] == x.shape[0])

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        single = False
        if x.dim() == 2:
            x = x.unsqueeze(0); single = True
            if src_mask is not None and src_mask.dim() == 2:
                src_mask = src_mask.unsqueeze(0)  # → [1,N,N]

        if src_mask is not None:
            src_mask = src_mask.bool()

        # Fast-path: Try passing [B,N,N] to Encoder via repeat_interleave(H)
        if (src_mask is not None) and self._fast_path_possible(x, src_mask):
            B, N, _ = x.shape
            attn_mask = src_mask.bool().repeat_interleave(self.n_heads, dim=0)  # [B*H, N, N]
            try:
                out = self.encoder(x, mask=attn_mask)
                out = self.norm(out)
                return out.squeeze(0) if single else out
            except Exception:
                pass  # Falling back to per-sample

        # Safe-path: forward independently sample by sample, passing [N,N] to each Encoder
        outs = []
        B = x.shape[0]
        for b in range(B):
            mask_b = None if src_mask is None else src_mask[b]
            out_b = self.encoder(x[b:b+1], mask=mask_b)     # [1,N,D]，mask_b: [N,N] or None
            outs.append(out_b)
        out = torch.cat(outs, dim=0)
        out = self.norm(out)
        return out.squeeze(0) if single else out


class JSSPEncoder(nn.Module):
    """
    Three-way structured encoder + gated fusion.
    Args:
    input_dim: The dimension of the environment feature (may be 4/5/9 depending on feature_type)
    d_model: The model dimension
    n_heads: The number of attention heads
    num_layers: The number of layers per channel
    use_global_encoder: Whether to enable the global encoder (True/False)
    """
    def __init__(self, input_dim: int = 9, d_model: int = 128, n_heads: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1, num_layers: int = 2,
                 use_global_encoder: bool = True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.job_encoder  = StructureAwareEncoder(d_model, n_heads, dim_feedforward, dropout, num_layers)
        self.mach_encoder = StructureAwareEncoder(d_model, n_heads, dim_feedforward, dropout, num_layers)
        self.global_encoder = (
            StructureAwareEncoder(d_model, n_heads, dim_feedforward, dropout, num_layers)
            if use_global_encoder else nn.Identity()
        )
        self.fusion = GatedWeightedFusion(d_model)

    def forward(
        self,
        x: Tensor,
        job_mask: Tensor,
        mach_mask: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Dict[str, Tensor]]]:
        """
        x: [B,N,input_dim] or [N,input_dim]
        job_mask: [B,N,N] or [N,N] (True if mask)
        mach_mask: [B,N,N] or [N,N] (True if mask)
        Returns: [B,N,d_model] or [N,d_model]
        """
        h = F.relu(self.input_proj(x))
        h = self.input_norm(h)

        e_job    = self.job_encoder(h,  src_mask=job_mask)
        e_mach   = self.mach_encoder(h, src_mask=mach_mask)
        e_global = self.global_encoder(h) if not isinstance(self.global_encoder, nn.Identity) else h
        return self.fusion(e_job, e_mach, e_global, return_aux=return_aux)
