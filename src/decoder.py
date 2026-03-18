import torch
import torch.nn as nn

class StepwiseSchedulingDecoder(nn.Module):
    """
    Single-step decoding with pointer-style additive attention.
    Each candidate token is scored directly against the current query embedding.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads  # Kept for config compatibility.
        self.token_proj = nn.Linear(d_model, d_model, bias=False)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.score_proj = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_outputs: torch.Tensor,      # [B,N,d] or [N,d]
        prev_selected_embed: torch.Tensor,  # [B,1,d] / [1,d] / [d]
        invalid_mask: torch.Tensor          # [B,N] or [N]，True=不可选
    ) -> torch.Tensor:
        single = False
        if encoder_outputs.dim() == 2:
            encoder_outputs = encoder_outputs.unsqueeze(0); single = True
        if invalid_mask.dim() == 1:
            invalid_mask = invalid_mask.unsqueeze(0)
        if prev_selected_embed.dim() == 1:
            prev_selected_embed = prev_selected_embed.unsqueeze(0).unsqueeze(0)
        elif prev_selected_embed.dim() == 2:
            prev_selected_embed = prev_selected_embed.unsqueeze(1)

        B, N, d = encoder_outputs.shape
        assert d == self.d_model, f"decoder dim mismatch: got {d}, expected {self.d_model}"
        key_padding_mask = invalid_mask.bool()

        query = prev_selected_embed.squeeze(1)                        # [B,d]
        token_term = self.token_proj(encoder_outputs)                 # [B,N,d]
        query_term = self.query_proj(query).unsqueeze(1)              # [B,1,d]
        fused = torch.tanh(self.norm(token_term + query_term))        # [B,N,d]
        fused = self.dropout(fused)
        logits = self.score_proj(fused).squeeze(-1)                   # [B,N]

        # Secondary hard mask, more numerically safe -inf handling
        neg_inf = -torch.finfo(logits.dtype).max
        logits = logits.masked_fill(key_padding_mask, neg_inf)

        return logits.squeeze(0) if single else logits
