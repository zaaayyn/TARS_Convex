import torch
import numpy as np
import torch.nn as nn
from typing import Any, Dict, Optional, Union

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableCategorical

from encoder import JSSPEncoder
from decoder import StepwiseSchedulingDecoder


class DummyExtractor(nn.Module):
    """
    SB3 requires an extractor + mlp_extractor, but I am going the custom encoding route.
    Here, only provide placeholders for latent_dim_pi / latent_dim_vf .
    """
    def __init__(self, latent_dim_pi: int, latent_dim_vf: int):
        super().__init__()
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf

    def forward(self, observations): return observations
    def forward_actor(self, features): return features
    def forward_critic(self, features): return features


class JSSPPolicy(MaskableActorCriticPolicy):
    """
    Custom strategy: Encoder (three-way structure + gated fusion) + single-step Decoder.
    Also overrides get_distribution / evaluate_actions / predict_values.
    """
    def __init__(
        self, observation_space, action_space, lr_schedule,
        encoder_kwargs=None, decoder_kwargs=None, **kwargs
    ):
        encoder_kwargs = dict(encoder_kwargs or {})
        decoder_kwargs = dict(decoder_kwargs or {})

        # === Automatically parse the features dimension from the obs space to avoid inconsistencies with the feature_type of the env ===
        try:
            feat_dim = int(observation_space.spaces["features"].shape[-1])  # type: ignore[attr-defined]
        except Exception:
            feat_dim = encoder_kwargs.get("input_dim", 9)
        encoder_kwargs.setdefault("input_dim", feat_dim)

        d_model = int(encoder_kwargs.get("d_model", 128))
        self._latent_dim_pi = d_model
        self._latent_dim_vf = d_model

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # dummy extractor + custom submodule
        self.pi_features_extractor = DummyExtractor(self._latent_dim_pi, self._latent_dim_vf)
        self.vf_features_extractor = self.pi_features_extractor

        self.encoder = JSSPEncoder(**encoder_kwargs)
        self.decoder = StepwiseSchedulingDecoder(**decoder_kwargs)

        # Critic: Masked-mean pooled representation of available actions/unscheduled actions
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self._build(lr_schedule)  # Initialize the optimizer/scheduler

    # Placeholder extractor (required for SB3)
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DummyExtractor(self._latent_dim_pi, self._latent_dim_vf)

    # ===== Encoding + Pooling =====
    def _get_latent(self, obs: Dict[str, torch.Tensor], return_encoder_aux: bool = False):
        feats = obs["features"].float()               # [B,N,F]
        job_mask  = obs["job_mask"].bool()           # [B,N,N] (True=mask)
        mach_mask = obs["mach_mask"].bool()          # [B,N,N]

        encoder_aux = None
        if return_encoder_aux:
            enc_out, encoder_aux = self.encoder(feats, job_mask, mach_mask, return_aux=True)   # type: ignore[misc]
        else:
            enc_out = self.encoder(feats, job_mask, mach_mask)   # [B,N,D]
        B, N, D = enc_out.shape
        device = enc_out.device

        # Priority: Available actions; Degrade: Not scheduled; Degrade again: All 1
        act_mask = obs.get("action_mask", None)
        if isinstance(act_mask, np.ndarray):
            act_mask = torch.as_tensor(act_mask, device=device)
        if act_mask is None:
            valid = None
        else:
            valid = act_mask.to(device).bool().float()
            if valid.dim() == 1:  # → [1,N]
                valid = valid.unsqueeze(0).float()

        if valid is None or (valid.sum(dim=1) == 0).any():
            assigned = obs.get("assigned_mask", None)
            if assigned is not None:
                if isinstance(assigned, np.ndarray):
                    assigned = torch.as_tensor(assigned, device=device)
                fallback = (~assigned.to(device).bool()).float()  # non-scheduled
                if fallback.dim() == 1:
                    fallback = fallback.unsqueeze(0)
                if valid is None:
                    valid = fallback
                else:
                    no_act = (valid.sum(dim=1, keepdim=True) == 0).float()
                    valid = valid * (1 - no_act) + fallback * no_act
            else:
                valid = torch.ones((B, N), device=device)

        # masked mean
        denom = valid.sum(dim=1, keepdim=True)
        safe_valid = torch.where(denom > 0, valid, torch.ones_like(valid) * 1e-6)
        weights = safe_valid / (safe_valid.sum(dim=1, keepdim=True) + 1e-8)
        pooled = (enc_out * weights.unsqueeze(-1)).sum(dim=1)        # [B,D]
        if return_encoder_aux:
            return enc_out, pooled, encoder_aux
        return enc_out, pooled

    # ===== Previous action embedding (start_token is used in the first step) =====
    def _build_prev_embed(self, obs: Dict[str, torch.Tensor], enc_out: torch.Tensor) -> torch.Tensor:
        B, N, D = enc_out.shape
        device = enc_out.device
        prev_onehot = obs.get("prev_op_idx", None)

        if prev_onehot is None:
            return self.start_token.to(device).expand(B, 1, -1)

        if isinstance(prev_onehot, np.ndarray):
            prev_onehot = torch.as_tensor(prev_onehot, device=device, dtype=torch.float32)
        else:
            prev_onehot = prev_onehot.to(device).float()

        if prev_onehot.dim() == 1:  # Single instance
            prev_onehot = prev_onehot.unsqueeze(0)

        prev_embed = torch.bmm(prev_onehot.unsqueeze(1), enc_out)    # [B,1,D]
        is_zero = (prev_onehot.abs().sum(dim=1, keepdim=True) == 0.0).unsqueeze(-1)  # [B,1,1]
        if is_zero.any():
            start_tok = self.start_token.to(device).expand(B, 1, -1)
            prev_embed = torch.where(is_zero, start_tok, prev_embed)
        return prev_embed

    def _ensure_action_mask(
        self, obs: Dict[str, torch.Tensor],
        action_masks: Optional[Union[torch.Tensor, np.ndarray]],
        B: int, N: int, device: torch.device
    ) -> torch.Tensor:
        mask = action_masks if action_masks is not None else obs.get("action_mask", None)
        if mask is None:
            return torch.ones((B, N), dtype=torch.bool, device=device)

        if isinstance(mask, np.ndarray):
            mask = torch.as_tensor(mask, device=device)
        else:
            mask = mask.to(device)

        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        assert mask.shape[-1] == N, f"mask last dim {mask.shape[-1]} != N {N}"
        return mask.bool()

    # ===== SB3 interfaces =====
    def get_distribution(
        self, obs: Dict[str, torch.Tensor],
        action_masks: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        enc_out, _ = self._get_latent(obs)                # [B,N,D]
        B, N, _ = enc_out.shape
        device = enc_out.device

        action_masks_t = self._ensure_action_mask(obs, action_masks, B, N, device)
        invalid_mask = ~action_masks_t
        prev_embed = self._build_prev_embed(obs, enc_out) # [B,1,D]

        logits = self.decoder(enc_out, prev_embed, invalid_mask=invalid_mask)  # [B,N]
        neg_inf = -torch.finfo(logits.dtype).max
        safe_logits = logits.masked_fill(invalid_mask, neg_inf)

        dist = MaskableCategorical(logits=safe_logits, masks=action_masks_t)
        dist.get_actions = lambda deterministic=False: (dist.logits.argmax(dim=1) if deterministic else dist.sample())  # type: ignore[attr-defined]
        return dist

    @torch.no_grad()
    def collect_step_diagnostics(
        self,
        obs: Dict[str, Any],
        action_masks: Optional[Union[torch.Tensor, np.ndarray]] = None,
        deterministic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        was_training = self.training
        self.set_training_mode(False)
        try:
            obs_t, _ = self.obs_to_tensor(obs)
            enc_out, _, encoder_aux = self._get_latent(obs_t, return_encoder_aux=True)
            assert encoder_aux is not None

            B, N, _ = enc_out.shape
            device = enc_out.device
            action_masks_t = self._ensure_action_mask(obs_t, action_masks, B, N, device)
            invalid_mask = ~action_masks_t
            prev_embed = self._build_prev_embed(obs_t, enc_out)

            logits = self.decoder(enc_out, prev_embed, invalid_mask=invalid_mask)  # [B,N]
            neg_inf = -torch.finfo(logits.dtype).max
            safe_logits = logits.masked_fill(invalid_mask, neg_inf)

            if deterministic:
                actions = safe_logits.argmax(dim=1)
            else:
                dist = MaskableCategorical(logits=safe_logits, masks=action_masks_t)
                actions = dist.sample()

            gates = encoder_aux["gates"]
            contrib_norms = encoder_aux["contrib_norms"]
            valid_mask_f = action_masks_t.float()
            valid_count = valid_mask_f.sum(dim=1).clamp_min(1.0)

            valid_gate_mean = (gates * valid_mask_f.unsqueeze(-1)).sum(dim=1) / valid_count.unsqueeze(-1)
            valid_gate_sq_mean = ((gates ** 2) * valid_mask_f.unsqueeze(-1)).sum(dim=1) / valid_count.unsqueeze(-1)

            gate_probs = gates.clamp_min(1e-8)
            gate_entropy = -(gate_probs * gate_probs.log()).sum(dim=-1)
            valid_gate_entropy_mean = (gate_entropy * valid_mask_f).sum(dim=1) / valid_count
            valid_contrib_mean = (contrib_norms * valid_mask_f.unsqueeze(-1)).sum(dim=1) / valid_count.unsqueeze(-1)

            gather_gate_idx = actions.long().view(B, 1, 1).expand(-1, 1, gates.shape[-1])
            chosen_gate = gates.gather(dim=1, index=gather_gate_idx).squeeze(1)
            chosen_contrib = contrib_norms.gather(dim=1, index=gather_gate_idx).squeeze(1)
            chosen_gate_entropy = gate_entropy.gather(dim=1, index=actions.long().unsqueeze(1)).squeeze(1)
            chosen_minus_valid = chosen_gate - valid_gate_mean

            return {
                "actions": actions,
                "valid_count": valid_count,
                "valid_gate_mean": valid_gate_mean,
                "valid_gate_sq_mean": valid_gate_sq_mean,
                "valid_gate_entropy_mean": valid_gate_entropy_mean,
                "valid_contrib_mean": valid_contrib_mean,
                "chosen_gate": chosen_gate,
                "chosen_gate_entropy": chosen_gate_entropy,
                "chosen_contrib": chosen_contrib,
                "chosen_minus_valid": chosen_minus_valid,
            }
        finally:
            self.set_training_mode(was_training)

    def forward(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False,
        action_masks: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        dist = self.get_distribution(obs, action_masks)
        actions = dist.get_actions(deterministic=deterministic)      # type: ignore[attr-defined]
        log_prob = dist.log_prob(actions)
        _, pooled = self._get_latent(obs)
        values = self.value_net(pooled).squeeze(-1)
        return actions, values, log_prob

    @torch.no_grad()
    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, pooled = self._get_latent(obs)
        return self.value_net(pooled).squeeze(-1)

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor,
        action_masks: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        enc_out, pooled = self._get_latent(obs)
        B, N, _ = enc_out.shape
        device = enc_out.device

        action_masks_t = self._ensure_action_mask(obs, action_masks, B, N, device)
        invalid_mask = ~action_masks_t
        prev_embed = self._build_prev_embed(obs, enc_out)

        logits = self.decoder(enc_out, prev_embed, invalid_mask=invalid_mask)  # [B,N]
        neg_inf = -torch.finfo(logits.dtype).max
        safe_logits = logits.masked_fill(invalid_mask, neg_inf)

        dist = MaskableCategorical(logits=safe_logits, masks=action_masks_t)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(pooled).squeeze(-1)
        return values, log_prob, entropy
