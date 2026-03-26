from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # ========= Project/Path =========
    project_name: str = "JSSP"
    run_name: str | None = None
    run_slug: str | None = None
    model_dir: str = "models"
    log_dir: str = "logs"
    tb_log: str = "tb_log"
    BASE_DIR = Path(__file__).resolve().parent
    train_dir = str(BASE_DIR / "jssp_dataset_6x6" / "train")
    valid_dir = str(BASE_DIR / "jssp_dataset_6x6" / "validate")
    RESUME_FROM: str | None = ""          # Manually specify recovery
    auto_resume: bool = False             # Automatically restore from the most recent ckpt

    # ========= General =========
    seed: int = 87             
    num_workers: int = 32
    device: str = "auto"
    wandb_online: bool = True

    # ========= Features/Network =========
    feature_type: str = "full"   # full | static_only | static_plus_start_time
    input_dim: int = 9                  # Will be automatically adjusted in train.py according to feature_type
    d_model: int = 256
    n_heads: int = 16
    n_layers: int = 3
    ff_mult: int = 2
    dropout: float = 0.1
    use_global_encoder: bool = False
    encoder_fusion: str = "convex"     # gated | convex

    # ========= Training =========
    total_timesteps: int = 1_500_000
    n_envs: int = num_workers
    n_steps: int = 216

    # —— Dynamic batch switching (sliding window + insurance)——
    switch_episodes: int = 128*8
    switch_gap_window: int = 16
    switch_gap_threshold: float = 0.005
    switch_patience: int = 16

    # —— Reward related —— 
    reward_alpha: float = 1.0
    reward_mode: str = "terminal"  # shaped | terminal | mixed

    # —— PPO Hyperparameters —— 
    batch_size: int = 1024
    n_epochs: int = 8
    learning_rate: float = 2e-4
    lr_final_ratio: float = 0.15
    lr_warmup_ratio: float = 1
    clip_range_max: float = 0.2
    clip_range_min: float = 0.05
    entropy_coef: float = 0.01
    end_entropy_coef: float = 0.001
    entropy_power: float = 1.5
    entropy_mode: str = "power_decay"  # | warmup_decay
    entropy_warmup_ratio: float = 0.1
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    target_kl: float = 0.01
    max_grad_norm: float = 1.0

    # ========= Evaluation/Save =========
    eval_freq: int = 1 * n_envs * n_steps
    save_freq: int = 8 * n_envs * n_steps
    eval_num_instances: int = 30
    log_every: int = n_envs * n_steps


def _coerce_like(old: Any, raw: str) -> Any:
    """Convert the string raw to the type of old (bool/int/float/str simplified version)."""
    if isinstance(old, bool):
        return raw.lower() in {"1", "true", "t", "yes", "y"}
    if isinstance(old, int) and raw.isdigit():
        return int(raw)
    if isinstance(old, float):
        try:
            return float(raw)
        except Exception:
            return old
    return raw


def parse_config() -> Config:
    """
    Read the default config; optional:
    - Override by specifying JSON/YAML via --config
    - Override individual configuration items via additional CLI options like --key value
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to JSON/YAML config.")
    parser.add_argument("--wandb_online", type=str, default="", help="force on/off (true/false)")
    args, unknown = parser.parse_known_args()

    cfg = Config()

    # 1) Overwrite from file
    file_path = args.config.strip()
    if file_path and Path(file_path).exists():
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                override = json.load(f)
            else:
                import yaml  
                override = yaml.safe_load(f)
        for k, v in (override or {}).items():
            if hasattr(cfg, k):
                val = v.get("value", v) if isinstance(v, dict) else v
                setattr(cfg, k, val)

    # 2) Handle temporary overrides like `--key value`
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok.lstrip("--")
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            raw = unknown[i + 1]
            i += 2
        else:
            raw = "true"
            i += 1
        if hasattr(cfg, key):
            old = getattr(cfg, key)
            setattr(cfg, key, _coerce_like(old, raw))

    # 3) wandb switch 
    if args.wandb_online:
        cfg.wandb_online = args.wandb_online.lower() in {"1", "true", "t", "yes", "y"}

    return cfg
