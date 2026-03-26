import sys, json, subprocess
from pathlib import Path
import wandb

from train_sweep import main as entry

FILE_DIR = Path(__file__).resolve().parent

# ---- Automatically detect project root: the directory containing jssp_dataset is preferred, otherwise the parent directory is used ----
if (FILE_DIR / "jssp_dataset").exists():
    PROJECT_ROOT = FILE_DIR
elif (FILE_DIR.parent / "jssp_dataset").exists():
    PROJECT_ROOT = FILE_DIR.parent
else:
    PROJECT_ROOT = FILE_DIR

def _find_train_script() -> Path:
    """Find an executable train script for "long run refresher" 
    (if not, fall back to train_sweep.py)."""
    candidates = [
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "current_workflow" / "train.py",
        FILE_DIR / "train.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return FILE_DIR / "train_sweep.py"  # fallback

TRAIN_SCRIPT = _find_train_script()

WANDB_ENTITY  = None  # If you need a fixed entity, fill in "your-entity"
WANDB_PROJECT = "JSSP"

def _rank_ascending(values):
    idx = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0] * len(values)
    for r, i in enumerate(idx, start=1):
        ranks[i] = r
    return ranks

def _get_metric(summary, key, default=float("inf")):
    v = summary.get(key, default)
    try:
        return float(v)
    except Exception:
        return float("inf")

def promote_top_and_longrun(
    sweep_id: str,
    top_n: int = 5,
    long_steps: int = 1_000_000,
    finetune: bool = False,
    replicate_seeds = (1001, 2002, 3003),
):
    """
    Select the top n runs from the sweep and use their hyperparameters for the long run:
    - finetune=False: Only reuse hyperparameters and train from scratch
    - finetune=True: Continue training on the best model
    """
    api = wandb.Api()
    swp = api.sweep(sweep_id)
    runs = [r for r in swp.runs if r.state == "finished" and r.summary is not None]
    if not runs:
        print("[promote] no finished runs, skip.")
        return

    online_vals = [_get_metric(r.summary, "eval/online_mean_gap") for r in runs]
    online_ranks = _rank_ascending(online_vals)

    scored = []
    for i, r in enumerate(runs):
        scored.append((online_ranks[i], online_vals[i], r))
    scored.sort(key=lambda x: (x[0], x[1]))
    picked = [t[2] for t in scored[:top_n]]

    print(f"[promote] picked {len(picked)} runs for long-run (rank by eval/online_mean_gap).")

    for i, r in enumerate(picked, 1):
        base_cfg_raw = dict(r.config)
        base_cfg = {k: (v.get("value", v) if isinstance(v, dict) else v) for k, v in base_cfg_raw.items()}
        base_slug = base_cfg.get("run_slug", r.name) or r.name

        for sd in (replicate_seeds or [base_cfg.get("seed", 0)]):
            cfg = dict(base_cfg)
            cfg["total_timesteps"] = int(long_steps)
            cfg["RESUME_FROM"] = ""
            cfg["auto_resume"] = False

            if finetune:
                cfg["finetune_from"] = base_slug
                suffix = "long"
            else:
                cfg["finetune_from"] = ""
                suffix = "long_fresh"

            cfg["seed"] = int(sd)
            cfg["run_slug"] = f"{base_slug}_{suffix}_s{sd}"
            cfg["dataset_dir"] = str((PROJECT_ROOT / "jssp_dataset").resolve())

            out_json = PROJECT_ROOT / f"longrun_{r.id}_s{sd}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            print(f"[promote] launching {i}/{len(picked)}  seed={sd}  from={base_slug}  mode={suffix}")
            subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT), "--config", str(out_json)],
                check=True,
                cwd=str(PROJECT_ROOT),
            )

# ======= Sweep Search Space =======
sweep_config = {
    "name": "sweep_fusion",
    "method": "bayes",
    "metric": {"name": "eval/online_mean_gap", "goal": "minimize"},
    "parameters": {
        # ================================================================
        # 1. Fixed parameters 
        # ================================================================
        
        # --- model ---
        "d_model":          {"value": 256},         
        "n_heads":          {"value": 16},          
        "n_layers":         {"value": 3},           
        "ff_mult":          {"value": 2},          
        
        # --- Experiment setup ---
        "feature_type":     {"values": ["full"]},
        "use_global_encoder": {"values": [False]},
        "encoder_fusion":   {"values": ["gated", "convex"]},
        "n_epochs":         {"values": [16]},           
        "n_envs":           {"value": 32},
        
        # --- timesteps ---
        "total_timesteps":  {"value": 1_000_000},      
        
        # --- PPO parameters ---
        "gamma":            {"values": [0.99]},
        "vf_coef":          {"value": 0.5},
        "max_grad_norm":    {"value": 1.0},
        "clip_range_min":   {"value": 0.05},
        
        # ================================================================
        # 2. Sweep parameters
        # ================================================================

        # --- lr ---
        "learning_rate":    {"value": 2e-4}, #{"distribution": "log_uniform_values", "min": 5e-5, "max": 3e-4},
        
        "reward_mode":      {"values": ["shaped"]},

        # --- Rollout Buffer ---
        "n_steps":          {"values": [216]}, # 6x6下，测试 36*4 和 36*6 两种长度
        "batch_size":       {"values": [512]},

        # --- PPO stability parameters ---
        "clip_range_max":   {"values": [0.2]},
        "gae_lambda":       {"values": [0.95]},
        "target_kl":        {"values": [0.01]},

        # --- Exploration and Regularization ---
        "entropy_coef":     {"values": [0.02]},
        "dropout":          {"values": [0.1]},
        
        "seed":             {"values": [87]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
    print("Created sweep:", sweep_id)
    # Let the agent call the entry point we provided (train_sweep.main）
    wandb.agent(sweep_id, function=entry, count=2)

    # If you need to do a "long-term refresher" on the optimal configurations, uncomment:
    # promote_top_and_longrun(sweep_id=sweep_id, top_n=5, long_steps=1_000_000, finetune=False)
