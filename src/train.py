import os
import sys
import gc
import glob
import random
import logging
from pathlib import Path
from typing import Any, List
from datetime import datetime

import torch
import wandb
from wandb import Settings
from sb3_contrib import MaskablePPO

from config import Config, parse_config
from gym_jssp_env import GymJSSPEnv
from utils.env_utils import build_train_env, seed_everything  
from callback import build_callbacks
from custom_policy import JSSPPolicy


if not hasattr(sys.stderr, "isatty"):
    sys.stderr.isatty = lambda: False

COUNTER_PATH = Path(__file__).parent / "run_counter.txt"

def setup_logging(cfg: Config) -> Path:
    """Initializes logging and wandb, and returns the log file path."""
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    if COUNTER_PATH.exists():
        raw = COUNTER_PATH.read_text().strip()
        run_idx = int(raw) + 1 if raw.isdigit() else 1
    else:
        run_idx = 1
    COUNTER_PATH.write_text(str(run_idx))

    arch = "withG" if cfg.use_global_encoder else "noG"
    run_name = f"{arch}_seed{cfg.seed}_{datetime.now():%Y%m%d-%H%M%S}"
    if cfg.run_slug is None:
        cfg.run_slug = run_name

    log_path = Path(cfg.log_dir) / f"{cfg.run_slug}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s 鈥?%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8", mode="a")],
    )

    mode = "online" if cfg.wandb_online else "offline"
    tmp_dir = Path(cfg.log_dir) / "wandb" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_TEMP_DIR", str(tmp_dir))

    wandb.init(
        project=cfg.project_name,
        name=run_name,
        config=vars(cfg),
        save_code=True,
        settings=Settings(console="wrap"),
        mode=mode,
        dir=str(Path(cfg.log_dir) / "wandb"),
    )
    return log_path


def setup_terminal_logger(log_path: str):
    """Write stdout/traceback to both a file and the terminal."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    class TeeLogger:
        def __init__(self, stream, log_f):
            self.stream = stream
            self.log_f = log_f

        def write(self, data):
            self.stream.write(data)
            self.log_f.write(data)
            self.log_f.flush()

        def flush(self):
            self.stream.flush()
            self.log_f.flush()

    sys.stdout = TeeLogger(sys.stdout, log_file)  # type: ignore
    sys.stderr = TeeLogger(sys.stderr, log_file)  # type: ignore


# ===== Learning rate/clip schedule =====
def make_lr_schedule(lr: float, final_ratio: float = 0.1, warmup_ratio: float = 0.0):
    lr = float(lr)
    final_ratio = float(final_ratio)
    warmup_ratio = float(warmup_ratio)

    def _schedule(progress: float) -> float:
        progress = float(progress)
        progress = min(max(progress, 0.0), 1.0)
        if warmup_ratio > 0 and progress > (1.0 - warmup_ratio):
        # Warm-up zone: The goal is to go from 0 -> lr
        # Calculates progress in the warm-up zone (0 -> 1)
            warmup_progress = (1.0 - progress) / warmup_ratio
            return max(1e-8, lr * warmup_progress)
        # The remaining interval: lr linearly decays to final_ratio*lr
        return max(1e-8, lr * (final_ratio + (1.0 - final_ratio) * progress))

    return _schedule


def make_clip_schedule(max_clip: float, min_clip: float):
    max_clip = float(max_clip)
    min_clip = float(min_clip)

    def _schedule(progress: float) -> float:
        progress = float(progress)
        return min_clip + (max_clip - min_clip) * progress

    return _schedule


# ===== Model construction/recovery =====
def init_model(cfg: Config, train_env):
    """
    Initialize or load models.
    - Priority 1: resuming_from -> seamlessly resumes training, loading all states and returning directly.
    - Priority 2: finetune_from -> fine-tunes, loading weights, but using a new scheduler.
    - Priority 3: creating a new model.
    """
    # --- Path 1: Seamless Resume (RESUME_FROM) ---
    resume_path = (cfg.RESUME_FROM or "").strip()
    if resume_path and Path(resume_path).exists() and resume_path.endswith(".zip"):
        logging.info(f"[RESUME] Loading checkpoint to continue training: {resume_path}")
        # Return directly after loading the model, preserving the complete optimizer and scheduler state
        model = MaskablePPO.load(
            resume_path,
            env=train_env,
            device=cfg.device,
            custom_objects={"policy_class": JSSPPolicy}
        )
        return model

    # --- Path 2: Fine-tuning (finetune_from) ---
    model = None
    finetune_path = (getattr(cfg, "finetune_from", None) or "").strip()
    if finetune_path:
        best_model_path = Path(cfg.model_dir) / finetune_path / "best_model" / "best_model.zip"
        if best_model_path.exists():
            logging.info(f"[FINETUNE] Loading weights from: {best_model_path}")
            model = MaskablePPO.load(
                str(best_model_path),
                env=train_env,
                device=cfg.device,
                custom_objects={"policy_class": JSSPPolicy}
            )
        else:
            logging.warning(f"[FINETUNE] 'finetune_from' was set, but model not found at {best_model_path}. Starting new.")

    policy_kwargs = {
        "encoder_kwargs": {
            "d_model": cfg.d_model, "n_heads": cfg.n_heads, "input_dim": cfg.input_dim,
            "num_layers": cfg.n_layers, "dropout": cfg.dropout,
            "dim_feedforward": int(cfg.ff_mult * cfg.d_model),
            "use_global_encoder": cfg.use_global_encoder,
        },
        "decoder_kwargs": {
            "d_model": cfg.d_model, "n_heads": cfg.n_heads, "dropout": cfg.dropout,
        },
    }

    # --- Path 3: Create a new model (if not loaded in the previous two cases) ---
    if model is None:
        logging.info("[NEW MODEL] Creating a new model from scratch.")
        model = MaskablePPO(
            policy=JSSPPolicy,
            env=train_env,
            learning_rate=cfg.learning_rate, # Temporary value, which will be overwritten by the scheduler immediately
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            clip_range=cfg.clip_range_max, # Temporary value
            ent_coef=cfg.entropy_coef,
            vf_coef=cfg.vf_coef,
            tensorboard_log=cfg.tb_log,
            policy_kwargs=policy_kwargs,
            device=cfg.device,
            verbose=1,
        )

    # --- For "new" or "fine-tuned" models, the scheduler is overwritten with the current configuration. ---
    logging.info("[CONFIG OVERRIDE] Applying schedules and PPO params from current config.")
    model.lr_schedule = make_lr_schedule(cfg.learning_rate, final_ratio=cfg.lr_final_ratio, warmup_ratio=cfg.lr_warmup_ratio)
    
    # Calculate the correct starting learning rate based on the current number of steps
    progress = max(0.0, 1.0 - model.num_timesteps / max(1, cfg.total_timesteps))
    start_lr = float(model.lr_schedule(progress))
    for g in model.policy.optimizer.param_groups:
        g["lr"] = start_lr

    model.clip_range = make_clip_schedule(cfg.clip_range_max, cfg.clip_range_min)
    model.target_kl = cfg.target_kl
    model.max_grad_norm = cfg.max_grad_norm
    model.gae_lambda = cfg.gae_lambda
    
    return model


def train_loop(model, callbacks, cfg: Config):
    model.learn(
        total_timesteps=cfg.total_timesteps,
        reset_num_timesteps=False,
        callback=callbacks,
        progress_bar=False,
    )


def safe_close_env(env) -> None:
    if env is None:
        return
    try:
        if hasattr(env, "close"):
            env.close()
        if hasattr(env, "env_method"):
            try:
                env.env_method("close")
            except Exception:
                pass
        if hasattr(env, "envs"):
            for sub_env in env.envs:
                try:
                    if hasattr(sub_env, "close"):
                        sub_env.close()
                except Exception:
                    continue
    except Exception as e:
        logging.warning(f"[Cleanup] safe close warning: {e}")


def release_runtime_resources(model=None, train_env=None, callbacks=None) -> None:
    if callbacks is not None:
        close_fn = getattr(callbacks, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as e:
                logging.warning(f"[Cleanup] callback close warning: {e}")

    model_env = None
    if model is not None:
        try:
            model_env = model.get_env()
        except Exception:
            model_env = None
    safe_close_env(model_env)
    if train_env is not None and train_env is not model_env:
        safe_close_env(train_env)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unwrap_env(env):
    """Peel off the wrappers to get the bottom layer GymJSSPEnv (for testing/statistics only)."""
    unwrapped: Any = env
    while True:
        if hasattr(unwrapped, "env"):
            inner = getattr(unwrapped, "env")
        elif hasattr(unwrapped, "venv"):
            inner = getattr(unwrapped, "venv")
        else:
            break
        unwrapped = inner
    assert isinstance(unwrapped, GymJSSPEnv), f"unwrap_env result is not GymJSSPEnv: {type(unwrapped)}"
    return unwrapped


def eval_loop(model, eval_envs) -> List[float]:
    """Perform a full rollout on eval_envs and return a list of gaps for each instance."""
    gaps: List[float] = []
    for env in eval_envs:
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.action_masks()  # type: ignore
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

        base_env = unwrap_env(env)
        final_ms = base_env.get_makespan()
        gap = final_ms / base_env.optimal_ms - 1.0
        gaps.append(gap)
    return gaps


# ===== main =====
def main():
    cfg = parse_config()
    train_env = None
    model = None
    callbacks = None

    if cfg.feature_type == "full":
        cfg.input_dim = 9
    elif cfg.feature_type == "static_only":
        cfg.input_dim = 4
    elif cfg.feature_type == "static_plus_start_time":
        cfg.input_dim = 5
    else:
        raise ValueError(f"Unknown feature_type: {cfg.feature_type}")

    log_file = setup_logging(cfg)
    setup_terminal_logger(str(log_file))

    seed_everything(cfg.seed)

    train_instances = sorted([Path(p) for p in glob.glob(os.path.join(cfg.train_dir, "*.txt"))])
    eval_instances = sorted([Path(p) for p in glob.glob(os.path.join(cfg.valid_dir, "*.txt"))])
    assert len(train_instances) >= cfg.n_envs, "训练实例数量必须 >= n_envs"

    try:
        train_files = random.sample(train_instances, cfg.n_envs)
        train_env = build_train_env(train_files, cfg)
        model = init_model(cfg, train_env)
        callbacks = build_callbacks(
            cfg,
            train_instances=train_instances,
            eval_instances=eval_instances[: cfg.eval_num_instances],
        )
        train_loop(model, callbacks, cfg)
    finally:
        release_runtime_resources(model=model, train_env=train_env, callbacks=callbacks)
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()


