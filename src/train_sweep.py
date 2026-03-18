import os, sys, math, random, logging, datetime, traceback, gc
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import wandb
from wandb.sdk.wandb_settings import Settings  

from config        import Config, parse_config
from gym_jssp_env  import GymJSSPEnv
from utils.env_utils     import build_train_env, make_env, seed_everything  
from callback      import build_callbacks
from custom_policy import JSSPPolicy
from sb3_contrib   import MaskablePPO

# If stderr doesn't have isatty, give it a fake implementation 
# (to avoid wandb progress bar errors)
if not hasattr(sys.stderr, "isatty"):
    sys.stderr.isatty = lambda: False

def make_lr_schedule(lr0: float, final_ratio: float = 0.08, warmup_ratio: float = 0.05):
    """SB3-style progress: progress_remaining. 
    Linear warmup followed by cosine annealing to lr0*final_ratio."""
    final_lr = lr0 * final_ratio
    def f(progress_remaining: float) -> float:
        t = 1.0 - float(progress_remaining)   # [0鈫?]
        if warmup_ratio > 0 and t < warmup_ratio:
            return lr0 * (t / max(1e-8, warmup_ratio))
        tt = (t - warmup_ratio) / max(1e-8, 1.0 - warmup_ratio)
        return final_lr + 0.5 * (lr0 - final_lr) * (1.0 + math.cos(math.pi * tt))
    return f

def make_clip_schedule(c0, c1):
    def f(progress_remaining):
        t = 1.0 - float(progress_remaining)
        return c1 + 0.5*(c0-c1)*(1.0+math.cos(math.pi*t))
    return f

# ======== Terminal and file dual logging ========
def setup_terminal_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    class TeeLogger:
        def __init__(self, stream, log_f):
            self.stream, self.log_f = stream, log_f
        def write(self, msg):
            self.stream.write(msg); self.log_f.write(msg)
        def flush(self):
            self.stream.flush(); self.log_f.flush()
        def isatty(self):  
            return False
        def fileno(self):  
            try:
                return self.stream.fileno()
            except Exception:
                return 2
    sys.stdout = TeeLogger(sys.stdout, log_file)
    sys.stderr = TeeLogger(sys.stderr, log_file)

# ======== Run naming and basic logging ========
COUNTER_PATH = Path(__file__).parent / "run_counter.txt"
def setup_logging(cfg):
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    if COUNTER_PATH.exists():
        raw = COUNTER_PATH.read_text().strip()
        run_idx = int(raw) + 1 if raw.isdigit() else 1
    else:
        run_idx = 1
    COUNTER_PATH.write_text(str(run_idx))

    cfg.run_name = f"exp{run_idx:03d}_{cfg.seed}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    cfg.run_slug = cfg.run_name

    log_path = Path(cfg.log_dir) / f"{cfg.run_name}.log"
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s 鈥?%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8", mode="a")]
    )
    return log_path

# ======== Model construction ========
def init_model(cfg: Config, env):
    return MaskablePPO(
        policy=JSSPPolicy,
        env=env,
        learning_rate=make_lr_schedule(
            cfg.learning_rate,
            final_ratio=getattr(cfg, "lr_final_ratio", 0.08),
            warmup_ratio=getattr(cfg, "lr_warmup_ratio", 0.05),
        ),
        clip_range=make_clip_schedule(
            getattr(cfg, "clip_range_max", 0.2),
            getattr(cfg, "clip_range_min", 0.1),
        ),
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.entropy_coef,
        vf_coef=cfg.vf_coef,
        target_kl=getattr(cfg, "target_kl", None),
        max_grad_norm=getattr(cfg, "max_grad_norm", 0.5),
        tensorboard_log=cfg.tb_log,
        policy_kwargs={
            "encoder_kwargs": {
                "input_dim": cfg.input_dim,              
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "num_layers": cfg.n_layers,
                "use_global_encoder": cfg.use_global_encoder,
            },
            "decoder_kwargs": {
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
            },
        },
        device=cfg.device,
        verbose=1,
    )

def train_loop(model, callbacks, cfg):
    model.learn(
        total_timesteps=int(cfg.total_timesteps),
        reset_num_timesteps=True,
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


def safe_close_envs(envs) -> None:
    if envs is None:
        return
    for env in envs:
        safe_close_env(env)


def release_runtime_resources(model=None, train_env=None, raw_eval_envs=None, callbacks=None) -> None:
    if callbacks is not None:
        close_fn = getattr(callbacks, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as e:
                logging.warning(f"[Cleanup] callback close warning: {e}")

    safe_close_envs(raw_eval_envs)

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

# ======== Eval ========
def unwrap_env(env):
    unwrapped: Any = env
    while True:
        if hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        elif hasattr(unwrapped, "venv"):
            unwrapped = unwrapped.venv
        else:
            break
    assert isinstance(unwrapped, GymJSSPEnv)
    return unwrapped

def eval_loop(model, eval_envs) -> List[float]:
    gaps = []
    for env in eval_envs:
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            mask = env.action_masks() if hasattr(env, "action_masks") else env.action_mask
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, terminated, truncated, _ = env.step(int(action))
        base = unwrap_env(env)
        final_ms = base.get_makespan()
        gaps.append(final_ms / base.optimal_ms - 1.0)
    return gaps

# ======== Main entry (for sweep) ========
def main():
    run = None
    train_env = None
    model = None
    callbacks = None
    raw_eval_envs = []
    try:
        cfg: Config = parse_config()
        log_path = setup_logging(cfg)
        setup_terminal_logger(str(log_path))
        logging.info(f"Training logs {log_path}")

        run = wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=vars(cfg),
            save_code=True,
            reinit=True,
            settings=Settings(console="wrap"),
        )

        for k, v in run.config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        feat_dim_map = {"full": 9, "static_plus_start_time": 5, "static_only": 4}
        ft = getattr(cfg, "feature_type", "full")
        cfg.input_dim = int(feat_dim_map.get(ft, 9))
        logging.info(f"[config] feature_type={ft}  encoder.input_dim={cfg.input_dim}")

        desired = str(getattr(cfg, "device", "auto")).lower()
        if desired in {"auto", "cuda", "cuda:auto"}:
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            cfg.device = desired
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            logging.warning("Requested CUDA but not available, fallback to CPU")
            cfg.device = "cpu"
        if cfg.device.startswith("cuda"):
            dev = torch.device(cfg.device)
            logging.info(f"Using GPU: {cfg.device} 鈥?{torch.cuda.get_device_name(dev)}")
            torch.backends.cudnn.benchmark = True
        else:
            logging.info("Using CPU")

        seed_everything(cfg.seed)

        train_files = sorted(Path(cfg.train_dir).glob("*.txt"))
        valid_files = sorted(Path(cfg.valid_dir).glob("*.txt"))
        random.shuffle(train_files); random.shuffle(valid_files)

        train_subset = train_files[: cfg.n_envs]
        eval_subset  = valid_files[: cfg.eval_num_instances]

        train_env    = build_train_env(train_subset, cfg)
        raw_eval_envs = [make_env(fp, cfg, i)() for i, fp in enumerate(eval_subset)]

        callbacks = build_callbacks(
            cfg,
            train_instances=train_subset,
            eval_instances=eval_subset,
        )
        model = init_model(cfg, train_env)

        assert cfg.clip_range_min < cfg.clip_range_max, "clip_range_min must be < clip_range_max"
        assert 0.8 <= cfg.gae_lambda < 1.0
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        train_loop(model, callbacks, cfg)

        gaps = eval_loop(model, raw_eval_envs)
        wandb.log({
            "final_eval/mean_gap": np.mean(gaps),
            "final_eval/std_gap":  np.std(gaps),
            "final_eval/min_gap":  np.min(gaps),
            "final_eval/max_gap":  np.max(gaps),
            "final_eval/gap_hist": wandb.Histogram(gaps),
        }, step=model.num_timesteps)

        final_path = Path(cfg.model_dir) / f"model_final_{model.num_timesteps}.zip"
        model.save(final_path)
        logging.info(f"Saved final model to {final_path}")
        wandb.summary["exit_code"] = 0

    except Exception:
        wandb.summary["exit_code"] = 1
        print(traceback.format_exc())
        err_file = "logs/exception.txt"
        os.makedirs(os.path.dirname(err_file), exist_ok=True)
        with open(err_file, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        try:
            import shutil
            dest = Path(wandb.run.dir) / "exception.txt"  # type: ignore
            shutil.copyfile(err_file, dest)
        except Exception:
            pass
    finally:
        release_runtime_resources(
            model=model,
            train_env=train_env,
            raw_eval_envs=raw_eval_envs,
            callbacks=callbacks,
        )
        try:
            wandb.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()

