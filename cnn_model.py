import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import os
import multiprocessing
import platform

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

from shared_core_config import CNN_TRAIN_CONFIG, CNN_CORE_CONFIG, SHARED_CORE_ENV_ID


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config: Dict[str, Any], render_mode: Optional[str] = None):
    env = gym.make(SHARED_CORE_ENV_ID, config=config, render_mode=render_mode)
    env.reset(seed=None)
    if render_mode == "rgb_array":
        env.unwrapped.viewer = None
        env.unwrapped.config["offscreen_rendering"] = True
    return env

# Frame stacking wrapper for CNN observations. Uses VecFrameStack from SB3 which stacks along the channel dimension.
def make_vec_env(config: Dict[str, Any], n_envs: int, n_stack: int):
    def make_env_fn():
        def _init():
            return make_env(config, render_mode=None)
        return _init
    env_fns = [make_env_fn() for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=n_stack, channels_order="first") # Output (C, H, W)
    return vec_env


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _build_run_dir(output_dir: Path, model_name: str, run_name: Optional[str]) -> Path:
    tag = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{model_name}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_cnn_hparams(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_initial_eps": args.exploration_initial_eps,
        "exploration_final_eps": args.exploration_final_eps,
        "verbose": args.verbose,
        "tensorboard_log": None,
        "seed": args.seed,
        "device": args.device,
    }


def train_cnn(args: argparse.Namespace) -> Tuple[Path, Dict[str, Any]]:
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    run_dir = _build_run_dir(output_dir=output_dir, model_name="cnn", run_name=args.run_name)

    train_config = copy.deepcopy(CNN_TRAIN_CONFIG)
    eval_config = copy.deepcopy(CNN_CORE_CONFIG)

    # Single source of truth: read desired temporal depth from config.
    frame_stack = int(train_config["observation"].get("stack_size", 4))

    # Keep env output to a single frame, then apply stacking only through VecFrameStack.
    train_config["observation"]["stack_size"] = 1
    eval_config["observation"]["stack_size"] = 1
    hparams = build_cnn_hparams(args)

    _json_dump(run_dir / "train_config.json", train_config)
    _json_dump(run_dir / "eval_config.json", eval_config)
    _json_dump(run_dir / "hparams.json", hparams)

    with (run_dir / "command.txt").open("w", encoding="utf-8") as f:
        f.write("python cnn_model.py ...\n")

    # Nombre d'envs en parallèles pour le training.
    n_envs_train = 16
    train_env = make_vec_env(train_config, n_envs=n_envs_train, n_stack=frame_stack)
    eval_env = make_vec_env(eval_config, n_envs=1, n_stack=frame_stack)

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        **hparams,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="dqn_cnn",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(1, args.eval_freq),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callback = CallbackList([checkpoint_cb, eval_cb])

    start = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=not args.no_progress_bar,
        callback=callback,
    )
    train_seconds = time.time() - start

    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))

    replay_path = run_dir / "replay_buffer.pkl"
    try:
        model.save_replay_buffer(str(replay_path))
    except Exception:
        replay_path = None

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    best_model_zip = run_dir / "best_model" / "best_model.zip"
    best_mean_reward = None
    best_std_reward = None
    if best_model_zip.exists():
        best_model = DQN.load(str(best_model_zip))
        best_mean_reward, best_std_reward = evaluate_policy(
            best_model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )

    metrics = {
        "train_seconds": train_seconds,
        "total_timesteps": args.total_timesteps,
        "eval_episodes": args.eval_episodes,
        "final_mean_reward": float(mean_reward),
        "final_std_reward": float(std_reward),
        "best_mean_reward": None if best_mean_reward is None else float(best_mean_reward),
        "best_std_reward": None if best_std_reward is None else float(best_std_reward),
        "device": str(model.device),
        "final_model_zip": str(final_model_path) + ".zip",
        "best_model_zip": str(best_model_zip) if best_model_zip.exists() else None,
        "replay_buffer": None if replay_path is None else str(replay_path),
    }
    _json_dump(run_dir / "metrics.json", metrics)

    with (run_dir / "README_RUN.txt").open("w", encoding="utf-8") as f:
        f.write("Artifacts in this folder:\n")
        f.write("- final_model.zip: final policy\n")
        f.write("- best_model/best_model.zip: best checkpoint from periodic eval\n")
        f.write("- checkpoints/: periodic checkpoints\n")
        f.write("- replay_buffer.pkl: replay buffer snapshot for warm restart\n")
        f.write("- train_config.json, eval_config.json, hparams.json, metrics.json\n")

    train_env.close()
    eval_env.close()
    return run_dir, metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNN DQN on highway-env image observations.")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=2)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=5_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # -------------------------
    # SYSTEM INFO
    # -------------------------
    print("\n" + "=" * 80)
    print("SYSTEM INFO")

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python processes (CPU cores): {multiprocessing.cpu_count()}")

    # PyTorch device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device: {device}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA devices: {torch.cuda.device_count()}")
    else:
        print("No GPU detected")

    # CPU threading info (important for SB3 + envs)
    print(f"PyTorch threads (intra-op): {torch.get_num_threads()}")

    # Env vars (useful for debugging performance)
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")

    print("=" * 80 + "\n")

    # -------------------------
    # TRAINING
    # -------------------------
    run_dir, metrics = train_cnn(args)

    # -------------------------
    # RESULTS
    # -------------------------
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Final mean reward: {metrics['final_mean_reward']:.3f} +/- {metrics['final_std_reward']:.3f}")

    if metrics["best_mean_reward"] is not None:
        print(f"Best  mean reward: {metrics['best_mean_reward']:.3f} +/- {metrics['best_std_reward']:.3f}")

    print("=" * 80)

    summary = {
        "model": getattr(args, "model", "cnn"),
        "run_dir": str(run_dir),
        "device_used": device,
        "cpu_cores": multiprocessing.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "final_mean_reward": metrics["final_mean_reward"],
        "final_std_reward": metrics["final_std_reward"],
        "best_mean_reward": metrics["best_mean_reward"],
        "best_std_reward": metrics["best_std_reward"],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
