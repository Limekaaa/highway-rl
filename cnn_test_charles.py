import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from shared_core_config import CNN_TRAIN_CONFIG, SHARED_CORE_ENV_ID

# --- NOUVEAU MODULE : SPATIAL ATTENTION ---

class SpatialAttention(nn.Module):
    """
    Module d'attention spatiale qui apprend à focaliser sur les zones 
    importantes de l'image (les autres véhicules).
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # On compresse l'info des canaux pour créer un masque 2D
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # On concatène les stats pour nourrir le masque
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        return x * self.sigmoid(res) # Applique le masque d'attention

# --- FEATURE EXTRACTOR MODIFIÉ ---

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # On divise le CNN en deux parties pour insérer l'attention au milieu
        self.low_level = nn.Sequential(
            # Block 1
            nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Couche d'attention
        self.attention = SpatialAttention()

        self.high_level = nn.Sequential(
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcul dynamique de la dimension aplatie
        with torch.no_grad():
            test_input = torch.zeros(
                1, n_input_channels, 
                observation_space.shape[1], 
                observation_space.shape[2]
            )
            # On simule le passage dans tout le réseau
            temp = self.low_level(test_input)
            temp = self.attention(temp)
            n_flatten = self.high_level(temp).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.low_level(observations)
        x = self.attention(x) # L'attention est appliquée ici
        x = self.high_level(x)
        return self.linear(x)

# --- RESTE DU SCRIPT (FONCTIONNELLEMENT IDENTIQUE) ---

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(config: Dict[str, Any], render_mode: str | None = None):
    env = gym.make(SHARED_CORE_ENV_ID, config=config, render_mode=render_mode)
    env.reset(seed=None)
    if render_mode == "rgb_array":
        env.unwrapped.viewer = None
        env.unwrapped.config["offscreen_rendering"] = True
    return env

def make_vec_env(config: Dict[str, Any]):
    vec_env = DummyVecEnv([lambda: make_env(config, render_mode=None)])
    if len(vec_env.observation_space.shape) == 3 and vec_env.observation_space.shape[-1] in (1, 3, 4):
        vec_env = VecTransposeImage(vec_env)
    return vec_env

def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def _build_run_dir(output_dir: Path, model_name: str, run_name: str | None) -> Path:
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

def train_cnn_attention(args: argparse.Namespace) -> Tuple[Path, Dict[str, Any]]:
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    run_dir = _build_run_dir(output_dir=output_dir, model_name="cnn_attention", run_name=args.run_name)

    train_config = dict(CNN_TRAIN_CONFIG)
    eval_config = dict(CNN_TRAIN_CONFIG)
    hparams = build_cnn_hparams(args)

    _json_dump(run_dir / "train_config.json", train_config)
    _json_dump(run_dir / "eval_config.json", eval_config)
    _json_dump(run_dir / "hparams.json", hparams)

    train_env = make_vec_env(train_config)
    eval_env = make_vec_env(eval_config)

    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, 512],
    )

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        **hparams,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="dqn_attention",
        save_replay_buffer=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(1, args.eval_freq),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
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

    model.save(str(run_dir / "final_model"))
    
    # Évaluation finale
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)

    metrics = {
        "train_seconds": train_seconds,
        "final_mean_reward": float(mean_reward),
        "final_std_reward": float(std_reward),
        "device": str(model.device),
    }
    _json_dump(run_dir / "metrics.json", metrics)

    train_env.close()
    eval_env.close()
    return run_dir, metrics

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNN Attention DQN.")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=2_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.3)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_dir, metrics = train_cnn_attention(args)
    print(f"Entraînement terminé. Résultats dans : {run_dir}")

if __name__ == "__main__":
    main()