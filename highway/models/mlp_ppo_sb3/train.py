from pathlib import Path
import json
import sys

import argparse

import gymnasium as gym

try:
    # Registers `highway-v0` into gymnasium
    import highway_env  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency `highway-env`. Install it (e.g. `python -m pip install -r requirements.txt`)"
    ) from e

try:
    # Prefer package import when run as a module: python -m highway.models.ppo_sb3.train
    from .ppo_sb3 import PPO_SB3
except ImportError:
    # Fallback when run as a script from this folder.
    from ppo_sb3 import PPO_SB3

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

try:
    from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG, TEST_CONFIG
except ModuleNotFoundError:
    # When running as a script (python highway/models/ppo_sb3/train.py), the repo root
    # isn't on sys.path; add it so shared_core_config.py can be imported.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from shared_core_config import SHARED_CORE_ENV_ID, SHARED_CORE_CONFIG, TEST_CONFIG

def make_env(render_mode=None):
    env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG, render_mode=render_mode)
    env.reset()
    if render_mode == "rgb_array":
        # Allow rendering in a notebook
        env.unwrapped.viewer = None
        env.unwrapped.config["offscreen_rendering"] = True
    return env

from stable_baselines3.common.logger import configure # <-- Add this import

def train(
    config: dict = {},
    ):

    seed = config.get("seed", 0)
    total_timesteps = config.get("total_timesteps", 1000)
    eval_freq = config.get("eval_freq", 100)
    eval_episodes = config.get("eval_episodes", 10)

    save_freq = config.get("save_freq", 500) 

    log_dir = config.get("log_dir", "logs/ppo_sb3")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Set up the SB3 Logger
    # This tells SB3 to print to the terminal ("stdout") and save to a CSV ("csv")
    # You can also add "tensorboard" to this list if you want Tensorboard logs
    new_logger = configure(str(log_dir), ["stdout", "csv"])

    train_env = make_env(render_mode=None)
    # The Monitor wrapper still handles raw episode rewards and lengths
    train_env = Monitor(train_env, filename=str(log_dir / "train_monitor.csv"))

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_model"
    )

    agent_ppo = PPO_SB3(
        "MlpPolicy",
        train_env, 
        **config.get("agent_config", {})
    )

    # 2. Attach the custom logger to your agent BEFORE learning
    agent_ppo.set_logger(new_logger)

    agent_ppo.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )
    train_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to a JSON config file. If not provided, defaults to `config.json` in the same folder as this script.",
    )
    args = parser.parse_args()
    config_path = Path(__file__).with_name(args.config)
    with config_path.open("r") as f:
        config = json.load(f)
    train(config)