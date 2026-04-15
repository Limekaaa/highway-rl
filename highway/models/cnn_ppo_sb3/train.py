import argparse
import logging
import json
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import highway_env  # noqa: F401

try:
    # Prefer package import when run as a module: python -m highway.models.cnn_ppo_sb3.train
    from .config import PPoConfig, PpoTrainConfig
except ImportError:
    # Fallback when run as a script from this folder.
    from config import PPoConfig, PpoTrainConfig

from highway.scripts.environment import ConfigType, get_env
from shared_core_config import CNN_TEST_CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and record a PPO on highway-env from images observation and CNNpolicy.")
    parser.add_argument("--output-root", type=str, default=r"./outputs_cnn_ppo/", help="Root directory for outputs (model, tensorboard logs).")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    tb_dir = output_root / "tb"
    eval_logs_dir = output_root / "eval_logs"
    best_model_dir = output_root / "best_model"
    model_path = output_root / "model"

    tb_dir.mkdir(parents=True, exist_ok=True)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Configurations
    config_model = PPoConfig()
    config_train = PpoTrainConfig()
    logging.info(f"Model Configuration (if not mentioned, default values used): {config_model}")
    logging.info(f"Training Configuration (if not mentioned, default values used): {config_train}")
    
    # Eval callback
    with (output_root / "eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(CNN_TEST_CONFIG, f, indent=2)
    
    eval_env = DummyVecEnv([lambda: Monitor(get_env(config_type=ConfigType.TEST_CNN))])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_logs_dir),
        eval_freq=1000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    # Train
    model = PPO(
        "CnnPolicy",
        DummyVecEnv([lambda: get_env(config_type=ConfigType.TEST_CNN)]),
        learning_rate=config_model.learning_rate,
        n_steps=config_model.n_steps,         
        batch_size=config_model.batch_size,         
        n_epochs=config_model.n_epochs,           
        gamma=config_model.gamma, 
        ent_coef=config_model.ent_coef,
        verbose=config_model.verbose,
        tensorboard_log=str(tb_dir),
    )
    model.learn(total_timesteps=config_train.total_timestamps, 
                tb_log_name=config_train.tb_log_name, 
                log_interval=config_train.log_interval,
                callback=eval_callback,)
    model.save(str(model_path))
    eval_env.close()

    print(f"Saved eval logs: {eval_logs_dir / 'evaluations.npz'}")
    print(f"Saved best model: {best_model_dir / 'best_model.zip'}")
    
    # Export compact curves for later plotting (reward + validation episode length).
    eval_npz = eval_logs_dir / "evaluations.npz"
    if eval_npz.exists():
        eval_data = np.load(eval_npz)
        timesteps = eval_data["timesteps"]
        rewards = eval_data["results"]
        ep_lengths = eval_data["ep_lengths"]

        curves_npz = output_root / "eval_curves.npz"
        np.savez(
            curves_npz,
            timesteps=timesteps,
            rewards=rewards,
            ep_lengths=ep_lengths,
            mean_rewards=rewards.mean(axis=1),
            std_rewards=rewards.std(axis=1),
            mean_ep_lengths=ep_lengths.mean(axis=1),
            std_ep_lengths=ep_lengths.std(axis=1),
        )
        print(f"Saved eval curves (reward + length): {curves_npz}")
