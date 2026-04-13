import os
import numpy as np


def get_training_paths(date_str: str):
    best_model_path = os.path.join("model_weights", "dqn", f"dqn_best_model_{date_str}.pth")

    losses = np.load(os.path.join("results", "dqn", "loss", f"dqn_losses_{date_str}.npy"))
    rewards = np.load(os.path.join("results", "dqn", "reward", f"dqn_rewards_{date_str}.npy"))
    lengths = np.load(os.path.join("results", "dqn", "length", f"dqn_lengths_{date_str}.npy"))

    ep_paths = [path for path in os.listdir(os.path.join("model_weights", "dqn")) if date_str in path and "_ep" in path]
    ep_paths = sorted(ep_paths, key=lambda x: int(x.split("_")[-2][2:]))

    return best_model_path, losses, rewards, lengths, ep_paths
