from typing import NamedTuple
from datetime import datetime

class DqnConfig(NamedTuple):
    """
    Configuration for the DQN agent.
    None mentioned parameters are set to their default values.
    cf https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    """
    learning_rate: float = 5e-4
    buffer_size: int = 10_000
    learning_starts: int = 200
    batch_size: int = 32
    gamma: float = 0.8
    train_freq: int = 1
    gradient_steps: int = 1
    optimize_memory_usage: bool = True
    target_update_interval: int = 50
    exploration_fraction: float = 0.7
    verbose: int = 1

class DqnTrainConfig(NamedTuple):
    """
    Configuration for the DQN training loop.
    Defaults parameters for the training.
    """

    total_timestamps: int = 100_000
    tb_log_name : str = f"cnn_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_interval : int = 4