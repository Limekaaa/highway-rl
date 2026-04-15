from typing import NamedTuple
from datetime import datetime

class PPoConfig(NamedTuple):
    """
    Configuration for the PPO agent.
    None mentioned parameters are set to their default values.
    cf https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    """
    learning_rate: float = 3e-4 
    batch_size: int = 64
    gamma: float = 0.98          
    n_steps: int = 1024          
    n_epochs: int = 15           
    ent_coef: float = 0.01       
    verbose: int = 1

class PpoTrainConfig(NamedTuple):
    """
    Configuration for the PPO training loop.
    Defaults parameters for the training.
    """

    total_timestamps: int = 500_000
    tb_log_name : str = f"ppo_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_interval : int = 4