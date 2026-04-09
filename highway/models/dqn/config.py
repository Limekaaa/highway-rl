from typing import NamedTuple


class DqnConfig(NamedTuple):
    """
    Configuration for the DQN agent.
    """

    gamma: float = 0.9
    batch_size: int = 64
    buffer_capacity: int = 15_000
    update_target_every: int = 100

    epsilon_start: float = 0.95
    decrease_epsilon_factor: int = 100
    epsilon_min: float = 0.05

    learning_rate: float = 1e-3


class DqnTrainConfig(NamedTuple):
    """
    Configuration for the DQN training loop.
    """

    n_episodes: int = 10000
    eval_every: int = 50
    reward_threshold: float = 35
    n_sim_per_eval: int = 20
