from enum import Enum

import gymnasium as gym
import highway_env

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TEST_CONFIG


class ConfigType(Enum):
    SHARED_CORE = 1
    TEST = 2


def get_env(seed=None, config_type=ConfigType.TEST):
    """
    Create and return the environment with the specified configuration and seed.
    """
    config = (
        SHARED_CORE_CONFIG if config_type == ConfigType.SHARED_CORE else TEST_CONFIG
    )
    env = gym.make(SHARED_CORE_ENV_ID, config=config, render_mode="rgb_array")
    env.reset(seed=seed)

    env.unwrapped.viewer = None
    env.unwrapped.config["offscreen_rendering"] = True
    return env
