from enum import Enum

import gymnasium as gym
import highway_env

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TEST_CONFIG, CNN_TEST_CONFIG, CNN_EVAL_CONFIG

class ConfigType(Enum):
    SHARED_CORE = 1
    TEST = 2
    EVAL_CNN = 3
    TEST_CNN = 4
    


def get_env(seed=None, config_type=ConfigType.TEST):
    """
    Create and return the environment with the specified configuration and seed.
    """
    config = (
        SHARED_CORE_CONFIG if config_type == ConfigType.SHARED_CORE else
        TEST_CONFIG if config_type == ConfigType.TEST else
        CNN_TEST_CONFIG if config_type == ConfigType.TEST_CNN else
        CNN_EVAL_CONFIG if config_type == ConfigType.EVAL_CNN
        else None)
    env = gym.make(SHARED_CORE_ENV_ID, config=config, render_mode="rgb_array")
    if config_type == ConfigType.TEST_CNN :
        env.unwrapped.config.update({"policy_frequency": 15, "duration": 20})
    env.reset(seed=seed)

    env.unwrapped.viewer = None
    env.unwrapped.config["offscreen_rendering"] = True
    
    return env