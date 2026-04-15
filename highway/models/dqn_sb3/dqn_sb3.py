from stable_baselines3 import DQN
from typing import Literal

class DQN_SB3:
    def __init__(self, policy_type: Literal["MlpPolicy", "CnnPolicy", "MultiInputPolicy"], env, *args, **kwargs):
        """
        Check https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html for more details on the parameters.
        """
        self.env = env
        self.model = DQN(policy_type, env, *args, **kwargs)

    def learn(self, total_timesteps, **kwargs):
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def get_action(self, state, seed=0):
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def set_logger(self, logger):
        """Passes the custom logger down to the underlying SB3 model."""
        self.model.set_logger(logger)

    def save(self, path):
        """Saves the underlying SB3 model."""
        self.model.save(path)

    @classmethod
    def load(cls, path, env=None, **kwargs):
        """
        Loads a saved SB3 model from disk and wraps it back into the PPO_SB3 class.
        """
        # 1. Create a blank instance bypassing the standard __init__
        instance = cls.__new__(cls)
        
        # 2. Assign the environment
        instance.env = env
        
        # 3. Load the actual trained model from the zip file and attach it
        instance.model = DQN.load(path, env=env, **kwargs)
        
        return instance


# Backwards-compatible alias (older code imported PPO_SB3 from this file).
PPO_SB3 = DQN_SB3