from stable_baselines3 import PPO
from typing import Literal

class PPO_SB3:
    def __init__(self, policy_type: Literal["MlpPolicy", "CnnPolicy"], env, *args, **kwargs):
        """
        Check https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html for more details on the parameters.
        """
        self.env = env
        self.model = PPO(policy_type, env, *args, **kwargs)

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
        # 1. Create a blank instance of PPO_SB3 bypassing the standard __init__ 
        # (because we don't want to initialize a brand-new, untrained PPO model)
        instance = cls.__new__(cls)
        
        # 2. Assign the environment
        instance.env = env
        
        # 3. Load the actual trained model from the zip file and attach it
        instance.model = PPO.load(path, env=env, **kwargs)
        
        return instance