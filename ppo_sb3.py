from stable_baselines3 import PPO
from typing import Literal

class PPO_SB3:
    def __init__(self, policy_type: Literal["MlpPolicy", "CnnPolicy"], env, *args, **kwargs):
        """
        Check https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html for more details on the parameters.
        """
        self.env = env
        self.model = PPO(policy_type, env, *args, **kwargs)

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps)

    def get_action(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action