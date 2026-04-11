import gymnasium as gym


class IdleAgent:
    def __init__(
        self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
    ):
        self.action_space = action_space
        return

    def get_action(self, state, *args):
        return 1  # IDLE

    def update(self, *data):
        pass
