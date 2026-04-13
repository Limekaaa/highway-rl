class SB3GreedyAgent:
    def __init__(self, model):
        self.model = model

    def get_action(self, state, epsilon=None):
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)
