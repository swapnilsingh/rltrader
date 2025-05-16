class RewardAgent:
    def __init__(self, evaluator_agent=None, state_tracker=None):
        self.evaluator_agent = evaluator_agent
        self.state_tracker = state_tracker

    def compute_reward_from_ticks(self, state, action, next_state):
        return 1.0  # placeholder static reward
