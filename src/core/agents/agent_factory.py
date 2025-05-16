class DummyAgent:
    def vote(self, state):
        return "BUY"

def load_indicator_agents():
    return [DummyAgent(), DummyAgent()]
