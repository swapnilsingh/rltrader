class FeatureStateBuilder:
    def build_state(self, candles):
        return [{ 'price': c['close'] } for c in candles]  # simplistic mock state
