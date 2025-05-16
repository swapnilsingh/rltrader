
import pytest
from core.reward.reward_traceker import RewardTracker

def test_reward_tracker_init():
    print("🧪 TEST: RewardTracker initialization")
    print("➡️ EXPECTED: Should construct without crashing")
    tracker = RewardTracker()
    print("✅ ACTUAL: Tracker instance created.")
    assert tracker is not None
