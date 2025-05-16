import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mocks():
    with patch("trainer.agent.dqn_trainer.fetch_initial_ohlcv") as mock_fetch_ohlcv, \
         patch("trainer.agent.dqn_trainer.FeatureStateBuilder.build_state") as mock_build_state, \
         patch("trainer.agent.dqn_trainer.load_indicator_agents") as mock_load_agents, \
         patch("trainer.agent.dqn_trainer.RewardAgent.compute_reward_from_ticks") as mock_reward:

        mock_fetch_ohlcv.return_value = [{"timestamp": i, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100} for i in range(100)]
        mock_build_state.return_value = [{"price": 100 + i} for i in range(20)]
        mock_reward.return_value = 1.0

        agent1 = MagicMock()
        agent1.vote.return_value = "BUY"
        agent2 = MagicMock()
        agent2.vote.return_value = "SELL"
        mock_load_agents.return_value = [agent1, agent2]

        yield {
            "fetch_ohlcv": mock_fetch_ohlcv,
            "build_state": mock_build_state,
            "load_agents": mock_load_agents,
            "reward": mock_reward,
            "agent1": agent1,
            "agent2": agent2
        }

@pytest.mark.parametrize("_", [None], ids=["Voting should handle conflicting votes"])
def test_vote_conflict_resolution(_, trainer, mocks):
    print("⚖️ Ensuring aggregate_votes handles conflict without crash")
    result = trainer.aggregate_votes(["BUY", "SELL"])
    assert result in ["BUY", "SELL"], "Unexpected vote resolution fallback"

@pytest.mark.parametrize("_", [None], ids=["Reward error should not crash bootstrap"])
def test_reward_error_does_not_crash(_, trainer, mocks):
    print("💥 Simulate reward computation error in one transition")
    mocks["reward"].side_effect = [1.0] * 10 + [Exception("Boom")] + [1.0] * 9

    trainer.buffer = MagicMock()
    trainer.buffer.add = MagicMock()
    try:
        trainer.bootstrap()
    except Exception as e:
        pytest.fail(f"Reward exception should be handled internally, got: {e}")

@pytest.mark.parametrize("_", [None], ids=["Trainer should call train_on_buffer"])
def test_train_on_buffer_called(_, trainer, mocks):
    print("🎓 Ensuring train_on_buffer is triggered after bootstrap")
    trainer.buffer = MagicMock()
    trainer.train_on_buffer = MagicMock()
    trainer.bootstrap()
    trainer.train_on_buffer.assert_called_once()
