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

        dummy_agent = MagicMock()
        dummy_agent.vote.return_value = "BUY"
        mock_load_agents.return_value = [dummy_agent, dummy_agent]

        yield {
            "fetch_ohlcv": mock_fetch_ohlcv,
            "build_state": mock_build_state,
            "load_agents": mock_load_agents,
            "reward": mock_reward,
        }

@pytest.mark.parametrize("_", [None], ids=["Bootstrap should call fetch_initial_ohlcv"])
def test_fetch_ohlcv_called(_, trainer, mocks):
    print("🔍 Ensuring that OHLCV data is fetched at the start of bootstrap.")
    trainer.buffer = MagicMock()
    trainer.bootstrap()
    assert mocks["fetch_ohlcv"].called

@pytest.mark.parametrize("_", [None], ids=["Bootstrap should call FeatureStateBuilder.build_state"])
def test_state_builder_called(_, trainer, mocks):
    print("🧱 Ensuring that state builder is used to transform OHLCV to feature state.")
    trainer.buffer = MagicMock()
    trainer.bootstrap()
    assert mocks["build_state"].called

@pytest.mark.parametrize("_", [None], ids=["Bootstrap should compute rewards and store experiences"])
def test_experience_added_to_buffer(_, trainer, mocks):
    print("🎯 Verifying that rewards are calculated and experiences are added to the buffer.")
    trainer.buffer = MagicMock()
    trainer.buffer.add = MagicMock()
    trainer.bootstrap()
    assert trainer.buffer.add.call_count == 19

@pytest.mark.parametrize("_", [None], ids=["Bootstrap should exit early if not enough states"])
def test_bootstrap_exits_on_insufficient_states(_, trainer, mocks):
    print("🚧 Verifying bootstrap exits if state builder returns < 2 states.")
    trainer.buffer = MagicMock()
    mocks["build_state"].return_value = [{"price": 100.0}]  # Only 1 state
    trainer.bootstrap()
    trainer.buffer.add.assert_not_called()

@pytest.mark.parametrize("_", [None], ids=["Bootstrap should not raise exceptions"])
def test_bootstrap_executes_without_crashing(_, trainer, mocks):
    print("🧪 Ensuring bootstrap executes cleanly under mocked setup.")
    trainer.buffer = MagicMock()
    trainer.buffer.add = MagicMock()
    try:
        trainer.bootstrap()
    except Exception as e:
        pytest.fail(f"Bootstrap raised an unexpected exception: {e}")
