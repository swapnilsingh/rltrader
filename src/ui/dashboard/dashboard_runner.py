import streamlit as st
import redis
from core.utils.config_loader import load_config

from ui.components.equity_chart import render_equity_chart
from ui.components.signal_view import render_signal_view
from ui.components.trade_history import render_trade_history
from ui.components.reward_weights import render_reward_weights
from ui.components.vote_breakdown import render_vote_breakdown
from ui.components.pnl_distribution import render_pnl_distribution
from ui.components.live_decision_params import render_live_decision_params

class RLTraderDashboard:
    def __init__(self):
        # ✅ Load config the same way as trainer
        self.config = load_config(env="local", path="configs/ui/config.yaml")
        redis_host = self.config.get("redis", {}).get("host", "localhost")
        redis_port = self.config.get("redis", {}).get("port", 6379)
        self.symbol = self.config.get("symbol", "btcusdt")

        self.redis_conn = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

        self.tabs = {
            "📈 Equity Curve": render_equity_chart,
            "🤖 Agent Signals": render_signal_view,
            "💰 Trade History": render_trade_history,
            "🧠 Vote Breakdown": render_vote_breakdown,
            "🎯 PnL Distribution": render_pnl_distribution,
            "🧪 Reward Weights": render_reward_weights,
            "🔧 Live Decision Parameters": render_live_decision_params
        }

        self.run()

    def run(self):
        st.set_page_config(layout="wide")
        selected_tab = st.sidebar.radio("🧭 Navigation", list(self.tabs.keys()))
        render_func = self.tabs.get(selected_tab)
        if render_func:
            render_func(self.redis_conn, self.symbol, self.config)
