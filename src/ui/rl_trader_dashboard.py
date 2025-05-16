from core.decorators.decorators import inject_logger
from ui.dashboard.dashboard_runner import RLTraderDashboard
import streamlit as st

@inject_logger()
class NeurotradeApp:
    def __init__(self):
        self.logger.info("🚀 Starting Neurotrade Streamlit Dashboard...")
        self.runner = RLTraderDashboard()

if __name__ == "__main__":
    NeurotradeApp()
