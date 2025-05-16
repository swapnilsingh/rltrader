
import streamlit as st
import pandas as pd
import plotly.express as px

def render_pnl_distribution(redis_conn, symbol, config):
    st.subheader("🎯 PnL Distribution")

    try:
        queue = config.get("redis_queues", {}).get("pnl_distribution", "").format(symbol=symbol)
        records = redis_conn.lrange("trade_log:btcusdt", -100, -1)
        if not records:
            st.warning("No trade logs found for PnL distribution.")
            return

        trades = [eval(x) for x in records]
        df = pd.DataFrame(trades)

        if 'pnl' not in df.columns:
            st.info("No PnL column found in trades.")
            return

        fig = px.histogram(df, x='pnl', nbins=30, title="Distribution of Trade PnL")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load PnL distribution: {e}")