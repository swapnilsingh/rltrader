
import streamlit as st
import pandas as pd
import plotly.express as px

def render_reward_weights(redis_conn, symbol, config):
    st.subheader("🧪 Reward Weights")

    try:
        queue = config.get("redis_queues", {}).get("reward_weights", "").format(symbol=symbol)
        records = redis_conn.lrange("reward_metrics:btcusdt", -100, -1)
        if not records:
            st.warning("No reward metrics found in Redis.")
            return

        parsed = [eval(rec) for rec in records]
        df = pd.DataFrame(parsed)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp")

        # Line chart for each reward component
        reward_cols = [col for col in df.columns if col not in ['timestamp']]
        fig = px.line(df, x="timestamp", y=reward_cols, title="Reward Component Weights")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to render reward weights: {e}")