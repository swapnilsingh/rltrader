
import streamlit as st
import pandas as pd

def render_signal_view(redis_conn, symbol, config):
    st.subheader("🤖 Latest Agent Signals")

    try:
        queue = config.get("redis_queues", {}).get("signal_view", "").format(symbol=symbol)
        records = redis_conn.lrange("signal:btcusdt", -50, -1)  # last 50 signals
        if not records:
            st.warning("No signals found in Redis.")
            return

        parsed = [eval(rec) for rec in records]
        df = pd.DataFrame(parsed)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp", ascending=False)

        st.dataframe(df[['timestamp', 'action', 'confidence', 'quantity', 'cooldown', 'reason']])
    except Exception as e:
        st.error(f"Failed to load agent signals: {e}")