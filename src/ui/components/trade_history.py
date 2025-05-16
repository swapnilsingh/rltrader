
import streamlit as st
import pandas as pd

def render_trade_history(redis_conn, symbol, config):
    st.subheader("💰 Trade History")

    try:
        queue = config.get("redis_queues", {}).get("trade_history", "").format(symbol=symbol)
        records = redis_conn.lrange("trade_log:btcusdt", -100, -1)  # latest 100 trades
        if not records:
            st.warning("No trade history found in Redis.")
            return

        parsed = [eval(rec) for rec in records]
        df = pd.DataFrame(parsed)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp", ascending=False)

        st.dataframe(df[['timestamp', 'side', 'price', 'quantity', 'pnl', 'reason']])
    except Exception as e:
        st.error(f"Failed to load trade history: {e}")