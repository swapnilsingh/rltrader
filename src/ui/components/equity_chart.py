import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_equity_chart(redis_conn, symbol, config):
    st.subheader("📈 Live Equity Curve")

    # Fetch recent equity data from Redis
    try:
        queue = config.get("redis_queues", {}).get("equity_chart", "").format(symbol=symbol)
        raw_data = redis_conn.lrange("equity:btcusdt", -200, -1)  # last 200 equity points
        if not raw_data:
            st.warning("No equity data available in Redis.")
            return

        records = [eval(x) for x in raw_data]  # each x is a dict
        df = pd.DataFrame(records)

        # Ensure proper formatting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["equity"],
            mode="lines+markers",
            name="Equity"
        ))
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Equity",
            title="Live Equity Curve",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to render equity chart: {e}")