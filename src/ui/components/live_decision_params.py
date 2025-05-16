import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_live_decision_params(redis_conn, symbol, config):
    st.subheader("🔧 Live Decision Parameters")

    try:
        queue = config.get("redis_queues", {}).get("live_decision_params", "").format(symbol=symbol)
        records = redis_conn.lrange("signal:btcusdt", -50, -1)
        if not records:
            st.warning("No signal data found.")
            return

        data = [eval(r) for r in records]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp")

        fig = go.Figure()
        for param in ["confidence", "cooldown", "quantity"]:
            if param in df.columns:
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df[param], mode="lines+markers", name=param))

        fig.update_layout(title="Live Decision Parameters Over Time", xaxis_title="Timestamp", height=400)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering live decision params: {e}")