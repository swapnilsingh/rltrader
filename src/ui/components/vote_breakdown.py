
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_vote_breakdown(redis_conn, symbol, config):
    st.subheader("🧠 Agent Vote Breakdown")

    try:
        queue = config.get("redis_queues", {}).get("vote_breakdown", "").format(symbol=symbol)
        records = redis_conn.lrange("signal:btcusdt", -1, -1)
        if not records:
            st.warning("No recent signal data found.")
            return

        latest = eval(records[0])
        logits = latest.get("signal_logits", {})
        if not logits:
            st.info("No signal logits available in latest signal.")
            return

        labels = list(logits.keys())
        values = list(logits.values())

        fig = go.Figure([go.Bar(x=labels, y=values)])
        fig.update_layout(title="Agent Vote Strengths (Signal Logits)", yaxis_title="Logit Value")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to show vote breakdown: {e}")