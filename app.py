import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
from agents.adaptive_window_agent import AdaptiveWindowAgent
from agents.sensor_agent import SensorAgent
from agents.global_anomaly_agent import GlobalAnomalyAgent
from agents.master_agent import MasterAgent
from utils.data_stream import simulate_stream

# --- Initialize agents ---
adaptive_agent = AdaptiveWindowAgent()
sensor_agents = {f"sensor_{i}": SensorAgent(f"sensor_{i}") for i in range(1, 6)}
global_agent = GlobalAnomalyAgent()
master_agent = MasterAgent(sensor_agents, global_agent, adaptive_agent)

# --- UI Setup ---
st.set_page_config(layout="wide")
st.title("🛰️ Agentic IoT Anomaly Detection System")

st.sidebar.title("⚙️ Controls")
speed = st.sidebar.slider("⏩ Playback speed", 0.1, 5.0, 1.0)
st.sidebar.markdown("---")
st.sidebar.markdown("👤 Built by [Your Name](https://example.com)")

# --- Layout ---
chart = st.empty()
metrics = st.columns(5)
alert = st.empty()
reasoning = st.container()
logbox = st.expander("🧠 Agent Logs", expanded=False)

# --- Simulated Data Stream ---
stream = simulate_stream()

history = []

for packet in stream:
    timestamp, readings = packet["timestamp"], packet["readings"]
    history.append({"timestamp": timestamp, **readings})
    df = pd.DataFrame(history).set_index("timestamp")

    # --- Adaptive Window ---
    window_size = adaptive_agent.predict_window(df)
    window_df = df.tail(window_size)

    # --- Sensor Agents ---
    local_alerts = []
    for i, (sensor, agent) in enumerate(sensor_agents.items()):
        latest_val = readings[sensor]
        metrics[i].metric(sensor, round(latest_val, 2))
        if agent.detect(latest_val):
            local_alerts.append(sensor)

    # --- Global Anomaly Detection ---
    global_anomaly = global_agent.detect(window_df)

    # --- Master Agent Orchestration ---
    final_decision, reasoning_text = master_agent.decide(local_alerts, global_anomaly)

    # --- UI Updates ---
    chart.line_chart(df)

    if final_decision:
        alert.warning("⚠️ Anomaly Detected by Agent Network!")
    else:
        alert.success("✅ System Stable")

    with reasoning:
        st.markdown("### 🤖 Agent Reasoning")
        st.info(reasoning_text)

    with logbox:
        st.write(master_agent.get_logs())

    time.sleep(1.0 / speed)
