import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time

# --- Settings ---
SENSOR_COUNT = 5
THRESHOLD = 70

# --- Generate Dummy Data ---
def create_dummy_df(n=200, anomaly_prob=0.05, seed=42):
    np.random.seed(seed)
    ts = [dt.datetime.now() + dt.timedelta(seconds=i) for i in range(n)]
    data = {}
    for i in range(1, SENSOR_COUNT + 1):
        base = np.random.normal(50, 5, n)
        mask = np.random.rand(n) < anomaly_prob
        base[mask] += np.random.normal(20, 5, mask.sum())  # anomalies
        data[f"sensor_{i}"] = base
    df = pd.DataFrame(data)
    df.insert(0, "timestamp", ts)
    return df

# --- UI Setup ---
st.set_page_config(layout="wide")

st.title("🛰️ Real-Time Industrial IoT Sensor Dashboard")
st.markdown("""
Welcome to the **Simulated IoT Monitoring App**.

🔹 Simulates real-time sensor readings
🔹 Displays streaming line charts & anomalies
🔹 Shows agent-style reasoning for detected events
""")

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Controls")
speed = st.sidebar.slider("⏩ Playback speed", 0.1, 5.0, 1.0)
window = st.sidebar.slider("🕓 Window size (sec)", 5, 60, 20)
st.sidebar.markdown("---")
st.sidebar.markdown("👤 Built by [Your Name](https://example.com)")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/44/Industrial_icon.png", width=150)

# --- Load Data ---
df = create_dummy_df()
sensor_cols = df.columns[1:]
selected_sensors = st.multiselect("📊 Select sensors to display", sensor_cols.tolist(), default=sensor_cols[:3])

# --- App Layout ---
alert = st.empty()
chart = st.empty()
metrics = st.columns(len(selected_sensors))
agent_box = st.container()

# --- Streaming Loop ---
for i in range(len(df)):
    now = df['timestamp'].iloc[i]
    window_df = df[df['timestamp'] >= now - pd.Timedelta(seconds=window)]

    # --- Chart Update ---
    with chart.container():
        st.line_chart(window_df.set_index("timestamp")[selected_sensors])

    # --- Real-Time Metrics ---
    for j, sensor in enumerate(selected_sensors):
        latest_val = window_df[sensor].iloc[-1]
        metrics[j].metric(sensor, round(latest_val, 2))

    # --- Anomaly Check ---
    anomalies = (window_df[selected_sensors] > THRESHOLD).any(axis=1)
    if anomalies.any():
        alert.warning("⚠️ Anomaly detected in the selected window!")
        agent_message = "Agent: Sudden spike detected. Recommend checking affected sensors."
    else:
        alert.success("✅ System operating normally.")
        agent_message = "Agent: All sensor values within expected range."

    # --- Agent Reasoning Panel ---
    with agent_box:
        st.markdown("### 🤖 Agent Reasoning")
        st.info(agent_message)

    time.sleep(1.0 / speed)
