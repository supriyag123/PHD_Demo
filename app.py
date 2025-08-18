# app.py - Complete Agentic Anomaly Detection Application
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agents.adaptive_window_agent import AdaptiveWindowAgent
from agents.sensor_agent import SensorAgent
from agents.global_anomaly_agent import GlobalAnomalyAgent
from agents.master_agent import MasterAgent
from utils.data_stream import simulate_stream

# --- Initialize agents ---
@st.cache_resource
def initialize_agents():
    print("🤖 Initializing agent network...")
    adaptive_agent = AdaptiveWindowAgent()
    sensor_agents = {f"sensor_{i}": SensorAgent(f"sensor_{i}") for i in range(1, 6)}
    global_agent = GlobalAnomalyAgent()
    master_agent = MasterAgent(sensor_agents, global_agent, adaptive_agent)
    print("✅ Agent network initialized")
    return adaptive_agent, sensor_agents, global_agent, master_agent

adaptive_agent, sensor_agents, global_agent, master_agent = initialize_agents()

# --- UI Setup ---
st.set_page_config(
    layout="wide", 
    page_title="Agentic IoT Anomaly Detection",
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.title("🛰️ Agentic IoT Anomaly Detection System")
st.markdown("**Real-time MetroPT data analysis with Enhanced MLP-based adaptive windowing**")

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ System Controls")
    
    # Playback controls
    st.subheader("📊 Data Stream")
    speed = st.slider("⏩ Playback Speed", 0.1, 5.0, 1.0, 0.1)
    
    auto_run = st.checkbox("🔄 Auto-run", value=True)
    
    if st.button("⏸️ Pause/Resume"):
        st.session_state.paused = not st.session_state.get('paused', False)
    
    if st.button("🔄 Reset System"):
        st.cache_resource.clear()
        st.rerun()
    
    # Model info
    st.subheader("🤖 MLP Model Status")
    model_info = adaptive_agent.get_model_info()
    st.write("**Enhanced MLP:**", "✅ Loaded" if model_info['model_available'] else "❌ Not Available")
    st.write("**Training Status:**", "✅ Trained" if model_info['is_trained'] else "⏳ Learning")
    st.write("**Feature Selection:**", "✅ Active" if model_info['has_selector'] else "📊 Basic")
    
    # Agent status
    st.subheader("🔍 Agent Network")
    st.write("**Adaptive Window Agent:**", "🟢 Active")
    st.write("**Sensor Agents:**", f"🟢 {len(sensor_agents)} Active")
    st.write("**Global Anomaly Agent:**", "🟢 Active")
    st.write("**Master Agent:**", "🟢 Orchestrating")
    
    st.markdown("---")
    st.markdown("👤 **Agentic IoT System**")
    st.markdown("*Powered by Enhanced Feature Engineering MLP*")

# --- Initialize session state ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'window_history' not in st.session_state:
    st.session_state.window_history = []
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = []
if 'stream_iterator' not in st.session_state:
    st.session_state.stream_iterator = simulate_stream()

# --- Main Layout ---
# Top row - Metrics and Controls
metrics_row = st.container()
with metrics_row:
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    
    # Sensor metrics will be updated in the loop
    sensor_metric_containers = [col1, col2, col3, col4, col5]
    window_metric_container = col6

# Alert banner
alert_container = st.container()

# Main charts row
charts_row = st.container()
with charts_row:
