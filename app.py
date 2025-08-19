# app.py - Complete Agentic Anomaly Detection Application
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import agents
from agents.adaptive_window_agent import AdaptiveWindowAgent
from agents.sensor_agent import SensorAgent
from agents.global_anomaly_agent import GlobalAnomalyAgent
from agents.master_agent import MasterAgent
from utils.data_stream import simulate_stream

# Initialize agents
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

# UI Setup
st.set_page_config(
    layout="wide", 
    page_title="Agentic IoT Anomaly Detection",
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

# Header
st.title("🛰️ Agentic IoT Anomaly Detection System")
st.markdown("**Real-time labeled subsequence analysis with Enhanced MLP-based adaptive windowing**")

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ System Controls")
    
    # Data source configuration
    st.subheader("📁 Data Source")
    data_dir = st.text_input("Data Directory", value="data/", help="Directory containing your labeled subsequences")
    
    # Playback controls
    st.subheader("📊 Data Stream")
    speed = st.slider("⏩ Playback Speed", 0.1, 5.0, 1.0, 0.1)
    auto_run = st.checkbox("🔄 Auto-run", value=True)
    
    if st.button("⏸️ Pause/Resume"):
        st.session_state.paused = not st.session_state.get('paused', False)
    
    if st.button("🔄 Reset System"):
        st.cache_resource.clear()
        st.session_state.history = []
        st.session_state.window_history = []
        st.session_state.anomaly_history = []
        st.session_state.stream_iterator = simulate_stream(data_dir)
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

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'window_history' not in st.session_state:
    st.session_state.window_history = []
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = []
if 'stream_iterator' not in st.session_state:
    st.session_state.stream_iterator = simulate_stream(data_dir)

# Main Layout - Metrics Row
metrics_row = st.container()
with metrics_row:
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    sensor_metric_containers = [col1, col2, col3, col4, col5]
    window_metric_container = col6

# Alert banner
alert_container = st.container()

# Main charts
st.subheader("📈 Real-time Data & Window Prediction")
chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.write("**Sensor Data Stream**")
    main_chart = st.empty()

with chart_col2:
    st.write("**MLP Window Prediction**")
    window_chart = st.empty()

# Reasoning and logs
st.subheader("🤖 System Intelligence")
reasoning_col, logs_col = st.columns([1, 1])

with reasoning_col:
    st.write("**Agent Reasoning**")
    reasoning_container = st.empty()

with logs_col:
    st.write("**System Logs**")
    logs_container = st.empty()

# Main Data Processing Loop
if auto_run and not st.session_state.get('paused', False):
    try:
        # Get next data packet
        packet = next(st.session_state.stream_iterator)
        timestamp, readings = packet["timestamp"], packet["readings"]
        
        # Add to history
        current_data = {"timestamp": timestamp, **readings}
        st.session_state.history.append(current_data)
        
        # Keep history manageable (last 200 points)
        if len(st.session_state.history) > 200:
            st.session_state.history = st.session_state.history[-200:]
        
        # Create DataFrame
        df = pd.DataFrame(st.session_state.history)
        df.set_index("timestamp", inplace=True)
        
        # MLP-based Adaptive Window Prediction
        predicted_window = adaptive_agent.predict_window(df.reset_index())
        st.session_state.window_history.append({
            'timestamp': timestamp,
            'window_size': predicted_window,
            'true_window': packet.get('true_window', predicted_window)
        })
        
        # Keep window history manageable
        if len(st.session_state.window_history) > 100:
            st.session_state.window_history = st.session_state.window_history[-100:]
        
        # Extract window from data
        window_df = df.tail(predicted_window)
        
        # Sensor Agent Processing
        local_alerts = []
        for i, (sensor_id, agent) in enumerate(sensor_agents.items()):
            latest_val = readings[sensor_id]
            is_anomaly = agent.detect(latest_val)
            if is_anomaly:
                local_alerts.append(sensor_id)
            
            # Update UI metric
            with sensor_metric_containers[i]:
                delta_color = "inverse" if is_anomaly else "normal"
                
                # Show ground truth comparison if available
                if 'true_anomaly' in packet:
                    ground_truth = packet['true_anomaly']
                    if ground_truth:
                        delta_text = "TRUE ⚠️" if is_anomaly else "MISSED ❌"
                    else:
                        delta_text = "FALSE ⚠️" if is_anomaly else "CORRECT ✅"
                else:
                    delta_text = "ALERT" if is_anomaly else "Normal"
                
                st.metric(
                    label=f"🔧 {sensor_id.title()}",
                    value=f"{latest_val:.2f}",
                    delta=delta_text,
                    delta_color=delta_color
                )
        
        # Update window size metric with MLP vs Ground Truth comparison
        with window_metric_container:
            if 'true_window' in packet:
                true_window = packet['true_window']
                window_error = abs(predicted_window - true_window)
                delta_text = f"GT:{true_window} (±{window_error})"
                delta_color = "normal" if window_error <= 3 else "inverse"
            else:
                delta_text = f"±{predicted_window - 20}" if len(st.session_state.window_history) > 1 else None
                delta_color = "normal"
            
            st.metric(
                label="🎯 MLP Window",
                value=f"{predicted_window}",
                delta=delta_text,
                delta_color=delta_color
            )
        
        # Global Anomaly Detection
        global_anomaly = global_agent.detect(window_df)
        
        # Master Agent Decision
        final_decision, reasoning_text = master_agent.decide(local_alerts, global_anomaly)
        
        # Store anomaly decision with ground truth comparison
        anomaly_record = {
            'timestamp': timestamp,
            'anomaly': final_decision,
            'local_alerts': len(local_alerts),
            'global_anomaly': global_anomaly,
            'sample_id': packet.get('sample_id', len(st.session_state.anomaly_history))
        }
        
        # Add ground truth comparison if available
        if 'true_anomaly' in packet:
            true_anomaly = packet['true_anomaly']
            anomaly_record['true_anomaly'] = true_anomaly
            anomaly_record['correct_prediction'] = (final_decision == true_anomaly)
        
        st.session_state.anomaly_history.append(anomaly_record)
        
        # Update UI Components
        
        # Alert banner with ground truth comparison
        with alert_container:
            if 'true_anomaly' in packet:
                true_anomaly = packet['true_anomaly']
                if final_decision and true_anomaly:
                    st.success(f"✅ **CORRECT DETECTION** - {reasoning_text}")
                elif final_decision and not true_anomaly:
                    st.warning(f"⚠️ **FALSE POSITIVE** - {reasoning_text}")
                elif not final_decision and true_anomaly:
                    st.error(f"❌ **MISSED ANOMALY** - {reasoning_text}")
                else:
                    st.success(f"✅ **CORRECT NORMAL** - {reasoning_text}")
            else:
                if final_decision:
                    st.error(f"🚨 **ANOMALY DETECTED** - {reasoning_text}")
                else:
                    st.success(f"✅ **SYSTEM NORMAL** - {reasoning_text}")
        
        # Main sensor data chart
        with main_chart:
            if len(df) > 1:
                fig = px.line(
                    df.reset_index(), 
                    x='timestamp', 
                    y=[col for col in df.columns if col.startswith('sensor_')],
                    title=f"Predicted: {predicted_window}" + 
                          (f", True: {packet.get('true_window', 'N/A')}" if 'true_window' in packet else ""),
                    labels={'value': 'Sensor Value', 'variable': 'Sensor'}
                )
                
                # Highlight anomalous sensors
                for sensor_id in local_alerts:
                    if sensor_id in df.columns:
                        fig.add_scatter(
                            x=[timestamp], 
                            y=[readings[sensor_id]], 
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='x'),
                            name=f'{sensor_id} ALERT',
                            showlegend=False
                        )
                
                # Add window boundary
                if len(df) >= predicted_window:
                    window_start = df.index[-predicted_window]
                    fig.add_vline(
                        x=window_start, 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text=f"MLP Window Start (Size: {predicted_window})"
                    )
                
                # Add ground truth annotation if available
                if 'true_anomaly' in packet and packet['true_anomaly']:
                    fig.add_annotation(
                        x=timestamp,
                        y=max([readings[col] for col in readings.keys()]),
                        text="TRUE ANOMALY",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="red",
                        bgcolor="red",
                        bordercolor="red",
                        font=dict(color="white")
                    )
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        # Window prediction chart with ground truth comparison
        with window_chart:
            if len(st.session_state.window_history) > 1:
                window_df_chart = pd.DataFrame(st.session_state.window_history)
                
                fig_window = go.Figure()
                
                # MLP predictions
                fig_window.add_trace(go.Scatter(
                    x=window_df_chart['timestamp'], 
                    y=window_df_chart['window_size'],
                    mode='lines+markers',
                    name='MLP Prediction',
                    line=dict(color='blue')
                ))
                
                # Ground truth if available
                if 'true_window' in window_df_chart.columns:
                    fig_window.add_trace(go.Scatter(
                        x=window_df_chart['timestamp'], 
                        y=window_df_chart['true_window'],
                        mode='lines+markers',
                        name='Ground Truth',
                        line=dict(color='red', dash='dash')
                    ))
                
                fig_window.add_hline(
                    y=20, 
                    line_dash="dot", 
                    line_color="gray",
                    annotation_text="Default (20)"
                )
                
                fig_window.update_layout(
                    title="MLP vs Ground Truth",
                    xaxis_title="Time",
                    yaxis_title="Window Size",
                    height=400
                )
                st.plotly_chart(fig_window, use_container_width=True)
        
        # Reasoning display with ground truth comparison
        with reasoning_container:
            # Ground truth comparison if available
            performance_info = ""
            if 'true_anomaly' in packet:
                true_anomaly = packet['true_anomaly']
                if final_decision == true_anomaly:
                    performance_info = f"<p><strong>🎯 Prediction:</strong> <span style='color: green;'>CORRECT</span></p>"
                else:
                    if final_decision and not true_anomaly:
                        performance_info = f"<p><strong>⚠️ Prediction:</strong> <span style='color: orange;'>FALSE POSITIVE</span></p>"
                    else:
                        performance_info = f"<p><strong>❌ Prediction:</strong> <span style='color: red;'>MISSED ANOMALY</span></p>"
            
            window_performance = ""
            if 'true_window' in packet:
                true_window = packet['true_window']
                window_error = abs(predicted_window - true_window)
                accuracy = max(0, 100 - (window_error / true_window * 100))
                window_performance = f"<p><strong>🎯 Window Accuracy:</strong> {accuracy:.1f}% (error: ±{window_error})</p>"
            
            reasoning_html = f"""
            <div style="padding: 10px; border-left: 4px solid {'#ff4444' if final_decision else '#44ff44'}; background-color: {'#ffe6e6' if final_decision else '#e6ffe6'};">
                <h4>🧠 Decision Process</h4>
                <p><strong>Sample ID:</strong> {packet.get('sample_id', 'N/A')}</p>
                <p><strong>MLP Window Prediction:</strong> {predicted_window} samples</p>
                {window_performance}
                <p><strong>Local Alerts:</strong> {len(local_alerts)}/5 sensors ({', '.join(local_alerts) if local_alerts else 'None'})</p>
                <p><strong>Global Pattern:</strong> {'Anomalous' if global_anomaly else 'Normal'}</p>
                <p><strong>Final Decision:</strong> {reasoning_text}</p>
                {performance_info}
                <p><strong>Timestamp:</strong> {timestamp.strftime('%H:%M:%S')}</p>
            </div>
            """
            st.markdown(reasoning_html, unsafe_allow_html=True)
        
        # System logs
        with logs_container:
            recent_logs = master_agent.get_logs()
            if recent_logs:
                log_text = "\n".join([f"• {log}" for log in reversed(recent_logs[-10:])])
                st.text_area("Recent Decisions", value=log_text, height=200, disabled=True)
        
        # Sleep based on speed
        time.sleep(1.0 / speed)
        
        # Auto-refresh the app
        time.sleep(0.1)
        st.rerun()
        
    except StopIteration:
        # Reset stream if it ends
        st.session_state.stream_iterator = simulate_stream(data_dir)
        st.rerun()
    except Exception as e:
        st.error(f"❌ Stream processing error: {e}")
        time.sleep(1)

# Manual Controls
else:
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Stream", use_container_width=True):
                st.session_state.paused = False
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause Stream", use_container_width=True):
                st.session_state.paused = True
        
        with col3:
            if st.button("🔄 Reset Data", use_container_width=True):
                st.session_state.history = []
                st.session_state.window_history = []
                st.session_state.anomaly_history = []
                st.session_state.stream_iterator = simulate_stream(data_dir)
                st.rerun()

# Statistics Dashboard
if st.session_state.history:
    st.markdown("---")
    st.subheader("📊 System Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    
    with stats_col1:
        total_points = len(st.session_state.history)
        st.metric("📈 Data Points", total_points)
    
    with stats_col2:
        if st.session_state.anomaly_history:
            anomaly_rate = sum(1 for a in st.session_state.anomaly_history if a['anomaly']) / len(st.session_state.anomaly_history) * 100
            st.metric("🚨 Anomaly Rate", f"{anomaly_rate:.1f}%")
        else:
            st.metric("🚨 Anomaly Rate", "0.0%")
    
    with stats_col3:
        if st.session_state.window_history:
            avg_window = np.mean([w['window_size'] for w in st.session_state.window_history])
            st.metric("🎯 Avg Window Size", f"{avg_window:.1f}")
        else:
            st.metric("🎯 Avg Window Size", "20.0")
    
    with stats_col4:
        master_stats = master_agent.get_statistics()
        st.metric("🤖 Decisions Made", master_stats['total_decisions'])
    
    with stats_col5:
        # Ground truth performance metrics
        if st.session_state.anomaly_history:
            records_with_gt = [r for r in st.session_state.anomaly_history if 'correct_prediction' in r]
            if records_with_gt:
                accuracy = np.mean([r['correct_prediction'] for r in records_with_gt]) * 100
                st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
            else:
                st.metric("🎯 Accuracy", "N/A")
        else:
            st.metric("🎯 Accuracy", "N/A")

# Performance Metrics (Expandable)
if st.session_state.anomaly_history:
    records_with_gt = [r for r in st.session_state.anomaly_history if 'correct_prediction' in r]
    if records_with_gt:
        with st.expander("📊 Detailed Performance Metrics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            true_positives = sum(1 for r in records_with_gt if r['anomaly'] and r['true_anomaly'])
            false_positives = sum(1 for r in records_with_gt if r['anomaly'] and not r['true_anomaly'])
            true_negatives = sum(1 for r in records_with_gt if not r['anomaly'] and not r['true_anomaly'])
            false_negatives = sum(1 for r in records_with_gt if not r['anomaly'] and r['true_anomaly'])
            
            with col1:
                st.metric("✅ True Positives", true_positives)
            with col2:
                st.metric("⚠️ False Positives", false_positives)
            with col3:
                st.metric("✅ True Negatives", true_negatives)
            with col4:
                st.metric("❌ False Negatives", false_negatives)
            
            # Calculate derived metrics
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
                st.write(f"**Recall (Sensitivity):** {recall:.3f}")
            
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
                st.write(f"**Precision:** {precision:.3f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>🛰️ Agentic IoT Anomaly Detection System</strong></p>
    <p>Powered by Enhanced Feature Engineering MLP • Real-time Labeled Subsequence Analysis</p>
</div>
""", unsafe_allow_html=True)
