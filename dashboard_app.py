import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================
REFRESH_RATE = 4  # minutes
API_URL = "http://127.0.0.1:5000/api/data"  # backend API

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Cognitive Drift Detection", page_icon="🧠", layout="wide")

# ================= DARK THEME =================
st.markdown("""
<style>
.stApp {background-color: #0E1117 !important; color: #FFFFFF !important;}
section[data-testid="stSidebar"] {background-color: #000000 !important;}
section[data-testid="stSidebar"] * {color: #FFFFFF !important;}
header[data-testid="stHeader"], div[data-testid="stToolbar"] {background-color: #000000 !important;}
h1,h2,h3,h4,h5,h6,p,li,span,div {color: #FFFFFF !important;}
[data-testid="stMetricValue"] {color: #FFFFFF !important; font-size: 28px !important;}
[data-testid="stMetricLabel"] {color: #AAAAAA !important;}
.drift-box {padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.2rem; margin: 10px 0;}
.drift-detected {background-color: #8B0000 !important; color: white !important; border: 2px solid #FF4444;}
.drift-stable {background-color: #006400 !important; color: white !important; border: 2px solid #44FF44;}
.stTabs [data-baseweb="tab-list"] {background-color: #1E1E1E;}
.stTabs [data-baseweb="tab"] {color: #FFFFFF !important;}
.stDataFrame {background-color: #1E1E1E !important; color: #FFFFFF !important;}
.streamlit-expanderHeader {background-color: #1E1E1E !important; color: #FFFFFF !important;}
.stButton > button {background-color: #1E1E1E !important; color: white !important; border: 1px solid #333333 !important;}
.stButton > button:hover {background-color: #2A2A2A !important;}
.metric-card {background-color: #1E1E1E; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(255,255,255,0.1); border: 1px solid #333333;}
.js-plotly-plot, .stPlot {background-color: #1E1E1E !important;}
.sidebar-section {font-weight: bold; font-size: 1.2rem; margin-top: 20px; border-bottom: 1px solid #333333; padding-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()
if 'batch_samples' not in st.session_state:
    st.session_state.batch_samples = []
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'dataset_stats' not in st.session_state:
    st.session_state.dataset_stats = None

# ================= HEADER =================
st.markdown('<h1 class="main-header">🧠 Cognitive Drift Detection System</h1>', unsafe_allow_html=True)
st.markdown("### Real-time Analysis of Human Cognitive Performance")
st.markdown("---")

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown('<div class="sidebar-section">⚙️ Controls</div>', unsafe_allow_html=True)
    refresh_rate = st.slider("Refresh Rate (minutes)", 1, 10, REFRESH_RATE)
    st.markdown('<div class="sidebar-section">📊 Monitor</div>', unsafe_allow_html=True)
    monitored_metric = st.selectbox("Select Metric to Monitor", ["Cognitive Score", "Reaction Time", "Memory Test Score"])
    st.markdown('<div class="sidebar-section">📈 Chart Settings</div>', unsafe_allow_html=True)
    show_timeline = st.checkbox("Drift Timeline", value=True)
    show_pvalue_trend = st.checkbox("p-value Trend", value=True)
    show_distribution = st.checkbox("Distribution Analysis", value=True)
    show_correlations = st.checkbox("Feature Correlations", value=True)
    show_speedometer = st.checkbox("Drift Speedometer", value=True)
    show_anomaly = st.checkbox("Anomaly Detection", value=True)
    show_demographics = st.checkbox("Demographic Analysis", value=True)

# ================= LOAD CSV =================
if st.session_state.dataset_stats is None:
    df = pd.read_csv("human_cognitive_performance.csv")
    st.session_state.dataset_stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "Cognitive_Score_mean": df[df.columns[0]].mean(),  # first column as numeric
        "Age_mean": df[df.columns[5]].mean()  # adjust based on CSV
    }

# ================= BACKEND CHECK =================
try:
    r = requests.get("http://127.0.0.1:5000/", timeout=2)
    if r.status_code == 200:
        st.sidebar.success("✅ Backend Connected")
    else:
        st.sidebar.error("❌ Backend Error")
        st.stop()
except:
    st.sidebar.error("❌ Cannot connect to backend! Start backend and refresh.")
    st.stop()

# ================= AUTO REFRESH =================
st_autorefresh(interval=refresh_rate*60*1000, key="datarefresh")

# ================= FETCH DATA =================
def fetch_data():
    try:
        resp = requests.get(API_URL, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None

data = fetch_data()
if data:
    timestamp = datetime.now()
    st.session_state.counter += 1
    st.session_state.batch_samples = data['sample_data']
    
    # convert to DataFrame
    batch_df = pd.DataFrame(st.session_state.batch_samples)
    batch_df['timestamp'] = [timestamp + pd.Timedelta(seconds=i) for i in range(len(batch_df))]
    batch_df['drift_status'] = data['drift']
    batch_df['p_value'] = data['p_value']
    
    # append to historical
    st.session_state.historical_data = pd.concat([st.session_state.historical_data, batch_df], ignore_index=True)
    if len(st.session_state.historical_data) > 1000:
        st.session_state.historical_data = st.session_state.historical_data.tail(1000)
    
    # ================= METRICS =================
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if data['drift']:
            st.markdown('<div class="drift-box drift-detected">⚠️ DRIFT DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="drift-box drift-stable">✅ STABLE</div>', unsafe_allow_html=True)
    with col2:
        st.metric("p-value", f"{data['p_value']:.4f}", delta="Significant" if data['p_value'] < 0.05 else "Not Significant")
    with col3:
        metric_col = {"Cognitive Score": "Cognitive_Score",
                      "Reaction Time": "Reaction_Time",
                      "Memory Test Score": "Memory_Test_Score"}
        st.metric(f"Avg {monitored_metric}", f"{batch_df[metric_col[monitored_metric]].mean():.2f}")
    with col4:
        st.metric("Batch Size", len(batch_df))
    with col5:
        drift_rate = st.session_state.historical_data['drift_status'].tail(50).mean()
        st.metric("Drift Rate (50 batches)", f"{drift_rate:.1%}")

st.markdown("✅ **Dashboard updated successfully!**")
