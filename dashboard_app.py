import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as stats
from datetime import datetime
from sklearn.linear_model import LinearRegression
from pandas.plotting import autocorrelation_plot

# Configuration
API_URL = "http://127.0.0.1:5000/api/data"
STATS_URL = "http://127.0.0.1:5000/api/stats"
REFRESH_RATE = 3

# Page config
st.set_page_config(
    page_title="Cognitive Drift Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for DARK THEME
st.markdown("""
    <style>
    /* Main background and text colors */
    .stApp {
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1wrcr25 {
        background-color: #1E1E1E !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Text color */
    p, li, span, div {
        color: #FFFFFF !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 28px !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #AAAAAA !important;
    }
    
    /* Drift status boxes */
    .drift-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    
    .drift-detected {
        background-color: #8B0000 !important;
        color: white !important;
        border: 2px solid #FF4444;
    }
    
    .drift-stable {
        background-color: #006400 !important;
        color: white !important;
        border: 2px solid #44FF44;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E1E1E;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    
    /* Sidebar text */
    .css-17lntkn, .css-1v3fvcr {
        color: #FFFFFF !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #333333;
    }
    
    /* Success/Error messages */
    .st-success, .st-error {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #FFFFFF !important;
    }
    
    /* Checkbox labels */
    .stCheckbox label {
        color: #FFFFFF !important;
    }
    
    /* Main header */
    .main-header {
        font-size: 3rem;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Metric containers */
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
        border: 1px solid #333333;
    }
    
    /* Plot background */
    .js-plotly-plot {
        background-color: #1E1E1E !important;
    }
    
    /* Matplotlib figure background */
    .stPlot {
        background-color: #1E1E1E !important;
    }
    
    /* Sidebar section headers */
    .sidebar-section {
        color: #FFFFFF !important;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 20px;
        border-bottom: 1px solid #333333;
        padding-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame(columns=['timestamp', 'cognitive_score', 'reaction_time', 'memory_score', 'drift_status', 'p_value'])
if 'batch_samples' not in st.session_state:
    st.session_state.batch_samples = []
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'dataset_stats' not in st.session_state:
    st.session_state.dataset_stats = None

# Header
st.markdown('<h1 class="main-header">🧠 Cognitive Drift Detection System</h1>', unsafe_allow_html=True)
st.markdown("### Real-time Analysis of Human Cognitive Performance")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">⚙️ Controls</div>', unsafe_allow_html=True)
    
    # Refresh rate control
    refresh_rate = st.slider("Refresh Rate (seconds)", min_value=1, max_value=10, value=REFRESH_RATE)
    
    # Metric selection
    st.markdown('<div class="sidebar-section">📊 Monitor</div>', unsafe_allow_html=True)
    monitored_metric = st.selectbox(
        "Select Metric to Monitor",
        ["Cognitive Score", "Reaction Time", "Memory Test Score"]
    )
    
    st.markdown('<div class="sidebar-section">📈 Chart Settings</div>', unsafe_allow_html=True)
    
    # Chart selection
    show_timeline = st.checkbox("Drift Timeline", value=True)
    show_pvalue_trend = st.checkbox("p-value Trend", value=True)
    show_distribution = st.checkbox("Distribution Analysis", value=True)
    show_correlations = st.checkbox("Feature Correlations", value=True)
    show_speedometer = st.checkbox("Drift Speedometer", value=True)
    show_anomaly = st.checkbox("Anomaly Detection", value=True)
    show_demographics = st.checkbox("Demographic Analysis", value=True)
    
    st.markdown('<div class="sidebar-section">ℹ️ Dataset Info</div>', unsafe_allow_html=True)
    
    # Fetch dataset stats
    if st.session_state.dataset_stats is None:
        try:
            response = requests.get(STATS_URL, timeout=2)
            if response.status_code == 200:
                st.session_state.dataset_stats = response.json()
        except:
            pass
    
    if st.session_state.dataset_stats:
        stats = st.session_state.dataset_stats
        st.info(f"""
        **Dataset Statistics:**
        - Total Samples: {stats['total_samples']:,}
        - Cognitive Score: {stats['cognitive_score_mean']:.1f} ± {stats['cognitive_score_std']:.1f}
        - Age Range: {stats['age_mean']:.0f} years (avg)
        - Stress Level: {stats['stress_level_mean']:.1f}/10
        """)
    
    # Reset button
    if st.button("🔄 Reset Detector & Start Over", use_container_width=True):
        try:
            requests.post("http://127.0.0.1:5000/api/reset", timeout=2)
            st.success("✅ Detector reset successfully!")
            st.session_state.historical_data = pd.DataFrame(columns=['timestamp', 'cognitive_score', 'reaction_time', 'memory_score', 'drift_status', 'p_value'])
            st.session_state.batch_samples = []
        except:
            st.error("❌ Could not connect to backend")

# Check if backend is running
try:
    response = requests.get("http://127.0.0.1:5000/", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ Backend Connected")
    else:
        st.sidebar.error("❌ Backend Error")
        st.stop()
except:
    st.sidebar.error("""
    ❌ **Cannot connect to backend!**
    
    Please start the Flask backend:
    1. Open a new terminal
    2. Run: `python backend_api.py`
    3. Then refresh this page
    """)
    st.stop()

# Create placeholders for dynamic content
header_placeholder = st.empty()
metrics_placeholder = st.empty()
charts_placeholder = st.empty()
stats_placeholder = st.empty()

def fetch_data():
    """Fetch data from backend"""
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Main loop
while True:
    data = fetch_data()
    
    if data:
        timestamp = datetime.now()
        st.session_state.counter += 1
        
        # Extract values based on selected metric
        if monitored_metric == "Cognitive Score":
            values = data['data']
            y_label = "Cognitive Score"
            y_range = [0, 100]
        elif monitored_metric == "Reaction Time":
            # Extract reaction times from sample data
            values = [s['Reaction_Time'] for s in data['sample_data']]
            y_label = "Reaction Time (ms)"
            y_range = [0, 1000]
        else:  # Memory Test Score
            values = [s['Memory_Test_Score'] for s in data['sample_data']]
            y_label = "Memory Test Score"
            y_range = [0, 100]
        
        # Update historical data
        for i, (value, sample) in enumerate(zip(values, data['sample_data'])):
            new_row = pd.DataFrame({
                'timestamp': [timestamp + pd.Timedelta(seconds=i)],
                'cognitive_score': [sample['Cognitive_Score']],
                'reaction_time': [sample['Reaction_Time']],
                'memory_score': [sample['Memory_Test_Score']],
                'drift_status': [data['drift']],
                'p_value': [data['p_value']],
                'age': [sample['Age']],
                'gender': [sample['Gender']],
                'stress_level': [sample['Stress_Level']]
            })
            
            if len(st.session_state.historical_data) > 0:
                st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_row], ignore_index=True)
            else:
                st.session_state.historical_data = new_row
        
        # Store batch samples for display
        st.session_state.batch_samples = data['sample_data']
        
        # Keep only last 1000 records for performance
        if len(st.session_state.historical_data) > 1000:
            st.session_state.historical_data = st.session_state.historical_data.tail(1000)
        
        # Display status and batch info
        header_placeholder.info(
            f"🕐 Last updated: {timestamp.strftime('%H:%M:%S')} | "
            f"📊 Processing batch {data['batch_info']['start_idx']}-{data['batch_info']['end_idx']} "
            f"of {data['batch_info']['total_rows']:,}"
        )
        
        # Display metrics
        with metrics_placeholder.container():
            st.markdown("## 📊 Current Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Drift status
            with col1:
                if data['drift']:
                    st.markdown('<div class="drift-box drift-detected">⚠️ DRIFT DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="drift-box drift-stable">✅ STABLE</div>', unsafe_allow_html=True)
            
            with col2:
                st.metric("p-value", f"{data['p_value']:.4f}", 
                         delta="Significant" if data['p_value'] < 0.05 else "Not Significant")
            
            with col3:
                st.metric(f"Avg {y_label}", f"{np.mean(values):.1f}")
            
            with col4:
                st.metric("Batch Size", len(values))
            
            with col5:
                if len(st.session_state.historical_data) > 0:
                    overall_drift_rate = st.session_state.historical_data['drift_status'].tail(50).mean()
                    st.metric("Drift Rate (50 batches)", f"{overall_drift_rate:.1%}")
        
        # Create charts container
        with charts_placeholder.container():
            st.markdown("## 📈 Visual Analytics")
            
            # Current data DataFrame
            current_df = pd.DataFrame({
                y_label: values,
                'Index': range(len(values))
            })
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Time Series", "📉 Statistical", "👥 Demographics", "🎯 Advanced"])
            
            with tab1:
                # Time Series Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    if show_timeline:
                        st.subheader("Drift Timeline")
                        if len(st.session_state.historical_data) > 1:
                            drift_history = st.session_state.historical_data[['timestamp', 'drift_status']].copy()
                            drift_history['drift_value'] = drift_history['drift_status'].astype(int)
                            st.line_chart(drift_history.set_index('timestamp')['drift_value'])
                
                with col2:
                    if show_pvalue_trend:
                        st.subheader("p-value Trend")
                        if len(st.session_state.historical_data) > 1:
                            pvalue_df = st.session_state.historical_data[['timestamp', 'p_value']].copy()
                            st.line_chart(pvalue_df.set_index('timestamp'))
                            st.caption("p-value < 0.05 indicates significant drift")
                
                # Main values chart
                st.subheader(f"{y_label} Over Time")
                st.line_chart(current_df.set_index('Index')[y_label])
            
            with tab2:
                # Statistical Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    if show_distribution:
                        st.subheader(f"{y_label} Distribution")
                        fig, ax = plt.subplots()
                        ax.hist(values, bins=15, edgecolor='white', alpha=0.7, color='skyblue')
                        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, 
                                  label=f'Mean: {np.mean(values):.1f}')
                        ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2, 
                                  label=f'Median: {np.median(values):.1f}')
                        ax.set_xlabel(y_label, color='white')
                        ax.set_ylabel('Frequency', color='white')
                        ax.tick_params(colors='white')
                        ax.legend(facecolor='#1E1E1E', labelcolor='white')
                        ax.set_facecolor('#1E1E1E')
                        fig.patch.set_facecolor('#0E1117')
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    if show_correlations and len(st.session_state.historical_data) > 10:
                        st.subheader("Feature Correlations")
                        recent = st.session_state.historical_data.tail(100)
                        corr_cols = ['cognitive_score', 'reaction_time', 'memory_score', 'age', 'stress_level']
                        corr_matrix = recent[corr_cols].corr()
                        
                        fig, ax = plt.subplots()
                        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                        ax.set_xticks(range(len(corr_cols)))
                        ax.set_yticks(range(len(corr_cols)))
                        ax.set_xticklabels(corr_cols, rotation=45, ha='right', color='white')
                        ax.set_yticklabels(corr_cols, color='white')
                        
                        # Add correlation values
                        for i in range(len(corr_cols)):
                            for j in range(len(corr_cols)):
                                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                              ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'black')
                        
                        fig.patch.set_facecolor('#0E1117')
                        ax.set_facecolor('#1E1E1E')
                        plt.colorbar(im, ax=ax)
                        st.pyplot(fig)
                        plt.close()
            
            with tab3:
                # Demographics
                col1, col2 = st.columns(2)
                
                with col1:
                    if show_demographics and len(st.session_state.batch_samples) > 0:
                        st.subheader("Age Distribution")
                        ages = [s['Age'] for s in st.session_state.batch_samples]
                        fig, ax = plt.subplots()
                        ax.hist(ages, bins=10, edgecolor='white', alpha=0.7, color='lightgreen')
                        ax.set_xlabel('Age', color='white')
                        ax.set_ylabel('Count', color='white')
                        ax.tick_params(colors='white')
                        ax.set_facecolor('#1E1E1E')
                        fig.patch.set_facecolor('#0E1117')
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    if show_demographics and len(st.session_state.batch_samples) > 0:
                        st.subheader("Gender Distribution")
                        genders = [s['Gender'] for s in st.session_state.batch_samples]
                        gender_counts = pd.Series(genders).value_counts()
                        
                        fig, ax = plt.subplots()
                        ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                               colors=['#ff9999', '#66b3ff'], textprops={'color': 'white'})
                        fig.patch.set_facecolor('#0E1117')
                        st.pyplot(fig)
                        plt.close()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if show_demographics and len(st.session_state.batch_samples) > 0:
                        st.subheader("Stress Level Distribution")
                        stress = [s['Stress_Level'] for s in st.session_state.batch_samples]
                        fig, ax = plt.subplots()
                        ax.hist(stress, bins=10, edgecolor='white', alpha=0.7, color='salmon')
                        ax.set_xlabel('Stress Level (1-10)', color='white')
                        ax.set_ylabel('Count', color='white')
                        ax.tick_params(colors='white')
                        ax.set_facecolor('#1E1E1E')
                        fig.patch.set_facecolor('#0E1117')
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    if show_demographics and len(st.session_state.batch_samples) > 0:
                        st.subheader("Cognitive Score by Gender")
                        df_gender = pd.DataFrame(st.session_state.batch_samples)
                        gender_scores = df_gender.groupby('Gender')['Cognitive_Score'].mean()
                        
                        fig, ax = plt.subplots()
                        ax.bar(gender_scores.index, gender_scores.values, color=['#ff9999', '#66b3ff'])
                        ax.set_xlabel('Gender', color='white')
                        ax.set_ylabel('Avg Cognitive Score', color='white')
                        ax.tick_params(colors='white')
                        ax.set_facecolor('#1E1E1E')
                        fig.patch.set_facecolor('#0E1117')
                        st.pyplot(fig)
                        plt.close()
            
            with tab4:
                # Advanced Analytics
                col1, col2 = st.columns(2)
                
                with col1:
                    if show_speedometer and len(st.session_state.historical_data) > 0:
                        st.subheader("Drift Risk Meter")
                        recent_drift_rate = st.session_state.historical_data['drift_status'].tail(20).mean() * 100
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=recent_drift_rate,
                            title={'text': "Drift Risk (%)", 'font': {'color': 'white'}},
                            number={'font': {'color': 'white'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                                'bar': {'color': "darkblue"},
                                'bgcolor': '#1E1E1E',
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 95
                                }
                            }
                        ))
                        fig.update_layout(
                            height=300,
                            paper_bgcolor='#0E1117',
                            font={'color': 'white'}
                        )
                        unique_key = f"speedometer_{st.session_state.counter}"
                        st.plotly_chart(fig, use_container_width=True, key=unique_key)
                
                with col2:
                    if show_anomaly and len(st.session_state.historical_data) > 20:
                        st.subheader("Anomaly Detection")
                        df_anomaly = st.session_state.historical_data.tail(50).copy()
                        mean = df_anomaly['cognitive_score'].mean()
                        std = df_anomaly['cognitive_score'].std()
                        df_anomaly['anomaly'] = (abs(df_anomaly['cognitive_score'] - mean) > 2*std)
                        
                        fig, ax = plt.subplots()
                        colors = df_anomaly['anomaly'].map({True: 'red', False: 'skyblue'})
                        ax.scatter(range(len(df_anomaly)), df_anomaly['cognitive_score'], c=colors, alpha=0.6, s=50)
                        ax.axhline(y=mean, color='green', linestyle='--', linewidth=2, label='Mean')
                        ax.axhline(y=mean+2*std, color='orange', linestyle=':', linewidth=2, label='+2σ')
                        ax.axhline(y=mean-2*std, color='orange', linestyle=':', linewidth=2, label='-2σ')
                        ax.set_xlabel('Sample', color='white')
                        ax.set_ylabel('Cognitive Score', color='white')
                        ax.tick_params(colors='white')
                        ax.legend(facecolor='#1E1E1E', labelcolor='white')
                        ax.set_facecolor('#1E1E1E')
                        fig.patch.set_facecolor('#0E1117')
                        st.pyplot(fig)
                        plt.close()
        
        # Statistics and Raw Data
        with stats_placeholder.container():
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Current Batch Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                    'Value': [
                        f"{np.mean(values):.2f}",
                        f"{np.std(values):.2f}",
                        f"{np.min(values):.2f}",
                        f"{np.percentile(values, 25):.2f}",
                        f"{np.median(values):.2f}",
                        f"{np.percentile(values, 75):.2f}",
                        f"{np.max(values):.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("🕒 Recent History")
                if len(st.session_state.historical_data) > 0:
                    recent = st.session_state.historical_data.tail(10)[
                        ['timestamp', 'cognitive_score', 'drift_status', 'p_value']
                    ].copy()
                    recent['timestamp'] = recent['timestamp'].dt.strftime('%H:%M:%S')
                    recent['drift_status'] = recent['drift_status'].map({True: '⚠ Drift', False: '✅ Stable'})
                    recent.columns = ['Time', 'Score', 'Status', 'p-value']
                    recent['Score'] = recent['Score'].round(2)
                    recent['p-value'] = recent['p-value'].round(4)
                    st.dataframe(recent, use_container_width=True, hide_index=True)
            
            # Raw data expander
            with st.expander("🔍 View Current Batch Data"):
                if st.session_state.batch_samples:
                    df_display = pd.DataFrame(st.session_state.batch_samples)
                    st.dataframe(df_display, use_container_width=True)
    
    time.sleep(refresh_rate)






