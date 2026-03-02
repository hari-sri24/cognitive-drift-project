import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime
import time

# --------------------------
# CONFIGURATION
# --------------------------
REFRESH_RATE = 4

st.set_page_config(
    page_title="Cognitive Drift Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# DARK THEME CSS
# --------------------------
st.markdown("""
<style>
.stApp {background-color: #0E1117; color: #FFFFFF;}
section[data-testid="stSidebar"] {background-color: #000000;}
section[data-testid="stSidebar"] * {color: #FFFFFF;}
header[data-testid="stHeader"], div[data-testid="stToolbar"] {background-color: #000000;}
h1,h2,h3,h4,h5,h6,p,li,span,div {color: #FFFFFF;}
[data-testid="stMetricValue"] {color: #FFFFFF !important; font-size:28px !important;}
[data-testid="stMetricLabel"] {color: #AAAAAA !important;}
.stDataFrame {background-color: #1E1E1E !important; color:#FFFFFF !important;}
.streamlit-expanderHeader {background-color:#1E1E1E !important;color:#FFFFFF !important;}
.metric-card {background-color: #1E1E1E; padding:1rem; border-radius:10px; border:1px solid #333333;}
.js-plotly-plot, .stPlot {background-color: #1E1E1E !important;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# SESSION STATE
# --------------------------
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame(columns=[
        'timestamp', 'cognitive_score', 'reaction_time', 'memory_score', 'drift_status', 'p_value', 'Age', 'Gender', 'Stress_Level'
    ])
if 'batch_samples' not in st.session_state:
    st.session_state.batch_samples = []
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# --------------------------
# HEADER
# --------------------------
st.markdown('<h1 style="text-align:center;">🧠 Cognitive Drift Detection System</h1>', unsafe_allow_html=True)
st.markdown("### Real-time Analysis of Human Cognitive Performance")
st.markdown("---")

# --------------------------
# SIDEBAR
# --------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    refresh_rate = st.slider("Refresh Rate (seconds)", 1, 30, REFRESH_RATE)
    
    st.markdown("### 📊 Monitor Metric")
    monitored_metric = st.selectbox("Select Metric to Monitor", ["Cognitive Score", "Reaction Time", "Memory Test Score"])
    
    st.markdown("### 📈 Chart Settings")
    show_timeline = st.checkbox("Drift Timeline", value=True)
    show_pvalue_trend = st.checkbox("p-value Trend", value=True)
    show_distribution = st.checkbox("Distribution Analysis", value=True)
    show_correlations = st.checkbox("Feature Correlations", value=True)
    show_speedometer = st.checkbox("Drift Speedometer", value=True)
    show_anomaly = st.checkbox("Anomaly Detection", value=True)
    show_demographics = st.checkbox("Demographic Analysis", value=True)

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

data_file = "human_cognitive_performance.csv"  # replace with your CSV path
df = load_data(data_file)

# --------------------------
# SELECT METRIC VALUES
# --------------------------
if monitored_metric == "Cognitive Score":
    values = df['Cognitive_Score'].tolist()
    y_label = "Cognitive Score"
elif monitored_metric == "Reaction Time":
    values = df['Reaction_Time'].tolist()
    y_label = "Reaction Time"
else:
    values = df['Memory_Test_Score'].tolist()
    y_label = "Memory Test Score"

# --------------------------
# UPDATE HISTORICAL DATA
# --------------------------
timestamp = datetime.now()
new_rows = df.copy()
new_rows['timestamp'] = timestamp
new_rows['drift_status'] = new_rows['Cognitive_Score'] > 70
new_rows['p_value'] = new_rows['Cognitive_Score'].apply(lambda x: 0.01 if x > 70 else 0.5)

st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_rows], ignore_index=True)
st.session_state.batch_samples = new_rows.to_dict(orient='records')
st.session_state.counter += 1

# --------------------------
# METRICS DISPLAY
# --------------------------
metrics_placeholder = st.empty()
charts_placeholder = st.empty()
stats_placeholder = st.empty()

with metrics_placeholder.container():
    st.markdown("## 📊 Current Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        drift_detected = np.mean(values) > 70
        st.markdown(
            '<div class="drift-box drift-detected">⚠️ DRIFT DETECTED</div>' if drift_detected else
            '<div class="drift-box drift-stable">✅ STABLE</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        if len(values) > 1:
            t_stat, p_value = stats.ttest_1samp(values, 70)
        else:
            p_value = 1.0
        st.metric("p-value", f"{p_value:.4f}", delta="Significant" if p_value < 0.05 else "Not Significant")
    
    with col3:
        st.metric(f"Avg {y_label}", f"{np.mean(values):.1f}")
    
    with col4:
        st.metric("Batch Size", len(values))
    
    with col5:
        if len(st.session_state.historical_data) > 0:
            overall_drift_rate = st.session_state.historical_data['drift_status'].tail(50).mean()
            st.metric("Drift Rate (50 batches)", f"{overall_drift_rate:.1%}")

# --------------------------
# VISUALIZATION TABS
# --------------------------
with charts_placeholder.container():
    st.markdown("## 📈 Visual Analytics")
    
    current_df = pd.DataFrame({y_label: values, 'Index': range(len(values))})
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Time Series", "📉 Statistical", "👥 Demographics", "🎯 Advanced"])
    
    # ---------- TIME SERIES ----------
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if show_timeline and len(st.session_state.historical_data) > 1:
                st.subheader("Drift Timeline")
                drift_history = st.session_state.historical_data[['timestamp','drift_status']].copy()
                drift_history['drift_value'] = drift_history['drift_status'].astype(int)
                st.line_chart(drift_history.set_index('timestamp')['drift_value'])
        with col2:
            if show_pvalue_trend and len(st.session_state.historical_data) > 1:
                st.subheader("p-value Trend")
                pvalue_df = st.session_state.historical_data[['timestamp','p_value']].copy()
                st.line_chart(pvalue_df.set_index('timestamp'))
                st.caption("p-value < 0.05 indicates significant drift")
        st.subheader(f"{y_label} Over Time")
        st.line_chart(current_df.set_index('Index')[y_label])
    
    # ---------- STATISTICAL ----------
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if show_distribution:
                st.subheader(f"{y_label} Distribution")
                fig, ax = plt.subplots()
                ax.hist(values, bins=15, edgecolor='white', alpha=0.7, color='skyblue')
                ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.1f}')
                ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.1f}')
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
                corr_cols = ['Cognitive_Score','Reaction_Time','Memory_Test_Score','Age','Stress_Level']
                corr_matrix = recent[corr_cols].corr()
                fig, ax = plt.subplots()
                im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr_cols)))
                ax.set_yticks(range(len(corr_cols)))
                ax.set_xticklabels(corr_cols, rotation=45, ha='right', color='white')
                ax.set_yticklabels(corr_cols, color='white')
                for i in range(len(corr_cols)):
                    for j in range(len(corr_cols)):
                        ax.text(j,i,f'{corr_matrix.iloc[i,j]:.2f}',ha='center',va='center',
                                color='white' if abs(corr_matrix.iloc[i,j])<0.5 else 'black')
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

            
st.success("✅ Dashboard loaded successfully!")



