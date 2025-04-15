import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

# Configure port for Render
PORT = int(os.environ.get("PORT", 10000))

# Page config with improved layout
st.set_page_config(
    page_title="CKD Model Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding: 2rem;}
    .metric-box {border-radius: 10px; padding: 1rem; background: #f0f2f6; margin-bottom: 1rem;}
    .header {color: #1e3a8a;}
    .stDataFrame {width: 100% !important;}
</style>
""", unsafe_allow_html=True)

# Title with colored header
st.markdown("<h1 class='header'>Chronic Kidney Disease Model Performance Dashboard</h1>", unsafe_allow_html=True)

# ========== DATA PREPARATION ==========
@st.cache_data
def load_data():
    # Model Performance Dataset
    model_performance = pd.DataFrame({
        "Model_Type": ["Traditional", "LLM", "Hybrid"],
        "Accuracy": [0.978995799, 0.227645529, None],
        "AUC_Score": [0.999769112, 0.527665394, 0.999767319],
        "Precision_Avg": [0.98, 0.23, None],
        "Recall_Avg": [0.98, 0.23, None],
        "F1_Score_Avg": [0.98, 0.21, None]
    })

    # Classification Reports
    traditional_class_report = pd.DataFrame({
        "Class": [0, 1, 2, 3, 4],
        "Precision": [0.98, 0.98, 0.98, 0.98, 0.99],
        "Recall": [0.98, 0.98, 0.97, 0.99, 0.97],
        "F1_Score": [0.98, 0.98, 0.98, 0.98, 0.98],
        "Support": [1029, 967, 984, 1005, 1014]
    })

    llm_class_report = pd.DataFrame({
        "Class": [0, 1, 2, 3, 4],
        "Precision": [0.23, 0.24, 0.22, 0.22, 0.23],
        "Recall": [0.33, 0.11, 0.26, 0.10, 0.33],
        "F1_Score": [0.27, 0.15, 0.24, 0.14, 0.27],
        "Support": [1029, 967, 984, 1005, 1014]
    })

    # Confusion Matrices (unpivoted for better visualization)
    def unpivot_confusion(df):
        return df.melt(id_vars="Actual", 
                      var_name="Predicted", 
                      value_name="Count").replace({
                          'Predicted_0': 0,
                          'Predicted_1': 1,
                          'Predicted_2': 2,
                          'Predicted_3': 3,
                          'Predicted_4': 4
                      })

    traditional_confusion = pd.DataFrame({
        "Actual": [0, 1, 2, 3, 4],
        "Predicted_0": [1005, 3, 8, 5, 9],
        "Predicted_1": [8, 951, 6, 2, 4],
        "Predicted_2": [6, 4, 958, 4, 7],
        "Predicted_3": [4, 7, 7, 993, 7],
        "Predicted_4": [6, 2, 5, 1, 987]
    })

    llm_confusion = pd.DataFrame({
        "Actual": [0, 1, 2, 3, 4],
        "Predicted_0": [338, 285, 279, 314, 279],
        "Predicted_1": [87, 102, 83, 67, 82],
        "Predicted_2": [221, 230, 259, 231, 223],
        "Predicted_3": [95, 84, 89, 100, 91],
        "Predicted_4": [288, 266, 274, 293, 339]
    })

    return {
        "model_perf": model_performance,
        "trad_report": traditional_class_report,
        "llm_report": llm_class_report,
        "trad_confusion": unpivot_confusion(traditional_confusion),
        "llm_confusion": unpivot_confusion(llm_confusion)
    }

data = load_data()

# ========== DASHBOARD LAYOUT ==========
# Top Metrics Row
st.markdown("## Overall Model Performance")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("Traditional Model Accuracy", f"{data['model_perf'].iloc[0]['Accuracy']*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("LLM Model Accuracy", f"{data['model_perf'].iloc[1]['Accuracy']*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("Hybrid Model AUC", f"{data['model_perf'].iloc[2]['AUC_Score']:.6f}")
    st.markdown("</div>", unsafe_allow_html=True)

# Performance Charts
st.markdown("---")
st.markdown("## Performance Comparison")

fig1, ax1 = plt.subplots(figsize=(10, 5))
data['model_perf'].dropna().set_index("Model_Type")[["Accuracy", "AUC_Score"]].plot.bar(ax=ax1, rot=0)
ax1.set_title("Model Accuracy vs AUC Score", pad=20)
ax1.set_ylabel("Score")
ax1.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig1)

# Classification Reports
st.markdown("---")
st.markdown("## Detailed Classification Metrics")

tab1, tab2 = st.tabs(["Traditional Model", "LLM Model"])

with tab1:
    st.dataframe(data['trad_report'].style.background_gradient(cmap='Blues'))
    
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    data['trad_report'].plot(x="Class", y=["Precision", "Recall", "F1_Score"], 
                           kind="bar", ax=ax2, title="Precision/Recall/F1 by Class")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    sns.heatmap(pd.crosstab(index=data['trad_confusion']['Actual'], 
                          columns=data['trad_confusion']['Predicted'],
                          values=data['trad_confusion']['Count'],
                          aggfunc='sum').fillna(0),
               annot=True, fmt='g', cmap='Blues', ax=ax3)
    ax3.set_title("Traditional Model Confusion Matrix")
    st.pyplot(fig2)

with tab2:
    st.dataframe(data['llm_report'].style.background_gradient(cmap='Oranges'))
    
    fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 5))
    data['llm_report'].plot(x="Class", y=["Precision", "Recall", "F1_Score"], 
                           kind="bar", ax=ax4, title="Precision/Recall/F1 by Class")
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    sns.heatmap(pd.crosstab(index=data['llm_confusion']['Actual'], 
                          columns=data['llm_confusion']['Predicted'],
                          values=data['llm_confusion']['Count'],
                          aggfunc='sum').fillna(0),
               annot=True, fmt='g', cmap='Oranges', ax=ax5)
    ax5.set_title("LLM Model Confusion Matrix")
    st.pyplot(fig3)

# ========== RENDER DEPLOYMENT FIX ==========
# This ensures Streamlit uses the correct port
if __name__ == "__main__":
    st._config.set_option("server.port", PORT)
