import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Dashboard Title ---
st.set_page_config(page_title="CKD Dashboard", layout="wide")
st.title("🩺 CKD Model Performance Dashboard")

# --- 1. Model Performance Table & Chart ---
st.header("📊 Model Performance Metrics")

performance_data = {
    "Model_Type": ["Traditional", "LLM", "Hybrid"],
    "Accuracy": [0.978995799, 0.227645529, None],
    "AUC_Score": [0.999769112, 0.527665394, 0.999767319],
    "Precision_Avg": [0.98, 0.23, None],
    "Recall_Avg": [0.98, 0.23, None],
    "F1_Score_Avg": [0.98, 0.21, None]
}
df_perf = pd.DataFrame(performance_data)
st.dataframe(df_perf)

# Plotting Accuracy & AUC
st.subheader("📈 Accuracy and AUC Comparison")
fig, ax = plt.subplots()
df_perf.dropna().plot(kind='bar', x="Model_Type", y=["Accuracy", "AUC_Score"], ax=ax, color=["#1f77b4", "#ff7f0e"])
plt.ylabel("Score")
plt.title("Accuracy vs AUC Score")
st.pyplot(fig)

# --- 2. Classification Reports ---
st.header("📋 Classification Reports by Class")

trad_classification = pd.DataFrame({
    "Class": [0, 1, 2, 3, 4],
    "Precision": [0.98, 0.98, 0.98, 0.98, 0.99],
    "Recall": [0.98, 0.98, 0.97, 0.99, 0.97],
    "F1_Score": [0.98, 0.98, 0.98, 0.98, 0.98],
    "Support": [1029, 967, 984, 1005, 1014]
})

llm_classification = pd.DataFrame({
    "Class": [0, 1, 2, 3, 4],
    "Precision": [0.23, 0.24, 0.22, 0.22, 0.23],
    "Recall": [0.33, 0.11, 0.26, 0.10, 0.33],
    "F1_Score": [0.27, 0.15, 0.24, 0.14, 0.27],
    "Support": [1029, 967, 984, 1005, 1014]
})

col1, col2 = st.columns(2)
with col1:
    st.subheader("🧠 Traditional Model")
    st.dataframe(trad_classification)
with col2:
    st.subheader("🤖 LLM Model")
    st.dataframe(llm_classification)

# --- 3. Confusion Matrices ---
st.header("🔀 Confusion Matrices")

conf_trad = pd.DataFrame([
    [1005, 8, 6, 4, 6],
    [3, 951, 4, 7, 2],
    [8, 6, 958, 7, 5],
    [5, 2, 4, 993, 1],
    [9, 4, 7, 7, 987]
], columns=[0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4])

conf_llm = pd.DataFrame([
    [338, 87, 221, 95, 288],
    [285, 102, 230, 84, 266],
    [279, 83, 259, 89, 274],
    [314, 67, 231, 100, 293],
    [279, 82, 223, 91, 339]
], columns=[0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Traditional Model Confusion Matrix")
    fig1, ax1 = plt.subplots()
    sns.heatmap(conf_trad, annot=True, fmt='d', cmap="Blues", ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

with col2:
    st.subheader("LLM Model Confusion Matrix")
    fig2, ax2 = plt.subplots()
    sns.heatmap(conf_llm, annot=True, fmt='d', cmap="Reds", ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)