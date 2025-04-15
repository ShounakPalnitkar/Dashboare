import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(page_title="CKD Dashboard", layout="wide")

st.title("CKD Model Performance Dashboard")

# Model Performance Dataset
model_performance = pd.DataFrame({
    "Model_Type": ["Traditional", "LLM", "Hybrid"],
    "Accuracy": [0.978995799, 0.227645529, None],
    "AUC_Score": [0.999769112, 0.527665394, 0.999767319],
    "Precision_Avg": [0.98, 0.23, None],
    "Recall_Avg": [0.98, 0.23, None],
    "F1_Score_Avg": [0.98, 0.21, None]
})

# Classification Report Dataset
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

# Confusion Matrices
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

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance Metrics")
    st.dataframe(model_performance)
    fig, ax = plt.subplots(figsize=(8, 4))
    model_performance.dropna().set_index("Model_Type")[["Accuracy", "AUC_Score", "F1_Score_Avg"]].plot.bar(ax=ax)
    plt.title("Performance Comparison")
    plt.ylabel("Score")
    st.pyplot(fig)

with col2:
    st.subheader("Classification Reports")
    st.markdown("**Traditional Model**")
    st.dataframe(traditional_class_report)
    st.markdown("**LLM Model**")
    st.dataframe(llm_class_report)

st.subheader("Confusion Matrices (Heatmaps)")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Traditional Model**")
    fig, ax = plt.subplots()
    sns.heatmap(traditional_confusion.drop("Actual", axis=1), annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

with c2:
    st.markdown("**LLM Model**")
    fig, ax = plt.subplots()
    sns.heatmap(llm_confusion.drop("Actual", axis=1), annot=True, fmt="d", cmap="YlOrRd", cbar=False)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

# Ensure proper port usage for Render
os.environ["PORT"] = os.environ.get("PORT", "10000")
