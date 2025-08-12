import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "cancer_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(
    page_title="AI-Based Early Cancer Biomarker Detector",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI-Based Early Cancer Biomarker Detector")
st.write("""
This tool uses **gene expression data** and a trained Machine Learning model  
to predict the likelihood of **cancer presence** based on biomarkers.  
Upload your CSV file or enter values manually to test the model.
""")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload Gene Expression CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    preds = model.predict(data)
    data['Prediction'] = np.where(preds == 1, "Cancer", "Normal")

    st.write("### Predictions")
    st.dataframe(data)

    fig, ax = plt.subplots()
    data['Prediction'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
    plt.title("Prediction Distribution")
    st.pyplot(fig)

# Manual Input
st.write("### Or Enter Gene Expression Values Manually")
gene1 = st.number_input("Gene1 Expression", min_value=0.0, step=0.1)
gene2 = st.number_input("Gene2 Expression", min_value=0.0, step=0.1)
gene3 = st.number_input("Gene3 Expression", min_value=0.0, step=0.1)

if st.button("Predict from Manual Input"):
    input_df = pd.DataFrame([[gene1, gene2, gene3]], columns=['Gene1', 'Gene2', 'Gene3'])
    pred = model.predict(input_df)[0]
    result = "Cancer" if pred == 1 else "Normal"
    st.success(f"Prediction: **{result}**")
