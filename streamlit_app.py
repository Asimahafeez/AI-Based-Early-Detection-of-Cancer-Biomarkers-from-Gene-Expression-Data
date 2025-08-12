import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model, preprocess_data

st.set_page_config(page_title="AI Cancer Biomarker Detector", layout="wide")

st.title("🧬 AI-Based Early Detection of Cancer Biomarkers")
st.write("Upload your gene expression dataset to detect potential cancer biomarkers.")

model = load_model()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    processed = preprocess_data(df)
    predictions = model.predict(processed)
    probs = model.predict_proba(processed)[:, 1]

    result_df = df.copy()
    result_df["Cancer_Prediction"] = predictions
    result_df["Cancer_Probability"] = probs

    st.subheader("🧾 Prediction Results")
    st.dataframe(result_df)

    fig, ax = plt.subplots()
    sns.histplot(probs, bins=10, kde=True, ax=ax, color="red")
    ax.set_title("Cancer Probability Distribution")
    st.pyplot(fig)
else:
    st.info("Please upload a CSV file to proceed.")

