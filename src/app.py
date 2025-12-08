import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_processing
import stage1_baseline
import stage2_imbalance
import stage3_features
import stage4_benchmark
import os

st.set_page_config(page_title="Credit Risk Analysis", layout="wide")

st.title("Credit Risk Analysis & Improvement Plan")

st.markdown("""
This dashboard visualizes the transformation of the credit risk model from a flawed state (Data Leakage)
to a robust predictive solution.
""")

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Project Status", "Data Analysis", "Model Evolution", "Feature Importance", "Benchmark"])

# --- Helper Functions ---
@st.cache_data
def load_data():
    if not os.path.exists("evaluaciones_2022.csv"):
        st.warning("Dataset not found. Generating synthetic data...")
        import create_synthetic_data
        create_synthetic_data.create_synthetic_data()
    return data_processing.load_and_clean_data()

# --- Pages ---

if page == "Project Status":
    st.header("Current Status & Improvement Plan")
    try:
        with open("STATUS_AND_PLAN.md", "r") as f:
            content = f.read()
        st.markdown(content)
    except FileNotFoundError:
        st.error("STATUS_AND_PLAN.md not found.")

elif page == "Data Analysis":
    st.header("Data Overview")
    df = load_data()

    if df is not None:
        st.write(f"**Shape:** {df.shape}")
        st.write("### First 5 Rows")
        st.dataframe(df.head())

        st.write("### Target Distribution (CD_SUBESTADO_PRODUCTO)")
        target_counts = df['CD_SUBESTADO_PRODUCTO'].value_counts()
        fig, ax = plt.figure(), plt.gca()
        sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write("### Numerical Distributions")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_col = st.selectbox("Select Column", num_cols)
        fig, ax = plt.figure(), plt.gca()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

elif page == "Model Evolution":
    st.header("Model Performance Evolution")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stage 1: Real Baseline")
        st.markdown("*Removed Data Leakage variables (`CD_CAJON_MORA`, etc.)*")
        if st.button("Run Stage 1"):
            with st.spinner("Training Stage 1 Model..."):
                acc1, report1 = stage1_baseline.run_baseline()
                st.metric("Accuracy", f"{acc1:.4f}")
                st.json(report1)
                st.session_state['report1'] = report1

    with col2:
        st.subheader("Stage 2: Addressing Imbalance")
        st.markdown("*Added SMOTE and Class Weight Balancing*")
        if st.button("Run Stage 2"):
            with st.spinner("Training Stage 2 Model..."):
                acc2, report2 = stage2_imbalance.run_imbalance_fix()
                st.metric("Accuracy", f"{acc2:.4f}")
                st.json(report2)
                st.session_state['report2'] = report2

    if 'report1' in st.session_state and 'report2' in st.session_state:
        st.write("### Comparison (Macro Avg F1-Score)")
        f1_1 = st.session_state['report1']['macro avg']['f1-score']
        f1_2 = st.session_state['report2']['macro avg']['f1-score']

        fig, ax = plt.figure(), plt.gca()
        sns.barplot(x=["Stage 1 (Baseline)", "Stage 2 (Balanced)"], y=[f1_1, f1_2], ax=ax)
        ax.set_ylabel("Macro F1 Score")
        st.pyplot(fig)

elif page == "Feature Importance":
    st.header("Stage 3: Feature Engineering & Importance")

    if st.button("Run Stage 3 Analysis"):
        with st.spinner("Running Feature Engineering Pipeline..."):
            feat_imp_df, acc3, report3 = stage3_features.run_feature_engineering()

            st.metric("Model Accuracy", f"{acc3:.4f}")

            st.subheader("Top Predictors")
            st.dataframe(feat_imp_df.head(15))

            fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
            sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(10), ax=ax)
            ax.set_title("Top 10 Feature Importances (Random Forest)")
            st.pyplot(fig)

elif page == "Benchmark":
    st.header("Stage 4: Model Benchmarking")
    st.markdown("Comparing Random Forest, XGBoost, LightGBM, and Logistic Regression.")

    if st.button("Run Benchmark"):
        with st.spinner("Training models... this may take a moment."):
            results_df = stage4_benchmark.run_benchmark()
            st.write("### Benchmark Results")
            st.dataframe(results_df)

            fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
            sns.barplot(x="Model", y="F1-Score (Weighted)", data=results_df, ax=ax)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
