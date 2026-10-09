"""
Streamlit Web Application for Titanic Survival Prediction.
Features:
- Interactive Single Passenger Survival Predictor with Live Probability
- Model Performance Diagnostics, ROC-AUC, and Confusion Matrix
- Exploratory Data Analysis & Historical Dataset Explorer
- Batch Prediction & CSV Export
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Ensure the repository root is on Python's import path.
# Expected layout: project_root/app/app.py and project_root/src/*.py
import sys

_APP_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_APP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_PROJECT_ROOT))

from src.data_loader import PROJECT_ROOT, load_raw_data, load_processed_data
from src.predict import TitanicPredictor
from src.evaluate import plot_confusion_matrix, plot_roc_curve, plot_feature_importance

# Set page config
st.set_page_config(
    page_title="Titanic ML - Survival Prediction System",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .survival-badge-survived {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }
    .survival-badge-perished {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3);
    }
    .factor-tag {
        display: inline-block;
        background-color: #F3F4F6;
        color: #1F2937;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 0.2rem;
        border-left: 3px solid #3B82F6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    return TitanicPredictor()


@st.cache_data
def get_metadata():
    metrics_path = PROJECT_ROOT / "models" / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_historical_data():
    return load_raw_data()


predictor = get_predictor()
metadata = get_metadata()
raw_df = get_historical_data()

# Header
st.markdown('<div class="main-title">🚢 Titanic Survival Prediction & Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">End-to-End Machine Learning System with Feature Engineering, Hyperparameter Optimization, and Diagnostic Analytics.</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Passenger Predictor", "📊 Model Performance", "🔍 Dataset & EDA", "📁 Batch Prediction"]
)

# ================= TAB 1: PASSENGER PREDICTOR =================
with tab1:
    st.subheader("Interactive Passenger Survival Predictor")
    st.write("Configure passenger profile attributes below to evaluate simulated survival probability.")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        with st.container():
            st.markdown("##### 👤 Passenger Identity & Ticket Class")
            c_name, c_pclass = st.columns([2, 1])
            name = c_name.text_input("Passenger Full Name", value="Astor, Mrs. John Jacob (Madeleine Talmadge Force)")
            pclass = c_pclass.selectbox("Ticket Class (Pclass)", options=[1, 2, 3], index=0, format_func=lambda x: f"Class {x} ({'1st Upper' if x==1 else '2nd Middle' if x==2 else '3rd Lower'})")

            c_sex, c_age = st.columns(2)
            sex = c_sex.selectbox("Sex", options=["female", "male"], index=0, format_func=lambda x: x.capitalize())
            age = c_age.slider("Age (years)", min_value=0.5, max_value=80.0, value=25.0, step=0.5)

            st.markdown("##### 👨‍👩‍👧 Family & Travel Companions")
            c_sibsp, c_parch = st.columns(2)
            sibsp = c_sibsp.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=1)
            parch = c_parch.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=6, value=0)

            st.markdown("##### 🎫 Fare, Cabin & Embarkation")
            c_fare, c_embarked, c_cabin = st.columns([1, 1, 1])
            fare = c_fare.number_input("Fare Paid (£)", min_value=0.0, max_value=600.0, value=55.0, step=5.0)
            embarked = c_embarked.selectbox(
                "Port of Embarkation",
                options=["S", "C", "Q"],
                index=1,
                format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x],
            )
            cabin = c_cabin.text_input("Cabin Number (optional)", value="C62")

    passenger_dict = {
        "Pclass": int(pclass),
        "Name": str(name),
        "Sex": str(sex),
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Ticket": "LINE 101",
        "Fare": float(fare),
        "Cabin": str(cabin) if cabin.strip() else "Unknown",
        "Embarked": str(embarked),
    }

    result = predictor.predict_detailed(passenger_dict)
    prob = result["survival_probability"]
    survived = result["prediction"] == 1

    with col2:
        st.markdown("##### 🎯 Prediction Outcome")
        if survived:
            st.markdown(
                f'<div class="survival-badge-survived">✅ PREDICTION: SURVIVED<br><span style="font-size: 0.95rem; font-weight: normal;">Confidence: {result["confidence_percentage"]}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="survival-badge-perished">❌ PREDICTION: DID NOT SURVIVE<br><span style="font-size: 0.95rem; font-weight: normal;">Survival Probability: {result["confidence_percentage"]}</span></div>',
                unsafe_allow_html=True,
            )

        st.write("**Survival Probability Meter:**")
        st.progress(prob)

        st.markdown("##### 📌 Key Contributing Profile Factors")
        for factor in result["risk_and_protective_factors"]:
            st.markdown(f'<div class="factor-tag">🔹 {factor}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 🧬 Computed Derived Features")
        family_size = sibsp + parch + 1
        st.write(f"- **Family Size:** {family_size} ({'Alone' if family_size==1 else 'Small Family' if family_size<=4 else 'Large Family'})")
        st.write(f"- **Fare Per Person:** £{round(fare / family_size, 2)}")
        st.write(f"- **Deck Assigned:** {str(cabin).strip()[0].upper() if cabin.strip() and cabin.strip().lower()!='unknown' else 'Unknown / None'}")

# ================= TAB 2: MODEL PERFORMANCE =================
with tab2:
    st.subheader("Model Benchmarking & Performance Diagnostics")
    
    if metadata:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Selected Model", metadata.get("model_type", "Random Forest"))
        col_m2.metric("Cross-Validation Accuracy", f"{metadata['tuning_info']['best_cv_accuracy']*100:.2f}%")
        col_m3.metric("Training Accuracy", f"{metadata['metrics']['accuracy']*100:.2f}%")
        col_m4.metric("ROC-AUC Score", f"{metadata['metrics']['roc_auc']:.4f}")

        st.markdown("---")
        
        # Benchmark comparison table
        st.markdown("##### 🏆 Model Benchmark Comparison (5-Fold Stratified CV)")
        benchmarks = metadata.get("benchmark_results", {})
        if benchmarks:
            bench_df = pd.DataFrame(benchmarks).T.reset_index()
            bench_df.columns = ["Model Architecture", "CV Accuracy (Mean)", "CV Accuracy (Std)", "CV ROC-AUC (Mean)", "CV ROC-AUC (Std)"]
            bench_df["CV Accuracy (Mean)"] = bench_df["CV Accuracy (Mean)"].apply(lambda x: f"{x*100:.2f}%")
            bench_df["CV ROC-AUC (Mean)"] = bench_df["CV ROC-AUC (Mean)"].apply(lambda x: f"{x:.4f}")
            st.dataframe(bench_df.sort_values(by="CV Accuracy (Mean)", ascending=False), use_container_width=True)

        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.markdown("##### 🎯 Confusion Matrix")
            cm = np.array(metadata["metrics"]["confusion_matrix"])
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Perished (0)", "Survived (1)"], yticklabels=["Perished (0)", "Survived (1)"], ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        with col_diag2:
            st.markdown("##### 🌲 Top Feature Importances")
            feat_names = metadata.get("feature_names", [])
            importances = np.array(metadata.get("feature_importances", []))
            if len(feat_names) > 0 and len(importances) > 0:
                fig_imp = plot_feature_importance(feat_names, importances, top_n=10)
                st.pyplot(fig_imp)
    else:
        st.info("Train the model via `python -m src.train` to view saved evaluation metrics.")

# ================= TAB 3: DATASET & EDA =================
with tab3:
    st.subheader("Historical Dataset & Exploratory Data Analysis")

    st.write(f"Total historical passengers loaded: **{len(raw_df)}**")

    # Filters
    c_f1, c_f2, c_f3 = st.columns(3)
    pclass_filter = c_f1.multiselect("Filter by Class", options=[1, 2, 3], default=[1, 2, 3])
    sex_filter = c_f2.multiselect("Filter by Sex", options=["male", "female"], default=["male", "female"])
    surv_filter = c_f3.multiselect("Filter by Survival", options=[0, 1], default=[0, 1], format_func=lambda x: "Survived (1)" if x==1 else "Perished (0)")

    filtered_df = raw_df[
        (raw_df["Pclass"].isin(pclass_filter))
        & (raw_df["Sex"].isin(sex_filter))
        & (raw_df["Survived"].isin(surv_filter))
    ]

    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📈 Historical Survival Rate Trends")
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        fig_s1, ax_s1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=raw_df, x="Sex", y="Survived", palette="Blues_r", ax=ax_s1)
        ax_s1.set_title("Survival Rate by Sex", fontsize=12, fontweight="bold")
        ax_s1.set_ylabel("Survival Rate")
        st.pyplot(fig_s1)

    with c_chart2:
        fig_s2, ax_s2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=raw_df, x="Pclass", y="Survived", hue="Sex", palette="Blues", ax=ax_s2)
        ax_s2.set_title("Survival Rate by Ticket Class and Sex", fontsize=12, fontweight="bold")
        ax_s2.set_ylabel("Survival Rate")
        st.pyplot(fig_s2)

# ================= TAB 4: BATCH PREDICTION =================
with tab4:
    st.subheader("Batch Passenger Inference & CSV Export")
    st.write("Upload a CSV file with passenger records to generate batch survival predictions.")

    sample_csv = raw_df.drop(columns=["Survived"]).head(10).to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Batch CSV Template",
        data=sample_csv,
        file_name="titanic_batch_sample.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_input_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded **{len(batch_input_df)}** passenger records.")

        if st.button("🚀 Run Batch Predictions"):
            with st.spinner("Processing inferences..."):
                detailed_preds = predictor.predict_detailed(batch_input_df)
                if isinstance(detailed_preds, dict):
                    detailed_preds = [detailed_preds]

                res_df = batch_input_df.copy()
                res_df["Predicted_Survival"] = [p["prediction"] for p in detailed_preds]
                res_df["Predicted_Status"] = [p["status"] for p in detailed_preds]
                res_df["Survival_Probability"] = [p["survival_probability"] for p in detailed_preds]

                st.success(f"Predictions completed for {len(res_df)} records!")
                st.dataframe(res_df, use_container_width=True)

                out_csv = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=out_csv,
                    file_name="titanic_predictions.csv",
                    mime="text/csv",
                )
