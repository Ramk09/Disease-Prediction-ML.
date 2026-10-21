
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(
    page_title="AI Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- DASHBOARD STYLING --------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1830 100%);
        color: #e7f2ff;
    }
    [data-testid="stSidebar"] {
        background: #0b1729;
        border-right: 1px solid #173652;
    }
    h1, h2, h3 {
        color: #e9f7ff !important;
    }
    .hero {
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #164e63;
        background: linear-gradient(110deg, #10263c, #0b3141);
        margin-bottom: 20px;
    }
    .hero p {
        color: #9dc6dc;
        margin-bottom: 0;
    }
    .metric-card {
        background: #101f33;
        border: 1px solid #1e4058;
        border-radius: 14px;
        padding: 16px;
        min-height: 105px;
    }
    .metric-label {
        color: #9db7cc;
        font-size: 13px;
    }
    .metric-value {
        color: #5ee7f7;
        font-size: 25px;
        font-weight: 700;
        margin-top: 7px;
    }
    .stButton > button {
        background: #08b9ce;
        color: #06121f;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }
    .stButton > button:hover {
        background: #5ee7f7;
        color: #06121f;
    }
    div[data-testid="stMetric"] {
        background: #101f33;
        border: 1px solid #1e4058;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- CONFIGURATION --------------------

DATASETS = {
    "Breast Cancer": "breast_cancer",
    "Heart Disease": "heart_disease",
    "Diabetes": "diabetes",
}

MODEL_NAMES = {
    "Logistic Regression": "logistic_regression",
    "SVM": "svm",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}

st.sidebar.markdown("## 🩺 Clinical AI")
st.sidebar.caption("Educational decision-support prototype")

disease_label = st.sidebar.selectbox(
    "Select disease dataset",
    list(DATASETS.keys()),
)
dataset_name = DATASETS[disease_label]

model_label = st.sidebar.selectbox(
    "Select prediction model",
    list(MODEL_NAMES.keys()),
)
model_name = MODEL_NAMES[model_label]

st.sidebar.divider()
st.sidebar.caption("Models trained on public datasets.")
st.sidebar.caption("Not intended for clinical diagnosis.")

st.markdown(
    """
    <div class="hero">
        <h1>AI Clinical Decision Support System</h1>
        <p>
            Machine-learning predictions, model performance,
            and transparent feature analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- LOAD DATA AND MODEL --------------------

model_path = MODEL_DIR / f"{dataset_name}_{model_name}.joblib"
data_path = DATA_DIR / f"{dataset_name}.csv"
metrics_path = OUTPUT_DIR / "model_metrics.csv"

if not model_path.exists():
    st.error(f"Model file not found: {model_path.name}")
    st.stop()

if not data_path.exists():
    st.error(f"Dataset file not found: {data_path.name}")
    st.stop()

try:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]
    class_names = bundle["class_names"]

    df = pd.read_csv(data_path)
    X_data = df[feature_columns].copy()

except Exception as error:
    st.error(f"Could not load the selected model or dataset: {error}")
    st.stop()


def default_value(column):
    """Return a representative starting value from the dataset."""
    series = X_data[column]

    if pd.api.types.is_numeric_dtype(series):
        value = series.median()
        return float(value) if pd.notna(value) else 0.0

    modes = series.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else ""


# -------------------- MODEL PERFORMANCE --------------------

st.markdown("## 📊 Model overview")

metric_df = pd.DataFrame()

if metrics_path.exists():
    all_metrics = pd.read_csv(metrics_path)

    if {"dataset", "model"}.issubset(all_metrics.columns):
        metric_df = all_metrics[
            (all_metrics["dataset"] == dataset_name)
            & (all_metrics["model"] == model_name)
        ]

if not metric_df.empty:
    selected_metrics = metric_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Test accuracy",
        f"{selected_metrics['accuracy']:.1%}",
    )
    c2.metric(
        "Weighted precision",
        f"{selected_metrics['precision_weighted']:.1%}",
    )
    c3.metric(
        "Weighted recall",
        f"{selected_metrics['recall_weighted']:.1%}",
    )
    c4.metric(
        "Weighted F1",
        f"{selected_metrics['f1_weighted']:.1%}",
    )
else:
    st.info(
        "Evaluation metrics for this model are not available yet. "
        "Predictions can still be explored."
    )

st.caption(
    f"Dataset: {disease_label} · Model: {model_label} · "
    f"Dataset records: {len(df):,}"
)

st.divider()

# -------------------- PATIENT INPUT FORM --------------------

st.markdown("## 🧑‍⚕️ Patient data")

st.write(
    "Enter feature values for one record. Initial values are set to "
    "dataset medians or the most common category."
)

with st.form("patient_form"):
    user_values = {}
    input_columns = st.columns(3)

    for index, feature in enumerate(feature_columns):
        series = X_data[feature]
        target_column = input_columns[index % 3]

        with target_column:
            label = feature.replace("_", " ").title()

            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(
                    series,
                    errors="coerce",
                ).dropna()

                median = default_value(feature)

                minimum = (
                    float(numeric_series.min())
                    if not numeric_series.empty
                    else 0.0
                )
                maximum = (
                    float(numeric_series.max())
                    if not numeric_series.empty
                    else 1.0
                )

                if minimum == maximum:
                    value = st.number_input(
                        label,
                        value=float(median),
                        key=f"input_{feature}",
                    )
                else:
                    value = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(
                            max(float(median), minimum),
                            maximum,
                        ),
                        key=f"input_{feature}",
                    )

                user_values[feature] = value

            else:
                choices = sorted(
                    series.dropna().astype(str).unique().tolist()
                )

                if not choices:
                    choices = [default_value(feature)]

                initial = default_value(feature)
                default_index = (
                    choices.index(initial)
                    if initial in choices
                    else 0
                )

                user_values[feature] = st.selectbox(
                    label,
                    choices,
                    index=default_index,
                    key=f"input_{feature}",
                )

    submitted = st.form_submit_button(
        "Generate prediction",
        use_container_width=True,
    )

# -------------------- PREDICTION AND EXPLANATION --------------------

if submitted:
    patient_df = pd.DataFrame(
        [user_values],
        columns=feature_columns,
    )

    try:
        encoded_prediction = int(
            pipeline.predict(patient_df)[0]
        )

        predicted_label = label_encoder.inverse_transform(
            [encoded_prediction]
        )[0]

        probabilities = pipeline.predict_proba(patient_df)[0]

        probability_df = pd.DataFrame({
            "Class": class_names,
            "Probability": probabilities,
        }).sort_values(
            "Probability",
            ascending=False,
        )

        st.divider()
        st.markdown("## 🔬 Prediction results")

        result_col, chart_col = st.columns([1, 1.5])

        with result_col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted class</div>
                    <div class="metric-value">
                        {str(predicted_label).replace('_', ' ').title()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.caption(
                "These are model-estimated class probabilities, "
                "not validated clinical risk estimates."
            )

        with chart_col:
            fig = px.bar(
                probability_df,
                x="Probability",
                y="Class",
                orientation="h",
                text=probability_df["Probability"].map(
                    lambda value: f"{value:.1%}"
                ),
                title="Model class probabilities",
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_tickformat=".0%",
                xaxis_range=[0, 1],
                height=300,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown("### 🔎 Input summary")
        st.dataframe(
            patient_df.T.rename(
                columns={0: "Input value"}
            ),
            use_container_width=True,
        )

        # -------------------- FEATURE IMPORTANCE --------------------

        st.markdown("### 🧭 Model explainability")

        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        feature_importance = None
        explanation_note = ""

        if hasattr(classifier, "feature_importances_"):
            encoded_names = preprocessor.get_feature_names_out()
            importances = classifier.feature_importances_

            feature_importance = pd.DataFrame({
                "Feature": encoded_names,
                "Importance": importances,
            }).sort_values(
                "Importance",
                ascending=False,
            ).head(15)

        elif hasattr(classifier, "coef_"):
            encoded_names = preprocessor.get_feature_names_out()
            coefficients = np.asarray(classifier.coef_)

            if coefficients.ndim == 2 and coefficients.shape[0] == 1:
                importance_values = np.abs(coefficients[0])
            else:
                importance_values = np.abs(coefficients).mean(axis=0)

            feature_importance = pd.DataFrame({
                "Feature": encoded_names,
                "Importance": importance_values,
            }).sort_values(
                "Importance",
                ascending=False,
            ).head(15)

        elif model_name == "svm":
            explanation_note = (
                "This SVM does not expose native feature importance. "
                "Model-agnostic explanations can be added separately."
            )

        if feature_importance is not None:
            feature_importance["Feature"] = (
                feature_importance["Feature"]
                .str.replace("numeric__", "", regex=False)
                .str.replace("categorical__", "", regex=False)
            )

            importance_fig = px.bar(
                feature_importance.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Most influential model features",
            )

            importance_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
            )

            st.plotly_chart(
                importance_fig,
                use_container_width=True,
            )

            st.caption(
                "Feature importance describes model behavior; "
                "it does not establish causation or explain this "
                "individual prediction by itself."
            )

        elif explanation_note:
            st.info(explanation_note)

        st.warning(
            "Educational prototype only. Do not use these results to "
            "diagnose, rule out, or treat a disease. Consult a qualified "
            "healthcare professional for medical decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# IMPORTANT: This line is outside the try/except and if blocks.
st.divider()

# -------------------- MODEL COMPARISON --------------------

st.markdown("## 📈 Compare available models")

if metrics_path.exists():
    comparison = pd.read_csv(metrics_path)

    if {"dataset", "model", "f1_weighted"}.issubset(comparison.columns):
        comparison = comparison[
            comparison["dataset"] == dataset_name
        ]

        if not comparison.empty:
            comparison = comparison.sort_values(
                "f1_weighted",
                ascending=False,
            )

            metric_columns = [
                column
                for column in [
                    "accuracy",
                    "precision_weighted",
                    "recall_weighted",
                    "f1_weighted",
                ]
                if column in comparison.columns
            ]

            compare_fig = px.bar(
                comparison,
                x="model",
                y=metric_columns,
                barmode="group",
                title=f"{disease_label}: model performance comparison",
                labels={
                    "model": "Model",
                    "value": "Score",
                    "variable": "Metric",
                },
            )

            compare_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1],
                height=420,
            )

            st.plotly_chart(
                compare_fig,
                use_container_width=True,
            )
        else:
            st.info(
                "No evaluation metrics found for this dataset."
            )
    else:
        st.warning(
            "The metrics file does not contain the expected columns."
        )
else:
    st.info(
        "The model metrics file has not been generated yet."
    )

st.caption(
    "AI Clinical Decision Support · Academic machine-learning project"
)
