from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="📊",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"

DEFAULT_MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError("model/metadata.json was not found.")

    import json

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def get_model_files(metadata):
    model_files = metadata.get("model_files", {})

    if model_files:
        return model_files

    return {
        model_name: DEFAULT_MODEL_FILES[model_name]
        for model_name in metadata.get("models", DEFAULT_MODEL_FILES.keys())
        if model_name in DEFAULT_MODEL_FILES
    }


def normalize_target(values):
    """Convert common Adult Income labels to 0/1."""
    series = pd.Series(values).copy()

    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(series.dropna().astype(int).unique())
        if unique_values.issubset({0, 1}):
            return series.astype(int)

    normalized = series.astype(str).str.strip()

    mapping = {
        "<=50K": 0,
        "<=50K.": 0,
        ">50K": 1,
        ">50K.": 1,
        "0": 0,
        "1": 1,
    }

    mapped = normalized.map(mapping)

    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(
            "The income column contains unsupported target labels: "
            + ", ".join(map(str, unknown))
        )

    return mapped.astype(int)


def calculate_metrics(model, X, y):
    predictions = np.asarray(model.predict(X)).astype(int)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    mcc = matthews_corrcoef(y, predictions)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X)
    else:
        probabilities = predictions

    auc = roc_auc_score(y, probabilities)

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "predictions": predictions,
    }


def build_confusion_matrix_table(matrix):
    return pd.DataFrame(
        matrix,
        index=["Actual <=50K", "Actual >50K"],
        columns=["Predicted <=50K", "Predicted >50K"],
    )


def build_classification_report(y_true, predictions):
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["<=50K", ">50K"],
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).T


def show_metric(label, value):
    st.metric(label, f"{value:.4f}")


try:
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Could not load model metadata: {exc}")
    st.stop()


target_column = metadata.get("target_column", "income")
expected_features = metadata.get("feature_columns", [])
model_files = get_model_files(metadata)

st.title("Adult Income Classification")
st.caption("Machine Learning Assignment 2 — UCI Adult Income Dataset")

st.sidebar.header("Evaluation")

available_models = [
    model_name for model_name in model_files
    if model_name in DEFAULT_MODEL_FILES
]

if not available_models:
    st.error("No supported model files are configured.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select classification model",
    available_models,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"],
    help="Upload the test data used for model evaluation.",
)

st.sidebar.markdown(
    "The uploaded CSV must contain the target column "
    f"`{target_column}` and the same feature columns used during training."
)

if uploaded_file is None:
    st.info("Upload the test CSV to begin evaluation.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if target_column not in uploaded_df.columns:
    st.error(
        f"The uploaded CSV is missing the target column `{target_column}`. "
        "Please upload the generated test_data.csv."
    )
    st.stop()

missing_features = [
    feature for feature in expected_features
    if feature not in uploaded_df.columns
]

if missing_features:
    st.error(
        "The uploaded CSV is missing required feature columns: "
        + ", ".join(missing_features)
    )
    st.stop()

extra_columns = [
    column
    for column in uploaded_df.columns
    if column not in expected_features and column != target_column
]

if extra_columns:
    st.warning(
        "The following extra columns will be ignored: "
        + ", ".join(extra_columns)
    )

X_eval = uploaded_df[expected_features].copy()

try:
    y_eval = normalize_target(uploaded_df[target_column])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# ---------------------------------------------------------------------
# Uploaded data
# ---------------------------------------------------------------------
st.subheader("Dataset Preview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(uploaded_df):,}")
col2.metric("Features", len(expected_features))
col3.metric("Target", target_column)

preview_columns = expected_features + [target_column]
st.dataframe(
    uploaded_df[preview_columns].head(10),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Target class distribution", expanded=False):
    target_distribution = (
        y_eval.value_counts()
        .sort_index()
        .rename(index={0: "<=50K", 1: ">50K"})
        .rename("Samples")
        .to_frame()
    )
    st.dataframe(target_distribution, use_container_width=True)

# ---------------------------------------------------------------------
# Evaluate all models on the uploaded test data
# ---------------------------------------------------------------------
st.subheader("Model Comparison")

all_results = {}
model_predictions = {}
model_errors = {}

progress = st.progress(0)

for index, (model_name, model_file) in enumerate(model_files.items(), start=1):
    model_path = MODEL_DIR / model_file

    try:
        model = load_model(str(model_path))
        result = calculate_metrics(model, X_eval, y_eval)

        model_predictions[model_name] = result.pop("predictions")
        all_results[model_name] = result
    except Exception as exc:
        model_errors[model_name] = str(exc)

    progress.progress(index / len(model_files))

progress.empty()

if model_errors:
    error_text = "\n".join(
        f"- {name}: {error}"
        for name, error in model_errors.items()
    )
    st.error("One or more models could not be evaluated:\n" + error_text)

if not all_results:
    st.stop()

comparison_table = (
    pd.DataFrame(all_results)
    .T.reset_index()
    .rename(columns={"index": "ML Model Name"})
)

metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

comparison_table[metric_columns] = comparison_table[metric_columns].round(4)

comparison_table = comparison_table.sort_values(
    "F1",
    ascending=False,
).reset_index(drop=True)

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True,
)

winner = comparison_table.iloc[0]["ML Model Name"]
winner_f1 = comparison_table.iloc[0]["F1"]

st.success(
    f"Best model on the uploaded test data: {winner} "
    f"(F1-score: {winner_f1:.4f})"
)

# ---------------------------------------------------------------------
# Selected model details
# ---------------------------------------------------------------------
st.subheader(f"Evaluation — {selected_model}")

if selected_model not in all_results:
    st.error(f"{selected_model} could not be evaluated.")
    st.stop()

selected_result = all_results[selected_model]
selected_predictions = model_predictions[selected_model]

metric_cols = st.columns(6)

for column, metric_name in zip(metric_cols, metric_columns):
    with column:
        show_metric(metric_name, selected_result[metric_name])

report_col, matrix_col = st.columns(2)

with report_col:
    st.markdown("### Classification Report")
    report_table = build_classification_report(
        y_eval,
        selected_predictions,
    ).round(4)

    st.dataframe(
        report_table,
        use_container_width=True,
    )

with matrix_col:
    st.markdown("### Confusion Matrix")
    matrix = confusion_matrix(
        y_eval,
        selected_predictions,
        labels=[0, 1],
    )

    st.dataframe(
        build_confusion_matrix_table(matrix),
        use_container_width=True,
    )

# ---------------------------------------------------------------------
# Sample predictions
# ---------------------------------------------------------------------
with st.expander("Actual vs Predicted (sample)", expanded=False):
    sample_size = min(20, len(uploaded_df))

    sample_predictions = pd.DataFrame(
        {
            "Actual": np.where(
                y_eval.iloc[:sample_size].to_numpy() == 1,
                ">50K",
                "<=50K",
            ),
            "Predicted": np.where(
                selected_predictions[:sample_size] == 1,
                ">50K",
                "<=50K",
            ),
        }
    )

    st.dataframe(
        sample_predictions,
        use_container_width=True,
        hide_index=True,
    )
