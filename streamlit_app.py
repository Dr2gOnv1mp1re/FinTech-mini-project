import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "results.json"
RESULTS_FIGURE_PATH = PROJECT_DIR / "fig_results.png"
ARCHITECTURE_FIGURE_PATH = PROJECT_DIR / "fig_architecture.png"
DATA_PATH = PROJECT_DIR / "synthetic_upi_transactions.csv"

st.set_page_config(page_title="UPI Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_data
def load_results() -> dict:
    with RESULTS_PATH.open(encoding="utf-8") as results_file:
        return json.load(results_file)


@st.cache_data
def load_transactions() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


st.title("UPI Fraud Detection")
st.caption("University mini-project dashboard · Assignment 3")
st.warning(
    "This prototype uses synthetic transactions for demonstration only. "
    "Its scores do not establish real-world fraud detection performance."
)

if not RESULTS_PATH.is_file():
    st.error("results.json was not found. Run upi_fraud_prototype.py first.")
    st.stop()

results = load_results()
model_results = results["res"]
best_model = results["best"]
best_metrics = model_results[best_model]

st.subheader("Model selected using validation data")
st.success(
    f"**{best_model}** was selected based on validation PR-AUC "
    f"({best_metrics['validation_pr_auc']:.3f}). "
    "Test metrics below are reported separately."
)

metric_columns = st.columns(4)
metric_columns[0].metric("Test PR-AUC", f"{best_metrics['pr_auc']:.3f}")
metric_columns[1].metric("Test precision", f"{best_metrics['precision']:.1%}")
metric_columns[2].metric("Test recall", f"{best_metrics['recall']:.1%}")
metric_columns[3].metric("Test F1-score", f"{best_metrics['f1']:.3f}")

with st.expander("What do these metrics mean?"):
    st.markdown(
        "- **PR-AUC** summarizes precision and recall across score thresholds; "
        "it is useful when fraud is rare.\n"
        "- **Precision** is the fraction of fraud alerts that were fraud.\n"
        "- **Recall** is the fraction of fraud cases the model detected.\n"
        "- **F1-score** combines precision and recall into one score."
    )

st.subheader("Compare the models")
comparison = pd.DataFrame.from_dict(model_results, orient="index")
comparison.index.name = "Model"
display_columns = [
    "validation_pr_auc",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "fpr",
]
st.dataframe(
    comparison[display_columns].rename(
        columns={
            "validation_pr_auc": "Validation PR-AUC",
            "roc_auc": "Test ROC-AUC",
            "pr_auc": "Test PR-AUC",
            "precision": "Test precision",
            "recall": "Test recall",
            "f1": "Test F1",
            "fpr": "Test false-positive rate",
        }
    ).style.format("{:.3f}"),
    width="stretch",
)

figure_columns = st.columns(2)
with figure_columns[0]:
    st.markdown("**Precision–recall and feature importance**")
    if RESULTS_FIGURE_PATH.is_file():
        st.image(str(RESULTS_FIGURE_PATH), width="stretch")
    else:
        st.info("Run upi_fraud_prototype.py to generate fig_results.png.")
with figure_columns[1]:
    st.markdown("**Proposed system architecture**")
    if ARCHITECTURE_FIGURE_PATH.is_file():
        st.image(str(ARCHITECTURE_FIGURE_PATH), width="stretch")
    else:
        st.info("Run upi_fraud_prototype.py to generate fig_architecture.png.")

st.subheader("Test-set outcomes")
outcome_columns = st.columns(4)
for column, label, key in zip(
    outcome_columns,
    ("Fraud detected", "False alerts", "Fraud missed", "Legitimate transactions cleared"),
    ("tp", "fp", "fn", "tn"),
):
    column.metric(label, f"{best_metrics[key]:,}")

st.subheader("Synthetic transaction data")
st.caption(
    f"Train: {results['n_train']:,} · Validation: {results['n_validation']:,} · "
    f"Test: {results['n_test']:,} transactions"
)
if DATA_PATH.is_file():
    with st.expander("Preview the generated transactions"):
        transactions = load_transactions()
        st.dataframe(transactions.head(100), width="stretch")
        st.caption(
            f"{len(transactions):,} synthetic rows · "
            f"{transactions['fraud'].sum():,} fraud labels"
        )
else:
    st.info("Run upi_fraud_prototype.py to generate the synthetic transaction CSV.")

st.caption(
    "Educational demonstration only. Do not use this prototype to approve, block, "
    "or make decisions about real payments."
)
