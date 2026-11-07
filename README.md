# AI-Based Real-Time Fraud Detection for UPI Payments

University mini-project for **21HCSF007 – Financial Technologies Foundations (Assignment 3)**.

The project creates a synthetic UPI transaction dataset, trains and compares three machine-learning classifiers, and presents the saved evaluation results in a Streamlit dashboard.

> **Important:** This is an educational prototype using synthetic data, not real customer or payment data. Its results do not demonstrate real-world fraud detection performance. Do not use it to make decisions about real transactions.

## Live dashboard

Open the deployed app: [UPI Fraud Detection Dashboard](https://fintech-mini-project-fv2tc2gaeehlnk4x48aa5p.streamlit.app/)

## What the project does

- Generates 200,000 synthetic transactions with an approximately 0.8% fraud rate and three simulated fraud scenarios: collect-request scams, vishing with remote-access apps, and mule/account-takeover activity.
- Splits the generated data into stratified training, validation, and test sets (60% / 20% / 20%). Since the generated dataset has no timestamps, a chronological split is not used.
- Trains and evaluates Logistic Regression, Random Forest, and HistGradientBoosting classifiers.
- Selects the best model using validation PR-AUC and selects its decision threshold using validation data. The test set is reserved for final evaluation.
- Saves charts, metrics, and the synthetic transactions for review in the dashboard.

In the recorded run, Random Forest had the highest validation PR-AUC (about 0.511). Its test PR-AUC was about 0.572. Results can change if the code, dependencies, or random seeds change.

## Run locally on Windows

Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe upi_fraud_prototype.py
```

The prototype writes or replaces `synthetic_upi_transactions.csv`, `results.json`, `fig_results.png`, and `fig_architecture.png` in the project folder. Running the training script may take a few minutes.

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

The dashboard displays the saved results; it does not train models or score newly entered transactions.

## Streamlit Community Cloud

This repository is configured for Streamlit Community Cloud:

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
2. Create or deploy an app from `Dr2gOnv1mp1re/FinTech-mini-project`.
3. Select branch `main` and main file `streamlit_app.py`.
4. Deploy. Streamlit installs the packages listed in `requirements.txt`.

The generated transaction CSV is included in the repository and is approximately 24 MB. The dashboard uses it to show a transaction preview.

## Files

| File | Purpose |
|---|---|
| `upi_fraud_prototype.py` | Generates synthetic data, trains the models, evaluates them, and creates the output files. |
| `streamlit_app.py` | Streamlit dashboard for the saved metrics, charts, model comparison, and data preview. |
| `requirements.txt` | Python package requirements. |
| `results.json` | Saved model metrics and run summary. |
| `fig_results.png` | Precision-recall curves and feature-importance chart. |
| `fig_architecture.png` | Proposed fraud-detection system architecture diagram. |
| `synthetic_upi_transactions.csv` | Generated synthetic transaction dataset. |

## Metrics

- **ROC-AUC:** Measures how well the model ranks fraud above legitimate transactions across thresholds.
- **PR-AUC:** Summarizes the precision-recall trade-off; useful when fraud is uncommon.
- **Precision:** Of the transactions flagged as fraud, the fraction that were fraud.
- **Recall:** Of the actual fraud transactions, the fraction detected.
- **F1-score:** A combined measure of precision and recall.
- **False-positive rate:** The fraction of legitimate transactions incorrectly flagged as fraud.

These are measurements on synthetic data and should be interpreted only as a demonstration of the workflow.
