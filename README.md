# Loan Defaulters (Lending Club) 🏦💸

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow.svg)](https://scikit-learn.org/stable/)

---

## 📘 Overview

This project predicts **loan defaults** using the Lending Club dataset by applying **Machine Learning (Random Forest, XGBoost)** and **Deep Learning (TensorFlow ANN)** models. The pipeline includes EDA, preprocessing, feature encoding/scaling, model training, and evaluation with accuracy, ROC-AUC, and classification metrics.

---

## 🧰 Tools & Libraries

**Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, TensorFlow, XGBoost, hvPlot, SciPy.

**Environment**: Jupyter Notebook / Google Colab.

**Visualization**: Seaborn, hvPlot, Matplotlib.

---


## 1) Problem Statement & Business Relevance

 **Problem**: Predict whether a borrower will default (loan_status) using LendingClub-style loan application and credit features (≈395K records, 27 features).
 
**Why it matters**: Accurate default prediction reduces charge-offs, improves underwriting decisions, enables risk-based pricing, and increases portfolio profitability. In production, this model supports loan approval, pricing, and manual-review prioritization.

## 2) Approach & Model Selection Rationale

**Data prep & feature engineering (summary)**:
Clean missing values and apply domain-driven outlier filters (e.g., cap annual_inc, dti, open_acc).
One-hot encode categorical variables (sub_grade, verification_status, purpose, home_ownership, zip_code).
Scale numeric features with MinMaxScaler to support ANN training.

**Models compared**:
 1)Random Forest: Robust baseline, interpretable via feature importance.
 
 2)XGBoost: Strong tabular performer capturing non-linear interactions.
 
 3)ANN (Keras dense network): Evaluated to test if representation learning helps on large dataset.

**Rationale**: Start with well-understood tree models for tabular credit data; include ANN to test whether deeper representations provide measurable lift. Use ROC-AUC as the primary separability metric, but analyze precision/recall and business cost metrics for threshold selection.

**Important evaluation note**: Ensure ROC AUC is calculated on model probability scores (predict_proba[:,1]) — not discrete predictions — and use stratified cross-validation for stability.

## 3) Results & Business Impact (Metrics)
Key metrics observed in notebook:

**ANN (selected run)**: ROC AUC ≈ 0.902 (test); validation AUC during training ≈ 0.896–0.901.

**XGBoost**: ROC AUC ≈ 0.734 (test).

**Random Forest**: ROC AUC ≈ 0.724 (test).

**Accuracy (single eval)**: ≈ 88.9% (note: inflated by class imbalance).

**Example confusion matrix printed**: [[25932 25733] [1490 208988]] (shows many false positives; very high recall for positives).

## Business interpretation:

ANN’s AUC lift suggests better risk ranking vs tree baselines in this run. Higher recall on defaults reduces missed defaulters (lower expected loss) but many false positives create potential revenue loss (denials or higher manual reviews).

Suggested business KPIs to compute and present:

Precision / recall for default class, FPR/FNR at chosen thresholds.

Expected monetary impact: convert FP/FN counts into expected dollars (loss avoided vs revenue lost).

Calibration and lift/gain charts (to map probability buckets to action thresholds).

## 4) Challenges & Learnings

Main challenges found:

**Class imbalance (~80% non-default, 20% default)** — affects thresholding and interpretation; requires cost-sensitive evaluation.

**Metric misuse risk** — notebook occasionally computes AUC from predict(); must use predict_proba.

**Outlier/filtering decisions(e.g., annual_inc <= 250k) need business validation** — filters can bias coverage.

**Explainability gap** — ANN lacks SHAP/LIME explainability in the current notebook; necessary for credit decisions.

## Key learnings & next steps:

Prioritize threshold tuning for business value (monetize FP vs FN).

Add probability calibration + SHAP for model trust and regulatory transparency.

Use stratified K-fold CV and report metric variance to show stability.

## 5) Scalability & Deployment Plan

**Production design (short)**:
Package model + preprocessing artifacts (feature list, encoders, scaler) and register in a model registry.

Serving options: (a) Real-time: REST API (FastAPI) for single-applicant scoring; (b) Batch: Spark/Pandas job for nightly scoring and dashboard updates.

Containerize (Docker) and orchestrate on Kubernetes; schedule retraining jobs with Airflow or similar.

## Monitoring & alerts (must-have):

Data drift (PSI), feature distribution changes, null-rate increases.

**Model performance**: AUC, AUC-PR, precision@k, recall, calibration (Brier score).

**Business metrics**: approval rate, actual default rate, manual review volume, expected loss.

**Retrain triggers**: drift thresholds, AUC drop > X%, or change in default prevalence.

**Governance**:

Save inference logs for auditability, register feature lineage, keep reproducible preprocessing code, and produce per-decision rationales (top-5 features via SHAP).

Quick reproducibility & recommended code checks (snippets to add to notebook)

**Compute AUC correctly**:

from sklearn.metrics import roc_auc_score
probs = model.predict_proba(X_test)[:,1]
roc_auc_score(y_test, probs)

**Threshold tuning for expected value**: compute expected_loss(threshold) by mapping FP/FN to monetary costs and choose threshold maximizing expected profit.

**Calibration check**: sklearn.calibration.CalibratedClassifierCV or sklearn.isotonic/Platt.

**Explainability**: compute shap.TreeExplainer(xgb_clf) for tree models and shap.KernelExplainer() or surrogate explanation for ANN.

Readme footer — Next steps for an interviewer-ready demo.

Replace any roc_auc_score(..., predict(...)) calls with predict_proba.

Add cost matrix calculations and threshold selection tied to dollars.

Add SHAP visualizations and provide 2–3 example-case explainability notes.

---

## 🚀 Reproducibility

### Installation

```bash
# Clone repo
git clone https://github.com/yourusername/Loan-Defaulters-LendingClub.git
cd Loan-Defaulters-LendingClub

# Create venv & install dependencies
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
hvplot
scipy
```

---
