# Loan Defaulters (Lending Club) 🏦💸

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow.svg)](https://scikit-learn.org/stable/)

---

## 📘 Overview

This project predicts **loan defaults** using the Lending Club dataset by applying **Machine Learning (Random Forest, XGBoost)** and **Deep Learning (TensorFlow ANN)** models. The pipeline includes EDA, preprocessing, feature encoding/scaling, model training, and evaluation with accuracy, ROC-AUC, and classification metrics.

---

## 🧰 Tools & Libraries

**Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, TensorFlow, XGBoost, hvPlot, SciPy
**Environment**: Jupyter Notebook / Google Colab
**Visualization**: Seaborn, hvPlot, Matplotlib

---

## 📂 Dataset

* Dataset: `lending_club_loan_two.csv`
* Target: `loan_status` (Fully Paid / Charged Off)
* Features: Loan amount, grade, sub_grade, purpose, annual income, DTI, revolving balance, etc.

> If running locally, ensure the dataset is placed in the same folder as the notebook.

---

## 🔍 EDA Highlights

* Explored feature distributions and outliers.
* Visualized correlation heatmaps and histograms.
* Analyzed loan grades and purpose-wise default rates.
* Checked missing values and performed imputation.

---

## ⚙️ Preprocessing

* **One-hot encoding** for categorical features (`pd.get_dummies`).
* **Feature scaling** using `MinMaxScaler`.
* **Train/test split** for robust evaluation.

---

## 🤖 Models Implemented

* Random Forest Classifier
* XGBoost Classifier
* TensorFlow Artificial Neural Network (ANN)

---

## 📈 Results

| Metric              | Value           |
| ------------------- | --------------- |
| **Accuracy**        | **88.48%**      |
| **Validation AUC**  | **0.89 – 0.90** |
| **Precision (1.0)** | 0.89            |
| **Recall (1.0)**    | 0.97            |
| **F1-Score (1.0)**  | 0.93            |

📊 **Confusion Matrix (Hold-out)**

```
[[13462, 12018],
 [ 3011, 101932]]
```

✅ The ANN model achieved the best overall validation AUC (~0.90).

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
