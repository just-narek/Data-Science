# Credit Card Fraud Detection Pipeline

A highly optimized Machine Learning classification pipeline designed to detect fraudulent credit card transactions. Operating under extreme class imbalance conditions (where fraudulent actions represent a tiny fraction of total data), this project implements specialized sampling techniques, robust feature scaling, and precision-recall-focused modeling to flag suspicious behavior effectively while minimizing false alarms.

## 🚀 Key Features
* **Imbalance Management:** Employs advanced resampling techniques (e.g., SMOTE, Random Under-Sampling) to address skewed class distributions.
* **Anonymized PCA Feature Engineering:** Smooth processing of highly confidential, pre-transformed PCA numerical components ($V_1$ to $V_{28}$) alongside transaction `Time` and `Amount`.
* **Cost-Sensitive Evaluation:** Bypasses misleading accuracy metrics in favor of Area Under the Precision-Recall Curve (AUPRC), Recall, and F1-Score.
* **Classifier Benchmarking:** Compares strong ensemble and boosting frameworks (Random Forest, XGBoost, LightGBM) to pinpoint the strongest detector.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Wrangling:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Imbalanced Data Tooling:** `imbalanced-learn` (`imblearn`)
* **Gradient Boosting:** `xgboost`, `lightgbm`

---

## 📊 Project Workflow

### 1. Exploratory Data Analysis & Scaling
* Inspect the immense skew between legitimate (Class 0) and fraudulent (Class 1) transactions.
* Apply robust scaling (`RobustScaler`) to the raw `Amount` feature to minimize the distortion caused by massive outlier transaction values.

### 2. Addressing Class Imbalance
To ensure models do not simply predict "not fraud" for every row and claim 99.9% accuracy, the pipeline integrates:
* **SMOTE (Synthetic Minority Over-sampling Technique):** Synthesizing new fraud examples along the feature space boundaries.
* **Strategic Under-sampling:** Reducing majority class dominance to give the classifiers balanced exposure.

### 3. Model Suite & Metrics
Evaluates multi-tree classifiers optimized for classification thresholds:
* Random Forest Classifier
* XGBoost Classifier
* LightGBM Classifier

**Primary Metric:** **Recall** (ensuring we catch as much fraud as possible) balanced against **Precision** (avoiding freezing the cards of innocent customers).

---

## 📈 Getting Started

### Installation
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm
