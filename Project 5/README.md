# House Price Prediction

A comprehensive, end-to-end Machine Learning pipeline designed to predict housing prices using regression techniques. This project implements strict data preprocessing, handles categorical/continuous features seamlessly, and evaluates a diverse suite of algorithms—ranging from linear baselines to advanced gradient boosting ensembles—to identify the most accurate predictive model.

## 🚀 Key Features
* **Robust Data Preprocessing:** Automated handling of missing values, ordinal encoding for categorical features, and feature scaling using `StandardScaler`.
* **Leakage-Free Validation:** Strict separation of test data prior to scaling and training to ensure unbiased performance metrics.
* **Multi-Model Benchmarking:** Evaluates 8 different algorithms to find the optimal balance between bias and variance.
* **Performance Metrics:** Models are scored and compared using Mean Absolute Error (MAE), Mean Squared Error (MSE), and $R^2$ Score.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning Framework:** `scikit-learn`
* **Gradient Boosting Ecosystem:** `xgboost`, `lightgbm`

---

## 📊 Pipeline Architecture

### 1. Exploratory Data Analysis & Preprocessing
* Missing values are imputed or cleaned based on distribution strategies.
* Categorical variables are transformed via `OrdinalEncoder` to maintain potential structural hierarchies.
* Continuous features are normalized using `StandardScaler` to optimize gradient descent convergence for linear models and distance-based metrics.

### 2. Evaluated Models
The following algorithms were trained, cross-validated, and tested:
* **Linear Baselines:** Linear Regression, Ridge Regression, Lasso Regression
* **Tree-Based Models:** Decision Tree Regressor
* **Ensemble Methods:** Random Forest Regressor, Gradient Boosting Regressor
* **Advanced Boosting Frameworks:** XGBoost Regressor, LightGBM Regressor

---

## 📈 Getting Started

### Prerequisites
Ensure you have Python installed, then install the required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
