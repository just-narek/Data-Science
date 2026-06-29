# Used Car Price Prediction using Linear Regression

This project builds a regression system to estimate the selling price of used cars based on multiple features such as the car's age, driven kilometers, fuel type, transmission type, and owner history. The model utilizes a **Linear Regression** framework combined with categorical text encoding and numerical scaling.

## 📈 Model Performance & Metrics

* **Training $R^2$ Score:** **88.72%**
* **Test $R^2$ Score:** **84.78%**
* *Engineering Insight:* The $R^2$ score of ~84.8% on the test set demonstrates that the model captures nearly 85% of the variance in car prices. The small gap between training and testing performance proves that the model generalizes successfully without overfitting.

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.12
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib
* **Dataset:** Car Data CSV (Features include: Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner).

## 📂 Engineering & Data Preprocessing Steps
1. **Categorical Encoding:** Converted categorical text columns (like `Fuel_Type`, `Seller_Type`, and `Transmission`) into numeric representations so the regression algorithm can process them mathematically.
2. **Feature & Target Separation:** Isolated the target variable (`Selling_Price`) from the independent features.
3. **Data Partitioning:** Split the dataset into 80% training and 20% testing sets using `train_test_split`.
4. **Regression Training:** Fitted a `LinearRegression` model to map the correlation between car specifications and market prices.
5. **Evaluation:** Evaluated performance metrics using the $R^2$ (R-squared) error score on both splits to monitor model health.

## 🚀 Future Upgrades: Handling Category Pipelines
To improve this codebase for automated production environments, the manual categorical mapping scripts can be upgraded. In the next iteration, a `ColumnTransformer` wrapping a `OneHotEncoder` or `OrdinalEncoder` alongside the `LinearRegression` model will be bound into a single, clean `sklearn.pipeline.Pipeline`. This ensures any raw incoming car data row can be processed and scored in a single operational step
