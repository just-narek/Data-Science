# Sonar Signal Classification: Mine vs. Rock Predictor

This is my first Machine Learning project! It focuses on classifying underwater sonar signals reflected off metal cylinders (mines) and rocks using **Logistic Regression**. The project walks through data preprocessing, feature scaling, model training, and evaluating test performance, followed by a custom single-instance predictive system.

## 📊 Model Performance & Results

* **Training Accuracy:** **93.98%**
* **Test Accuracy:** **69.05%**
* *Note:* The gap between training and testing performance indicates potential overfitting, which serves as an excellent starting point for future model optimization (e.g., hyperparameter tuning or trying ensemble models).

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.12
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib

## 📂 Project Workflow
1. **Data Loading:** Handled 60 continuous numeric sonar frequency features with a binary target column (`M` for Mine, `R` for Rock).
2. **Data Splitting:** Partitioned the dataset using `train_test_split` with a 20% test size and enforced `stratify=y` to maintain class balance.
3. **Feature Scaling:** Applied `StandardScaler` to normalize the feature ranges (`fit_transform` on train data, and strictly `transform` on test data to prevent leakage).
4. **Model Training:** Trained a `LogisticRegression` classifier on the scaled features.
5. **Predictive System:** Built a custom interface that reshapes and scales any raw 60-element sonar input vector to output a real-time prediction (`Mine` or `Rock`).

## 🚀 Core Code Snippet (Predictive System)
Here is how the final model makes an inference on raw unseen data:
```python
# Raw input data vector (60 features)
input_data = (0.0179, 0.0136, 0.0408, ..., 0.0160, 0.0085)

# Process and predict safely
as_numpy_array = np.asarray(input_data)
input_data_reshape = as_numpy_array.reshape(1, -1)
input_data_scaled = scaler.transform(input_data_reshape)
prediction = model.predict(input_data_scaled)

if prediction[0] == "R":
    print('The object is a Rock')
else:
    print('The object is a Mine')
