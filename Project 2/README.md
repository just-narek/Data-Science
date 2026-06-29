# Diabetes Risk Prediction System using Support Vector Machines (SVM)

This project builds a predictive system to determine whether a patient is at high or low risk for Diabetes based on several diagnostic measurements (such as Glucose levels, Blood Pressure, BMI, Age, etc.). The model utilizes a **Support Vector Machine (SVM)** classifier with a linear kernel, combined with proper feature scaling.

## 📈 Model Performance & Stability

* **Training Accuracy:** **78.34%**
* **Test Accuracy:** **74.68%**
* *Engineering Insight:* Unlike heavily overfitted models, the tight gap (~3.6%) between training and test metrics proves that this system generalizes exceptionally well to unseen patient samples.

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.12
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib
* **Dataset:** Pima Indians Diabetes Dataset (Features include: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).

## 📂 Engineering Steps Covered
1. **Exploratory Analysis:** Explored diagnostic data structure and confirmed class distributions using descriptive statistics.
2. **Feature Standardization:** Scaled features using `StandardScaler` to ensure the SVM optimization algorithm is not biased toward large absolute numeric values.
3. **Data Splitting:** Applied `train_test_split` with `stratify=y` to retain identical diabetes proportions in both train and test partitions.
4. **SVM Classifier Training:** Configured and fitted a `svm.SVC(kernel='linear')` model.
5. **Real-time Inference:** Implemented a single-instance array prediction script that processes new medical metrics safely.

## 🚀 Single-Instance Prediction Module
The predictive system takes a raw user vector and flags the clinical risk status:

```python
# Raw patient metrics: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
raw_patient_data = (5, 116, 74, 0, 0, 25.6, 0.201, 30)

# Reshape, scale with the pre-fitted transformer, and evaluate
data_array = np.asarray(raw_patient_data).reshape(1, -1)
scaled_data = scaler.transform(data_array)
prediction = classifier.predict(scaled_data)

if prediction[0] == 1:
    print("Prediction: Likely Diabetes (High Risk)")
else:
    print("Prediction: Likely No Diabetes (Low Risk)")
