# Sonar Signal Classification using Scikit-Learn Pipeline

An end-to-end Machine Learning system designed to classify underwater sonar signals into Mines (`M`) or Rocks (`R`) using **Logistic Regression**. This project implements a unified `scikit-learn` Pipeline architecture to automate data flow and completely eliminate data leakage.

## 📊 Model Performance
* **Training Accuracy:** 93.98%
* **Test Accuracy:** 69.05%

## 🚀 Unified Pipeline Architecture
Instead of separating data scaling and model training into manual scripts, this project encapsulates them into a single, reliable conveyor belt:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Built with zero data leakage architecture
sonar_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Fits the scaler and trains the model safely in one shot
sonar_pipeline.fit(X_train, y_train)

# --- Production Real-Time Inference ---
raw_input = np.array([0.0179, 0.0136, 0.0408, 0.0205, 0.0368, 0.1098, 0.1276, 0.0598, 0.1264, 0.0881, 0.1992, 0.0184, 0.2261, 0.1729, 0.2131, 0.0693, 0.2281, 0.4060, 0.3973, 0.2741, 0.3690, 0.5556, 0.4846, 0.3140, 0.5334, 0.5256, 0.2520, 0.2090, 0.3559, 0.6260, 0.7340, 0.6120, 0.3497, 0.3953, 0.3012, 0.5408, 0.8814, 0.9857, 0.9167, 0.6121, 0.5006, 0.3210, 0.3202, 0.4295, 0.3654, 0.2655, 0.1576, 0.0681, 0.0294, 0.0241, 0.0121, 0.0036, 0.0150, 0.0085, 0.0073, 0.0050, 0.0044, 0.0040, 0.0117, 0.0062]).reshape(1, -1)

# Pipeline automatically scales raw input data and predicts
prediction = sonar_pipeline.predict(raw_input)
print("Object is a Rock" if prediction[0] == "R" else "Object is a Mine")
