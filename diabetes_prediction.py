import pandas as pd
import numpy as np
import joblib

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
df = pd.read_csv("diabetes.csv")


# 2. Select features and target
features = [
    "Glucose",
    "BMI",
    "Age",
    "Pregnancies",
    "DiabetesPedigreeFunction"
]

X = df[features].copy()
y = df["Outcome"]


# 3. Handle missing values
X = X.fillna(X.median())


# 4. Apply Box-Cox transformation to Age
X["Age"], age_lambda = stats.boxcox(X["Age"])


# 5. Apply Yeo-Johnson transformation to Pregnancies
X["Pregnancies"], preg_lambda = stats.yeojohnson(
    X["Pregnancies"]
)


# 6. Apply Log Transformation to DiabetesPedigreeFunction
X["skewed_dbf"] = np.log1p(
    X["DiabetesPedigreeFunction"]
)

X.drop("DiabetesPedigreeFunction", axis=1, inplace=True)


# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 8. Standard Scaling for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 9. Logistic Regression Model
logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)


# 10. Random Forest Model
random_forest = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

random_forest.fit(X_train, y_train)


# 11. Model Evaluation

lr_predictions = logistic_model.predict(X_test_scaled)

rf_predictions = random_forest.predict(X_test)


print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))


print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))


# 12. Save Models and Preprocessing Information

model_data = {
    "logistic_regression": logistic_model,
    "random_forest": random_forest,
    "scaler": scaler,
    "age_lambda": age_lambda,
    "preg_lambda": preg_lambda,
    "features": features
}

joblib.dump(model_data, "diabetes_model.pkl")

print("\nModels successfully trained and saved!")