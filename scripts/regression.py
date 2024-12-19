# model_training.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
import joblib


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/anurag1703/Climate-Change-Modeling-and-Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "anurag1703"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "7dd797a7be400379260b080ab37bcd63b4bc455e"


# Load training and testing data
X_train = pd.read_csv(r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\data splits\regression data\X_train.csv")
X_test = pd.read_csv(r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\data splits\regression data\X_test.csv")
y_train = pd.read_csv(r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\data splits\regression data\y_train.csv")
y_test = pd.read_csv(r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\data splits\regression data\y_test.csv")

# Experiment: Linear Regression
mlflow.set_tracking_uri("https://dagshub.com/anurag1703/Climate-Change-Modeling-and-Prediction.mlflow")
mlflow.set_experiment("linear_regression_experiment")
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "linear_regression_model")

# Experiment: Random Forest
mlflow.set_experiment("random_forest_experiment")
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Predictions and metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "random_forest_model")


# Create the 'models' folder if it doesn't exist
model_dir = r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\models"
os.makedirs(model_dir, exist_ok=True)

# Save Linear Regression model
linear_model_path = os.path.join(model_dir, "linear_regression_model.pkl")
joblib.dump(model, linear_model_path)

# Save Random Forest model
rf_model_path = os.path.join(model_dir, "random_forest_model.pkl")
joblib.dump(model, rf_model_path)

print(f"Models saved in {model_dir}")











