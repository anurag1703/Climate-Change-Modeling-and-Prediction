import os
import mlflow
import mlflow.keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os
import joblib


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/anurag1703/Climate-Change-Modeling-and-Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "anurag1703"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "7dd797a7be400379260b080ab37bcd63b4bc455e"


# Load prepared time-series data
data = pd.read_csv(r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\data splits\trend data\prepared_trend_data.csv", 
                   index_col="date", parse_dates=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)


# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


seq_length = 10
X, y = create_sequences(scaled_data, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Define the directory for saving models
model_dir = r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\models"
os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

# Train LSTM Model
mlflow.set_tracking_uri("https://dagshub.com/anurag1703/Climate-Change-Modeling-and-Prediction.mlflow")
mlflow.set_experiment("trend_prediction_experiment")
with mlflow.start_run():
    # Define the model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("root_mean_squared_error", rmse)
    mlflow.log_metric("mean_absolute_error", mae)
    mlflow.log_metric("validation_loss", history.history['val_loss'][-1])

    # Save the model locally in the specified directory
    model_save_path = os.path.join(model_dir, "lstm_trend_model.h5")
    model.save(model_save_path)
    print(f"LSTM Trend Prediction Model saved at {model_save_path}")

    # Log the model in MLflow
    mlflow.keras.log_model(model, "lstm_trend_model")
    print("LSTM Trend Prediction Model Training Complete and Logged to MLflow!")
