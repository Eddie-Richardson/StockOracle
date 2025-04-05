# app/models/linear_model.py
import os
from app.core.config import DB_PATH
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi import HTTPException
from dateutil import parser
from joblib import dump, load
import sqlite3
import schedule
import time
import threading

class LinearStockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.previous_X = None  # Store training history indices
        self.previous_y = None  # Store training history values

    def fit(self, X, y):
        self.previous_X = X  # Save training indices
        self.previous_y = y  # Save training values
        self.model.fit(np.array(X).reshape(-1, 1), np.array(y))

    def predict(self, future_indices, X_train=None, y_train=None):
        preds = self.model.predict(np.array(future_indices).reshape(-1, 1))
        metrics = {}
        if X_train is not None and y_train is not None:
            y_pred = self.model.predict(np.array(X_train).reshape(-1, 1))
            metrics["MAE"] = mean_absolute_error(y_train, y_pred)
            metrics["MSE"] = mean_squared_error(y_train, y_pred)
        return preds, metrics

    def retrain(self, updated_X, updated_y):
        """
        Retrains the linear model with updated combined data.

        :param updated_X: Combined array of indices (existing + new).
        :param updated_y: Combined array of closing prices (existing + new).
        """
        self.fit(updated_X, updated_y)

def predict_stock_trend(data, forecast_steps=1):
    # Sort and preprocess data
    data_sorted = sorted(data, key=lambda x: parser.parse(x["Date"]) if isinstance(x["Date"], str) else x["Date"])
    closing_prices = [float(row["Close"]) for row in data_sorted if "Close" in row and row["Close"] is not None]

    # Ensure enough data exists for predictions
    if len(closing_prices) < 2:
        print("Not enough data for prediction.")
        raise HTTPException(status_code=400, detail="Not enough data for prediction.")

    # Prepare data for Linear Regression
    X = np.arange(len(closing_prices)).reshape(-1, 1)
    y = np.array(closing_prices)

    # Train the regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_indices = np.arange(len(closing_prices), len(closing_prices) + forecast_steps).reshape(-1, 1)
    predictions = model.predict(future_indices)

    # Evaluate the model
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    return {
        "predictions": predictions.tolist(),
        "metrics": {"MAE": mae, "MSE": mse}
    }

# Saving and Loading Training History (Database)
def save_training_history_to_db(db_path, X, y):
    # Convert numpy array `X` to a flattened list
    X_as_list = X.flatten().tolist()

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO linear_training_history (X, y) VALUES (?, ?)", zip(X_as_list, y))
    conn.commit()
    conn.close()
    print("Linear training history saved to database.")


def load_training_history_from_db(db_path):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT X, y FROM linear_training_history")
    data = cursor.fetchall()
    conn.close()
    X = np.array([row[0] for row in data]).reshape(-1, 1)
    y = np.array([row[1] for row in data])
    print("Training history loaded from database.")
    return X, y

# Saving and Loading Model (Joblib)
def save_model(file_path, model, X, y):
    dump({"model": model, "X": X, "y": y}, file_path)
    print(f"Model and history saved to {file_path}")

def load_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs("data/Linear", exist_ok=True)
        raise FileNotFoundError(f"No saved model found at {file_path}. Initialize a new model.")
    data = load(file_path)
    model = data["model"]
    X = data["X"]
    y = data["y"]
    print(f"Model and history loaded from {file_path}")
    return model, X, y

# Retraining Logic
def retrain_linear_model(predictor, new_X, new_y):
    # Combine old and new data
    combined_X = np.concatenate([predictor.previous_X, new_X])
    combined_y = np.concatenate([predictor.previous_y, new_y])
    predictor.retrain(combined_X, combined_y)
    return predictor


def daily_workflow():
    print("Fetching daily stock data...")

    # Step 1: Load previous model and history
    try:
        model, previous_X, previous_y = load_model("data/Linear/linear_model.joblib")
        predictor = LinearStockPredictor()
        predictor.model = model
        predictor.previous_X = previous_X
        predictor.previous_y = previous_y
    except FileNotFoundError:
        print("No saved model found, initializing a new one.")
        predictor = LinearStockPredictor()
        previous_X = np.arange(0, 100).reshape(-1, 1)  # Example indices
        previous_y = np.random.rand(100) * 50  # Simulated closing prices
        predictor.fit(previous_X, previous_y)
        save_training_history_to_db("stock_data.db", previous_X, previous_y)
        save_model("data/Linear/linear_model.joblib", predictor.model, previous_X, previous_y)

    # Step 2: Fetch new data after market close
    last_X = predictor.previous_X[-1][0] if predictor.previous_X is not None else 0
    new_X, new_y = fetch_new_data_from_db("stock_data.db", last_X)

    if new_X is not None and new_y is not None:
        # Step 3: Retrain the model
        combined_X = np.concatenate([predictor.previous_X, new_X])
        combined_y = np.concatenate([predictor.previous_y, new_y])
        predictor.retrain(combined_X, combined_y)

        # Step 4: Save updated model and training history
        save_training_history_to_db("stock_data.db", combined_X, combined_y)
        save_model("data/Linear/linear_model.joblib", predictor.model, combined_X, combined_y)

        print("Daily workflow completed successfully.")
    else:
        print("No new data fetched for retraining.")


def fetch_new_data_from_db(db_path, last_X):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Query for data where X > last_X (fetch new entries)
    cursor.execute("""
        SELECT X, y FROM linear_training_history
        WHERE X > ?
        ORDER BY X ASC
    """, (last_X,))
    data = cursor.fetchall()
    conn.close()

    if data:
        new_X = np.array([row[0] for row in data]).reshape(-1, 1)
        new_y = np.array([row[1] for row in data])
        return new_X, new_y
    else:
        print("No new data available.")
        return None, None


# Schedule daily workflow to run after market close (e.g., 6:00 PM)
schedule.every().day.at("18:00").do(daily_workflow)


def start_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start the scheduler in a background thread
scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
scheduler_thread.start()

