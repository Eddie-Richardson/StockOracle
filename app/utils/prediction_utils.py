# app/utils/prediction_utils.py
import sqlite3
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi import HTTPException
from dateutil import parser
from app.core.config import DB_PATH

def predict_stock_trend(data, forecast_steps=1):
    data_sorted = sorted(data, key=lambda x: parser.parse(x["Date"]) if isinstance(x["Date"], str) else x["Date"])
    closing_prices = [float(row["Close"]) for row in data_sorted]
    if len(closing_prices) < 2:
        raise HTTPException(status_code=400, detail="Not enough data for prediction.")
    X = np.arange(len(closing_prices)).reshape(-1, 1)
    y = np.array(closing_prices)
    model = LinearRegression()
    model.fit(X, y)
    future_indices = np.arange(len(closing_prices), len(closing_prices) + forecast_steps).reshape(-1, 1)
    predictions = model.predict(future_indices)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return {
        "predictions": predictions.tolist(),
        "metrics": {"MAE": mae, "MSE": mse}
    }

def save_prediction_to_database(ticker, prediction_data, model_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    predicted_close = prediction_data["predictions"][0] if prediction_data.get("predictions") else None
    mae = prediction_data["metrics"]["MAE"] if "metrics" in prediction_data else None
    mse = prediction_data["metrics"]["MSE"] if "metrics" in prediction_data else None
    prediction_model = model_name
    cursor.execute("""
        INSERT INTO predictions (ticker, prediction_timestamp, predicted_close, mae, mse, prediction_model)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ticker, prediction_timestamp, predicted_close, mae, mse, prediction_model))
    conn.commit()
    conn.close()
