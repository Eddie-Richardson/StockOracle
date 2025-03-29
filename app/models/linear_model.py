# app/models/linear_model.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi import HTTPException
from dateutil import parser

class LinearStockPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(np.array(X).reshape(-1, 1), np.array(y))

    def predict(self, future_indices, X_train=None, y_train=None):
        preds = self.model.predict(np.array(future_indices).reshape(-1, 1))
        metrics = {}
        if X_train is not None and y_train is not None:
            y_pred = self.model.predict(np.array(X_train).reshape(-1, 1))
            metrics["MAE"] = mean_absolute_error(y_train, y_pred)
            metrics["MSE"] = mean_squared_error(y_train, y_pred)
        return preds, metrics

def predict_stock_trend(data, forecast_steps=1):
    data_sorted = sorted(data, key=lambda x: parser.parse(x["Date"]) if isinstance(x["Date"], str) else x["Date"])
    closing_prices = [float(row["Close"]) for row in data_sorted]
    if len(closing_prices) < 2:
        raise HTTPException(status_code=400, detail="Not enough data for prediction.")
    X_train = list(range(len(closing_prices)))
    predictor = LinearStockPredictor()
    predictor.fit(X_train, closing_prices)
    future_indices = list(range(len(closing_prices), len(closing_prices) + forecast_steps))
    predictions, metrics = predictor.predict(future_indices, X_train=X_train, y_train=closing_prices)
    return {
        "predictions": predictions.tolist(),
        "metrics": metrics
    }
