# app/models/ml_models.py
from .linear_model import predict_stock_trend as linear_predict_stock_trend
from .arima_model import predict_stock_trend as arima_predict_stock_trend
from .lstm_model import predict_stock_trend as lstm_predict_stock_trend

def get_prediction_method(method="linear"):
    if method == "linear":
        return linear_predict_stock_trend
    elif method == "arima":
        return arima_predict_stock_trend
    elif method == "lstm":
        return lstm_predict_stock_trend
    else:
        raise ValueError(f"Undefined prediction method: {method}")