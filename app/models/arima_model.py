# app/models/arima_model.py
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def predict_stock_trend(data, forecast_steps=5):
    """
    Predict stock trends using the ARIMA model.

    Parameters:
    - data (list of dict): Historical stock data (must contain 'Close' key).
    - forecast_steps (int): Number of future steps to forecast.

    Returns:
    - dict: Contains predictions and metrics like MAE and MSE.
    """
    try:
        closing_prices = [row["Close"] for row in data]

        # Ensure data is a numpy array
        closing_prices = np.array(closing_prices)

        # Fit ARIMA model
        model = ARIMA(closing_prices, order=(5, 1, 0))  # Example order (p, d, q)
        model_fit = model.fit()

        # Forecast future values
        forecast = model_fit.forecast(steps=forecast_steps)

        # Evaluate metrics (MAE and MSE on in-sample predictions)
        predictions = model_fit.predict()
        mae = np.mean(np.abs(predictions - closing_prices))
        mse = np.mean((predictions - closing_prices) ** 2)

        return {
            "predictions": forecast.tolist(),
            "metrics": {
                "MAE": mae,
                "MSE": mse
            }
        }
    except Exception as e:
        raise ValueError(f"Error fitting ARIMA model: {e}")

def retrain_arima_model(existing_series, new_values):
    updated_series = np.concatenate([existing_series, new_values])
    model = ARIMA(updated_series, order=(p, d, q))
    model_fit = model.fit()
    return model_fit