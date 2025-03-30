# app/models/db_models.py

from pydantic import BaseModel
from datetime import datetime

class StockData(BaseModel):
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

class Prediction(BaseModel):
    ticker: str
    prediction_timestamp: datetime
    predicted_close: float
    mae: float
    mse: float
    prediction_model: str
