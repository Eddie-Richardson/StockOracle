# app/api/stock.py
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from app.core.config import DEFAULT_PERIOD, DEFAULT_INTERVAL
from app.utils.data_utils import fetch_stock_data, filter_data, generate_metadata, save_csv_with_metadata
from app.utils.normalization import normalize_param
from app.utils.plot_utils import save_plot_chart
from app.utils.prediction_utils import save_prediction_to_database
from app.models.linear_model import predict_stock_trend


router = APIRouter()


@router.get("/stock/{ticker}")
def get_stock_data(
        ticker: str,
        format: str = "json",
        period: str = Query(None),
        interval: str = Query(None),
        start_time: str = Query(None),
        end_time: str = Query(None),
        min_close_price: float = Query(None),
        max_close_price: float = Query(None),
        min_volume: int = Query(None),
        max_volume: int = Query(None)
):
    period = normalize_param(period, DEFAULT_PERIOD)
    interval = normalize_param(interval, DEFAULT_INTERVAL)
    start_time = normalize_param(start_time)
    end_time = normalize_param(end_time)

    raw_data = fetch_stock_data(ticker, period, interval, start_time, end_time)
    filtered_data = filter_data(raw_data, start_time, end_time, min_close_price, max_close_price, min_volume,
                                max_volume)
    metadata = generate_metadata(ticker, start_time, end_time, period, interval, extra_filters={
        "Min Close Price": min_close_price,
        "Max Close Price": max_close_price,
        "Min Volume": min_volume,
        "Max Volume": max_volume
    })
    ticker_folder = os.path.join("data", ticker)
    os.makedirs(ticker_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        prediction_results = predict_stock_trend(raw_data, forecast_steps=1)
    except Exception as e:
        prediction_results = {"error": str(e)}
    csv_file_path = save_csv_with_metadata(metadata, filtered_data, ticker, timestamp, prediction=prediction_results)
    try:
        save_prediction_to_database(ticker, prediction_results)
    except Exception as e:
        print(f"Error saving prediction to DB: {e}")
    if format == "json":
        return {"metadata": {"csv_file_path": csv_file_path},
                "stock_data": filtered_data,
                "prediction": prediction_results}
    elif format == "table":
        return {"metadata": {"csv_file_path": csv_file_path},
                "headers": ["Date", "Open", "High", "Low", "Close", "Volume"],
                "rows": filtered_data,
                "prediction": prediction_results}
    else:
        raise HTTPException(status_code=400, detail="Invalid format specified")


@router.get("/visualize/{ticker}")
def visualize_stock(
        ticker: str,
        period: str = Query(None),
        interval: str = Query(None),
        start_time: str = Query(None),
        end_time: str = Query(None)
):
    period = normalize_param(period, DEFAULT_PERIOD)
    interval = normalize_param(interval, DEFAULT_INTERVAL)
    start_time = normalize_param(start_time)
    end_time = normalize_param(end_time)

    raw_data = fetch_stock_data(ticker, period, interval, start_time, end_time)
    metadata = generate_metadata(ticker, start_time, end_time, period, interval)
    ticker_folder = os.path.join("data", ticker)
    os.makedirs(ticker_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        prediction_results = predict_stock_trend(raw_data, forecast_steps=1)
    except Exception as e:
        prediction_results = {"error": str(e)}
    csv_file_path = save_csv_with_metadata(metadata, raw_data, ticker, timestamp, prediction=prediction_results)
    png_file_path = save_plot_chart(ticker, raw_data, ticker_folder, timestamp, prediction=prediction_results)
    try:
        save_prediction_to_database(ticker, prediction_results)
    except Exception as e:
        print(f"Error saving prediction to DB: {e}")
    headers = {"Content-Disposition": f'inline; filename="{ticker}_closing_prices_{timestamp}.png"'}
    return FileResponse(png_file_path, media_type="image/png", headers=headers)
