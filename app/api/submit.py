# app/api/submit.py
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from app.models.ml_models import get_prediction_method
from app.utils.data_utils import save_csv_with_metadata, generate_metadata, fetch_or_get_data
from app.utils.normalization import normalize_param
from app.core.config import templates
import os
from datetime import datetime

from app.utils.plot_utils import save_plot_chart
from app.utils.prediction_utils import save_prediction_to_database

router = APIRouter()



@router.post("/submit", response_class=HTMLResponse)
def submit_form(
        request: Request,
        tickers: str = Form(...),
        period: str = Form(None),
        interval: str = Form(None),
        start_time: str = Form(None),
        end_time: str = Form(None),
        prediction_model: str = Form(None)
):
    # Normalize input parameters (empty fields become None)
    period = normalize_param(period, None)
    interval = normalize_param(interval, None)
    start_time = normalize_param(start_time, None)
    end_time = normalize_param(end_time, None)

    ticker_list = [t.strip() for t in tickers.split(",")]
    results = []

    for ticker in ticker_list:
        try:
            raw_data = fetch_or_get_data(ticker, period, interval, start_time, end_time)
            metadata = generate_metadata(ticker, start_time, end_time, period, interval)
            ticker_folder = os.path.join("data", ticker)
            os.makedirs(ticker_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if prediction_model:  # Prediction model provided
                prediction_func = get_prediction_method(prediction_model.lower())
                prediction_results = prediction_func(raw_data, forecast_steps=1)
            else:  # No prediction model provided
                prediction_results = {}

            csv_file_path = save_csv_with_metadata(metadata, raw_data, ticker, timestamp, prediction=prediction_results)
            png_file_path = save_plot_chart(ticker, raw_data, ticker_folder, timestamp, prediction=prediction_results)

            try:
                save_prediction_to_database(ticker, prediction_results)
            except Exception as e:
                print(f"Error saving prediction to DB: {e}")

            results.append({
                "ticker": ticker,
                "csv_url": f"/data/{ticker}/{os.path.basename(csv_file_path)}",
                "png_url": f"/data/{ticker}/{os.path.basename(png_file_path)}",
                "prediction": prediction_results,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "ticker": ticker,
                "error": str(e),
                "status": "failed",
                "prediction": {}
            })

    return templates.TemplateResponse("results.html", {"request": request, "results": results})

