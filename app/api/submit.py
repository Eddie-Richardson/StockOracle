# app/api/submit.py
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from app.models.ml_models import get_prediction_method
from app.utils.compare_predictions import compare_predictions_to_actual
from app.utils.data_utils import save_csv_with_metadata, generate_metadata, fetch_stock_data, create_ticker_folder, create_model_folder
from app.utils.normalization import normalize_param
from app.core.config import templates
from app.utils.plot_utils import save_plot_chart
from app.utils.prediction_utils import save_prediction_to_database
from datetime import datetime
import os

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
    # Normalize input parameters
    period = normalize_param(period, None)
    interval = normalize_param(interval, None)
    start_time = normalize_param(start_time, None)
    end_time = normalize_param(end_time, None)

    ticker_list = [t.strip() for t in tickers.split(",")]
    results = []

    for ticker in ticker_list:
        try:
            ticker_folder = os.path.join("data", ticker)
            model_folder = os.path.join("data", ticker, prediction_model)
            create_ticker_folder(ticker_folder)
            create_model_folder(model_folder)

            try:
                raw_data = fetch_stock_data(ticker, start_time, end_time, period, interval)
                if not raw_data or len(raw_data) == 0:
                    results.append({
                        "ticker": ticker,
                        "error": f"No data retrieved for ticker '{ticker}'.",
                        "status": "failed",
                        "prediction": {}
                    })
                    continue

                # Validate raw_data structure
                if not isinstance(raw_data, list) or any(not isinstance(row, dict) for row in raw_data):
                    raise ValueError(f"Invalid data format for ticker {ticker}. Expected a list of dictionaries.")

                # Normalize raw_data dates to strings
                raw_data = [{"Date": str(row["Date"]), **row} for row in raw_data]
            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "error": f"Error processing raw data: {e}",
                    "status": "failed",
                    "prediction": {}
                })
                continue

            metadata = generate_metadata(ticker, start_time, end_time, period, interval)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if prediction_model:
                prediction_func = get_prediction_method(prediction_model.lower())
                prediction_results = prediction_func(raw_data, forecast_steps=1)
            else:
            #    print("No model selected. Returning empty predictions.")
                prediction_results = {}

            mae = prediction_results.get("metrics", {}).get("MAE")
            mse = prediction_results.get("metrics", {}).get("MSE")

            csv_file_path = save_csv_with_metadata(metadata, raw_data, ticker, timestamp, prediction=prediction_results,
                                                   model_name=prediction_model)
            png_file_path = save_plot_chart(ticker, raw_data, ticker_folder, timestamp, prediction=prediction_results,
                                            model_name=prediction_model)

            try:
                save_prediction_to_database(ticker, prediction_results, prediction_model, end_time, mae=mae, mse=mse)
                compare_predictions_to_actual(ticker, raw_data, prediction_model)
            except Exception as e:
                print(f"Error saving prediction or comparison to DB: {e}")

            results.append({
                "ticker": ticker,
                "csv_url": f"/data/{ticker}/{os.path.basename(csv_file_path)}",
                "png_url": f"/data/{ticker}/{prediction_model}/{os.path.basename(png_file_path)}",
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
