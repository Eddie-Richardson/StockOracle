# app/utils/prediction_utils.py
import sqlite3
from app.core.config import DB_PATH
from app.utils.data_utils import fetch_stock_data, fetch_data_from_database, save_to_database
from fastapi import HTTPException
from dateutil import parser
from datetime import timedelta, datetime

def save_prediction_to_database(ticker, prediction_results, model_name, end_time, mae=None, mse=None):
    """
    Save predictions to the predictions table in the database.

    :param ticker: The stock ticker symbol.
    :param prediction_results: A dictionary containing predictions and metrics.
    :param model_name: The prediction model used.
    :param mae: Mean Absolute Error (optional).
    :param mse: Mean Squared Error (optional).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Extract predictions (expecting a list in this case)
    predicted_values = prediction_results.get("predictions", [])
    if not isinstance(predicted_values, list):
        raise ValueError(f"Invalid predictions format for ticker {ticker}: {predicted_values}")

    for index, predicted_close in enumerate(predicted_values):
        prediction_date = end_time

        # Validate predicted_close
        if not isinstance(predicted_close, (float, int)):
            raise ValueError(f"Invalid predicted close value for ticker {ticker}: {predicted_close}")

        try:
            # Check for duplicates before inserting
            cursor.execute("""
                SELECT COUNT(*)
                FROM predictions
                WHERE ticker = ? AND prediction_timestamp = ? AND model = ?
            """, (ticker, prediction_date, model_name))
            record_exists = cursor.fetchone()[0] > 0

            if not record_exists:
                # Insert prediction into the database
                cursor.execute("""
                    INSERT INTO predictions (ticker, prediction_timestamp, predicted_close, mae, mse, model)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ticker, prediction_date, predicted_close, mae, mse, model_name))
            #    print(f"Prediction saved: {ticker}, Date: {prediction_date}, Predicted Close: {predicted_close}, Model: {model_name}, MAE: {mae}, MSE: {mse}")
           # else:
            #    print(f"Duplicate prediction found for {ticker}, Date: {prediction_date}, Model: {model_name}")
        except Exception as e:
            print(f"Error inserting prediction into the database for ticker {ticker}, Date: {prediction_date}: {e}")

    conn.commit()
    conn.close()



def merge_and_predict(ticker, start_time=None, end_time=None, period=None, interval=None):
    """
    Merge data from the database and API and prepare it for external prediction functions.

    :param ticker: The stock ticker symbol.
    :param start_time: Start date for fetching data.
    :param end_time: End date for fetching data.
    :param period: Period parameter for API fetch (optional).
    :param interval: Interval parameter for API fetch (optional).
    :return: Combined dataset ready for predictions.
    """
    # Step 1: Fetch existing data from the database
    historical_data = fetch_data_from_database(ticker, start_date=start_time, end_date=end_time)

    # Step 2: Fetch missing data from the API
    if historical_data:
        max_date_in_db = max(parser.parse(row["Date"]) for row in historical_data)
        new_data_start_date = (max_date_in_db + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        new_data_start_date = start_time or "2025-03-01"  # Default start date
    #    print(f"No data in database. Fetching full range starting from {new_data_start_date}.")

    new_data = fetch_stock_data(
        ticker,
        start_time=new_data_start_date,
        period=period,
        interval=interval,
        end_time=end_time
    )

    # Step 3: Combine datasets
    combined_data = sorted(historical_data + new_data, key=lambda x: parser.parse(x["Date"]))

    # Save new data to the database if available
    if new_data:
        save_to_database(ticker, new_data)

    # Step 4: Return the combined dataset
    if len(combined_data) < 2:
        raise HTTPException(status_code=400, detail="Not enough data available after combining datasets.")
   # print("Merged dataset ready for predictions.")
    return combined_data