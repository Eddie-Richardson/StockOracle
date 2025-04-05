# app/utils/compare_predictions.py
import sqlite3
from dateutil import parser
from app.core.config import DB_PATH


def compare_predictions_to_actual(ticker, data, model):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Normalize and sort data by date
    data = [
        {"Date": parser.parse(row["Date"]) if isinstance(row["Date"], str) else row["Date"], **row}
        for row in data
    ]
    data = sorted(data, key=lambda row: row["Date"])

    for entry in data:
        prediction_date = entry["Date"]  # Date for which the prediction was made

        # Find next day's date and close price
        actual_date = next((row["Date"] for row in data if row["Date"] > prediction_date), None)
        if isinstance(actual_date, str):
            actual_date = parser.parse(actual_date)
        if actual_date is None:
        #    print(f"No next day's date found for prediction_date: {prediction_date}")
            continue

        actual_close = next((row["Close"] for row in data if row["Date"] == actual_date), None)
        if actual_close is None:
            print(f"No closing price found for actual_date: {actual_date}")
            continue

        # Fetch prediction from database
        cursor.execute("""
            SELECT predicted_close 
            FROM predictions 
            WHERE ticker = ? AND prediction_timestamp = ? AND model = ?
        """, (ticker, prediction_date.strftime("%Y-%m-%d"), model))
        prediction_row = cursor.fetchone()

        if not prediction_row:
            continue

        predicted_close = prediction_row[0]
        error = abs(predicted_close - actual_close)

        # Check for duplicates before saving comparison
        cursor.execute("""
            SELECT COUNT(*)
            FROM prediction_comparisons
            WHERE ticker = ? AND date = ? AND model = ?
        """, (ticker, actual_date.strftime("%Y-%m-%d"), model))
        record_exists = cursor.fetchone()[0] > 0

        if not record_exists:
            # Save comparison results to the database
            try:
                cursor.execute("""
                    INSERT INTO prediction_comparisons (ticker, date, model, predicted_close, actual_close, error)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    actual_date.strftime("%Y-%m-%d"),  # Ensure proper formatting
                    model,
                    predicted_close,
                    actual_close,
                    error
                ))
                conn.commit()
       #         print(f"Comparison saved: {ticker}, Prediction Date: {prediction_date}, Actual Date: {actual_date}, Model: {model}, Error: {error}")
            except Exception as e:
                print(f"Error saving comparison to DB: {e}")
       # else:
       #     print(f"Duplicate comparison found: {ticker}, Date: {actual_date.strftime('%Y-%m-%d')}, Model: {model}")

    conn.close()
