# app/utils/data_utils.py
import sqlite3
from datetime import datetime
import yfinance as yf
from fastapi import HTTPException
from dateutil import parser
from app.core.config import DB_PATH

def save_to_database(ticker, data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for row in data:
        date_str = row["Date"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(row["Date"], "strftime") else str(row["Date"])
        cursor.execute("""
            INSERT INTO stock_data (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, date_str, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]))
    conn.commit()
    conn.close()

def get_data_from_database(ticker, start_date=None, end_date=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = "SELECT * FROM stock_data WHERE ticker = ?"
    params = [ticker]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

def filter_data(data, start_time=None, end_time=None, min_close_price=None, max_close_price=None,
                min_volume=None, max_volume=None):
    filtered = []
    for row in data:
        try:
            row_time = parser.parse(row["Date"]) if isinstance(row["Date"], str) else row["Date"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing date: {e}")
        if start_time and start_time.strip() != "" and row_time < parser.parse(start_time):
            continue
        if end_time and end_time.strip() != "" and row_time > parser.parse(end_time):
            continue
        close = float(row["Close"])
        if min_close_price is not None and close < min_close_price:
            continue
        if max_close_price is not None and close > max_close_price:
            continue
        volume = int(row["Volume"])
        if min_volume is not None and volume < min_volume:
            continue
        if max_volume is not None and volume > max_volume:
            continue
        filtered.append(row)
    return filtered

def fetch_stock_data(ticker: str, period: str = None, interval: str = None):
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period=period, interval=interval)
    if hist_data.empty:
        raise HTTPException(status_code=404, detail="No data available for the given ticker or time period.")
    hist_data.reset_index(inplace=True)
    return hist_data.to_dict(orient="records")

def fetch_or_get_data(ticker, period=None, interval=None, start_time=None, end_time=None):
    db_data = get_data_from_database(ticker, start_date=start_time, end_date=end_time)
    if db_data:
        return [
            {"Date": row[2], "Open": row[3], "High": row[4], "Low": row[5], "Close": row[6], "Volume": row[7]}
            for row in db_data
        ]
    else:
        raw_data = fetch_stock_data(ticker, period=period, interval=interval)
        save_to_database(ticker, raw_data)
        return raw_data


def generate_metadata(ticker: str, start_time, end_time, period, interval, extra_filters=None):
    """
    Generate a metadata dictionary for a given ticker request.
    """
    metadata = {
        "Stock Ticker": ticker,
        "Request Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filters Applied": {
            "Start Time": start_time,
            "End Time": end_time,
            "Period": period,
            "Interval": interval
        }
    }
    if extra_filters:
        metadata["Filters Applied"].update(extra_filters)
    return metadata


def save_csv_with_metadata(metadata: dict, data: list, ticker: str, timestamp: str, prediction: dict = None) -> str:
    """
    Save a CSV file containing metadata and stock data.
    Optionally appends prediction data.
    Returns the CSV file path.
    """
    import os, csv
    ticker_folder = os.path.join("data", ticker)
    os.makedirs(ticker_folder, exist_ok=True)
    csv_filename = f"{ticker}_stock_data_{timestamp}.csv"
    csv_file_path = os.path.join(ticker_folder, csv_filename)

    try:
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Metadata"])
            writer.writerow(["Stock Ticker", metadata["Stock Ticker"]])
            writer.writerow(["Request Timestamp", metadata["Request Timestamp"]])
            for key, value in metadata["Filters Applied"].items():
                writer.writerow([key, value])
            writer.writerow([])  # Blank line for separation
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
            for row in data:
                writer.writerow([row["Date"], row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
            if prediction is not None and "predictions" in prediction and "metrics" in prediction:
                writer.writerow([])  # Extra blank line
                writer.writerow(["Prediction Results"])
                writer.writerow(["Predicted Closing Price", prediction["predictions"][0]])
                writer.writerow(["MAE", prediction["metrics"]["MAE"], "MSE", prediction["metrics"]["MSE"]])
    except Exception as e:
        print(f"Error saving CSV: {e}")
        raise Exception("Error saving CSV")

    return csv_file_path
