# app/utils/data_utils.py
import os
import csv
import time
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from fastapi import HTTPException
from dateutil import parser
from app.core.config import DB_PATH


def debug_data_structure(data, context=""):
    """
    Utility function to print the structure of the provided data.
    """
    print(f"Debugging Data Structure: {context}")
    for i, row in enumerate(data[:5]):  # Print first 5 rows
        print(f"Row {i + 1}: {row}")
    print("...")

def save_to_database(ticker, data):
    """
    Save non-duplicated stock data to the stock_data table in the database.

    :param ticker: The stock ticker symbol.
    :param data: A list of dictionaries representing stock data rows.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    for entry in data:
        # Ensure the date is a string in the proper format
        date_str = entry["Date"].strftime("%Y-%m-%d") if isinstance(entry["Date"], pd.Timestamp) else entry["Date"]

        # Check if the record already exists
        cursor.execute("""
            SELECT COUNT(*)
            FROM stock_data
            WHERE ticker = ? AND date = ?
        """, (ticker, date_str))
        record_exists = cursor.fetchone()[0] > 0

        if not record_exists:
            # Insert the new record
            cursor.execute("""
                INSERT INTO stock_data (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                date_str,  # Use the formatted date string
                entry["Open"],
                entry["High"],
                entry["Low"],
                entry["Close"],
                entry["Volume"]
            ))

    conn.commit()
    conn.close()



def fetch_data_from_database(ticker, start_date=None, end_date=None, columns=None):
    """
    Fetch historical data from the database with optional filtering and selected columns.

    :param ticker: The stock ticker to fetch data for.
    :param start_date: Optional start date for filtering (inclusive).
    :param end_date: Optional end date for filtering (inclusive).
    :param columns: Optional list of columns to fetch (defaults to all columns).
    :return: List of dictionaries representing rows from the database.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Default to all columns if none are specified
    if not columns:
        columns = ["id", "ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
    selected_columns = ", ".join(columns)

    # Build the query dynamically
    query = f"SELECT {selected_columns} FROM stock_data WHERE ticker = ?"
    params = [ticker]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    # Execute the query
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to a list of dictionaries
    rows_as_dicts = [dict(zip(columns, row)) for row in rows]
    return rows_as_dicts

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

def fetch_stock_data(ticker, start_time=None, end_time=None, period=None, interval=None):
    """
    Fetch stock data from an external API (e.g., Yahoo Finance).

    :param ticker: The stock ticker symbol to fetch data for.
    :param start_time: The start date for fetching data (optional).
    :param end_time: The end date for fetching data (optional).
    :param period: Period parameter for API fetch (optional).
    :param interval: Interval parameter for API fetch (optional).
    :return: List of rows fetched from the API.
    """
    stock = yf.Ticker(ticker)

    # Adjust end time to include the full day's data
    if end_time:
        adjusted_end_time = (datetime.strptime(end_time, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        adjusted_end_time = None

    # Fetch historical data from the API
    hist_data = stock.history(start=start_time, end=adjusted_end_time, period=period, interval=interval)
    if hist_data.empty:
        raise HTTPException(status_code=404, detail="No data available for the given ticker or time period.")

    hist_data.reset_index(inplace=True)
    stock_data = hist_data.to_dict(orient="records")
    if stock_data:
        save_to_database(ticker, stock_data)
    return stock_data



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


def save_csv_with_metadata(metadata: dict, data: list, ticker: str, timestamp: str, prediction: dict = None, model_name: str = None) -> str:
    """
    Save a CSV file containing metadata and stock data.
    Optionally appends prediction data.
    Returns the CSV file path.
    """
    ticker_folder = os.path.join("data", ticker)
    model_folder = os.path.join("data", ticker, model_name)
    os.makedirs(ticker_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    csv_filename = f"{ticker}_stock_data_{timestamp}.csv"
    csv_file_path = os.path.join(ticker_folder, model_name, csv_filename)

    try:
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Metadata"])
            writer.writerow(["Stock Ticker", metadata["Stock Ticker"]])
            writer.writerow(["Request Timestamp", metadata["Request Timestamp"]])
            writer.writerow(["Prediction Model Used", model_name])
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

def create_ticker_folder(ticker_folder):
    """
    Ensures the ticker folder exists, creating it if necessary.
    """
    if not os.path.exists(ticker_folder):
        os.makedirs(ticker_folder)

def create_model_folder(model_folder):
    """
    Ensures the ticker folder exists, creating it if necessary.
    """
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)


def fetch_data_with_retries(ticker, period=None, interval=None, start_time=None, end_time=None, retries=3):
    for attempt in range(retries):
        try:
            raw_data = fetch_stock_data(ticker, start_time=start_time, period=period, interval=interval,
                                        end_time=end_time)

            if raw_data and len(raw_data) > 0:
                return raw_data

            print(f"Attempt {attempt + 1}/{retries}: No data fetched for {ticker}. Retrying...")
        except Exception as e:
            print(
                f"Attempt {attempt + 1}/{retries}: Error fetching data for {ticker}. Parameters: start_date={start_time}, end_time={end_time}, period={period}, interval={interval}")
            print(f"Error details: {e}")
        time.sleep(2)

    raise HTTPException(status_code=500, detail=f"Failed to fetch data for {ticker} after {retries} attempts.")
