import yfinance as yf
import csv
from datetime import datetime
import os
import sqlite3
from fastapi import FastAPI, HTTPException, Query, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dateutil import parser
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for rendering
import matplotlib.pyplot as plt

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database Setup
conn = sqlite3.connect("data/stock_data.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS stock_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER
)
""")
conn.commit()
conn.close()


# Helper Functions
def save_to_database(ticker, data):
    conn = sqlite3.connect("data/stock_data.db")
    cursor = conn.cursor()

    for row in data:
        # Convert the date field to a string
        date_str = row["Date"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row["Date"], datetime) else str(row["Date"])
        cursor.execute("""
        INSERT INTO stock_data (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, date_str, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]))

    conn.commit()
    conn.close()


def get_data_from_database(ticker, start_date=None, end_date=None):
    conn = sqlite3.connect("data/stock_data.db")
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
    filtered_data = []
    for row in data:
        try:
            row_time = parser.parse(row["Date"]) if isinstance(row["Date"], str) else row["Date"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing date: {e}")

        # Apply time-based filtering
        if start_time and row_time < parser.parse(start_time):
            continue
        if end_time and row_time > parser.parse(end_time):
            continue

        # Apply price-based filters
        close_price = float(row["Close"])
        if min_close_price and close_price < min_close_price:
            continue
        if max_close_price and close_price > max_close_price:
            continue

        # Apply volume-based filters
        volume = int(row["Volume"])
        if min_volume and volume < min_volume:
            continue
        if max_volume and volume > max_volume:
            continue

        filtered_data.append(row)
    return filtered_data


def fetch_or_get_data(ticker, period="1mo", interval="1d", start_time=None, end_time=None):
    db_data = get_data_from_database(ticker, start_date=start_time, end_date=end_time)

    if db_data:  # Data exists in the database
        return [
            {"Date": row[2], "Open": row[3], "High": row[4], "Low": row[5], "Close": row[6], "Volume": row[7]}
            for row in db_data
        ]
    else:  # Fetch new data from yfinance
        raw_data = fetch_stock_data(ticker, period=period, interval=interval)
        save_to_database(ticker, raw_data)
        return raw_data


# Visualization Functions
def plot_closing_prices(ticker, data):
    # Extract dates and closing prices
    dates = [row["Date"] for row in data]
    closing_prices = [row["Close"] for row in data]

    # Create the plot
    plt.figure(figsize=(12, 6))  # Adjust figure size for readability
    plt.plot(dates, closing_prices, label=f"{ticker} Closing Prices", color="blue", linewidth=2)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.title(f"{ticker} Closing Prices Over Time", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.5)

    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=45, fontsize=10, ha='right')  # Rotate and align for better readability
    plt.tight_layout()  # Automatically adjust layout to prevent overlap

    # Save the plot
    file_path = f"{ticker}_closing_prices.png"
    plt.savefig(file_path)
    plt.close()
    return file_path

def plot_closing_prices_with_save_location(ticker, data, file_path):
    # Extract dates and closing prices
    dates = [row["Date"] for row in data]
    closing_prices = [row["Close"] for row in data]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, closing_prices, label=f"{ticker} Closing Prices", color="blue", linewidth=2)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.title(f"{ticker} Closing Prices Over Time", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.5)

    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=45, fontsize=10, ha='right')  # Rotate and align x-axis labels
    plt.tight_layout()  # Automatically adjust layout to prevent overlap

    # Save the plot to the provided file path
    plt.savefig(file_path)
    plt.close()


def add_moving_average(data, window=5):
    closing_prices = [row["Close"] for row in data]
    moving_avg = [sum(closing_prices[i:i+window]) / window for i in range(len(closing_prices) - window + 1)]
    return moving_avg


def plot_with_moving_average(ticker, data):
    dates = [row["Date"] for row in data]
    closing_prices = [row["Close"] for row in data]

    moving_avg = add_moving_average(data, window=5)
    avg_dates = dates[4:]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, closing_prices, label=f"{ticker} Closing Prices", color="blue")
    plt.plot(avg_dates, moving_avg, label="5-Day Moving Average", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} Closing Prices with Moving Average")
    plt.legend()
    plt.grid()

    file_path = f"{ticker}_closing_prices_with_avg.png"
    plt.savefig(file_path)
    plt.close()
    return file_path


# Fetch Stock Data
def fetch_stock_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period=period, interval=interval)
    if historical_data.empty:
        raise HTTPException(status_code=404, detail="No data available for the given ticker or time period.")
    historical_data.reset_index(inplace=True)
    return historical_data.to_dict(orient="records")


# API Endpoints
@app.get("/stock/{ticker}")
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
    # Fetch or get data
    raw_data = fetch_or_get_data(ticker, period, interval, start_time, end_time)

    # Filter the data
    filtered_data = filter_data(raw_data, start_time, end_time, min_close_price, max_close_price, min_volume, max_volume)

    # Prepare metadata
    metadata = {
        "Stock Ticker": ticker,
        "Request Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filters Applied": {
            "Start Time": start_time,
            "End Time": end_time,
            "Period": period,
            "Interval": interval,
            "Min Close Price": min_close_price,
            "Max Close Price": max_close_price,
            "Min Volume": min_volume,
            "Max Volume": max_volume
        }
    }

    # Save filtered data to CSV (with metadata)
    ticker_folder = os.path.join("data", ticker)  # Subfolder for this ticker
    os.makedirs(ticker_folder, exist_ok=True)  # Create if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{ticker}_stock_data_{timestamp}.csv"
    file_path = os.path.join(ticker_folder, file_name)
    print(f"Saving CSV to: {file_path}")  # Debugging output

    try:
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write metadata at the top
            writer.writerow(["Metadata"])
            writer.writerow(["Stock Ticker", metadata["Stock Ticker"]])
            writer.writerow(["Request Timestamp", metadata["Request Timestamp"]])
            writer.writerow(["Start Time", metadata["Filters Applied"]["Start Time"]])
            writer.writerow(["End Time", metadata["Filters Applied"]["End Time"]])
            writer.writerow(["Period", metadata["Filters Applied"]["Period"]])
            writer.writerow(["Interval", metadata["Filters Applied"]["Interval"]])
            writer.writerow(["Min Close Price", metadata["Filters Applied"]["Min Close Price"]])
            writer.writerow(["Max Close Price", metadata["Filters Applied"]["Max Close Price"]])
            writer.writerow(["Min Volume", metadata["Filters Applied"]["Min Volume"]])
            writer.writerow(["Max Volume", metadata["Filters Applied"]["Max Volume"]])
            writer.writerow([])  # Add a blank line for separation
            # Write the data headers and rows
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
            for row in filtered_data:
                writer.writerow([row["Date"], row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
        print(f"CSV saved successfully at {file_path}.")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Return filtered data
    if format == "json":
        return {"metadata": {"csv_file_path": file_path}, "stock_data": filtered_data}
    elif format == "table":
        return {"metadata": {"csv_file_path": file_path},
                "headers": ["Date", "Open", "High", "Low", "Close", "Volume"],
                "rows": filtered_data}
    else:
        raise HTTPException(status_code=400, detail="Invalid format specified")

@app.get("/visualize/{ticker}")
def visualize_stock(
        ticker: str,
        period: str = Query(None),
        interval: str = Query(None),
        start_time: str = Query(None),
        end_time: str = Query(None)
):
    # Fetch or get data
    raw_data = fetch_or_get_data(ticker, period, interval, start_time, end_time)

    # Prepare metadata
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


    # Save data to CSV
    ticker_folder = os.path.join("data", ticker)  # Subfolder for this ticker
    os.makedirs(ticker_folder, exist_ok=True)  # Create if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{ticker}_stock_data_{timestamp}.csv"
    file_path = os.path.join(ticker_folder, file_name)
    print(f"Saving CSV to: {file_path}")  # Debugging output

    try:
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write metadata at the top
            writer.writerow(["Metadata"])
            writer.writerow(["Stock Ticker", metadata["Stock Ticker"]])
            writer.writerow(["Request Timestamp", metadata["Request Timestamp"]])
            writer.writerow(["Start Time", metadata["Filters Applied"]["Start Time"]])
            writer.writerow(["End Time", metadata["Filters Applied"]["End Time"]])
            writer.writerow(["Period", metadata["Filters Applied"]["Period"]])
            writer.writerow(["Interval", metadata["Filters Applied"]["Interval"]])
            writer.writerow([])  # Add a blank line for separation
            # Write the data headers and rows
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
            for row in raw_data:
                writer.writerow([row["Date"], row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
        print(f"CSV saved successfully at {file_path}.")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Generate the chart and save it with the same timestamp
    ticker_folder = os.path.join("data", ticker)  # Subfolder for this ticker
    os.makedirs(ticker_folder, exist_ok=True)  # Create if it doesn't exist
    plot_file_name = f"{ticker}_closing_prices_{timestamp}.png"
    plot_file_path = os.path.join(ticker_folder, plot_file_name)
    print(f"Saving plot to: {plot_file_path}")  # Debugging output
    plot_closing_prices_with_save_location(ticker, raw_data, plot_file_path)

    # Return the PNG file to be displayed inline in the browser
    headers = {"Content-Disposition": f'inline; filename="{plot_file_name}"'}
    return FileResponse(plot_file_path, media_type="image/png", headers=headers)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    print("Serving template for root route")  # Add this debug message
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit")
def submit_form(
    tickers: str = Form(...),
    period: str = Form(None),
    interval: str = Form(None),
    start_time: str = Form(None),
    end_time: str = Form(None)
):
    ticker_list = [ticker.strip() for ticker in tickers.split(",")]  # Parse comma-separated tickers
    results = []

    for ticker in ticker_list:
        try:
            # Fetch data for the current ticker
            raw_data = fetch_or_get_data(ticker, period, interval, start_time, end_time)

            # Prepare metadata
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

            # Ensure the ticker folder exists
            ticker_folder = os.path.join("data", ticker)
            os.makedirs(ticker_folder, exist_ok=True)

            # Save data to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{ticker}_stock_data_{timestamp}.csv"
            file_path = os.path.join(ticker_folder, file_name)
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Metadata"])
                writer.writerow(["Stock Ticker", metadata["Stock Ticker"]])
                writer.writerow(["Request Timestamp", metadata["Request Timestamp"]])
                writer.writerow(["Start Time", metadata["Filters Applied"]["Start Time"]])
                writer.writerow(["End Time", metadata["Filters Applied"]["End Time"]])
                writer.writerow(["Period", metadata["Filters Applied"]["Period"]])
                writer.writerow(["Interval", metadata["Filters Applied"]["Interval"]])
                writer.writerow([])  # Separation
                writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
                for row in raw_data:
                    writer.writerow([row["Date"], row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])

            # Generate and save PNG chart
            plot_file_name = f"{ticker}_closing_prices_{timestamp}.png"
            plot_file_path = os.path.join(ticker_folder, plot_file_name)
            plot_closing_prices_with_save_location(ticker, raw_data, plot_file_path)

            # Append success result
            results.append({
                "ticker": ticker,
                "csv_path": file_path,
                "plot_path": plot_file_path,
                "status": "success"
            })

        except Exception as e:
            # Append failure result
            results.append({
                "ticker": ticker,
                "error": str(e),
                "status": "failed"
            })

    return {"results": results}
