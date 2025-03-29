# app/utils/plot_utils.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_plot_chart(ticker: str, data: list, ticker_folder: str, timestamp: str, prediction: dict = None) -> str:
    """
    Generate a PNG chart of the ticker's closing prices.
    If prediction data is provided, annotate the chart.
    Returns the PNG file path.
    """
    png_filename = f"{ticker}_closing_prices_{timestamp}.png"
    png_file_path = os.path.join(ticker_folder, png_filename)

    dates = [row["Date"] for row in data]
    closing_prices = [row["Close"] for row in data]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, closing_prices, label=f"{ticker} Closing Prices", color="blue", linewidth=2)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.title(f"{ticker} Closing Prices Over Time", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.5)
    plt.xticks(rotation=45, fontsize=10, ha='right')

    if prediction is not None and "predictions" in prediction and "metrics" in prediction:
        predicted_close = prediction["predictions"][0]
        mae = prediction["metrics"]["MAE"]
        mse = prediction["metrics"]["MSE"]
        annotation_text = f"Predicted: {predicted_close:.2f}\nMAE: {mae:.2f}\nMSE: {mse:.2f}"
        plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=12,
                     backgroundcolor="white", verticalalignment="top")

    plt.tight_layout()
    plt.savefig(png_file_path)
    plt.close()
    print(f"Plot saved successfully at {png_file_path}.")
    return png_file_path
