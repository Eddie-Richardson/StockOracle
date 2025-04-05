# app/utils/plot_utils.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_plot_chart(ticker: str, data: list, ticker_folder: str, timestamp: str, prediction: dict = None, model_name: str = None) -> str:
    png_filename = f"{ticker}_closing_prices_{timestamp}.png"
    png_file_path = os.path.join(ticker_folder, model_name, png_filename)

    # Ensure dates are strings for compatibility
    dates = [str(row["Date"]) for row in data]
    closing_prices = [row["Close"] for row in data]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, closing_prices, label=f"{ticker} Closing Prices", color="blue", linewidth=2)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.title(f"{ticker} Closing Prices Over Time\n(Prediction Model: {model_name})", fontsize=14)

    # Format X-axis to display dates without time
    plt.xticks(dates, [d.split(' ')[0] for d in dates], rotation=45, fontsize=10, ha='right')

    plt.legend(fontsize=10)
    plt.grid(alpha=0.5)

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
    return png_file_path
