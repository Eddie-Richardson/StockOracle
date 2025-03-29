StockOracle
StockOracle is a powerful stock data analysis and prediction tool designed to streamline financial insights for traders, analysts, and enthusiasts. Using historical stock data fetched via APIs like yfinance, StockOracle allows users to visualize trends, filter stock information, and apply machine learning models for future predictions.

Features
Fetch historical stock data using customizable parameters (e.g., time period, interval).

Visualize closing price trends with detailed plots.

Save data in CSV format with metadata and optional predictions.

Predict future closing prices using modular machine learning models such as Linear Regression.

Installation
Clone the repository:

bash
git clone https://github.com/Eddie-Richardson/StockOracle.git
Navigate into the project directory:

bash
cd StockOracle
Create a virtual environment (optional but recommended):

bash
python -m venv .venv
Activate the virtual environment:

On Windows:

bash
.venv\Scripts\activate
On macOS/Linux:

bash
source .venv/bin/activate
Install dependencies:

bash
pip install -r requirements.txt
Usage
Start the server:

bash
uvicorn main:app --reload
Open the browser and navigate to:

http://127.0.0.1:8000
Use the interactive form to input tickers, define parameters, and select prediction models.

Folder Structure
StockOracle/
├── app/
│   ├── api/
│   │   ├── stock.py         # Endpoints for stock data visualization and filtering
│   │   └── submit.py        # Endpoint for bulk form submission
│   ├── core/
│   │   ├── config.py        # Configuration settings
│   │   └── db.py            # Database initialization logic
│   ├── models/
│   │   ├── linear_model.py  # Linear regression implementation
│   │   ├── ml_models.py     # ML model aggregator
│   │   └── db.models.py     # Database model
│   ├── utils/
│   │   ├── data_utils.py    # Data fetching and saving
│   │   ├── plot_utils.py    # Plot generation
│   │   ├── prediction_utils.py  # Prediction management
│   │   └── normalization.py     # Normalizes Data
│   └── templates/           # HTML templates (index.html, results.html)
├── static/                  # Static assets (CSS, JS)
├── data/                    # Data storage (SQLite DB, CSV files)
└── main.py                  # Entry point

Contributing
Feel free to contribute! Fork the repository, make your changes, and open a pull request.

Future Enhancements
Support for additional machine learning models like ARIMA and LSTM.

Enhanced UI with dynamic filtering options.

Deployment instructions for hosting on platforms like Heroku or AWS.