# app/core/config.py
import os
from fastapi.templating import Jinja2Templates

# Default parameters
DEFAULT_PERIOD = None
DEFAULT_INTERVAL = None

# Database configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "stock_data.db")

# BASE_DIR is the directory of this config.py, which is StockOracle/app/core
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go to the parent directory (StockOracle/app) and then to templates
TEMPLATES_DIR = os.path.join(BASE_DIR, "..", "templates")
TEMPLATES_DIR = os.path.normpath(TEMPLATES_DIR)  # Optional: Normalize the path

# Initialize Jinja2Templates with the correct path
templates = Jinja2Templates(directory=TEMPLATES_DIR)
