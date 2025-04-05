# app/core/db.py
import sqlite3
from app.core.config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_timestamp Date,
            predicted_close REAL,
            mae REAL,
            mse REAL,
            model TEXT
        )
    """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            model TEXT NOT NULL,
            predicted_close REAL,
            actual_close REAL,
            error REAL
        )
        """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS linear_training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            X REAL,
            y REAL
        )
        """)

    conn.commit()
    conn.close()
