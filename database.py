import os
import sqlite3
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database", "bank.db")


def create_database():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            description TEXT,
            debit REAL,
            credit REAL,
            balance REAL,
            category TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_transactions():
    create_database()

    conn = sqlite3.connect(DATABASE_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM transactions",
        conn
    )

    conn.close()

    return df