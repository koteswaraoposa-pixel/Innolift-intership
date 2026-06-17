import os
import sqlite3


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "loan_predictions.db")
DATASET_PATH = os.path.join(DATA_DIR, "loan_default_dataset.csv")


print("DB_PATH:", DB_PATH)
print("DB exists:", os.path.exists(DB_PATH))
print("Dataset exists:", os.path.exists(DATASET_PATH))

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

conn.execute(
    """
    CREATE TABLE IF NOT EXISTS LoanPredictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        loan_id TEXT,
        gender TEXT,
        married TEXT,
        dependents TEXT,
        education TEXT,
        self_employed TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount REAL,
        loan_amount_term REAL,
        credit_history TEXT,
        property_area TEXT,
        risk_score REAL,
        risk_level TEXT,
        prediction TEXT
    )
    """
)
conn.commit()

rows = conn.execute("SELECT COUNT(*) AS c FROM LoanPredictions").fetchone()
print("Loan prediction rows:", rows["c"] if rows else None)

conn.close()
print("OK")