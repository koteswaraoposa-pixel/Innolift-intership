import csv
import os
import sqlite3
from statistics import mean

from flask import Flask, flash, redirect, render_template, request


app = Flask(__name__)
app.secret_key = "dev-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "loan_predictions.db")
DATASET_PATH = os.path.join(DATA_DIR, "loan_default_dataset.csv")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_table():
    conn = get_db_connection()
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
    conn.close()


def load_dataset_rows():
    if not os.path.exists(DATASET_PATH):
        return []

    with open(DATASET_PATH, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


DATASET_ROWS = load_dataset_rows()


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dataset_summary():
    if not DATASET_ROWS:
        return {
            "total_records": 0,
            "default_records": 0,
            "default_rate": 0,
            "avg_income": 0,
            "avg_loan_amount": 0,
        }

    total_records = len(DATASET_ROWS)
    default_records = sum(1 for row in DATASET_ROWS if row.get("Loan_Default") == "1")
    incomes = [to_float(row.get("ApplicantIncome")) for row in DATASET_ROWS]
    loan_amounts = [to_float(row.get("LoanAmount")) for row in DATASET_ROWS]

    return {
        "total_records": total_records,
        "default_records": default_records,
        "default_rate": round((default_records / total_records) * 100, 2),
        "avg_income": round(mean(incomes), 2),
        "avg_loan_amount": round(mean(loan_amounts), 2),
    }


SUMMARY = dataset_summary()


def default_rate_for(field, value):
    matches = [row for row in DATASET_ROWS if row.get(field) == value]
    if not matches:
        return SUMMARY["default_rate"] / 100 if SUMMARY["total_records"] else 0.25

    defaults = sum(1 for row in matches if row.get("Loan_Default") == "1")
    return defaults / len(matches)


def predict_default(form_data):
    applicant_income = to_float(form_data["applicant_income"])
    coapplicant_income = to_float(form_data["coapplicant_income"])
    loan_amount = to_float(form_data["loan_amount"])
    loan_amount_term = to_float(form_data["loan_amount_term"])
    total_income = max(applicant_income + coapplicant_income, 1)

    risk = SUMMARY["default_rate"] / 100 if SUMMARY["total_records"] else 0.25
    risk += default_rate_for("Credit_History", form_data["credit_history"]) * 0.35
    risk += default_rate_for("Property_Area", form_data["property_area"]) * 0.12
    risk += default_rate_for("Education", form_data["education"]) * 0.08
    risk += default_rate_for("Self_Employed", form_data["self_employed"]) * 0.08

    monthly_loan_pressure = loan_amount / total_income
    if form_data["credit_history"] == "0":
        risk += 0.26
    if monthly_loan_pressure > 0.035:
        risk += 0.12
    elif monthly_loan_pressure > 0.025:
        risk += 0.07
    if loan_amount_term <= 180:
        risk += 0.06
    if applicant_income < SUMMARY["avg_income"]:
        risk += 0.05
    if coapplicant_income == 0:
        risk += 0.04

    risk_score = max(0, min(round(risk * 100, 2), 100))

    if risk_score >= 55:
        return risk_score, "High Risk", "Likely to Default"
    if risk_score >= 35:
        return risk_score, "Medium Risk", "Needs Manual Review"
    return risk_score, "Low Risk", "Low Default Risk"


def get_form_data():
    return {
        "loan_id": request.form.get("loan_id", "").strip(),
        "gender": request.form.get("gender", "").strip(),
        "married": request.form.get("married", "").strip(),
        "dependents": request.form.get("dependents", "").strip(),
        "education": request.form.get("education", "").strip(),
        "self_employed": request.form.get("self_employed", "").strip(),
        "applicant_income": request.form.get("applicant_income", "").strip(),
        "coapplicant_income": request.form.get("coapplicant_income", "").strip(),
        "loan_amount": request.form.get("loan_amount", "").strip(),
        "loan_amount_term": request.form.get("loan_amount_term", "").strip(),
        "credit_history": request.form.get("credit_history", "").strip(),
        "property_area": request.form.get("property_area", "").strip(),
    }


@app.route("/")
def home():
    return render_template("index.html", summary=SUMMARY)


@app.route("/predict", methods=["GET", "POST"])
@app.route("/register", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = get_form_data()
        missing = [label.replace("_", " ").title() for label, value in data.items() if not value]
        if missing:
            flash(f"Missing fields: {', '.join(missing)}", "error")
            return render_template("register.html", data=data), 400

        risk_score, risk_level, prediction = predict_default(data)

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO LoanPredictions
            (loan_id, gender, married, dependents, education, self_employed,
             applicant_income, coapplicant_income, loan_amount, loan_amount_term,
             credit_history, property_area, risk_score, risk_level, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["loan_id"],
                data["gender"],
                data["married"],
                data["dependents"],
                data["education"],
                data["self_employed"],
                to_float(data["applicant_income"]),
                to_float(data["coapplicant_income"]),
                to_float(data["loan_amount"]),
                to_float(data["loan_amount_term"]),
                data["credit_history"],
                data["property_area"],
                risk_score,
                risk_level,
                prediction,
            ),
        )
        conn.commit()
        conn.close()

        flash(f"Prediction saved: {prediction} ({risk_score}% risk)", "success")
        return render_template(
            "register.html",
            data=data,
            risk_score=risk_score,
            risk_level=risk_level,
            prediction=prediction,
        )

    return render_template("register.html", data={})


@app.route("/predictions")
@app.route("/students")
def predictions():
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT loan_id, applicant_income, loan_amount, credit_history,
               property_area, risk_score, risk_level, prediction
        FROM LoanPredictions
        ORDER BY id DESC
        """
    ).fetchall()
    conn.close()

    return render_template("students.html", predictions=rows)


@app.route("/about")
def about():
    return render_template("about.html", summary=SUMMARY)


create_table()


if __name__ == "__main__":
    create_table()
    app.run(debug=True)
