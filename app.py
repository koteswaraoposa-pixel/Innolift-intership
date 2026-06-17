import csv
import os
import sqlite3
from statistics import mean
from functools import wraps

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from models.model import LoanDefaultModel
from auth_google import init_google_oauth
from google_auth_routes import register_google_routes




app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "loan_predictions.db")
DATASET_PATH = os.path.join(DATA_DIR, "loan_default_dataset.csv")
LEGACY_DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "Loan_Default_dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "final_model.pkl")

try:
    ml_model = LoanDefaultModel(MODEL_PATH)
except Exception:
    ml_model = None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15)
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
            confidence REAL,
            risk_level TEXT,
            prediction TEXT
        )
        """
    )

    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(LoanPredictions)").fetchall()
    }
    expected_columns = {
        "risk_score": "REAL",
        "confidence": "REAL",
        "risk_level": "TEXT",
        "prediction": "TEXT",
    }
    for column_name, column_type in expected_columns.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE LoanPredictions ADD COLUMN {column_name} {column_type}"
            )

    conn.commit()
    conn.close()


# --- Auth tables ---

def init_auth_tables():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            google_sub TEXT UNIQUE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()

    # SQLite migration for older DBs
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(Users)").fetchall()}
    if "google_sub" not in existing:
        # SQLite cannot add UNIQUE constraints via ALTER TABLE in older versions.
        conn.execute("ALTER TABLE Users ADD COLUMN google_sub TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub ON Users(google_sub)")
    if "password_hash" not in existing:
        conn.execute("ALTER TABLE Users ADD COLUMN password_hash TEXT")


    conn.commit()
    conn.close()



def get_user_by_username(username: str):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM Users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row


def create_user(username: str, password: str) -> int:
    conn = get_db_connection()
    password_hash = generate_password_hash(password)
    conn.execute(
        "INSERT INTO Users (username, password_hash) VALUES (?, ?)",
        (username, password_hash),
    )
    conn.commit()
    user_id = conn.execute(
        "SELECT id FROM Users WHERE username = ?", (username,)
    ).fetchone()["id"]
    conn.close()
    return user_id


def is_logged_in():
    return session.get("user_id") is not None


def login_required(view_fn):
    @wraps(view_fn)
    def wrapped(*args, **kwargs):
        if not is_logged_in():
            return redirect(url_for("login", next=request.path))
        return view_fn(*args, **kwargs)

    return wrapped


# --- App data/model ---

def load_dataset_rows():
    dataset_path = DATASET_PATH if os.path.exists(DATASET_PATH) else LEGACY_DATASET_PATH
    if not os.path.exists(dataset_path):
        return []
    with open(dataset_path, newline="", encoding="utf-8") as csv_file:
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


# --- Auth routes ---


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        if get_user_by_username(username):
            flash("Username already exists. Please login.", "error")
            return render_template("signup.html")

        try:
            user_id = create_user(username, password)
        except sqlite3.IntegrityError:
            flash("Username already exists. Please login.", "error")
            return render_template("signup.html")

        session["user_id"] = user_id
        flash("Account created. Welcome!", "success")

        next_url = request.args.get("next") or url_for("predict")
        return redirect(next_url)

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        user = get_user_by_username(username)
        if not user:
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        if not user["password_hash"] or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        session["user_id"] = user["id"]
        flash("Login successful.", "success")

        next_url = request.args.get("next") or url_for("predict")
        return redirect(next_url)

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))


# --- Protected app routes ---


@app.route("/")
@login_required
def home():
    return render_template("index.html", summary=SUMMARY)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        data = get_form_data()
        missing = [label.replace("_", " ").title() for label, value in data.items() if not value]
        if missing:
            flash(f"Missing fields: {', '.join(missing)}", "error")
            return render_template("register.html", data=data), 400

        if ml_model is not None:
            prediction, confidence, risk_level = ml_model.predict(data)
            risk_score = round(confidence, 2) if confidence is not None and confidence >= 0 else None
        else:
            risk_score, risk_level, prediction = predict_default(data)
            confidence = risk_score

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO LoanPredictions
            (loan_id, gender, married, dependents, education, self_employed,
             applicant_income, coapplicant_income, loan_amount, loan_amount_term,
             credit_history, property_area, risk_score, confidence, risk_level, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                confidence if confidence is not None and confidence >= 0 else None,
                risk_level,
                prediction,
            ),
        )
        conn.commit()
        conn.close()

        flash(f"Prediction saved: {prediction} ({risk_score}% risk)", "success")
        return render_template(
            "loan_prediction.html",
            data=data,
            risk_score=risk_score,
            confidence=confidence,
            risk_level=risk_level,
            prediction=prediction,
        )

    return render_template("register.html", data={})


@app.route("/predictions")
@login_required
def predictions():
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT loan_id, applicant_income, loan_amount, credit_history,
               property_area, risk_score, confidence, risk_level, prediction
        FROM LoanPredictions
        ORDER BY id DESC
        """
    ).fetchall()
    conn.close()
    return render_template("students.html", predictions=rows)


@app.route("/about")
@login_required
def about():
    return render_template("about.html", summary=SUMMARY)


@app.route("/flow")
@login_required
def flow():
    return render_template("flowchart.html")


@app.errorhandler(404)
def not_found_error(error):
    flash("Page not found. Please use the navigation links.", "error")
    return redirect(url_for("login" if not is_logged_in() else "home"))


@app.errorhandler(500)
def internal_error(error):
    app.logger.exception("Unhandled server error: %s", error)
    flash("Something went wrong. Please try again.", "error")
    return redirect(url_for("login" if not is_logged_in() else "home"))



# Init
create_table()
init_auth_tables()

# Google OAuth setup (Authlib)
oauth = init_google_oauth(app)
register_google_routes(app, oauth)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

