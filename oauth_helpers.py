from __future__ import annotations

import sqlite3
import os


# IMPORTANT: avoid importing `app` here (circular import).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "loan_predictions.db")


def _get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def get_user_by_google_sub(google_sub: str):
    conn = _get_db_connection()
    row = conn.execute(
        "SELECT * FROM Users WHERE google_sub = ?", (google_sub,)
    ).fetchone()
    conn.close()
    return row


def create_user_from_google(google_sub: str, email: str | None, name: str | None):
    # Build a stable username.
    base = (email or name or "google_user").strip().lower()
    if not base:
        base = "google_user"

    conn = _get_db_connection()

    existing_google_user = conn.execute(
        "SELECT id FROM Users WHERE google_sub = ?", (google_sub,)
    ).fetchone()
    if existing_google_user:
        user_id = existing_google_user["id"]
        conn.close()
        return user_id

    username = base
    suffix = 0
    while conn.execute("SELECT id FROM Users WHERE username = ?", (username,)).fetchone():
        suffix += 1
        username = f"{base}_{google_sub[:8]}_{suffix}"

    conn.execute(
        "INSERT INTO Users (username, password_hash, google_sub) VALUES (?, ?, ?)",
        (username, "", google_sub),
    )
    conn.commit()

    user_id = conn.execute(
        "SELECT id FROM Users WHERE google_sub = ?", (google_sub,)
    ).fetchone()["id"]
    conn.close()
    return user_id


