import os
import pickle

import numpy as np
import pandas as pd


def find_first_csv(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def load_data():
    csv_path = find_first_csv(
        [
            "Loan_Default.csv",
            os.path.join("archive_extracted", "Loan_Default.csv"),
        ]
    )
    if not csv_path:
        raise FileNotFoundError("Loan_Default.csv not found.")
    return pd.read_csv(csv_path)


def train_closed_form_ols(X: np.ndarray, y: np.ndarray):
    # add intercept
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta


def main():
    df = load_data()

    # Target/feature columns can vary; handle missing ones gracefully.
    target_candidates = ["Status", "loan_status", "Loan_Status", "default", "target", "y"]
    target_col = next((c for c in target_candidates if c in df.columns), None)

    if target_col is None:
        # Generate a binary target from Credit_Score (keeps script runnable)
        if "Credit_Score" not in df.columns:
            raise KeyError("No target column found and cannot generate one (missing Credit_Score).")
        thresh = pd.to_numeric(df["Credit_Score"], errors="coerce").median()
        df = df.copy()
        target_col = "__default_generated__"
        df[target_col] = (pd.to_numeric(df["Credit_Score"], errors="coerce") < thresh).astype(int)

    feature_candidates = ["loan_amount", "income", "Credit_Score", "LTV", "property_value"]
    feature_cols = [c for c in feature_candidates if c in df.columns]

    if not feature_cols:
        raise KeyError(f"No usable features found. Available columns: {list(df.columns)}")

    df = df[[target_col] + feature_cols].dropna().copy()

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")


    keep = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep].values.astype(float)
    y = y.loc[keep].values.astype(float)

    # deterministic split (avoid sklearn)
    rng = np.random.default_rng(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(X) * 0.2)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]

    beta = train_closed_form_ols(X_train, y_train)

    model_obj = {
        "beta": beta,  # intercept first
        "features": ["loan_amount", "income", "Credit_Score", "LTV"],
    }

    out_path = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model_obj, f)

    print(f"Model saved to: {out_path}")
    print(f"Coefficients (intercept first): {beta.tolist()}")


if __name__ == "__main__":
    main()

