import os

import numpy as np
import pandas as pd


def load_data():
    # Prefer repo-local file
    for p in [
        "Loan_Default.csv",
        os.path.join("archive_extracted", "Loan_Default.csv"),
    ]:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Loan_Default.csv not found in repo")


def detect_target_col(df: pd.DataFrame):
    for c in ["Status", "loan_status", "Loan_Status", "default", "target", "y"]:
        if c in df.columns:
            return c
    return None


def main():
    df = load_data()

    target_col = detect_target_col(df)
    if target_col is None:
        # Create a usable target if missing
        if "Credit_Score" not in df.columns:
            raise KeyError(
                "No target column found and cannot generate one (missing Credit_Score)."
            )
        thresh = pd.to_numeric(df["Credit_Score"], errors="coerce").median()
        df = df.copy()
        target_col = "__default_generated__"
        df[target_col] = (
            pd.to_numeric(df["Credit_Score"], errors="coerce") < thresh
        ).astype(int)

    # Use numeric coercion for feature/target
    features = ["loan_amount", "income", "Credit_Score", "LTV"]

    results = []

    for feature in features:
        if feature not in df.columns:
            continue

        tmp = df[[feature, target_col]].copy()
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
        tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
        tmp = tmp.dropna()

        if len(tmp) < 5:
            results.append((feature, None, len(tmp)))
            continue

        X = tmp[[feature]].values
        y = tmp[target_col].values

        # Simple deterministic split (avoid sklearn dependency)
        rng = np.random.default_rng(42)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(X) * 0.2)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Closed-form linear regression with intercept
        X_design = np.column_stack([np.ones(len(X_train)), X_train])
        beta, *_ = np.linalg.lstsq(X_design, y_train, rcond=None)

        X_test_design = np.column_stack([np.ones(len(X_test)), X_test])
        pred = X_test_design @ beta

        rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))

        results.append((feature, rmse, len(tmp)))

    for feature, rmse, n in results:
        if rmse is None:
            print(f"{feature}: insufficient data (n={n})")
        else:
            print(f"{feature} RMSE = {rmse:.6f} (n={n})")


if __name__ == "__main__":
    main()

