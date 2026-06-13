import os
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
            "Titanic-Dataset.csv",
            "student-mat.csv",
        ]
    )
    if not csv_path:
        raise FileNotFoundError(
            "No CSV found. Expected one of: "
            "Loan_Default.csv, archive_extracted/Loan_Default.csv, "
            "Titanic-Dataset.csv, student-mat.csv"
        )
    return pd.read_csv(csv_path), csv_path


def ensure_target_and_features(df: pd.DataFrame):
    # Try common target names
    target_candidates = ["Status", "loan_status", "Loan_Status", "default", "target", "y"]
    target_col = next((c for c in target_candidates if c in df.columns), None)

    desired_features = ["loan_amount", "income", "Credit_Score", "LTV", "property_value"]
    feature_cols = [c for c in desired_features if c in df.columns]

    if not feature_cols:
        raise KeyError(
            "None of the expected feature columns were found. "
            f"Expected at least one of: {desired_features}. Available columns: {list(df.columns)}"
        )

    # If there is no target column, generate one from Credit_Score so the model can run.
    if target_col is None:
        if "Credit_Score" in df.columns:
            thresh = pd.to_numeric(df["Credit_Score"], errors="coerce").median()
            df = df.copy()
            target_col = "__default_generated__"
            df[target_col] = (pd.to_numeric(df["Credit_Score"], errors="coerce") < thresh).astype(int)
        else:
            raise KeyError(
                "Could not find a target column and could not generate one. "
                f"Available columns: {list(df.columns)}"
            )

    return df, target_col, feature_cols


def train_test_split_indices(n, test_size=0.2, seed=42, stratify_y=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)

    # Optional simple stratification (no sklearn)
    if stratify_y is not None and pd.Series(stratify_y).nunique(dropna=False) > 1:
        y = np.asarray(stratify_y)
        classes = pd.Series(y).value_counts(dropna=False).index.tolist()
        test_indices = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            m = int(round(len(cls_idx) * test_size))
            m = max(1, m) if len(cls_idx) > 1 else 0
            test_indices.extend(cls_idx[:m])
        test_indices = np.array(sorted(set(test_indices)), dtype=int)
    else:
        rng.shuffle(indices)
        m = int(round(n * test_size))
        m = max(1, m) if n > 1 else 0
        test_indices = indices[:m]

    train_indices = np.setdiff1d(np.arange(n), test_indices, assume_unique=False)
    return train_indices, test_indices


def linear_regression_fit(X: np.ndarray, y: np.ndarray):
    """Closed-form OLS with intercept using numpy."""
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta


def linear_regression_predict(X: np.ndarray, beta: np.ndarray):
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ beta


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def main():
    df, used_path = load_data()
    df = df.dropna().copy()

    df, target_col, feature_cols = ensure_target_and_features(df)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    y = df.loc[X.index, target_col]

    # ensure aligned indexes
    X = X.astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)

    # Align after numeric coercion
    keep_idx = X.index.intersection(y.dropna().index)
    X = X.loc[keep_idx]
    y = y.loc[keep_idx]

    print(f"Loaded: {used_path}")
    print(f"Using target: {target_col}")
    print(f"Using features: {feature_cols}")
    print(f"Rows used: {len(df)} -> {len(X)} after numeric cleaning")

    # Run quick evaluation for a single split (first regression)
    test_size = 0.2
    train_idx, test_idx = train_test_split_indices(
        n=len(X), test_size=test_size, seed=42, stratify_y=y
    )

    X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

    beta = linear_regression_fit(X_train, y_train)
    preds = linear_regression_predict(X_test, beta)

    print("\n=== First Regression (Linear Regression OLS) ===")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("Coefficients (intercept first):", beta.tolist())
    print(f"RMSE: {rmse(y_test, preds):.6f}")
    print(f"R2: {r2_score(y_test, preds):.6f}")


if __name__ == "__main__":
    main()
