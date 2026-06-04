import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        raise FileNotFoundError(
            "No CSV found. Expected one of: Loan_Default.csv, archive_extracted/Loan_Default.csv"
        )
    return pd.read_csv(csv_path), csv_path


def ensure_target_and_features(df: pd.DataFrame):
    target_candidates = ["Status", "loan_status", "Loan_Status", "default", "target", "y"]
    target_col = next((c for c in target_candidates if c in df.columns), None)

    desired_features = ["loan_amount", "income", "Credit_Score", "LTV", "property_value"]
    feature_cols = [c for c in desired_features if c in df.columns]

    if target_col is None:
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

    return df, target_col, feature_cols


def train_test_split_indices(n, test_size=0.2, seed=42, stratify_y=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)

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
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta


def linear_regression_predict(X: np.ndarray, beta: np.ndarray):
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design @ beta


def main():
    df, used_path = load_data()
    df = df.dropna().copy()

    df, target_col, feature_cols = ensure_target_and_features(df)

    # Coerce numeric
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    keep_idx = X.index.intersection(y.dropna().index)
    X = X.loc[keep_idx].astype(float)
    y = y.loc[keep_idx].astype(float)

    print(f"Loaded: {used_path}")
    print(f"Using target: {target_col}")
    print(f"Using features: {feature_cols}")
    print(f"Rows used: {len(df)} -> {len(X)} after numeric cleaning")

    train_idx, test_idx = train_test_split_indices(
        n=len(X), test_size=0.2, seed=42, stratify_y=y
    )

    X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

    beta = linear_regression_fit(X_train, y_train)
    pred = linear_regression_predict(X_test, beta)

    # Corrected plot: ensure scales and proper labels, and save to file.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.7)
    min_v = min(float(np.min(y_test)), float(np.min(pred)))
    max_v = max(float(np.max(y_test)), float(np.max(pred)))
    plt.plot([min_v, max_v], [min_v, max_v], color="red", linewidth=2)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.grid(True, alpha=0.25)

    out_path = os.path.join(os.path.dirname(__file__), "actual_vs_predicted.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()

