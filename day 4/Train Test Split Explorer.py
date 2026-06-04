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

    df = pd.read_csv(csv_path)
    return df, csv_path


def resolve_target_and_features(df: pd.DataFrame):
    # Preferred target columns
    target_candidates = ["Status", "loan_status", "Loan_Status", "default", "target", "y"]
    target_col = next((c for c in target_candidates if c in df.columns), None)

    desired_features = ["loan_amount", "income", "Credit_Score", "LTV"]
    feature_cols = [c for c in desired_features if c in df.columns]

    if not feature_cols:
        raise KeyError(
            "None of the expected feature columns were found. "
            f"Expected at least one of: {desired_features}. Available columns: {list(df.columns)}"
        )

    # If loan-default target column isn't present, create a usable one from Credit_Score.
    # This keeps the script runnable with the sample/minimal Loan_Default.csv shipped in the repo.
    if target_col is None:
        if "Credit_Score" in df.columns:
            # Simple binary label: default=1 if credit score < median, else 0
            thresh = pd.to_numeric(df["Credit_Score"], errors="coerce").median()
            target_col = "__default_generated__"
            df = df.copy()
            df[target_col] = (pd.to_numeric(df["Credit_Score"], errors="coerce") < thresh).astype(int)
        else:
            raise KeyError(
                "Could not find a target column and could not generate one. "
                f"Available columns: {list(df.columns)}"
            )

    return df, target_col, feature_cols


def stratified_split_indices(y, test_size, seed=42):
    """Return train_idx, test_idx using simple stratification without sklearn."""
    rng = np.random.default_rng(seed)

    y_arr = np.asarray(y)
    classes = pd.Series(y_arr).value_counts(dropna=False).index.tolist()

    test_indices = []
    for cls in classes:
        cls_idx = np.where(y_arr == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_size))
        n_test = max(1, n_test) if len(cls_idx) > 1 else 0
        test_indices.extend(cls_idx[:n_test])

    test_indices = np.array(sorted(set(test_indices)), dtype=int)

    all_indices = np.arange(len(y_arr))
    train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=False)
    return train_indices, test_indices


def random_split_indices(n, test_size, seed=42, stratify_y=None):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)

    if stratify_y is not None and pd.Series(stratify_y).nunique(dropna=False) > 1:
        return stratified_split_indices(stratify_y, test_size=test_size, seed=seed)

    rng.shuffle(indices)
    n_test = int(round(n * test_size))
    n_test = max(1, n_test) if n > 1 else 0

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx


def main():
    df, used_path = load_data()
    df = df.dropna()

    df, target_col, feature_cols = resolve_target_and_features(df)

    X = df[feature_cols].reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)

    print(f"Loaded: {used_path}")
    print(f"Using target: {target_col}")
    print(f"Using features: {feature_cols}")
    print(f"Rows after dropna: {len(df)}")

    for size in [0.1, 0.2, 0.3]:
        train_idx, test_idx = random_split_indices(
            n=len(df),
            test_size=size,
            seed=42,
            stratify_y=y,
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        print("\nTest Size:", size)
        print("Train:", len(X_train))
        print("Test:", len(X_test))
        print("Train target distribution:")
        print(y_train.value_counts(dropna=False))
        print("Test target distribution:")
        print(y_test.value_counts(dropna=False))


if __name__ == "__main__":
    main()
