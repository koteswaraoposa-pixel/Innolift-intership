import os
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


def pick_target_column(df: pd.DataFrame):
    candidates = [
        "Loan_Default",
        "loan_default",
        "Default",
        "default",
        "target",
        "Target",
        "label",
        "Label",
        "is_default",
        "Is_Default",
        "default_flag",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # Heuristic: prefer binary numeric columns with 0/1 values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna().unique()
            if len(vals) <= 3 and set(map(int, vals)) <= {0, 1}:
                return col

    raise ValueError(
        "Could not determine target column automatically. "
        f"Columns found: {list(df.columns)}"
    )


def main():
    dataset_path = "Loan_Default_Risk_Prediction_Dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. "
            "Place your CSV in the project root or update dataset_path."
        )

    df = pd.read_csv(dataset_path)

    target_col = pick_target_column(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Ensure y is 0/1
    if y.dtype == "object":
        # try to map common yes/no defaults
        y_lower = y.astype(str).str.lower().str.strip()
        mapping = {
            "yes": 1,
            "y": 1,
            "true": 1,
            "1": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
        }
        y = y_lower.map(mapping)

    y = pd.to_numeric(y, errors="coerce")
    if y.isna().any():
        raise ValueError("Target column contains values that could not be converted to 0/1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = [
        c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
    ]
    categorical_features = [
        c for c in X.columns if c not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        n_jobs=None,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=== Model Training: Logistic Regression ===")
    print(f"Dataset: {dataset_path}")
    print(f"Target column: {target_col}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("")
    print(f"Accuracy: {acc:.4f}")
    print("")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()

