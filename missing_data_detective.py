import os
import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Load Titanic dataset from repo-local path.

    Tries Titanic-Dataset.csv next to the script first; falls back to the old absolute path.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base_dir, "Titanic-Dataset.csv")

    if os.path.exists(candidate):
        return pd.read_csv(candidate)

    # Fallback for previous environment (kept to avoid breaking if file location differs)
    return pd.read_csv(r"C:\Users\supradeep\OneDrive\Desktop\python_internship\day3\Titanic-Dataset.csv")


def main() -> None:
    df = load_dataset()

    print("===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== MISSING VALUES BEFORE CLEANING =====")
    print(df.isnull().sum())

    # Numeric columns: fill with mean
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        mean_val = df[col].mean()
        if pd.notna(mean_val):
            df[col] = df[col].fillna(mean_val)

    # Text columns: fill with 'Unknown'
    # Use include=["object", "string"] to avoid pandas dtype warnings.
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].fillna("Unknown")


    print("\n===== MISSING VALUES AFTER CLEANING =====")
    print(df.isnull().sum())

    total_remaining = int(df.isnull().sum().sum())
    print("\nTotal Remaining Nulls:")
    print(total_remaining)

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())


if __name__ == "__main__":
    main()

