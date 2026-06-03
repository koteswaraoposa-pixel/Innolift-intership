import os
import pandas as pd


# EDA Function
def eda_report(df, name):

    print("\n" + "=" * 60)
    print(f"EDA REPORT : {name}")
    print("=" * 60)

    # Shape
    print("\n1. DATASET SHAPE")
    print(df.shape)

    # Columns
    print("\n2. COLUMN NAMES")
    print(df.columns.tolist())

    # Data Types
    print("\n3. DATA TYPES")
    print(df.dtypes)

    # Missing Values
    print("\n4. NULL VALUES")
    print(df.isnull().sum())

    # Numeric Columns
    print("\n5. NUMERIC COLUMN SUMMARY")
    print(df.describe())

    # Object Columns
    print("\n6. CATEGORICAL COLUMN ANALYSIS")

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        print("\n" + "-" * 40)
        print(f"Column: {col}")
        print(df[col].value_counts())

    # Duplicate Rows
    print("\n7. DUPLICATE ROWS")
    print(df.duplicated().sum())

    # Memory Usage
    print("\n8. MEMORY USAGE")
    print(df.memory_usage(deep=True))

    print("\nEDA COMPLETED")
    print("=" * 60)


# ----------------------------
# DATASET 1 : Student Dataset
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Student dataset (repo-local)
student_df = pd.read_csv(os.path.join(BASE_DIR, "student-mat.csv"), sep=";")
eda_report(student_df, "Student Performance Dataset")


# ----------------------------
# DATASET 2 : Titanic Dataset
# ----------------------------

titanic_df = pd.read_csv(os.path.join(BASE_DIR, "Titanic-Dataset.csv"))
eda_report(titanic_df, "Titanic Dataset")
