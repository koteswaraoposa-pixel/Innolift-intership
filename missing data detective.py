import pandas as pd

# Load Dataset
df = pd.read_csv(
    r"C:\Users\supradeep\OneDrive\Desktop\python_internship\day3\Titanic-Dataset.csv"
)

# Dataset Information
print("===== DATASET SHAPE =====")
print(df.shape)

# Check Missing Values
print("\n===== MISSING VALUES BEFORE CLEANING =====")
print(df.isnull().sum())

# Find Numeric Columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Fill Numeric Nulls with Mean
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Find Text Columns
text_cols = df.select_dtypes(include=["object"]).columns

# Fill Text Nulls with 'Unknown'
for col in text_cols:
    df[col] = df[col].fillna("Unknown")

# Verify No Nulls Remain
print("\n===== MISSING VALUES AFTER CLEANING =====")
print(df.isnull().sum())

# Total Remaining Nulls
print("\nTotal Remaining Nulls:")
print(df.isnull().sum().sum())

# Preview Cleaned Data
print("\n===== FIRST 5 ROWS =====")
print(df.head())