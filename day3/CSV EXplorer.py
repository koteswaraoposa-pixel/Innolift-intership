import os
import pandas as pd

# Load CSV File (use repo-local CSV instead of a machine-specific absolute path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "student-mat.csv")

df = pd.read_csv(CSV_PATH, sep=";")


# Basic Information
print("===== DATASET INFORMATION =====")
print("Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Column Names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# First and Last Records
print("\n===== FIRST 3 ROWS =====")
print(df.head(3))

print("\n===== LAST 3 ROWS =====")
print(df.tail(3))

# Data Types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# Dataset Summary
print("\n===== DATASET SUMMARY =====")
print(df.info())

# Missing Values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Statistical Summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# Internet Access Count
print("\n===== INTERNET ACCESS =====")
print(df["internet"].value_counts())

# Gender Count
print("\n===== GENDER COUNT =====")
print(df["sex"].value_counts())

# School Count
print("\n===== SCHOOL COUNT =====")
print(df["school"].value_counts())

# Average Final Grade
print("\n===== AVERAGE FINAL GRADE (G3) =====")
print(df["G3"].mean())

# Highest Final Grade
print("\n===== HIGHEST FINAL GRADE =====")
print(df["G3"].max())

# Lowest Final Grade
print("\n===== LOWEST FINAL GRADE =====")
print(df["G3"].min())

# Students with Internet
print("\n===== STUDENTS WITH INTERNET =====")
print(df[df["internet"] == "yes"].head())

# Students without Internet
print("\n===== STUDENTS WITHOUT INTERNET =====")
print(df[df["internet"] == "no"].head())

# Sort by Final Grade
print("\n===== TOP 5 STUDENTS BY GRADE =====")
print(df.sort_values(by="G3", ascending=False).head())

# Add Result Column
df["Result"] = df["G3"].apply(
    lambda x: "Pass" if x >= 10 else "Fail"
)

# Result Count
print("\n===== PASS / FAIL COUNT =====")
print(df["Result"].value_counts())

# Display Updated Data
print("\n===== UPDATED DATASET =====")
print(df.head())
