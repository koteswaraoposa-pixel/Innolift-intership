import os
import pandas as pd

# Try to import matplotlib. If it's not installed, fail with a clear message.
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "matplotlib is not installed. Install it with: pip install matplotlib"
    )

# Prefer Loan_Default.csv if available, otherwise fall back to Titanic-Dataset.csv.
CSV_PATHS = ["Loan_Default.csv", "Titanic-Dataset.csv"]
CSV_PATH = next((p for p in CSV_PATHS if os.path.exists(p)), None)

if CSV_PATH is None:
    raise FileNotFoundError(
        "None of these CSV files were found in the project folder: "
        f"{', '.join(CSV_PATHS)}\n"
        "Place the required CSV in this folder or update CSV_PATHS in the script."
    )

df = pd.read_csv(CSV_PATH)

print(f"Loaded: {CSV_PATH}")

# --- Charts (only create what the dataset supports) ---
# 1 Bar Chart: Average Loan Amount by Region
if {"Region", "loan_amount"}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    df.groupby("Region")["loan_amount"].mean().plot(kind="bar")
    plt.title("Average Loan Amount by Region")
    plt.tight_layout()
    plt.savefig("bar_chart.png")

# 2 Scatter Plot: income vs loan_amount
if {"income", "loan_amount"}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["income"], df["loan_amount"])
    plt.xlabel("Income")
    plt.ylabel("Loan Amount")
    plt.title("Income vs Loan Amount")
    plt.tight_layout()
    plt.savefig("scatter_plot.png")

# 3 Histogram: Credit_Score distribution
if "Credit_Score" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df["Credit_Score"].dropna(), bins=20)
    plt.title("Credit Score Distribution")
    plt.tight_layout()
    plt.savefig("histogram.png")

# 4 Line Chart: Average loan_amount by year
if {"year", "loan_amount"}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    avg = df.groupby("year")["loan_amount"].mean()
    plt.plot(avg.index, avg.values)
    plt.title("Average Loan Amount by Year")
    plt.tight_layout()
    plt.savefig("line_chart.png")

print("Done. Check generated PNG files in this folder.")

