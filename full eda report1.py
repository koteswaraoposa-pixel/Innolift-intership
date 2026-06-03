import pandas as pd

# Function to generate EDA report
def eda_report(df):

    print("\n" + "="*50)
    print("EDA REPORT")
    print("="*50)

    # Shape
    print("\n1. Shape of Dataset:")
    print(df.shape)

    # Columns
    print("\n2. Column Names:")
    print(df.columns.tolist())

    # Data Types
    print("\n3. Data Types:")
    print(df.dtypes)

    # Missing Values
    print("\n4. Missing Values:")
    print(df.isnull().sum())

    # Numeric Summary
    print("\n5. Numeric Columns Summary:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    print(numeric_cols.describe())

    # Object Columns Value Counts
    print("\n6. Categorical Columns Value Counts:")
    object_cols = df.select_dtypes(include=['object'])

    for col in object_cols.columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts())

    print("\n" + "="*50)
    print("END OF REPORT")
    print("="*50)


# ==========================
# TEST ON CSV 1
# ==========================
df1 = pd.read_csv("student-mat.csv", sep=";")

print("\n******** STUDENT DATASET ********")
eda_report(df1)

# ==========================
# TEST ON CSV 2
# ==========================
df2 = pd.read_csv("Titanic-Dataset.csv")

print("\n******** TITANIC DATASET ********")
eda_report(df2)