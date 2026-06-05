import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv('../data/Loan_Default_Risk_Prediction_Dataset.csv')

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# Select Required Columns
df = df[
    [
        'LoanAmount',
        'ApplicantIncome',
        'Credit_History',
        'Loan_Amount_Term',
        'Loan_Default'
    ]
].dropna()

# Features
X = df[
    [
        'LoanAmount',
        'ApplicantIncome',
        'Credit_History',
        'Loan_Amount_Term'
    ]
]

# Target
y = df['Loan_Default']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Save Model
with open('../models/loan_default_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as loan_default_model.pkl")