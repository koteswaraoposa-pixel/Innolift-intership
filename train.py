import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv('Loan_Default.csv')

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# Select Required Columns
df = df[
    [
        'loan_amount',
        'income',
        'Credit_Score',
        'LTV',
        'Status'
    ]
].dropna()

# Features
X = df[
    [
        'loan_amount',
        'income',
        'Credit_Score',
        'LTV'
    ]
]

# Target
y = df['Status']

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
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as loan_model.pkl")