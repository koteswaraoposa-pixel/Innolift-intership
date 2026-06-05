import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load model
model_path = '../models/loan_default_model.pkl'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Please train the model first using train.py")
    sys.exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("=== Loan Default Risk Predictor ===")
print("Enter Customer Details\n")

try:
    loan_amount = float(input("Loan Amount: "))
    applicant_income = float(input("Applicant Income: "))
    credit_history = float(input("Credit History (0 or 1): "))
    loan_amount_term = float(input("Loan Amount Term (months): "))

    prediction = model.predict(
        [[loan_amount, applicant_income, credit_history, loan_amount_term]]
    )

    if prediction[0] == 1:
        print("\n*** Prediction: HIGH RISK - Likely to Default ***")
    else:
        print("\n*** Prediction: LOW RISK - Unlikely to Default ***")
except ValueError:
    print("Error: Please enter valid numerical values.")
except Exception as e:
    print(f"Error during prediction: {str(e)}")