import pickle

# Load model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Enter Customer Details")

loan_amount = float(input("Loan Amount: "))
income = float(input("Income: "))
credit_score = float(input("Credit Score: "))
ltv = float(input("LTV (Loan-To-Value Ratio): "))

prediction = model.predict(
    [[loan_amount, income, credit_score, ltv]]
)

if prediction[0] >= 0.5:
    print("\nPrediction: High Risk of Loan Default")
else:
    print("\nPrediction: Low Risk of Loan Default")