import os
import pickle

import numpy as np


def main():
    model_path = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    # model_obj was saved as a dict: {"beta": ..., "features": [...]}
    beta = model_obj["beta"]
    features = model_obj.get("features", ["loan_amount", "income", "Credit_Score", "LTV"])

    # Example customer in the same feature order used during training.
    # Feature count must match the length of beta minus 1 (intercept).
    # beta length=4 => 3 features + intercept.
    new_customer = np.array([[200000, 60000, 750]], dtype=float)  # [loan_amount, income, Credit_Score]

    X_design = np.column_stack([np.ones((len(new_customer), 1)), new_customer])
    prediction = X_design @ beta

    print(f"Loaded model from: {model_path}")
    print(f"Using feature order: {features}")
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()

