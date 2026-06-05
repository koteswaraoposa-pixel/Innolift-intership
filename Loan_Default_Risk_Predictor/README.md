# Loan Default Risk Predictor

A machine learning project to predict the risk of loan default based on applicant and loan characteristics.

## Project Structure

```
Loan_Default_Risk_Predictor/
├── data/
│   └── Loan_Default_Risk_Prediction_Dataset.csv
├── models/
│   └── loan_default_model.pkl
├── scripts/
│   ├── train.py
│   ├── model.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Features

The model uses the following features to predict loan default:
- **LoanAmount**: The loan amount requested
- **ApplicantIncome**: The applicant's income
- **Credit_History**: Credit history indicator (0 or 1)
- **Loan_Amount_Term**: The loan term in months

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Navigate to the scripts directory:
   ```bash
   cd scripts
   ```

## Training the Model

To train the Random Forest model:

```bash
python train.py
```

This will:
- Load the dataset from `data/Loan_Default_Risk_Prediction_Dataset.csv`
- Split the data into training and testing sets
- Train a Random Forest classifier
- Display accuracy metrics
- Save the model to `models/loan_default_model.pkl`

## Making Predictions

To make predictions on new loan applications:

```bash
python predict.py
```

The script will prompt you to enter:
- Loan Amount
- Applicant Income
- Credit History (0 or 1)
- Loan Amount Term (in months)

The model will then predict whether the loan is at HIGH RISK or LOW RISK of default.

## Alternative Model

Alternatively, you can use the Logistic Regression model by running:

```bash
python model.py
```

This provides detailed classification metrics and confusion matrix.

## Dataset

The dataset contains 1200 loan records with the following fields:
- Loan_ID
- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Default (Target variable: 0 = No Default, 1 = Default)

## Results

The Random Forest model achieves good accuracy in predicting loan defaults, helping financial institutions assess credit risk effectively.

## Requirements

See `requirements.txt` for all dependencies. Key packages include:
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn
