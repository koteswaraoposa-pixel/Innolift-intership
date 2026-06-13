import os
import pickle
import warnings
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

try:
    import joblib
except ImportError:  # pragma: no cover - pickle fallback still works.
    joblib = None


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class LoanDefaultModel:
    """Wrapper around the serialized sklearn-like model.

    Expected to support either:
    - predict_proba(X) -> (n_samples, n_classes)
    - or only predict(X)

    Confidence is reported as:
    - max class probability (percent) if predict_proba exists
    - otherwise set to None
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load(model_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at: {path}. "
                "Place your trained '.pkl' model there or update the path in app.py."
            )
        if joblib is not None:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                    return joblib.load(path)
            except Exception:
                pass

        with open(path, "rb") as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                return pickle.load(f)

    def _encode_form_data(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        return {
            "Gender": {"Female": 0, "Male": 1}.get(form_data.get("gender"), 0),
            "Married": {"No": 0, "Yes": 1}.get(form_data.get("married"), 0),
            "Dependents": int(_to_float(form_data.get("dependents"))),
            "Education": {"Graduate": 0, "Not Graduate": 1}.get(form_data.get("education"), 0),
            "Self_Employed": {"No": 0, "Yes": 1}.get(form_data.get("self_employed"), 0),
            "ApplicantIncome": _to_float(form_data.get("applicant_income")),
            "CoapplicantIncome": _to_float(form_data.get("coapplicant_income")),
            "LoanAmount": _to_float(form_data.get("loan_amount")),
            "Loan_Amount_Term": _to_float(form_data.get("loan_amount_term")),
            "Credit_History": _to_float(form_data.get("credit_history")),
            "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}.get(
                form_data.get("property_area"), 0
            ),
        }

    def _build_feature_vector(self, form_data: Dict[str, str]):
        encoded = self._encode_form_data(form_data)
        feature_order: List[str] = list(
            getattr(
                self.model,
                "feature_names_in_",
                [
                    "Gender",
                    "Married",
                    "Dependents",
                    "Education",
                    "Self_Employed",
                    "ApplicantIncome",
                    "CoapplicantIncome",
                    "LoanAmount",
                    "Loan_Amount_Term",
                    "Credit_History",
                    "Property_Area",
                ],
            )
        )
        return pd.DataFrame([{name: encoded[name] for name in feature_order}], columns=feature_order)

    def predict(self, form_data: Dict[str, str]) -> Tuple[str, float, str]:
        """Returns: (prediction_text, confidence_percent_or_-1, risk_level_text)

        Because the original project uses High/Medium/Low risk labels,
        we map predicted class into risk buckets.
        """
        X = self._build_feature_vector(form_data)

        # Predict class
        y_pred = self.model.predict(X)
        pred_raw = y_pred[0] if hasattr(y_pred, "__len__") else y_pred

        # Risk score is the probability of class 1 (default), not merely
        # the model's maximum confidence in either class.
        risk_score = -1.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            proba_row = proba[0]
            classes = list(getattr(self.model, "classes_", []))
            default_index = classes.index(1) if 1 in classes else len(proba_row) - 1
            risk_score = float(proba_row[default_index] * 100.0)

        # Convert to business label
        # Common for loan default models: class 1 = Default, class 0 = Non-default.
        default_flag = None
        try:
            default_flag = int(pred_raw)
        except Exception:
            # If model uses strings already
            s = str(pred_raw).lower()
            if "1" in s or "default" in s or "yes" in s:
                default_flag = 1
            else:
                default_flag = 0

        if default_flag == 1:
            risk_level = "High Risk"
            prediction_text = "Likely to Default"
        else:
            risk_level = "Low Risk"
            prediction_text = "Low Default Risk"

        if risk_score >= 0:
            if risk_score >= 60:
                risk_level = "High Risk"
                prediction_text = "Likely to Default"
            elif risk_score >= 35:
                risk_level = "Medium Risk"
                prediction_text = "Needs Manual Review"
            else:
                risk_level = "Low Risk"
                prediction_text = "Low Default Risk"

        return prediction_text, risk_score, risk_level

