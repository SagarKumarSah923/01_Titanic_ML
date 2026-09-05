"""
Inference and prediction module for Titanic ML project.
Provides high-level TitanicPredictor class for single-passenger and batch inferences.
"""

from pathlib import Path
from typing import Dict, Any, Union, List
import joblib
import pandas as pd
import numpy as np

from src.data_loader import PROJECT_ROOT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "titanic_model.joblib"


class TitanicPredictor:
    """
    Inference class for predicting Titanic passenger survival.
    """

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            from src.train import run_training_pipeline
            model, _ = run_training_pipeline()
            return model
        return joblib.load(self.model_path)

    def _format_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input data must be a dictionary or a pandas DataFrame.")

        # Ensure expected raw columns exist or fill reasonable defaults
        default_columns = {
            "Pclass": 3,
            "Name": "Passenger, Mr. John",
            "Sex": "male",
            "Age": 28.0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "UNKNOWN",
            "Fare": 14.45,
            "Cabin": "Unknown",
            "Embarked": "S",
        }
        for col, default_val in default_columns.items():
            if col not in df.columns:
                df[col] = default_val

        return df

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Union[int, List[int]]:
        """
        Predict binary survival outcome (0 = Did not survive, 1 = Survived).
        """
        df = self._format_input(data)
        preds = self.model.predict(df)
        if len(df) == 1:
            return int(preds[0])
        return [int(p) for p in preds]

    def predict_proba(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Union[float, List[float]]:
        """
        Predict survival probability (confidence between 0.0 and 1.0).
        """
        df = self._format_input(data)
        probas = self.model.predict_proba(df)[:, 1]
        if len(df) == 1:
            return float(probas[0])
        return [float(p) for p in probas]

    def predict_detailed(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate rich detailed prediction output with explanation notes.
        """
        df = self._format_input(data)
        preds = self.model.predict(df)
        probas = self.model.predict_proba(df)[:, 1]

        results = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            survived = bool(preds[idx] == 1)
            prob = float(probas[idx])

            # Generate intuitive risk factors
            factors = []
            if str(row.get("Sex", "")).lower() == "female":
                factors.append("Female passenger (Historically highest survival rate ~74%)")
            else:
                factors.append("Male passenger (Lower historical survival rate ~19%)")

            pclass = int(row.get("Pclass", 3))
            if pclass == 1:
                factors.append("First Class passenger (Priority lifeboat access)")
            elif pclass == 3:
                factors.append("Third Class passenger (Lower deck location, lower survival chance)")

            age = float(row.get("Age", 28))
            if age <= 12:
                factors.append("Child passenger ('Women and children first' protocol)")

            family_size = int(row.get("SibSp", 0)) + int(row.get("Parch", 0)) + 1
            if family_size > 4:
                factors.append("Large family group (More difficult evacuation coordination)")
            elif family_size == 1:
                factors.append("Traveling alone")

            results.append({
                "passenger_index": idx,
                "prediction": int(preds[idx]),
                "status": "Survived" if survived else "Did not survive",
                "survival_probability": round(prob, 4),
                "confidence_percentage": f"{round(prob * 100, 1)}%",
                "risk_and_protective_factors": factors,
            })

        return results[0] if len(results) == 1 else results


def make_sample_prediction():
    predictor = TitanicPredictor()

    sample_passengers = [
        {
            "Pclass": 1,
            "Name": "Rose DeWitt Bukater, Miss.",
            "Sex": "female",
            "Age": 17,
            "SibSp": 1,
            "Parch": 1,
            "Fare": 100.0,
            "Cabin": "B52",
            "Embarked": "S",
        },
        {
            "Pclass": 3,
            "Name": "Jack Dawson, Mr.",
            "Sex": "male",
            "Age": 20,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 8.05,
            "Cabin": "Unknown",
            "Embarked": "S",
        },
    ]

    print("--- Single Passenger Detailed Inferences ---")
    for passenger in sample_passengers:
        result = predictor.predict_detailed(passenger)
        print(f"\nPassenger: {passenger['Name']}")
        print(f"Prediction: {result['status']} (Probability: {result['confidence_percentage']})")
        print("Key Factors:")
        for factor in result["risk_and_protective_factors"]:
            print(f"  - {factor}")


if __name__ == "__main__":
    make_sample_prediction()
