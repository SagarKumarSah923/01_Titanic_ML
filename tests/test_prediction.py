"""
Unit and integration tests for Titanic model prediction and inference pipeline.
"""

import pandas as pd
import pytest
from src.predict import TitanicPredictor


@pytest.fixture(scope="module")
def predictor():
    """Load or initialize TitanicPredictor instance."""
    return TitanicPredictor()


def test_single_passenger_prediction(predictor):
    passenger = {
        "Pclass": 1,
        "Name": "Astor, Mrs. John Jacob (Madeleine Talmadge Force)",
        "Sex": "female",
        "Age": 18.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 227.525,
        "Cabin": "C62",
        "Embarked": "C",
    }
    pred = predictor.predict(passenger)
    proba = predictor.predict_proba(passenger)
    detailed = predictor.predict_detailed(passenger)

    assert pred in [0, 1]
    assert 0.0 <= proba <= 1.0
    assert detailed["status"] in ["Survived", "Did not survive"]
    assert len(detailed["risk_and_protective_factors"]) > 0


def test_batch_prediction(predictor):
    batch_df = pd.DataFrame(
        [
            {
                "Pclass": 3,
                "Name": "Dooley, Mr. Patrick",
                "Sex": "male",
                "Age": 32.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 7.75,
                "Cabin": "Unknown",
                "Embarked": "Q",
            },
            {
                "Pclass": 1,
                "Name": "Cardeza, Mrs. James Warburton Martinez (Charlotte Wardle Drake)",
                "Sex": "female",
                "Age": 58.0,
                "SibSp": 0,
                "Parch": 1,
                "Fare": 512.3292,
                "Cabin": "B51",
                "Embarked": "C",
            },
        ]
    )
    preds = predictor.predict(batch_df)
    probas = predictor.predict_proba(batch_df)
    detailed = predictor.predict_detailed(batch_df)

    assert len(preds) == 2
    assert len(probas) == 2
    assert isinstance(detailed, list)
    assert len(detailed) == 2


def test_missing_field_robustness(predictor):
    minimal_passenger = {"Pclass": 2, "Sex": "female"}
    pred = predictor.predict(minimal_passenger)
    proba = predictor.predict_proba(minimal_passenger)

    assert pred in [0, 1]
    assert 0.0 <= proba <= 1.0
