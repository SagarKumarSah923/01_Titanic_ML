"""
Unit tests for data loader, preprocessing, and feature engineering modules.
"""

import pandas as pd
import numpy as np
import pytest

from src.data_loader import load_raw_data, get_train_val_split
from src.preprocessing import TitanicPreprocessor, extract_title_raw
from src.feature_engineering import TitanicFeatureEngineer, create_features


@pytest.fixture
def sample_raw_data():
    """Create a mock raw Titanic DataFrame for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 0, 0],
            "Pclass": [3, 1, 3, 1, 2],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
                "Montvila, Rev. Juozas",
            ],
            "Sex": ["male", "female", "female", "female", "male"],
            "Age": [22.0, 38.0, 26.0, np.nan, 27.0],
            "SibSp": [1, 1, 0, 1, 0],
            "Parch": [0, 0, 0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "211536"],
            "Fare": [7.25, 71.2833, 7.925, 53.1, 13.0],
            "Cabin": [np.nan, "C85", np.nan, "C123", np.nan],
            "Embarked": ["S", "C", "S", "S", np.nan],
        }
    )


def test_title_extraction():
    names = pd.Series([
        "Braund, Mr. Owen Harris",
        "Heikkinen, Miss. Laina",
        "Palsson, Master. Gosta Leonard",
        "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",
        "Dr. John Doe",
    ])
    titles = extract_title_raw(names)
    assert titles.iloc[0] == "Mr"
    assert titles.iloc[1] == "Miss"
    assert titles.iloc[2] == "Master"
    assert titles.iloc[3] == "Royalty"


def test_preprocessor_imputation(sample_raw_data):
    preprocessor = TitanicPreprocessor()
    transformed = preprocessor.fit_transform(sample_raw_data)

    # Check no missing values in key fields
    assert transformed["Age"].isnull().sum() == 0
    assert transformed["Embarked"].isnull().sum() == 0
    assert transformed["Deck"].isnull().sum() == 0
    assert "Title" in transformed.columns
    assert transformed.loc[1, "Deck"] == "C"
    assert transformed.loc[0, "Deck"] == "Unknown"


def test_feature_engineering(sample_raw_data):
    preprocessor = TitanicPreprocessor()
    cleaned = preprocessor.fit_transform(sample_raw_data)

    fe = TitanicFeatureEngineer(drop_raw_cols=True)
    engineered = fe.transform(cleaned)

    assert "FamilySize" in engineered.columns
    assert "IsAlone" in engineered.columns
    assert "FarePerPerson" in engineered.columns
    assert "AgeGroup" in engineered.columns
    assert "FareBin" in engineered.columns

    # Passenger 1 has 1 SibSp, 0 Parch -> FamilySize = 2, IsAlone = 0
    assert engineered.loc[0, "FamilySize"] == 2
    assert engineered.loc[0, "IsAlone"] == 0
    # Passenger 3 has 0 SibSp, 0 Parch -> FamilySize = 1, IsAlone = 1
    assert engineered.loc[2, "FamilySize"] == 1
    assert engineered.loc[2, "IsAlone"] == 1


def test_train_val_split(sample_raw_data):
    X_train, X_val, y_train, y_val = get_train_val_split(
        sample_raw_data, test_size=0.4, random_state=42, stratify=False
    )
    assert len(X_train) == 3
    assert len(X_val) == 2
    assert "Survived" not in X_train.columns
    assert len(y_train) == 3
