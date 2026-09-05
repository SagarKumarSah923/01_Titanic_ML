"""
Feature engineering module for Titanic ML project.
Generates domain features such as FamilySize, IsAlone, FarePerPerson, AgeGroup, and encoded categories.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "Deck",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
]


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms preprocessed Titanic DataFrame into engineered feature set.
    """

    def __init__(self, drop_raw_cols: bool = True):
        self.drop_raw_cols = drop_raw_cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Family Size & Alone Indicator
        sibsp = df["SibSp"] if "SibSp" in df.columns else 0
        parch = df["Parch"] if "Parch" in df.columns else 0
        df["FamilySize"] = sibsp + parch + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

        # 2. Fare per person
        if "Fare" in df.columns:
            df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
        else:
            df["FarePerPerson"] = 0.0

        # 3. Age Group Categorization
        if "Age" in df.columns:
            df["AgeGroup"] = pd.cut(
                df["Age"],
                bins=[-1, 12, 18, 35, 60, 120],
                labels=["Child", "Teen", "YoungAdult", "MiddleAged", "Senior"],
            ).astype(str)

        # 4. Fare Bins (fixed bin thresholds based on quartiles: ~7.91, ~14.45, ~31.0)
        if "Fare" in df.columns:
            df["FareBin"] = pd.cut(
                df["Fare"],
                bins=[-np.inf, 7.91, 14.45, 31.0, np.inf],
                labels=["Low", "MidLow", "MidHigh", "High"],
            ).astype(str)

        # 5. Clean / Fill defaults for mandatory features
        if "Title" not in df.columns:
            df["Title"] = "Mr"
        if "Deck" not in df.columns:
            df["Deck"] = "Unknown"
        if "Embarked" not in df.columns:
            df["Embarked"] = "S"
        if "Sex" not in df.columns:
            df["Sex"] = "male"

        if self.drop_raw_cols:
            cols_to_drop = [c for c in ["PassengerId", "Name", "Ticket", "Cabin", "_TempTitle"] if c in df.columns]
            df = df.drop(columns=cols_to_drop)

        return df


def create_features(df: pd.DataFrame, drop_raw_cols: bool = True) -> pd.DataFrame:
    """
    Convenience function to generate engineered features on a DataFrame.
    """
    fe = TitanicFeatureEngineer(drop_raw_cols=drop_raw_cols)
    return fe.transform(df)
