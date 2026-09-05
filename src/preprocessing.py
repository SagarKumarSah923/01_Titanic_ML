"""
Preprocessing module for Titanic ML project.
Handles missing value imputation, data cleaning, and data transformation.
"""

from typing import Optional, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def extract_title_raw(name_series: pd.Series) -> pd.Series:
    """Extract standard title from passenger name."""
    titles = name_series.str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Officer",
        "Rev": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Mlle": "Miss",
        "Mme": "Mrs",
        "Ms": "Miss",
        "Lady": "Royalty",
        "Sir": "Royalty",
        "Capt": "Officer",
        "Countess": "Royalty",
        "Jonkheer": "Royalty",
        "Dona": "Royalty",
        "Don": "Royalty",
    }
    return titles.map(title_mapping).fillna("Other")


def get_title_series(name_series: pd.Series) -> pd.Series:
    """Helper to extract title series from names."""
    return extract_title_raw(name_series)


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for Titanic dataset.
    Learns imputation statistics on training data and applies them consistently.
    """

    def __init__(self):
        self.age_lookup_: Dict[Tuple[int, str], float] = {}
        self.global_age_median_: float = 28.0
        self.fare_lookup_: Dict[int, float] = {}
        self.global_fare_median_: float = 14.45
        self.embarked_mode_: str = "S"

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        if "Name" in X_df.columns:
            titles = extract_title_raw(X_df["Name"])
        else:
            titles = pd.Series(["Mr"] * len(X_df), index=X_df.index)

        X_df["_TempTitle"] = titles

        # Learn Age medians grouped by Pclass and Title
        if "Age" in X_df.columns:
            self.global_age_median_ = float(X_df["Age"].median(skipna=True))
            grouped_age = (
                X_df.groupby(["Pclass", "_TempTitle"])["Age"]
                .median()
                .to_dict()
            )
            self.age_lookup_ = {k: float(v) for k, v in grouped_age.items() if not np.isnan(v)}

        # Learn Fare medians grouped by Pclass
        if "Fare" in X_df.columns:
            self.global_fare_median_ = float(X_df["Fare"].median(skipna=True))
            grouped_fare = X_df.groupby("Pclass")["Fare"].median().to_dict()
            self.fare_lookup_ = {k: float(v) for k, v in grouped_fare.items() if not np.isnan(v)}

        # Learn Embarked mode
        if "Embarked" in X_df.columns and not X_df["Embarked"].dropna().empty:
            self.embarked_mode_ = str(X_df["Embarked"].mode().iloc[0])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Extract title if Name exists
        if "Name" in df.columns:
            titles = extract_title_raw(df["Name"])
        else:
            titles = pd.Series(["Mr"] * len(df), index=df.index)
        df["Title"] = titles

        # Impute Age
        if "Age" in df.columns:
            def get_age(row):
                if pd.notna(row["Age"]):
                    return float(row["Age"])
                key = (row.get("Pclass", 3), row.get("Title", "Mr"))
                return self.age_lookup_.get(key, self.global_age_median_)

            df["Age"] = df.apply(get_age, axis=1)

        # Impute Fare
        if "Fare" in df.columns:
            def get_fare(row):
                if pd.notna(row["Fare"]):
                    return float(row["Fare"])
                pclass = row.get("Pclass", 3)
                return self.fare_lookup_.get(pclass, self.global_fare_median_)

            df["Fare"] = df.apply(get_fare, axis=1)

        # Impute Embarked
        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)

        # Clean Cabin -> Deck
        if "Cabin" in df.columns:
            df["Cabin"] = df["Cabin"].fillna("Unknown")
            df["Deck"] = df["Cabin"].astype(str).str[0].map(
                lambda x: x if x in ["A", "B", "C", "D", "E", "F", "G", "T"] else "Unknown"
            )
        else:
            df["Deck"] = "Unknown"

        # Standardize Sex
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].astype(str).str.lower().map({"male": "male", "female": "female"}).fillna("male")

        return df


def clean_titanic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to fit and transform raw Titanic data into a clean baseline DataFrame.
    """
    preprocessor = TitanicPreprocessor()
    preprocessor.fit(df)
    return preprocessor.transform(df)
