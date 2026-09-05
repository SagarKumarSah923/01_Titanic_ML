"""
Data loading module for Titanic ML project.
Provides utilities to load raw and processed data, and split datasets for modeling.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "titanic_clean.csv"


def load_raw_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load raw Titanic training dataset from CSV.

    Args:
        filepath: Optional path to raw dataset. Defaults to data/raw/train.csv.

    Returns:
        pd.DataFrame containing raw Titanic data.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    return pd.read_csv(path)


def load_processed_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load preprocessed Titanic dataset.

    Args:
        filepath: Optional path to processed dataset. Defaults to data/processed/titanic_clean.csv.

    Returns:
        pd.DataFrame containing preprocessed Titanic data.
    """
    path = Path(filepath) if filepath else PROCESSED_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found at: {path}")
    return pd.read_csv(path)


def save_processed_data(
    df: pd.DataFrame, filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Save processed DataFrame to CSV.

    Args:
        df: DataFrame to save.
        filepath: Destination path. Defaults to data/processed/titanic_clean.csv.

    Returns:
        Path to saved file.
    """
    path = Path(filepath) if filepath else PROCESSED_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def get_train_val_split(
    df: Optional[pd.DataFrame] = None,
    target_column: str = "Survived",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a dataset into train and validation sets.

    Args:
        df: Input DataFrame. If None, loads processed data.
        target_column: Name of target column ('Survived').
        test_size: Proportion of dataset for validation split.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify split by target column.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    if df is None:
        df = load_processed_data()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_target = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_target
    )
