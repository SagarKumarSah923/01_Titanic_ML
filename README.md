# Titanic ML Project

Machine learning project for predicting Titanic passenger survival.

## Project Structure

```
01_Titanic_ML/
├── data/              # Data files (raw and processed)
├── notebooks/         # Jupyter notebooks for analysis
├── src/               # Source code modules
├── models/            # Trained models
├── tests/             # Unit and integration tests
├── app/               # Web application
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Setup Instructions

1. Create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Explore notebooks for analysis and modeling

## Data

- **raw/**: Original Titanic dataset
- **processed/**: Cleaned data for modeling

## Notebooks

Sequential analysis and modeling pipeline:
1. `01_data_exploration.ipynb` - Data exploration
2. `02_data_cleaning.ipynb` - Data cleaning
3. `03_feature_engineering.ipynb` - Feature creation
4. `04_model_training.ipynb` - Model training
5. `05_model_evaluation.ipynb` - Model evaluation

## Source Code

- `data_loader.py` - Data loading utilities
- `preprocessing.py` - Data preprocessing functions
- `feature_engineering.py` - Feature engineering functions
- `train.py` - Model training logic
- `evaluate.py` - Model evaluation metrics
- `predict.py` - Prediction pipeline

## Testing

Run tests with: `pytest tests/`

## Application

Web interface in `app/app.py`
