import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

FEATURES = [
    'Year',
    'Food Category',
    'Avg Waste per Capita (Kg)',
    'Population (Million)',
    'Household Waste (%)',
]
TARGET = 'Total Waste (Tons)'

NUMERIC_FEATURES = [
    'Year',
    'Avg Waste per Capita (Kg)',
    'Population (Million)',
    'Household Waste (%)',
]
CATEGORICAL_FEATURES = ['Food Category']


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_features_target(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(drop='first'), CATEGORICAL_FEATURES),
    ])


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
