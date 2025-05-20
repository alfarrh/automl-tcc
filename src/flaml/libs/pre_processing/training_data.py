import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def read_data(path: str, target: str) -> tuple:

    df = pd.read_csv(path)
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def get_training_data(
    X: pd.DataFrame, y: pd.DataFrame, pipeline: Pipeline
) -> tuple:
    X_processed = pipeline.fit_transform(X)

    return train_test_split(
        X_processed, y.values, test_size=0.8, random_state=42
    )
