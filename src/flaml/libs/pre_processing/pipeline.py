import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols, categorical_cols = get_columns(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    return pipeline


def get_columns(X: pd.DataFrame) -> tuple:
    numeric_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numeric_cols, categorical_cols
