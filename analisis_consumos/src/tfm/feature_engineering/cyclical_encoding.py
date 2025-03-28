import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period: float):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period: float):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def add_sine_cosine_transformation(df: pd.DataFrame, source_column: str, period: float):
    augmented_df = df.copy()
    augmented_df[source_column + "_sin"] = sin_transformer(period).fit_transform(augmented_df)[source_column]
    augmented_df[source_column + "_cos"] = cos_transformer(period).fit_transform(augmented_df)[source_column]
    return augmented_df
