import numpy as np
import pandas as pd


def add_sine_cosine_transformation(df: pd.DataFrame, source_column: str, period: float) -> pd.DataFrame:
    """
    Applies sine and cosine transformations to specified column in a Pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - source_column (str): The name of the column to be transformed.
    - period (float): The period of the cycle.

    Returns:
    - pd.DataFrame: A new DataFrame with the original data and additional columns:
      - '<source_column>_sin': The sine transformation of the column.
      - '<source_column>_cos': The cosine transformation of the column.
    """
    augmented_df = df.copy()
    augmented_df[source_column + "_sin"] = np.sin(df[source_column] / period * 2 * np.pi)
    augmented_df[source_column + "_cos"] = np.cos(df[source_column] / period * 2 * np.pi)
    return augmented_df
