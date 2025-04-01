import pandas as pd


def add_timestamp_values(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Adds time-related values to a DataFrame from a datetime column.
    Parameters:
    - df (pd.DataFrame): The input DataFrame with a datetime index.
    - datetime_column (str): The name of the column

    Returns:
    - pd.DataFrame: A new DataFrame with additional columns:
      - 'hour': The hour of the timestamp.
      - 'dayofweek': The day of the week (0 = Monday, 6 = Sunday).
      - 'month': The month of the timestamp.
    """
    if not datetime_column in df.columns:
        raise ValueError(f"The following column is missing from the DataFrame: {datetime_column}")
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        raise TypeError(f"Column {datetime_column} must be of type datetime64.")
    augmented_df = df.copy()
    augmented_df["hour"] = augmented_df[datetime_column].dt.hour
    augmented_df["dayofweek"] = augmented_df[datetime_column].dt.dayofweek
    augmented_df["month"] = augmented_df[datetime_column].dt.month

    return augmented_df
