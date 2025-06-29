import pandas as pd

from tfmforecasting.dataset import AdditionalHouseholdFields, HouseholdColumns


def name_lagged_feature(feature: HouseholdColumns, lag: int) -> str:
    return f"prev_{feature}_{lag:02d}"


def add_datetime_column(df: pd.DataFrame, tz: str = 'Europe/Madrid') -> pd.DataFrame:
    def add_hour_to_datetime(row: pd.Series):
        time_values = row[HouseholdColumns.Time].split(':')
        hours = int(time_values[0])
        return row[AdditionalHouseholdFields.Datetime] + pd.Timedelta(hours=hours)

    datetime_column = AdditionalHouseholdFields.Datetime
    parsed_df = df.copy()
    parsed_df[datetime_column] = pd.to_datetime(parsed_df[HouseholdColumns.Date])
    parsed_df[datetime_column] = parsed_df.apply(add_hour_to_datetime, axis=1)
    parsed_df[datetime_column] = parsed_df[datetime_column].dt.tz_localize(tz)
    return parsed_df


def lag_consumption_feature(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    column_name = HouseholdColumns.Consumption
    lagged_df = df.copy()
    for lag in range(1, n_lags + 1):
        lagged_df[name_lagged_feature(column_name, lag)] = df[column_name].shift(lag)
    return lagged_df
