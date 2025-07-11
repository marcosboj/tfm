import pandas as pd
from sklego.preprocessing import RepeatingBasisFunction

from tfmforecasting.dataset import AdditionalHousingUnitFields, HousingUnitColumns


def name_lagged_feature(feature: HousingUnitColumns, lag: int) -> str:
    return f"lagged_{feature}_{lag:02d}h"


def add_rbf(df: pd.DataFrame, column: str, period: int, input_range: tuple[int, int]):
    rbf_df = df.copy()
    rbf = RepeatingBasisFunction(
        n_periods=period,
        column=column,
        input_range=input_range,
        remainder="drop"
    )
    rbf.fit(rbf_df)
    rbf_df = pd.DataFrame(index=rbf_df.index, data=rbf.transform(rbf_df))
    rbf_df = rbf_df.rename(columns=lambda df_column: f'{column}_rbf_{df_column}')
    return rbf_df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    datetime_column = AdditionalHousingUnitFields.Datetime.value
    parsed_df = df.copy()
    if not datetime_column in parsed_df.columns:
        return parsed_df
    parsed_df[AdditionalHousingUnitFields.Hour.value] = parsed_df[datetime_column].dt.hour
    parsed_df[AdditionalHousingUnitFields.Weekday.value] = parsed_df[datetime_column].dt.weekday
    return parsed_df


def add_datetime_to_housing_unit_dataset(df: pd.DataFrame, tz: str = 'Europe/Madrid') -> pd.DataFrame:
    def add_hour_to_datetime(row: pd.Series):
        time_values = row[HousingUnitColumns.Time.value].split(':')
        hours = int(time_values[0])
        return row[AdditionalHousingUnitFields.Datetime.value] + pd.Timedelta(hours=hours)

    datetime_column = AdditionalHousingUnitFields.Datetime.value
    parsed_df = df.copy()
    parsed_df[datetime_column] = pd.to_datetime(parsed_df[HousingUnitColumns.Date.value])
    parsed_df[datetime_column] = parsed_df.apply(add_hour_to_datetime, axis=1)
    parsed_df[datetime_column] = parsed_df[datetime_column].dt.tz_localize(tz)
    parsed_df[datetime_column] = parsed_df[datetime_column].dt.tz_convert('UTC')
    return parsed_df.sort_values(by=[datetime_column]).reset_index(drop=True)


def lag_consumption_feature(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    column_name = HousingUnitColumns.Consumption.value
    lagged_df = df.copy()
    for lag in range(1, n_lags + 1):
        lagged_df[name_lagged_feature(column_name, lag)] = df[column_name].shift(lag)
    return lagged_df
