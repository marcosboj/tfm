from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from context import DATA_DIR
from context import RANDOM_STATE
from context import TIME_ZONE
from tfm.feature_engineering.cyclical_encoding import add_sine_cosine_transformation
from tfm.utils.data_frame import add_timestamp_values


def main(csv_file: Path):
    df = pd.read_csv(csv_file)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TIME_ZONE)
    print(df)

    # Add time-related features
    df = add_timestamp_values(df, "datetime")
    df = add_sine_cosine_transformation(df, "hour", 24)

    # Clustering
    x = df[["a", "b", "hour", "hour_sin", "hour_cos"]].values
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init="auto")
    kmeans.fit(x)
    df["cluster"] = kmeans.predict(x)
    print(df[["datetime", "a", "b", "cluster"]])


if __name__ == "__main__":
    main(DATA_DIR / "dummy_data.csv")
