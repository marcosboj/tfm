from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from context import DATA_DIR
from context import RANDOM_STATE
from context import TIME_ZONE
from tfm.feature_engineering.cyclical_encoding import add_sine_cosine_transformation
from tfm.utils.data_frame import add_timestamp_values


def process_timestamp(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    modified_df = df.copy()
    modified_df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(tz=tz)

    return modified_df


def main(csv_file: Path):
    df = pd.read_csv(
        csv_file,
        dtype={'nif': str}
    )
    if "datetime" in df.columns:
        df = process_timestamp(df, TIME_ZONE)
        # Add time-related features
        df = add_timestamp_values(df, "datetime")
        df = add_sine_cosine_transformation(df, "hour", 24)

    # Clustering
    features = [
        "mean_consumption", "std_consumption", "min_consumption", "max_consumption"
    ]
    x = df[features].values
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
    kmeans.fit(x)
    df["cluster"] = kmeans.predict(x)
    groups = df.groupby(by="cluster")['nif'].unique()
    for cluster, value in groups.items():
        print(f"Cluster {cluster}: {", ".join(value)}")


if __name__ == "__main__":
    main(DATA_DIR / "dummy_data.csv")
