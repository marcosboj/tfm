from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from context import DATA_DIR
from context import RANDOM_STATE
from context import TIME_ZONE
from tfm.feature_engineering.cyclical_encoding import add_sine_cosine_transformation
from tfm.utils.data_frame import add_timestamp_values


def process_timestamp(df: pd.DataFrame, datetime_column: str, tz: str) -> pd.DataFrame:
    modified_df = df.copy()
    modified_df[datetime_column] = pd.to_datetime(df[datetime_column], utc=True).dt.tz_convert(tz=tz)

    return modified_df


def main(csv_file: Path, datetime_column: str = "datetime"):
    df = pd.read_csv(
        csv_file,
        dtype={'nif': str}
    )
    if datetime_column in df.columns:
        df = process_timestamp(df, datetime_column, TIME_ZONE)
        # Add time-related features
        df = add_timestamp_values(df, datetime_column)
        df = add_sine_cosine_transformation(df, "hour", 24)

    # Clustering
    cluster_column: str = "cluster"
    nif_column: str = "nif"
    features = [
        "media_consumo", "std_consumo", "min_consumo", "max_consumo"
    ]
    x = df[features].values
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
    kmeans.fit(x)
    df[cluster_column] = kmeans.predict(x)
    groups = df.groupby(by=cluster_column)[nif_column].unique()
    for cluster, value in groups.items():
        print(f"Cluster {cluster}: {", ".join(value)}")


if __name__ == "__main__":
    main(DATA_DIR / "dummy_data.csv")
