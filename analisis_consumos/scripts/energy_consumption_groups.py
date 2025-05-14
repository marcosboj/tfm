from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional
)

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


def clustering(
        df: pd.DataFrame, features: list[str], n_clusters: int, random_state: Optional[int] = None,
        n_init: int | Literal["auto"] = "auto", max_iter: int = 300
) -> tuple[Any, KMeans]:
    X = df[features].values
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans


def main(csv_file: Path, datetime_column: str = "datetime"):
    import pandas as pd
    df = pd.read_csv(
        csv_file,
        dtype={'archivo': str}
    )
    if datetime_column in df.columns:
        df = process_timestamp(df, datetime_column, TIME_ZONE)
        # Add time-related features
        df = add_timestamp_values(df, datetime_column)
        df = add_sine_cosine_transformation(df, "hour", 24)

    # Clustering
    archivo_column: str = "archivo"
    '''
    features = [
        "media_consumo", "std_consumo", "min_consumo", "max_consumo", "percentil_25_consumo", "percentil_50_consumo",
        "percentil_75_consumo","consumo_1.0","consumo_2.0","consumo_3.0","consumo_4.0","consumo_5.0","consumo_6.0",
        "consumo_7.0","consumo_8.0","consumo_9.0","consumo_10.0","consumo_11.0","consumo_12.0","consumo_13.0",
        "consumo_14.0","consumo_15.0","consumo_16.0","consumo_17.0","consumo_18.0","consumo_19.0","consumo_20.0",
        "consumo_21.0","consumo_22.0","consumo_23.0","Mañana","Tarde","Noche","Madrugada","Lunes","Martes","Miércoles"
        ,"Jueves","Viernes","Sábado","Domingo","Entre semana","Fin de semana"
        
    ]
    '''
    features = [
        "media_consumo", "std_consumo", "min_consumo", "max_consumo", "percentil_25_consumo", "percentil_50_consumo",
        "percentil_75_consumo","Mañana","Tarde","Noche","Madrugada","Lunes","Martes","Miércoles"
        ,"Jueves","Viernes","Sábado","Domingo","Entre semana","Fin de semana"
        
    ]
    for n_clusters in range(2, 5):
        clusters, _ = clustering(df, features, n_clusters, random_state=RANDOM_STATE)
        label_column = f"{n_clusters}_clusters_label"
        df[label_column] = clusters
        groups = df.groupby(by=label_column)[archivo_column].unique()
        print(f"KMeans(n_clusters={n_clusters}, random_state={RANDOM_STATE}) results:")
        for cluster, value in groups.items():
            print(f"Cluster {cluster}: {", ".join(value)}")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Escalar los datos si aún no lo has hecho
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Entrenar el modelo (usa el número de clusters que hayas elegido)
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE)
    kmeans.fit(X_scaled)

    # Obtener los centroides
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)

    # Calcular la varianza entre los centroides por cada feature (cuanto mayor, más discriminativa)
    variances = centroids.var().sort_values(ascending=False)

    # Mostrar las más importantes
    print("Features más relevantes para el clustering:")
    print(variances.head(10))
    variances.head(10).plot(kind='barh', figsize=(8, 6), title="Importancia relativa de las features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Carga de cada feature en los componentes principales
    pca_loadings = pd.Series(np.abs(pca.components_[0]), index=features)
    print("Features con mayor carga en el primer componente:")
    print(pca_loadings.sort_values(ascending=False).head(10))
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Suponemos que 'df' es tu DataFrame y 'features' es la lista de columnas usadas para el clustering
    X = df[features].values

    inertias = []
    silhouette_scores = []
    range_n_clusters = range(2, 10)

    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        
        # Inercia para el método del codo
        inertias.append(kmeans.inertia_)
        
        # Silhouette Score
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    # Graficar el método del codo
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range_n_clusters, inertias, marker='o')
    plt.title('Método del codo')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')

    # Graficar Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range_n_clusters, silhouette_scores, marker='o', color='orange')
    plt.title('Silhouette Score')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.show()


    

            


if __name__ == "__main__":
    #main(DATA_DIR / "dummy_data.csv")
    main(DATA_DIR / "resumen_consumos.csv")
