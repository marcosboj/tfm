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
    import matplotlib.pyplot as plt
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

    features = [
        "media_consumo", "std_consumo", "min_consumo", "max_consumo", "percentil_25_consumo", "percentil_50_consumo",
        "percentil_75_consumo","promedio_por_dia","consumo_medio_diario",
        "Mañana","Mediodia", "Tarde","Noche","Madrugada","sum_consumo",
        "Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo","Entre semana","Fin de semana",
        "s_Mañana","s_Mediodia","s_Tarde","s_Noche","s_Madrugada","s_Lunes","s_Martes","s_Miércoles","s_Jueves","s_Viernes",
        "s_Sábado","s_Domingo","s_Entre semana","s_Fin de semana","s_invierno","s_otoño","s_primavera","s_verano",
        "std_Mañana","std_Mediodia","std_Tarde","std_Noche","std_Madrugada","std_Lunes","std_Martes","std_Miércoles","std_Jueves","std_Viernes",
        "std_Sábado","std_Domingo","std_Entre semana","std_Fin de semana","std_invierno","std_otoño","std_primavera","std_verano",
        "Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
    ]
    # Elimina features que no están presentes en el DataFrame (por rango temporal)
    features = [f for f in features if f in df.columns]
    print(f"[INFO] Features en el DataFrame: {features}")

    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import pandas as pd
    import numpy as np

    # Escalar los datos si aún no lo has hecho
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Entrenar el modelo (usa el número de clusters que hayas elegido)
    kmeans = KMeans(n_clusters_analisis, random_state=RANDOM_STATE)
    kmeans.fit(X_scaled)
    df['cluster'] = kmeans.labels_
    print("\n[INFO] Viviendas por cluster:")
    grupos = df.groupby('cluster')['archivo'].unique()
    for cluster_id, archivos in grupos.items():
        print(f"Cluster {cluster_id} ({len(archivos)} viviendas): {', '.join(archivos)}")
    print("\n[DEBUG] Media de consumo y número de casos por cluster:")
    print(df.groupby('cluster')['media_consumo'].agg(['mean', 'count']))

    # Agrupar por cluster y calcular la media de todas las features
    df_cluster_summary = df.groupby('cluster')[features].mean()

    # Mostrar las estadísticas promedio por cluster
    print(df_cluster_summary)
    # Plot del perfil del cluster 0
    df_cluster_summary.loc[0].sort_values(ascending=False).plot(kind='barh', figsize=(8,10), title="Perfil del Cluster 0")
    '''
    for cluster_id in df_cluster_summary.index:
        df_cluster_summary.loc[cluster_id].sort_values(ascending=False).plot(
            kind='barh',
            figsize=(8, 10),
            title=f"Perfil del Cluster {cluster_id}"
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    '''



    # Obtener los centroides
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)

    # Calcular la varianza entre los centroides por cada feature (cuanto mayor, más discriminativa)
    variances = centroids.var().sort_values(ascending=False)

    # Mostrar las más importantes
    #print("Features más relevantes para el clustering:")
    #print(variances.head(20))
    variances.head(20).plot(kind='barh', figsize=(8, 6), title="Importancia relativa de las features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Carga de cada feature en los componentes principales
    pca_loadings = pd.Series(np.abs(pca.components_[0]), index=features)
    #print("Features con mayor carga en el primer componente:")
    #print(pca_loadings.sort_values(ascending=False).head(10))
    
    import pandas as pd
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

    ###guardar caracteristicas analisis
    import sys
    import os
    from contextlib import redirect_stdout
    from pathlib import Path

    # --- Configuración de parámetros ---
    nombre_filtro = csv_file.stem.replace("resumen_consumos_", "")  # Extrae lo que sigue a 'resumen_consumos'
    n_casos = len(df)  # Asumiendo que df tiene las 18 filas analizadas
    nombre_log = f"{nombre_filtro}_k{n_clusters_analisis}_c{n_casos}.txt"

    # Ruta donde guardar el archivo de log
    ruta_log = Path("logs") / nombre_log  # Crea una carpeta 'logs'
    ruta_log.parent.mkdir(exist_ok=True)  # Asegura que exista la carpeta

    # --- Redirigir salida estándar a archivo ---
    with open(ruta_log, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print(f"[INFO] Features en el DataFrame: {features}")
            print("\n🏠 Viviendas por cluster:")
            grupos = df.groupby('cluster')['archivo'].unique()
            for cluster_id, archivos in grupos.items():
                print(f"Cluster {cluster_id} ({len(archivos)} viviendas): {', '.join(archivos)}")

            # Aquí pones todo el análisis que quieras guardar:
            # Mostrar todas las filas y columnas completas sin truncar
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)  # para que no limite el ancho
            pd.set_option('display.max_colwidth', None)  # mostrar columnas completas
            print(f"\nPara {n_clusters_analisis} clusters las features mas importantes por cluster son:")
            print(df_cluster_summary)
            print("\nFeatures más relevantes para el clustering:")
            print(variances.head(20))

            print("\nFeatures con mayor carga en el primer componente:")
            print(pca_loadings.sort_values(ascending=False).head(10))

            print("\nSilhouette Scores por número de clusters:")
            for k, score in zip(range_n_clusters, silhouette_scores):
                print(f"k={k}: silhouette_score={score:.4f}")

            print("\nInertias por número de clusters (codo):")
            for k, inertia in zip(range_n_clusters, inertias):
                print(f"k={k}: inertia={inertia:.2f}")

    # --- Mostrar confirmación ---
    print(f"Resultado guardado en {ruta_log}")
    

            


if __name__ == "__main__":
    #main(DATA_DIR / "dummy_data.csv")
    n_clusters_analisis = 4
    main(DATA_DIR / "resumen_consumos_mes_10.csv")
