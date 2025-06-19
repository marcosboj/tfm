from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from contextlib import redirect_stdout

from context import DATA_DIR, RANDOM_STATE, TIME_ZONE
from tfm.feature_engineering.cyclical_encoding import add_sine_cosine_transformation
from tfm.utils.data_frame import add_timestamp_values


def process_timestamp(df: pd.DataFrame, datetime_column: str, tz: str) -> pd.DataFrame:
    modified_df = df.copy()
    modified_df[datetime_column] = pd.to_datetime(df[datetime_column], utc=True).dt.tz_convert(tz=tz)
    return modified_df


def graficar_dendrograma(X_scaled: np.ndarray, csv_file: Path, k_clusters: int, n_casos: int) -> None:
    plt.figure(figsize=(10, 6))
    Z = linkage(X_scaled, method='ward')
    dendrogram(Z)
    plt.title("Dendrograma (Agglomerative Clustering)")
    plt.xlabel("Casos")
    plt.ylabel("Distancia")
    nombre_filtro = csv_file.stem.replace("resumen_consumos_", "")
    dendro_path = Path("resultados") / f"{nombre_filtro}_dendrograma_k{k_clusters}_c{n_casos}.png"
    dendro_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(dendro_path)
    plt.close()
    print(f"[INFO] Dendrograma guardado en {dendro_path}")


def aplicar_agglomerative(df: pd.DataFrame, features: list[str], n_clusters: int = 4, csv_file: Path = None) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Dendrograma
    if csv_file:
        graficar_dendrograma(X_scaled, csv_file, k_clusters=n_clusters, n_casos=len(df))



    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(X_scaled)
    df['agg_cluster'] = labels

    # Resumen por cluster
    df_cluster_summary = df.groupby('agg_cluster')[features].mean()
    print("\nResumen por cluster (Agglomerative):")
    print(df_cluster_summary)

    # Visualización PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['cluster'] = labels

    plt.figure(figsize=(8, 6))
    for label in sorted(df_pca['cluster'].unique()):
        cluster = df_pca[df_pca['cluster'] == label]
        plt.scatter(cluster['PC1'], cluster['PC2'], label=f"Cluster {label}")
    plt.title(f"Agglomerative Clustering con {n_clusters} clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    nombre_filtro = csv_file.stem.replace("resumen_consumos_", "") if csv_file else "resultado"
    plot_path = Path("resultados") / f"{nombre_filtro}_clusters_k{n_clusters}_c{len(df)}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Gráfico de clusters guardado en {plot_path}")

    return df, df_cluster_summary

def obtener_cargas_pca(X_scaled: np.ndarray, features: list[str]) -> tuple[pd.Series, pd.Series]:
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    loadings_pc1 = pd.Series(np.abs(pca.components_[0]), index=features)
    loadings_pc2 = pd.Series(np.abs(pca.components_[1]), index=features)
    return loadings_pc1.sort_values(ascending=False), loadings_pc2.sort_values(ascending=False)

def sugerir_num_clusters(X_scaled: np.ndarray) -> int:
    from scipy.cluster.hierarchy import linkage
    Z = linkage(X_scaled, method='ward')
    distancias = Z[:, 2]  # tercer columna = distancia entre merges
    difs = np.diff(distancias[::-1])  # diferencia descendente
    idx_max_salto = np.argmax(difs)
    sugerido = idx_max_salto + 1
    return sugerido


def main(csv_file: Path, datetime_column: str = "datetime", n_clusters: Optional[int] = None):
    df = pd.read_csv(csv_file, dtype={'archivo': str})

    if datetime_column in df.columns:
        df = process_timestamp(df, datetime_column, TIME_ZONE)
        df = add_timestamp_values(df, datetime_column)
        df = add_sine_cosine_transformation(df, "hour", 24)

    archivo_column = "archivo"

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


    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Sugerencia automática de n_clusters
    n_clusters_sugerido = sugerir_num_clusters(X_scaled)

    # Obtener features más influyentes en PC1 y PC2
    loadings_pc1, loadings_pc2 = obtener_cargas_pca(X_scaled, features)
    # Determinar cuántos clusters usar (manual o sugerido)
    n_clusters_usado = n_clusters if n_clusters is not None else n_clusters_sugerido
    # Logging e impresión de análisis
    print(f"[INFO] Número de clusters sugerido (dendrograma): {n_clusters_sugerido}")
    print("\n🔎 Features más influyentes en PC1:")
    print(loadings_pc1.head(10))
    print("\n🔎 Features más influyentes en PC2:")
    print(loadings_pc2.head(10))

    nombre_filtro = csv_file.stem.replace("resumen_consumos_", "")
    n_casos = len(df)
    nombre_log = f"{nombre_filtro}_Agglomerative_k{n_clusters_usado}_c{n_casos}.txt"
    ruta_log = Path("logs") / nombre_log
    ruta_log.parent.mkdir(exist_ok=True)



    with open(ruta_log, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(f"[INFO] Features en el DataFrame: {features}")
            print(f"[INFO] Clustering Agglomerative para archivo: {csv_file.name}")
            df, summary = aplicar_agglomerative(df, features, n_clusters=n_clusters_usado, csv_file=csv_file)
            #df, summary = aplicar_agglomerative(df, features, n_clusters=n_clusters_sugerido, csv_file=csv_file)
            # Mostrar todo el contenido en el log sin truncar

            #print("\nResumen por cluster:")
            #print(summary)

            # Mostrar qué archivos están en cada cluster
            grupos = df.groupby('agg_cluster')[archivo_column].unique()
            print("\nCasos por cluster:")
            for cluster, archivos in grupos.items():
                print(f"Cluster {cluster}: {', '.join(archivos)}")
            print(f"\n[INFO] Número de clusters sugerido (dendrograma): {n_clusters_usado}")

            print("\n🔎 Features más influyentes en PC1:")
            print(loadings_pc1.head(10))
            print("\n🔎 Features más influyentes en PC2:")
            print(loadings_pc2.head(10))


    print(f"[INFO] Log guardado en: {ruta_log}")


if __name__ == "__main__":
     main(DATA_DIR / "resumen_consumos_mes_10.csv", n_clusters=4)
