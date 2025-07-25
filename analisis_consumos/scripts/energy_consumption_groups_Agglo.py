from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from context import DATA_DIR, RANDOM_STATE, TIME_ZONE
from tfm.feature_engineering.cyclical_encoding import add_sine_cosine_transformation
from tfm.utils.data_frame import add_timestamp_values


def process_timestamp(df: pd.DataFrame, datetime_column: str, tz: str) -> pd.DataFrame:
    df[datetime_column] = pd.to_datetime(df[datetime_column], utc=True).dt.tz_convert(tz=tz)
    return df


def find_optimal_k(X, k_range=(2, 10)):
    silhouettes = []
    for k in range(*k_range):
        agglom = AgglomerativeClustering(n_clusters=k)
        labels = agglom.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))
    best_k = range(*k_range)[np.argmax(silhouettes)]
    return best_k, silhouettes


def clustering_analysis(df: pd.DataFrame, features: list[str], k: int):
    X_scaled = StandardScaler().fit_transform(df[features])
    agglom = AgglomerativeClustering(n_clusters=k)
    df["cluster"] = agglom.fit_predict(X_scaled)

    # Resumen
    cluster_summary = df.groupby("cluster")[features].mean()
    # No centroid attribute; approximate centroids via cluster means
    centroids = cluster_summary
    variances = centroids.var().sort_values(ascending=False)

    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    pca_loadings = pd.Series(np.abs(pca.components_[0]), index=features)

    return df, cluster_summary, variances, pca_loadings


def main(
    csv_file: Path,
    datetime_column: str = "datetime",
    modo_manual: bool = True,
    n_clusters_analisis: Optional[int] = None,  
    ):
    df = pd.read_csv(csv_file, dtype={"archivo": str})
    if datetime_column in df.columns:
        df = process_timestamp(df, datetime_column, TIME_ZONE)
        df = add_timestamp_values(df, datetime_column)
        df = add_sine_cosine_transformation(df, "hour", 24)

    # Selección de features
    features_base = [
        "media_consumo", "std_consumo", "min_consumo", "max_consumo",
        "percentil_25_consumo", "percentil_50_consumo", "percentil_75_consumo",
        "promedio_por_dia", "consumo_medio_diario",
        "Mañana", "Mediodia", "Tarde", "Noche", "Madrugada", "sum_consumo",
        "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo",
        "Entre semana", "Fin de semana",
        "s_Mañana", "s_Mediodia", "s_Tarde", "s_Noche", "s_Madrugada",
        "s_Lunes", "s_Martes", "s_Miércoles", "s_Jueves", "s_Viernes",
        "s_Sábado", "s_Domingo", "s_Entre semana", "s_Fin de semana",
        "s_invierno", "s_otoño", "s_primavera", "s_verano",
        "std_Mañana", "std_Mediodia", "std_Tarde", "std_Noche", "std_Madrugada",
        "std_Lunes", "std_Martes", "std_Miércoles", "std_Jueves", "std_Viernes",
        "std_Sábado", "std_Domingo", "std_Entre semana", "std_Fin de semana",
        "std_invierno", "std_otoño", "std_primavera", "std_verano",
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
        "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    
    features = [f for f in features_base if f in df.columns]

    print(f"[INFO] Features en el DataFrame: {features}")

    if not modo_manual or n_clusters_analisis is None:
        print("\n[INFO] Buscando automáticamente el número óptimo de clusters...")
        n_clusters_analisis, silhouettes = find_optimal_k(df[features].values)
        print(f"\n[INFO] Número óptimo de clusters: {n_clusters_analisis}")
    else:
        print(f"\n[INFO] Usando número de clusters definido por el usuario: {n_clusters_analisis}")

    df, resumen_clusters, importancia_vars, pca_vars = clustering_analysis(df, features, n_clusters_analisis)

    # Mostrar resumen
    print("\n🏘 Viviendas por cluster:")
    for c_id, viviendas in df.groupby("cluster")["archivo"]:
        print(f"Cluster {c_id} ({len(viviendas)} viviendas)")

    print("\n📊 Media de consumo por cluster:")
    print(resumen_clusters["media_consumo"])

    print("\n🔥 Variables más importantes para el clustering:")
    print(importancia_vars.head(10))

    print("\n💡 Features con mayor carga en el primer componente (PCA):")
    print(pca_vars.sort_values(ascending=False).head(10))

    # Plot del perfil del cluster 0
    resumen_clusters.loc[0].sort_values(ascending=False).plot(kind='barh', figsize=(8,10), title="Perfil del Cluster 0")
    '''
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    '''
        # --- Guardar análisis en archivo .txt ---
    from contextlib import redirect_stdout

    nombre_filtro = csv_file.stem.replace("resumen_consumos_", "")
    n_casos = len(df)
    nombre_log = f"{nombre_filtro}_agglo{k}_c{n_casos}.txt"
    ruta_log = Path("logs/agglomerative") / nombre_log
    ruta_log.parent.mkdir(exist_ok=True)

    with open(ruta_log, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print(f"[INFO] Features utilizadas en el DataFrame: {features}")
            print("\n🏘 Viviendas por cluster:")
            grupos = df.groupby("cluster")["archivo"].unique()
            for cluster_id, archivos in grupos.items():
                print(f"Cluster {cluster_id} ({len(archivos)} viviendas): {', '.join(archivos)}")

            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", None)

            print(f"\nPara {n_clusters_analisis} clusters, las medias por cluster son:")
            print(resumen_clusters)

            print("\n🔥 Variables más importantes para el clustering (varianza entre centroides):")
            print(importancia_vars.head(20))

            print("\n💡 Features con mayor carga en el primer componente PCA:")
            print(pca_vars.sort_values(ascending=False).head(10))

            if not modo_manual:
                print("\n📈 Silhouette Scores por número de clusters:")
                for k_val, score in zip(range(2, 10), silhouettes):
                    print(f"k={k_val}: silhouette_score={score:.4f}")

    print(f"[INFO] Análisis guardado en: {ruta_log}")


if __name__ == "__main__":
    #for file in DATA_DIR.glob("resumenes/resumen_consumos_meses_*.csv"):
    #for file in DATA_DIR.glob("resumenes/resumen_consumos_todo*.csv"):
    #for file in DATA_DIR.glob("resumenes/resumen_consumos_estacion*.csv"):
    #for file in DATA_DIR.glob("resumenes/resumen_consumos_dia*.csv"):
    for file in DATA_DIR.glob("resumenes/resumen_consumos_tipo*.csv"):
        print(f"\n📂 Procesando archivo: {file.name}")
        for k in [2, 3, 4, 5]:
            print(f"\n🔢 Ejecutando clustering con k={k} clusters")
            main(
                csv_file=file,
                modo_manual=True,       # Forzamos el modo manual
                n_clusters_analisis=k   # Probamos con 2, 3 y 4 clusters
            )
