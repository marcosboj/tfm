from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from contextlib import redirect_stdout

from context import DATA_DIR, RANDOM_STATE, TIME_ZONE
from tfm.feature_engineering.cyclical_encoding import add_sine_cosine_transformation
from tfm.utils.data_frame import add_timestamp_values


def process_timestamp(df: pd.DataFrame, datetime_column: str, tz: str) -> pd.DataFrame:
    modified_df = df.copy()
    modified_df[datetime_column] = pd.to_datetime(df[datetime_column], utc=True).dt.tz_convert(tz=tz)
    return modified_df


def obtener_cargas_pca(X_scaled: np.ndarray, features: list[str]) -> tuple[pd.Series, pd.Series]:
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    loadings_pc1 = pd.Series(np.abs(pca.components_[0]), index=features)
    loadings_pc2 = pd.Series(np.abs(pca.components_[1]), index=features)
    return loadings_pc1.sort_values(ascending=False), loadings_pc2.sort_values(ascending=False)


def aplicar_gmm(df: pd.DataFrame, features: list[str], n_clusters: int, csv_file: Path = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
    labels = gmm.fit_predict(X_scaled)
    df['gmm_cluster'] = labels

    df_cluster_summary = df.groupby('gmm_cluster')[features].mean()
    print("\nResumen por cluster (GMM):")
    print(df_cluster_summary)

    # Visualización con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['cluster'] = labels

    plt.figure(figsize=(8, 6))
    for label in sorted(df_pca['cluster'].unique()):
        cluster = df_pca[df_pca['cluster'] == label]
        plt.scatter(cluster['PC1'], cluster['PC2'], label=f"Cluster {label}")
    plt.title(f"GMM con {n_clusters} clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    if csv_file:
        nombre_filtro = csv_file.stem.replace("resumen_consumos_", "")
        plot_path = Path("resultados") / f"{nombre_filtro}_gmm_clusters_k{n_clusters}_c{len(df)}.png"
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path)
        print(f"[INFO] Gráfico de clusters GMM guardado en {plot_path}")
    plt.close()

    return df, df_cluster_summary


def main(csv_file: Path, datetime_column: str = "datetime", n_clusters: Optional[int] = 4):
    df = pd.read_csv(csv_file, dtype={'archivo': str})

    if datetime_column in df.columns:
        df = process_timestamp(df, datetime_column, TIME_ZONE)
        df = add_timestamp_values(df, datetime_column)
        df = add_sine_cosine_transformation(df, "hour", 24)

    archivo_column = "archivo"

    features = [
        "media_consumo", "std_consumo", "min_consumo", "max_consumo", "percentil_25_consumo", "percentil_50_consumo",
        "percentil_75_consumo", "promedio_por_dia", "consumo_medio_diario",
        "Mañana", "Mediodia", "Tarde", "Noche", "Madrugada", "sum_consumo",
        "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo", "Entre semana", "Fin de semana",
        "s_Mañana", "s_Mediodia", "s_Tarde", "s_Noche", "s_Madrugada", "s_Lunes", "s_Martes", "s_Miércoles", "s_Jueves", "s_Viernes",
        "s_Sábado", "s_Domingo", "s_Entre semana", "s_Fin de semana", "s_invierno", "s_otoño", "s_primavera", "s_verano",
        "std_Mañana", "std_Mediodia", "std_Tarde", "std_Noche", "std_Madrugada", "std_Lunes", "std_Martes", "std_Miércoles", "std_Jueves", "std_Viernes",
        "std_Sábado", "std_Domingo", "std_Entre semana", "std_Fin de semana", "std_invierno", "std_otoño", "std_primavera", "std_verano",
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
        # Elimina features que no están presentes en el DataFrame (por rango temporal)
    features = [f for f in features if f in df.columns]
    print(f"[INFO] Features en el DataFrame: {features}")

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Cargas PCA
    loadings_pc1, loadings_pc2 = obtener_cargas_pca(X_scaled, features)

    # Preparar logging
    nombre_filtro = csv_file.stem.replace("resumen_consumos_", "")
    n_casos = len(df)
    nombre_log = f"{nombre_filtro}_GMM_k{n_clusters}_c{n_casos}.txt"
    ruta_log = Path("logs") / nombre_log
    ruta_log.parent.mkdir(exist_ok=True)

    with open(ruta_log, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(f"[INFO] Features en el DataFrame: {features}")

            print(f"[INFO] Clustering GMM para archivo: {csv_file.name}")
            df, summary = aplicar_gmm(df, features, n_clusters=n_clusters, csv_file=csv_file)
            

            print("\nCasos por cluster:")
            grupos = df.groupby('gmm_cluster')[archivo_column].unique()
            for cluster, archivos in grupos.items():
                print(f"Cluster {cluster}: {', '.join(archivos)}")

            print("\n🔎 Features más influyentes en PC1:")
            print(loadings_pc1.head(10))
            print("\n🔎 Features más influyentes en PC2:")
            print(loadings_pc2.head(10))

    print(f"[INFO] Log guardado en: {ruta_log}")


if __name__ == "__main__":
    main(DATA_DIR / "resumen_consumos_mes_10.csv", n_clusters=4)
