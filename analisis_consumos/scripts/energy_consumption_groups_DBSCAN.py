from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional
)

import pandas as pd
from sklearn.cluster import DBSCAN

from context import DATA_DIR
from context import RANDOM_STATE
from context import TIME_ZONE
from tfm.feature_engineering.cyclical_encoding import add_sine_cosine_transformation
from tfm.utils.data_frame import add_timestamp_values

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def aplicar_dbscan(df: pd.DataFrame, features: list[str], eps_values=[0.5], min_samples=5):
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    resultados = {}

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        etiquetas = dbscan.fit_predict(X_scaled)
        df[f'dbscan_eps_{eps}'] = etiquetas

        n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
        n_ruido = list(etiquetas).count(-1)

        # Silhouette solo si hay más de 1 cluster
        if n_clusters > 1:
            sil_score = silhouette_score(X_scaled, etiquetas)
        else:
            sil_score = np.nan

        resultados[eps] = {
            'n_clusters': n_clusters,
            'n_ruido': n_ruido,
            'silhouette_score': sil_score
        }

        print(f"\nDBSCAN eps={eps}: {n_clusters} clusters, {n_ruido} puntos de ruido, Silhouette={sil_score:.4f}")

    return df, resultados


def visualizar_clusters(df: pd.DataFrame, features: list[str], label_col: str):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca[label_col] = df[label_col]

    plt.figure(figsize=(8, 6))
    for label in sorted(df_pca[label_col].unique()):
        cluster = df_pca[df_pca[label_col] == label]
        plt.scatter(cluster["PC1"], cluster["PC2"], label=f"Cluster {label}")
    plt.title(f"Visualización PCA con etiquetas {label_col}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(csv_file: Path, datetime_column: str = "datetime"):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from contextlib import redirect_stdout
    from pathlib import Path

    df = pd.read_csv(csv_file, dtype={'archivo': str})

    if datetime_column in df.columns:
        df = process_timestamp(df, datetime_column, TIME_ZONE)
        df = add_timestamp_values(df, datetime_column)
        df = add_sine_cosine_transformation(df, "hour", 24)

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

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Configuración de DBSCAN
    eps_values = [2, 4, 6, 8]
    min_samples = 3

    nombre_filtro = csv_file.stem.replace("resumen_consumos_", "")
    n_casos = len(df)
    nombre_log = f"{nombre_filtro}_DBSCAN_c{n_casos}.txt"
    ruta_log = Path("logs") / nombre_log
    ruta_log.parent.mkdir(exist_ok=True)

    with open(ruta_log, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            for eps in eps_values:
                db = DBSCAN(eps=eps, min_samples=min_samples)
                etiquetas = db.fit_predict(X_scaled)
                df[f'dbscan_eps_{eps}'] = etiquetas

                n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
                n_ruido = list(etiquetas).count(-1)

                if n_clusters > 1:
                    sil_score = silhouette_score(X_scaled, etiquetas)
                else:
                    sil_score = np.nan

                print(f"\nDBSCAN eps={eps}:")
                print(f"- Clusters encontrados: {n_clusters}")
                print(f"- Puntos de ruido: {n_ruido}")
                print(f"- Silhouette Score: {sil_score:.4f}" if not np.isnan(sil_score) else "- Silhouette Score: N/A")

                # Agrupar por cluster y mostrar qué archivos están en cada grupo
                grupos = df.groupby(f'dbscan_eps_{eps}')[archivo_column].unique()
                for cluster, archivos in grupos.items():
                    print(f"Cluster {cluster}: {', '.join(archivos)}")

                # Estadísticas de cada cluster
                df_cluster_summary = df.groupby(f'dbscan_eps_{eps}')[features].mean()
                print("\nPromedios por cluster:")
                print(df_cluster_summary)

                # PCA + visualización
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
                df_pca["cluster"] = etiquetas

                plt.figure(figsize=(8, 6))
                for label in sorted(df_pca["cluster"].unique()):
                    cluster = df_pca[df_pca["cluster"] == label]
                    plt.scatter(cluster["PC1"], cluster["PC2"], label=f"Cluster {label}")
                plt.title(f"DBSCAN eps={eps}")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.legend()
                plt.tight_layout()
                plt.show()

    print(f"Resultado guardado en {ruta_log}")
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    def estimar_eps(X, min_samples=5):
        """
        Genera el gráfico de la distancia al min_samples-ésimo vecino más cercano
        para ayudar a estimar el parámetro eps en DBSCAN.
        
        X: array-like, datos ya escalados.
        min_samples: parámetro de DBSCAN (por defecto 5).
        """
        neigh = NearestNeighbors(n_neighbors=min_samples)
        neigh.fit(X)
        
        distances, indices = neigh.kneighbors(X)
        
        # Obtener la distancia al min_samples-ésimo vecino
        distancias_ordenadas = np.sort(distances[:, -1])
        
        # Graficar la curva de distancias ordenadas
        plt.figure(figsize=(8, 5))
        plt.plot(distancias_ordenadas)
        plt.title(f"Gráfico de distancia al {min_samples}-ésimo vecino más cercano")
        plt.xlabel("Punto (ordenado)")
        plt.ylabel("Distancia")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    X_scaled = scaler.fit_transform(df[features])
    estimar_eps(X_scaled, min_samples=5)



if __name__ == "__main__":
    #main(DATA_DIR / "dummy_data.csv")
    main(DATA_DIR / "resumen_consumos_2023-03-01_a_2024-12-01.csv")

