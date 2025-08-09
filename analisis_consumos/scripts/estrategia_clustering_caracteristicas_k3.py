import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer

# ————————————————— Input / Output —————————————————

def crear_carpetas_project(project_root: Path):
    clustering_dir = project_root / "resultados" / "clustering_k3"
    pca_dir = project_root / "resultados" / "pca_k3"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)
    return clustering_dir, pca_dir

# ————————————————— Carga de resúmenes —————————————————

def cargar_resumenes(carpeta: Path, sep: str = ',', decimal: str = '.') -> dict[str, pd.DataFrame]:
    """
    Carga todos los CSV de resúmenes de características en un diccionario:
    clave: nombre base de hogar, valor: DataFrame con índice en 'hogar'
    """
    resumenes = {}
    for csv in sorted(carpeta.glob('*.csv')):
        df = pd.read_csv(csv, sep=sep, decimal=decimal)
        if 'archivo' in df.columns:
            df = df.set_index('archivo')
            df.index.name = 'hogar'
        else:
            df.index.name = 'hogar'
        resumenes[csv.stem] = df
    return resumenes

# ————————————————— Interpretación de componentes —————————————————

def reorder_labels_by_mean(Y: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    dfp = Y.copy()
    dfp['cluster'] = labels
    dfp['total'] = dfp.drop('cluster', axis=1).sum(axis=1)
    means = dfp.groupby('cluster')['total'].mean()
    ordered = means.sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(ordered)}
    return np.array([remap[l] for l in labels])

def summary_by_cluster(Y: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    dfp = Y.copy()
    dfp['cluster'] = labels
    hourly = dfp.groupby('cluster').mean().T
    total = dfp.drop('cluster', axis=1).sum(axis=1)
    total_means = total.groupby(dfp['cluster']).mean()
    summary = hourly.copy()
    summary.loc['total_mean_consumption'] = total_means
    return summary

# ————————————————— Visualización de clusters —————————————————

def plot_pca_scatter(Xp: np.ndarray, labels: np.ndarray, names: np.ndarray, strategy: str,
                     algo_name: str, out_path: Path):
    plt.figure(figsize=(6, 5))
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(Xp[mask, 0], Xp[mask, 1], s=30, alpha=0.7, label=f'Cluster {c}')
        for x, y, name in zip(Xp[mask, 0], Xp[mask, 1], names[mask]):
            plt.text(x, y, name, fontsize=5, alpha=0.8, ha='right', va='bottom')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'{strategy} – {algo_name} PCA scatter')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    save_dir = out_path / strategy
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{algo_name}_pca_scatter.png', dpi=150)
    plt.close()

# ————————————————— Clustering y Métricas —————————————————

def aplicar_clustering(Y: pd.DataFrame, X: np.ndarray, clustering_dir: Path,
                       strategy: str, algo_name: str, k: int) -> dict:
    # Selección del modelo
    if algo_name == 'KMeans':
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X)
    elif algo_name == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
    elif algo_name == 'GMM':
        model = GaussianMixture(n_components=k, random_state=0)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f'Algo desconocido: {algo_name}')

    # Reordenar etiquetas
    labels_ord = reorder_labels_by_mean(Y, labels)

    # Guardar etiquetas
    out_path = clustering_dir / strategy
    out_path.mkdir(parents=True, exist_ok=True)
    colname = f"{algo_name}_k{k}"
    pd.DataFrame({'hogar': Y.index, 'cluster': labels_ord}) \
      .to_csv(out_path / f"{colname}.csv", index=False)

    # Scatter PCA
    names = Y.index.to_numpy()
    plot_pca_scatter(X, labels_ord, names, strategy, algo_name + f'_k{k}', clustering_dir)

    # Perfil de clusters
    summary = summary_by_cluster(Y, labels_ord)
    summary.to_csv(out_path / f"{colname}_profile.csv")

    # Métricas
    n_clusters = len(set(labels_ord))
    if n_clusters > 1:
        sil = silhouette_score(X, labels_ord)
        dbi = davies_bouldin_score(X, labels_ord)
        chi = calinski_harabasz_score(X, labels_ord)
    else:
        sil = dbi = chi = np.nan

    return {'estrategia': strategy, 'algoritmo': algo_name, 'k': k,
            'silhouette': sil, 'davies_bouldin': dbi, 'calinski_harabasz': chi}

# ————————————————— Main —————————————————

def main(resumen_dir: Path, project_root: Path):
    clustering_dir, pca_dir = crear_carpetas_project(project_root)
    resumenes = cargar_resumenes(resumen_dir)

    metrics = []
    feature_imps = []
    for strategy, df in resumenes.items():
        # Preparar datos
        Y = df.copy()

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
        features = [f for f in features_base if f in Y.columns]
        #print(f"[INFO] Features seleccionadas para '{strategy}': {features}")
        Y = Y[features]

        # INFORME DE VALORES FALTANTES
        missing_per_col = Y.isna().sum()
        total_missing = missing_per_col.sum()
        #print(f"Estrategia '{strategy}': valores faltantes por columna:\n{missing_per_col}")
        #print(f"Total valores faltantes: {total_missing}\n")

        # Imputar valores faltantes
        imputer = SimpleImputer(strategy='mean')
        X_imp = imputer.fit_transform(Y.values)
        # Escalar
        X_scaled = StandardScaler().fit_transform(X_imp)
        # PCA reduciendo al 90% de varianza
        pca = PCA(n_components=0.9, svd_solver='full')
        Xp = pca.fit_transform(X_scaled)

        # Guardar PCA
        np.save(pca_dir / f"{strategy}_Xp.npy", Xp)
        loadings = pd.DataFrame(
            pca.components_,
            index=[f'PC{i+1}' for i in range(pca.n_components_)],
            columns=Y.columns
        )
        loadings.to_csv(pca_dir / f"{strategy}_loadings.csv")

        # Clustering
        for algo in ['KMeans']:
            for k in [3]:
                m = aplicar_clustering(Y, Xp, clustering_dir, strategy, algo, k)
                metrics.append(m)

                # --- Feature importance: varianza relativa en datos escalados ---
                from sklearn.preprocessing import StandardScaler as _SS
                scaler_ft = _SS()
                Y_scaled = pd.DataFrame(
                    scaler_ft.fit_transform(Y),
                    columns=Y.columns,
                    index=Y.index
                )

                etiquetas = pd.read_csv(
                    clustering_dir/strategy/f"{algo}_k{k}.csv"
                )['cluster'].values
                perf_scaled = summary_by_cluster(Y_scaled, etiquetas)
                var_ft = perf_scaled.drop('total_mean_consumption').var(axis=1)
                rel_imp = var_ft / var_ft.sum()
                top70 = rel_imp.sort_values(ascending=False).head(70)
                for feature, importance in top70.items():
                    feature_imps.append({
                        'estrategia': strategy,
                        'algoritmo': algo,
                        'k': k,
                        'feature': feature,
                        'importance': importance
                    })

    # Guardar métricas unificadas
    df_met = pd.DataFrame(metrics)
    df_met.to_csv(project_root / 'resultados' / 'cluster_metrics_k3.csv', index=False)
    print('Clustering de resúmenes completado. Métricas guardadas en resultados/cluster_metrics_k3.csv')

    # Guardar feature importances
    df_imp = pd.DataFrame(feature_imps)
    df_imp.to_csv(project_root / 'resultados' / 'feature_importances_k3.csv', index=False)
    print('Feature importances guardadas en resultados/feature_importances.csv')

    # ————————————————— Unificación de etiquetas —————————————————
    all_labels = []
    for strat_dir in clustering_dir.iterdir():
        for csv_file in strat_dir.glob('*.csv'):
            df_lbl = pd.read_csv(csv_file)
            df_lbl['estrategia'] = strat_dir.name
            df_lbl['algoritmo'] = csv_file.stem
            all_labels.append(df_lbl)
    df_all = pd.concat(all_labels, ignore_index=True)
    df_all.to_csv(project_root / 'resultados' / 'all_labels_unified_k3.csv', index=False)
    print('All labels unified saved to resultados/all_labels_unified_k3.csv')

if __name__ == '__main__':
    base_dir = Path.cwd()
    resumen_dir = base_dir / 'data' / 'viviendas' / 'resumenes'
    project_root = base_dir
    main(resumen_dir, project_root)

