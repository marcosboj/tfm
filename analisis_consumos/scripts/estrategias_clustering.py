import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ————————————————— Input / Output —————————————————

def crear_carpetas_project(project_root: Path):
    clustering_dir = project_root / "resultados" / "clustering"
    pca_dir = project_root / "resultados" / "pca"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)
    return clustering_dir, pca_dir

# ————————————————— Carga y Preprocesado —————————————————

def cargar_todos_consumos(carpeta: Path, sep: str = ';') -> pd.DataFrame:
    archivos = sorted(f for f in os.listdir(carpeta) if f.endswith('.csv'))
    dfs = []
    for fn in archivos:
        df = pd.read_csv(carpeta / fn, sep=sep)
        df['hogar'] = fn.split('.')[0]
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    ########################################
    # 1) Crea la serie de fechas datetime (para el cálculo interno, NO crea columna)
    _dates = pd.to_datetime(full_df['date'], dayfirst=True)

    # 2) Sustituye “24:00:00” por “00:00” sólo para parsear
    _times = full_df['time'].replace({'24:00:00': '00:00'})

    # 3) Construye el timestamp local, sumando 1 día si time era “24:00:00”
    _full_local = (
        pd.to_datetime(
            _dates.dt.strftime('%Y-%m-%d') + ' ' + _times,
            format='%Y-%m-%d %H:%M',
            errors='coerce'
        )
        + pd.to_timedelta(full_df['time'].eq('24:00:00').astype(int), unit='d')
    )

    # 4) Localiza en Europe/Madrid y convierte a UTC, guardando en la misma columna
    full_df['timestamp'] = (
        _full_local
        .dt.tz_localize('Europe/Madrid', ambiguous='infer', nonexistent='shift_forward')
        .dt.tz_convert('UTC')
    )
    ########################################

    return full_df.dropna(subset=['timestamp'])


def preparar_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].map(lambda m: 
        'invierno' if m in (12,1,2) else
        'primavera' if m in (3,4,5) else
        'verano' if m in (6,7,8) else 'otono')
    return df


# ————————————————— Pivot Tables —————————————————

def pivot_global(df):
    return df.groupby(['hogar','hour'])['consumptionKWh'].mean().unstack(fill_value=0)

def pivot_estacional(df, season):
    return pivot_global(df[df['season']==season])

def pivot_mensual(df, month):
    return pivot_global(df[df['month']==month])
'''
def pivot_rolling(df, start, window_days=90):
    end = start + pd.Timedelta(days=window_days)
    win = df[(df['timestamp']>=start)&(df['timestamp']<end)]
    return pivot_global(win)
'''
def pivot_day_type(df, day_type):
    return pivot_global(df[df['day_type'] == day_type])

# ————————————————— Interpretación de componentes —————————————————

def reorder_labels_by_mean(pivot: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    dfp = pivot.copy()
    dfp['cluster'] = labels
    # total diario por hogar
    dfp['total'] = dfp.drop('cluster', axis=1).sum(axis=1)
    # media total por cluster
    means = dfp.groupby('cluster')['total'].mean()
    ordered = means.sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(ordered)}
    return np.array([remap[l] for l in labels])


def summary_by_cluster(pivot: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    dfp = pivot.copy()
    dfp['cluster'] = labels
    hourly = dfp.groupby('cluster').mean().T
    total = dfp.drop('cluster', axis=1).sum(axis=1)
    total_means = total.groupby(dfp['cluster']).mean()
    summary = hourly.copy()
    summary.loc['total_mean_consumption'] = total_means
    return summary

# ————————————————— Visualización de clusters —————————————————
import matplotlib.pyplot as plt

def plot_pca_scatter(Xp, labels, strategy, algo_name, out_path):
    plt.figure(figsize=(6,5))
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(Xp[mask,0], Xp[mask,1], s=30, alpha=0.7, label=f'Cluster {c}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'{strategy} – {algo_name} PCA scatter')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path/strategy/f'{algo_name}_pca_scatter.png')
    plt.close()


# ————————————————— Clustering y Métricas —————————————————

def aplicar_clustering(Y, X, algorithm, params, clustering_dir, strategy, algo_name, k=None):
    labels = None
    if algo_name=='KMeans':
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X)
    elif algo_name=='Agglomerative':
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
    elif algo_name=='GMM':
        model = GaussianMixture(n_components=k, random_state=0)
        labels = model.fit_predict(X)
    # Reordenar etiquetas por consumo medio
    labels_ord = reorder_labels_by_mean(Y, labels)
    # Guardar etiquetas
    out_path = clustering_dir / strategy
    out_path.mkdir(parents=True, exist_ok=True)
    colname = f"{algo_name}_k{k}" if k else algo_name
    pd.DataFrame({'hogar':Y.index, 'cluster':labels_ord})\
      .to_csv(out_path/f"{colname}.csv", index=False)
    
    # Visualización scatter PCA
    plot_pca_scatter(X, labels_ord, strategy, f"{algo_name}_k{k}" if k else algo_name, clustering_dir)
    out_path = clustering_dir / strategy
    out_path.mkdir(parents=True, exist_ok=True)
    colname = f"{algo_name}_k{k}" if k else algo_name
    pd.DataFrame({'hogar':Y.index, 'cluster':labels_ord})\
      .to_csv(out_path/f"{colname}.csv", index=False)
    # Guardar perfil horario de clusters
    summary = summary_by_cluster(Y, labels_ord)
    summary.to_csv(out_path / f"{colname}_profile.csv")
    # Calcular métricas si hay más de 1 cluster
    n_clusters = len(set(labels_ord)) - (1 if -1 in labels_ord else 0)
    if n_clusters>1:
        sil = silhouette_score(X, labels_ord)
        dbi = davies_bouldin_score(X, labels_ord)
        chi = calinski_harabasz_score(X, labels_ord)
    else:
        sil = dbi = chi = np.nan
    return {'estrategia':strategy, 'algoritmo':algo_name, 'k':k,
            'silhouette':sil, 'davies_bouldin':dbi, 'calinski_harabasz':chi}

# ————————————————— Main —————————————————

def main(data_dir:Path, project_root:Path):
    df = cargar_todos_consumos(data_dir)
    df = preparar_timestamp(df)
    clustering_dir, pca_dir = crear_carpetas_project(project_root)

    # 1) Marcar días festivos vs laborables (festivos desde CSV + sáb/dom)
    festivos = pd.read_csv(project_root/'data'/'festivos_zgz.csv')['fecha'].astype(str).tolist()
    df['date_only'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['weekday']   = df['timestamp'].dt.weekday
    df['day_type']  = np.where(
        df['date_only'].isin(festivos) | df['weekday'].isin([5,6]),
        'festivo','laborable'
    )

    # Preprocesar y PCA
    strategies = []
    pivots = {'global':pivot_global(df)}
    pivots = {'global': pivot_global(df)}
    # 2) Añadir estrategias laborable y festivo
    pivots['laborable'] = pivot_day_type(df, 'laborable')
    pivots['festivo']   = pivot_day_type(df, 'festivo')
    for s in ['invierno','primavera','verano','otono']:
        pivots[f'estacion_{s}'] = pivot_estacional(df,s)
    for m in range(1,13):
        pivots[f'mes_{m:02d}'] = pivot_mensual(df,m)
    df_sorted = df.sort_values('timestamp')
    cur = df_sorted['timestamp'].min()
    end = df_sorted['timestamp'].max()
    '''
    while cur + pd.Timedelta(days=90)<=end:
        lbl= f'rolling_{cur.date()}_{(cur+pd.Timedelta(days=90)).date()}'
        pivots[lbl] = pivot_rolling(df_sorted,cur)
        cur+=pd.Timedelta(days=30)
    '''
    # Algoritmos y parámetros
    algos = ['KMeans','Agglomerative','GMM']
    db_params = {'eps':0.5,'min_samples':2}
    k_list = [2,3,4,5]

    # Ejecutar clustering y recopilar métricas
    metrics = []
    for strat, pivot in pivots.items():
        # Escalado + PCA
        scaler = StandardScaler()
        Xs = scaler.fit_transform(pivot.values)
        pca = PCA(n_components=0.9, svd_solver='full')
        Xp = pca.fit_transform(Xs)
        # Guardar PCA
        np.save(pca_dir/f"{strat}_Xp.npy",Xp)
        loadings = pd.DataFrame(
            pca.components_,
            index=[f'PC{i+1}' for i in range(pca.n_components_)],
            columns=pivot.columns
        )
        loadings.to_csv(pca_dir/f"{strat}_loadings.csv")
        for algo in algos:
                for k in k_list:
                    m = aplicar_clustering(pivot,Xp,None,{},clustering_dir,strat,algo,k)
                    metrics.append(m)

        # Unificación de etiquetas
    all_labels = []
    for strat_dir in clustering_dir.iterdir():
        for csv_file in strat_dir.glob('*.csv'):
            df_lbl = pd.read_csv(csv_file)
            df_lbl['estrategia'] = strat_dir.name
            df_lbl['algoritmo'] = csv_file.stem
            all_labels.append(df_lbl)
    df_all = pd.concat(all_labels, ignore_index=True)
    df_all.to_csv(project_root / 'resultados' / 'all_labels_unified.csv', index=False)
    print('All labels unified saved to resultados/all_labels_unified.csv')

    # Guardar métricas
    df_met = pd.DataFrame(metrics)
    df_met.to_csv(project_root/'resultados'/'cluster_metrics.csv',index=False)
    print('Cluster metrics saved to resultados/cluster_metrics.csv')

if __name__=='__main__':
    project_root=Path.cwd()
    data_dir=project_root/'data'/'viviendas'/'consumos'
    main(data_dir,project_root)
