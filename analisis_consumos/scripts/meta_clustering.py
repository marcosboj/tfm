import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# —————————————————————————————————————————————————————————————————
# Script de Meta-Clustering basado en matriz de co-aparición
# Aprovecha los CSV generados por estrategias_clustering.py:
#   resultados/all_labels_unified.csv
# Para cada vivienda i,j cuenta en cuántas estrategias comparten cluster.
# Luego aplica clustering jerárquico sobre esa matriz.
# —————————————————————————————————————————————————————————————————

def load_labels(all_labels_path: Path) -> pd.DataFrame:
    """Carga el CSV unificado y filtra sólo las columnas necesarias: hogar, cluster, estrategia y algoritmo"""
    df = pd.read_csv(all_labels_path)
    # Filtrar sólo filas de etiquetas (no perfiles) y con valores válidos
    mask_hogar   = df['hogar'].notna()
    mask_cluster = df['cluster'].notna()
    mask_algo    = ~df['algoritmo'].str.contains('_profile', na=False)
    df_lbl = df[mask_hogar & mask_cluster & mask_algo].copy()
    # Asegurar tipos
    df_lbl['cluster'] = df_lbl['cluster'].astype(int)
    # Seleccionar sólo columnas clave
    return df_lbl[['hogar', 'cluster', 'estrategia', 'algoritmo']]


def build_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una matriz N×N donde M[i,j] = número de particiones
    en que hogar_i y hogar_j están en el mismo cluster.
    """
    hogares = df['hogar'].unique()
    idx = {h: i for i, h in enumerate(hogares)}
    N = len(hogares)
    M = np.zeros((N, N), dtype=int)

    for (_, grp) in df.groupby(['estrategia', 'algoritmo']):
        lab_map = grp.set_index('hogar')['cluster'].to_dict()
        for c in grp['cluster'].unique():
            members = [h for h, l in lab_map.items() if l == c]
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    ii = idx[members[i]]
                    jj = idx[members[j]]
                    M[ii, jj] += 1
                    M[jj, ii] += 1

    cooc = pd.DataFrame(M, index=hogares, columns=hogares)
    return cooc


def meta_clustering(cooc: pd.DataFrame, n_clusters: int) -> pd.Series:
    """
    Aplica clustering jerárquico a la matriz de co-ocurrencia.
    Usa la distancia 1 - cooc/max_cooc para obtener valores en [0,1].
    """
    max_val = cooc.values.max()
    dist = 1 - cooc.values / max_val
    d = pdist(dist)
    Z = linkage(d, method='complete')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    return pd.Series(labels, index=cooc.index, name='meta_cluster')


if __name__ == '__main__':
    project_root = Path.cwd()
    all_labels_path = project_root / 'resultados' / 'all_labels_unified_resumen.csv'
    out_dir = project_root / 'resultados' / 'meta_clustering_resumen'
    '''
    all_labels_path = project_root / 'resultados' / 'all_labels_unified.csv'
    out_dir = project_root / 'resultados' / 'meta_clustering'
    '''
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Carga y filtrado de etiquetas
    df_labels = load_labels(all_labels_path)

    # 2) Construcción de la matriz de co-aparición
    cooc = build_cooccurrence(df_labels)
    cooc.to_csv(out_dir / 'cooccurrence_matrix.csv')

    # 3) Meta-clustering jerárquico
    K = 4  # número de meta-clusters deseado
    meta_labels = meta_clustering(cooc, K)

    # 4) Ordenar meta-clusters por consumo medio diario
    from estrategias_clustering import pivot_global, cargar_todos_consumos, preparar_timestamp  # ajustar import según ruta
    
    df_consumos = cargar_todos_consumos(project_root / 'data' / 'viviendas' / 'consumos')
    df_consumos = preparar_timestamp(df_consumos)
    pivot = pivot_global(df_consumos)
    totales = pivot.sum(axis=1)

    df_meta = pd.DataFrame({
        'hogar': meta_labels.index,
        'cluster_old': meta_labels.values
    }).set_index('hogar')
    df_meta['total'] = totales

    medias = df_meta.groupby('cluster_old')['total'].mean()
    orden = medias.sort_values().index.tolist()
    mapping = {old: new for new, old in enumerate(orden)}
    meta_labels_ord = meta_labels.map(mapping).astype(int)
    meta_labels_ord.name = 'cluster_ordenado'

    # 5) Guardar resultados
    output_df = meta_labels_ord.reset_index().rename(
        columns={'index': 'hogar', 'cluster_ordenado': 'meta_cluster'}
    )
    output_df.to_csv(out_dir / f'meta_clusters_k{K}.csv', index=False)

    print(f'Meta-clusters (k={K}) guardados y ordenados en {out_dir}')
