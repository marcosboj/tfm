#!/usr/bin/env python3
# analyze_rolling_clusters_all.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parámetros de usuario: algoritmos y lista de k
ALGOS = ['KMeans', 'Agglomerative', 'GMM']
K_LIST = [2, 3, 4, 5]

# Rutas
PROJECT_ROOT = Path.cwd()
CLUSTER_DIR   = PROJECT_ROOT / 'resultados' / 'clustering'
OUTPUT_DIR    = PROJECT_ROOT / 'resultados' / 'rolling_analysis_all'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Función para procesar cada combinación

def analyze_combo(algo: str, k: int):
    # 1. Cargar etiquetas rolling
    rows = []
    for strat_dir in sorted(CLUSTER_DIR.iterdir()):
        if not strat_dir.name.startswith('rolling_'):
            continue
        csv_path = strat_dir / f"{algo}_k{k}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df['window'] = strat_dir.name
        rows.append(df[['hogar', 'window', 'cluster']])
    if not rows:
        return
    df_roll = pd.concat(rows, ignore_index=True)
    df_p = df_roll.pivot(index='hogar', columns='window', values='cluster')

    combo_out = OUTPUT_DIR / f'{algo}_k{k}'
    combo_out.mkdir(parents=True, exist_ok=True)

    # 2. Matrices de transición
    wins = sorted(df_p.columns)
    for i in range(len(wins) - 1):
        prev_win, next_win = wins[i], wins[i+1]
        pair = df_p[[prev_win, next_win]].dropna()
        m = pd.crosstab(pair[prev_win], pair[next_win], normalize='index')
        m.to_csv(combo_out / f'transition_{prev_win}_to_{next_win}.csv')

    # 3a. Proporciones
    prop = (df_roll
            .groupby(['window','cluster'])
            .size()
            .groupby(level=0)
            .apply(lambda x: x / x.sum())
            .unstack(fill_value=0))
    prop.to_csv(combo_out / 'proportions_by_window.csv')
    plt.figure(figsize=(10,5))
    prop.plot(kind='bar', stacked=True, colormap='tab10', width=0.8)
    plt.ylabel('Proporción de hogares')
    plt.xlabel('Ventana rolling')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Cluster', bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(combo_out / 'proportions_by_window.png')
    plt.close()

    # 3b. Heatmap evolución sin seaborn
    sample = df_p.sample(min(10, len(df_p)), random_state=0)
    data = sample.values.astype(int)
    plt.figure(figsize=(12,6))
    cax = plt.imshow(data, aspect='auto', interpolation='nearest', cmap='tab10')
    cbar = plt.colorbar(cax, ticks=np.unique(data))
    cbar.set_label('Cluster')
    plt.yticks(range(sample.shape[0]), sample.index)
    plt.xticks(range(sample.shape[1]), sample.columns, rotation=45, ha='right')
    plt.xlabel('Ventana rolling')
    plt.ylabel('Hogar')
    plt.title(f'Evolución cluster {algo}_k{k}')
    plt.tight_layout()
    plt.savefig(combo_out / 'heatmap_evolution.png')
    plt.close()

    # 4. Tasa de cambio
    diff = df_p.diff(axis=1)
    change_rate = (diff != 0).sum(axis=1) / (df_p.shape[1] - 1)
    change_rate.to_csv(combo_out / 'change_rate_per_hogar.csv', header=['change_rate'])
    descr = change_rate.describe()
    with open(combo_out / 'change_rate_summary.txt', 'w') as f:
        f.write(descr.to_string())

    print(f'Completado: {algo} k={k}')

# Main
def main():
    for algo in ALGOS:
        for k in K_LIST:
            analyze_combo(algo, k)
    print('Todos los análisis rolling completados. Revisa', OUTPUT_DIR)

if __name__ == '__main__':
    main()
