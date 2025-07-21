import pandas as pd

def recommend_clusters_lex(df):
    """
    Selección lexicográfica por:
      1) silhouette ↑
      2) davies_bouldin ↓
      3) calinski_harabasz ↑
    Devuelve un DataFrame con la mejor (algoritmo, k) por cada estrategia.
    """
    best_rows = []
    # iteramos por cada grupo de estrategia
    for estrategia, group in df.groupby('estrategia', sort=True):
        # 1) máximo silhouette
        cand = group[group['silhouette'] == group['silhouette'].max()]
        # 2) Si hay empate, máximo Calinski–Harabasz
        if len(cand) > 1:
            cand = cand[cand['calinski_harabasz'] == cand['calinski_harabasz'].max()]
        # 3) Si aún empata, mínimo Davies–Bouldin
        if len(cand) > 1:
            cand = cand[cand['davies_bouldin'] == cand['davies_bouldin'].min()]
        # nos quedamos con el primero en caso de empate total
        best_rows.append(cand.iloc[0])
    return pd.DataFrame(best_rows)[
        ['estrategia','algoritmo','k','silhouette','davies_bouldin','calinski_harabasz']
    ]

if __name__ == "__main__":
    df = pd.read_csv("resultados/cluster_metrics.csv")
    recomendaciones = recommend_clusters_lex(df)
    recomendaciones.to_csv("resultados/recomendaciones_lex.csv", index=False)
