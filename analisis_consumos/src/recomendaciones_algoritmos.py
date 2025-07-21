import pandas as pd
from sklearn.preprocessing import minmax_scale

def recommend_clusters(df,
                       weight_silhouette=1.0,
                       weight_db=1.0,
                       weight_ch=1.0):
    """
    Recibe un DataFrame con columnas:
      ['estrategia', 'algoritmo', 'k', 'silhouette',
       'davies_bouldin', 'calinski_harabasz']
    Devuelve un DataFrame con la mejor (algoritmo, k) por cada estrategia.
    """
    df = df.copy()
    
    # Normalizar métricas a [0, 1]:
    #   silhouette y calinski_harabasz -> a mayor mejor
    #   davies_bouldin -> invertimos (menor mejor)
    df['sil_norm'] = df.groupby('estrategia')['silhouette'] \
                   .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    df['ch_norm'] = df.groupby('estrategia')['calinski_harabasz'] \
                 .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    # invertimos DB y luego normalizamos
    df['db_inv']  = df['davies_bouldin'].max() - df['davies_bouldin']
    df['db_norm'] = df.groupby('estrategia')['db_inv'] \
                 .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Score ponderado
    df['score'] = (
        weight_silhouette * df['sil_norm'] +
        weight_db         * df['db_norm']  +
        weight_ch         * df['ch_norm']
    )
    
    # Para cada estrategia, quedarnos con la fila de mayor score
    best = (df
            .sort_values(['estrategia','score'], ascending=[True, False])
            .groupby('estrategia', as_index=False)
            .first()
            [['estrategia','algoritmo','k','silhouette',
              'davies_bouldin','calinski_harabasz','score']]
           )
    return best

if __name__ == "__main__":
    # Ejemplo de carga desde CSV
    df = pd.read_csv("resultados/cluster_metrics.csv")
    recomendaciones = recommend_clusters(df,
                                         weight_silhouette=1.0,
                                         weight_db=0.5,
                                         weight_ch=1.0)
    # Guardar en CSV
    recomendaciones.to_csv("resultados/recomendaciones_algoritmos.csv", index=False)
    print("Recomendaciones guardadas en 'recomendaciones.csv'")
