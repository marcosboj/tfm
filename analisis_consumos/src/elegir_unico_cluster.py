import pandas as pd
from sklearn.preprocessing import minmax_scale
import os

# ====== Configuration ======
# Define input and output paths
input_path = "resultados/cluster_metrics.csv"
output_weighted = "resultados/recomendaciones_algoritmos.csv"
output_lex = "resultados/recomendaciones_lex.csv"

# Ensure input exists
if not os.path.isfile(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Load data (automatic separator detection)
sep = "\t" if input_path.lower().endswith((".tsv", ".tab")) else ","
df = pd.read_csv(input_path, sep=sep)

# ====== Strategy 1: Weighted normalization ======
def recommend_clusters(df, weight_silhouette=1.0, weight_db=0.5, weight_ch=1.0):
    df = df.copy()
    # Normalize metrics by strategy group
    df['sil_norm'] = df.groupby('estrategia')['silhouette'] \
                     .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    df['ch_norm'] = df.groupby('estrategia')['calinski_harabasz'] \
                    .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    # Invert and normalize DB across whole df, then per strategy
    df['db_inv'] = df['davies_bouldin'].max() - df['davies_bouldin']
    df['db_norm'] = df.groupby('estrategia')['db_inv'] \
                    .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    # Compute weighted score
    df['score'] = (weight_silhouette * df['sil_norm'] +
                   weight_db         * df['db_norm'] +
                   weight_ch         * df['ch_norm'])
    # Select best per strategy
    best_weighted = (df.sort_values(['estrategia', 'score'], ascending=[True, False])
                       .groupby('estrategia', as_index=False)
                       .first()[['estrategia','algoritmo','k',
                                 'silhouette','davies_bouldin',
                                 'calinski_harabasz','score']])
    return best_weighted

# Run weighted recommendation
reco_weighted = recommend_clusters(df, weight_silhouette=1.0, weight_db=0.5, weight_ch=1.0)
reco_weighted.to_csv(output_weighted, index=False)

# ====== Strategy 2: Lexicographic ======
def recommend_clusters_lex(df):
    best_rows = []
    for estrategia, group in df.groupby('estrategia', sort=True):
        # 1) Max silhouette
        cand = group[group['silhouette'] == group['silhouette'].max()]
        # 2) If tie, max Calinski–Harabasz
        if len(cand) > 1:
            cand = cand[cand['calinski_harabasz'] == cand['calinski_harabasz'].max()]
        # 3) If still tie, min Davies–Bouldin
        if len(cand) > 1:
            cand = cand[cand['davies_bouldin'] == cand['davies_bouldin'].min()]
        best_rows.append(cand.iloc[0])
    return pd.DataFrame(best_rows)[['estrategia','algoritmo','k',
                                    'silhouette','davies_bouldin',
                                    'calinski_harabasz']]

# Run lexicographic recommendation
reco_lex = recommend_clusters_lex(df)
reco_lex.to_csv(output_lex, index=False)

print(f"Weighted recommendations saved to: {output_weighted}")
print(f"Lexicographic recommendations saved to: {output_lex}")