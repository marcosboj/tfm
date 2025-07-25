import pandas as pd
import os

# Rutas a adaptar según tu proyecto
WEIGHTED_CSV = "resultados/recomendaciones_algoritmos_resumen.csv"
LEX_CSV      = "resultados/recomendaciones_lex_resumen.csv"
FINAL_CSV    = "resultados/recomendacion_final_resumen.csv"

# 1) Cargamos ambas recomendaciones
df_w = pd.read_csv(WEIGHTED_CSV)
df_l = pd.read_csv(LEX_CSV)

# 2) Combinamos sólo las columnas de interés
combined = pd.concat([
    df_w[['estrategia','algoritmo','k']],
    df_l[['estrategia','algoritmo','k']]
], ignore_index=True)

# --- FILTRO: solo k > 2 (mínimo 3 clusters) ---
combined = combined[combined['k'] > 1]

# 3) Contamos cuántas veces aparece cada (algoritmo, k)
votes = (
    combined
    .groupby(['algoritmo','k'], as_index=False)
    .size()
    .rename(columns={'size':'votes'})
    .sort_values(['votes','algoritmo','k'], ascending=[False,True,True])
)

# 4) Elegimos el más votado
best = votes.iloc[0]

print(f"Recomendación FINAL por mayoría:")
print(f"  • Algoritmo: {best['algoritmo']}")
print(f"  • Número de clusters: {int(best['k'])}")
print(f"  • Votos: {int(best['votes'])} de {len(combined)} posibles")

# 5) Guardamos en CSV
best.to_frame().T[['algoritmo','k','votes']].to_csv(FINAL_CSV, index=False)
print(f"Guardado en: {FINAL_CSV}")
