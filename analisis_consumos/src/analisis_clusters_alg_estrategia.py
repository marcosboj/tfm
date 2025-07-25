#!/usr/bin/env python3
import pandas as pd

# ← Ajusta estas rutas según tu proyecto
INPUT_CSV  = "resultados/all_labels_unified_resumen.csv"  # CSV con columnas: hogar, cluster, estrategia, algoritmo, ...
OUTPUT_CSV = "resultados/labels_wide_resumen.csv"         # Salida deseada

def main():
    # 1) Carga el CSV (tab separado)
    df = pd.read_csv(INPUT_CSV, sep=',')

    # 2) Filtra solo las filas con etiqueta de cluster (descarta perfiles u otras secciones)
    df_clusters = df[df['cluster'].notna()]

    # 3) Reestructura a formato ancho: un hogar por fila,
    #    columnas = combinación de estrategia y algoritmo
    wide = df_clusters.pivot(
        index='hogar', #cambiar entre caracteristicas consumo y 24h
        columns=['estrategia', 'algoritmo'],
        values='cluster'
    )

    # 4) Aplana el MultiIndex de columnas: "estrategia_algoritmo"
    wide.columns = [
        f"{estrategia}_{algoritmo}" 
        for estrategia, algoritmo in wide.columns
    ]

    # 5) Convierte 'hogar' en columna normal
    wide = wide.reset_index()

    # 6) Guarda el resultado
    wide.to_csv(OUTPUT_CSV, index=False)
    print(f"Archivo generado: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
