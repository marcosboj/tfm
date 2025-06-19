import pandas as pd
import re
from pathlib import Path

# 📁 Ruta a los archivos de consumo y salida
DIR_CONSUMOS = Path("data/viviendas/consumos")  # REEMPLAZAR
DIR_SALIDA = Path("data/viviendas/por_mes_con_cluster")  # REEMPLAZAR
DIR_SALIDA.mkdir(exist_ok=True)

# 📄 Cargar cluster mensual por vivienda
df_clusters = pd.read_csv("data/viviendas_por_cluster_por_mes_sin2.csv")

# 🗂 Procesar cada archivo de consumo por vivienda
for file in DIR_CONSUMOS.glob("*_Consumos_????-??_????-??_*"):
    filename = file.stem

    # Extraer nombre de vivienda y fechas
    match = re.match(r"([A-Z]{3,})_Consumos_(\d{4})-(\d{2})_(\d{4})-(\d{2})", filename)
    if not match:
        print(f"⚠️ Nombre inválido: {file.name}")
        continue

    vivienda, y_ini, m_ini, y_fin, m_fin = match.groups()
    print(f"🏘 Procesando vivienda: {vivienda}")

    # Leer archivo de consumo
    df = pd.read_csv(file, sep=";")

    # Convertir columna 'date' a datetime
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["año"] = df["date"].dt.year
    df["mes"] = df["date"].dt.month

    for (año, mes), df_mes in df.groupby(["año", "mes"]):
        filas_clusters = df_clusters[
            (df_clusters["vivienda"] == vivienda) &
            (df_clusters["año"] == año) &
            (df_clusters["mes"] == mes)
        ]

        if filas_clusters.empty:
            print(f"⚠️ No hay cluster para {vivienda} en {año}-{mes:02}")
            continue

        for _, fila in filas_clusters.iterrows():
            n_clusters = fila["n_clusters"]
            cluster_id = fila["cluster"]

            colname = f"cluster_k{n_clusters}_{año}_{mes:02}"
            df_mes_copia = df_mes.copy()
            df_mes_copia[colname] = cluster_id

            nombre_archivo = f"{vivienda}_{año}_{mes:02}_cluster_k{n_clusters}.csv"
            df_mes_copia.to_csv(DIR_SALIDA / nombre_archivo, sep=";", index=False)
            print(f"✅ Guardado: {nombre_archivo}")
