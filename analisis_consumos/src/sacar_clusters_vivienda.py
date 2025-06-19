import re
from pathlib import Path
import pandas as pd

# Ruta a los archivos
LOG_DIR = Path("logs/24_meses/todos_por_ventana/k_2_3_4")  # Cambia esta ruta a tu ubicación real
datos = []

for log_file in LOG_DIR.glob("mes_??_????_k*_c*.txt"):
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extraer metadatos del nombre
    match = re.search(r"mes_(\d{2})_(\d{4})_k(\d+)", log_file.name)
    if not match:
        print(f"⚠️ Nombre no válido: {log_file.name}")
        continue

    mes, anio, n_clusters = match.groups()
    mes = int(mes)
    anio = int(anio)
    n_clusters = int(n_clusters)

    # Buscar dinámicamente las líneas que contienen clusters
    for i, linea in enumerate(lines):
        linea = linea.strip()
        if linea.startswith("Cluster"):
            # Extraer número de cluster
            cluster_match = re.match(r"Cluster\s+(\d+)", linea)
            if not cluster_match:
                continue

            cluster_id = int(cluster_match.group(1))
            viviendas = re.findall(r"\b[A-Z]{3,}\b", linea)

            print(f"📄 {log_file.name} - Cluster {cluster_id} → Viviendas: {viviendas}")

            for vivienda in viviendas:
                datos.append({
                    "vivienda": vivienda,
                    "año": anio,
                    "mes": mes,
                    "n_clusters": n_clusters,
                    "cluster": cluster_id
                })


# Crear DataFrame
df_clusters = pd.DataFrame(datos)
df_clusters.sort_values(by=["vivienda", "año", "mes"], inplace=True)

# Guardar CSV
df_clusters.to_csv("data/viviendas_por_cluster_por_mes.csv", index=False)
print("✅ Archivo generado: viviendas_por_cluster_por_mes.csv")
