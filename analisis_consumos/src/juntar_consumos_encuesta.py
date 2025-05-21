"""
Descripción:
    Junta en una misma tabla los datos de la encuesta realzada a cada CUPS con sus caracteristicas de consumo obtenidas. Todo en una misma fila.

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python juntar_consumos_encuesta.py, encuesta_consumos_oliver.csv, resumen_consumos.csv
    datos_combinados.csv
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""

import pandas as pd
from pathlib import Path

path_actual = Path.cwd()
#archivo_path = path_actual / 'Registro de autorizados Oliver.xlsx'
#df_registros = pd.read_excel(archivo_path)
#df_nif_cups = df_registros[["DNI/NIE","CUPS"]]

archivo_encuesta = path_actual / "data/encuesta_consumos_oliver.csv"
archivo_consumo =  path_actual / "data/viviendas/resumen_consumos.csv"


df_consumos = pd.read_csv(archivo_consumo)
df_encuesta = pd.read_csv(archivo_encuesta)


df_consumos = df_consumos.rename(columns={"archivo": "ID"})
#df_encuesta = df_encuesta.rename(columns={"ID": "nif"})
#df_nif_cups = df_nif_cups.rename(columns={"CUPS": "cups", "DNI/NIE": "nif"})
#print(df_consumos['ID'].duplicated().sum())  # ¿Hay IDs duplicados en df_consumos?
#print(df_encuesta['ID'].duplicated().sum())  # ¿Hay IDs duplicados en df_encuesta?
# **1️⃣ Unir datos de consumo con la tabla de relación usando CUPS**
df_merged = pd.merge(df_consumos, df_encuesta, on="ID", how="left")
'''
print("IDs en df_consumos:")
print(sorted(df_consumos["ID"].unique()))

print("\nIDs en df_encuesta:")
print(sorted(df_encuesta["ID"].unique()))
print("IDs en df_consumos:")
print(sorted(df_consumos["ID"].unique()))

print("\nIDs en df_encuesta:")
print(sorted(df_encuesta["ID"].unique()))
'''
# **2️⃣ Unir el resultado con las características de vivienda usando DNI**
#df_juntos = pd.merge(df_merged, df_encuesta, on="nif", how="left")
#columnas = ["nif"] + [col for col in df_juntos.columns if col != "nif"]
#df_juntos = df_juntos[columnas]

#print(df_merged)

# **3️⃣ Guardar el resultado en un nuevo CSV**
#df_merged.to_csv("datos_combinados.csv", index=False)

ruta_completa_archivo = path_actual/ "data/datos_combinados.csv"
df_merged.to_csv(ruta_completa_archivo, index=False)