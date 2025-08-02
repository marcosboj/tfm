"""
Descripción:
    Script para iterar sobre una carpeta distintos archivos csv que constienen consumos historicos. Cada archivo contiente hasta los dos ultimos años de consumos hora por hora

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python llamada_consumos.py, (carpeta con distintos archivos csv)
    Devuelve las características de todos los consumos en un archivo: resumen_consumos.csv
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""

import pandas as pd
from pathlib import Path
import os
import re



path_actual = Path.cwd()
print(path_actual)
carpeta_now = path_actual
print(carpeta_now)
carpeta = carpeta_now / "data/viviendas/consumos"
print(carpeta)


archivos = [f for f in os.listdir(carpeta) if f.endswith('.csv')]
df_resultados = pd.DataFrame()

for archivo in archivos:
    ruta_completa = os.path.join(carpeta, archivo)
    print(ruta_completa)      
    import caracteristicas_consumo
    #filtro = "fechas"
    #filtro = "completo"
    #filtro = "mes"
    #filtro = "estacion"
    #filtro = "dia_semana"
    filtro = "tipo_dia"

    df_resultado, nombre_filtro = caracteristicas_consumo.caracteristicas_consumo(ruta_completa,filtro)
    match = re.search(r"^[A-Z]+", archivo)
    if match:
        parte_archivo = match.group(0)  
    else:
        parte_archivo = "Desconocido"  

    df_resultado.insert(0, 'archivo', parte_archivo)
   
    df_resultados = pd.concat([df_resultados, df_resultado], ignore_index=True)

nombre_archivo = f"resumen_consumos_{nombre_filtro}.csv"
ruta_completa = carpeta_now / "data/viviendas/resumenes"
ruta_completa_archivo = ruta_completa / nombre_archivo
df_resultados.to_csv(ruta_completa_archivo, index=False)
