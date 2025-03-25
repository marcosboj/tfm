"""
Descripción:
    Consulta api aemet temperaturas

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python temperaturas.py
    temperaturas.csv

    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""


import requests
import json
import pandas as pd
import os

# URL donde están los datos
url_datos = "https://opendata.aemet.es/opendata/sh/ca111ff0"

# Hacer una solicitud HTTP para obtener los datos
response = requests.get(url_datos)

# Comprobar si la solicitud fue exitosa
if response.status_code == 200:
    # Parsear el contenido JSON
    datos = response.json()
    
    # Crear una lista para almacenar las fechas y tmed
    registros = []
    
    # Extraer solo los campos de interés (fecha y tmed)
    for registro in datos:
        fecha = registro['fecha']
        # Reemplazar la coma por punto para que sea un número decimal válido
        tmed = float(registro['tmed'].replace(',', '.'))
        registros.append({'fecha': fecha, 'tmed': tmed})
    
    # Convertir la lista en un DataFrame
    df = pd.DataFrame(registros)
    
    # Nombre del archivo CSV donde se guardarán los datos
    nombre_archivo = 'temperaturas.csv'
    
    # Comprobar si el archivo ya existe para agregar datos
    if os.path.exists(nombre_archivo):
        # Si el archivo existe, agregamos los nuevos datos sin sobrescribir los anteriores
        df.to_csv(nombre_archivo, mode='a', header=False, index=False)
    else:
        # Si el archivo no existe, lo creamos con los datos actuales
        df.to_csv(nombre_archivo, mode='w', header=True, index=False)

    print(f"Datos guardados exitosamente en {nombre_archivo}")
else:
    print(f"Error al obtener los datos: {response.status_code}")
