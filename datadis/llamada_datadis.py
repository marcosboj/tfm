
"""
Descripción:
  Scrip para iterar sobre un excel para consuultar consumos y potencias a traves de la API de DATADIS, se utiliza otro script datadis.py

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python llamada_datadis.py, datadis.py, Registro de autorizados uno.xlsx


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""

import datadis
import requests
import pandas as pd
from pathlib import Path
# Paso 1: Obtener el token de autenticación
url = 'https://datadis.es/nikola-auth/tokens/login'

# Definir los datos de autenticación
payload = {
    'username': 'G50503523',  # Tu NIF
    'password': 'Ecodes1992'   # Tu contraseña
}

# Realizar la solicitud POST
response = requests.post(url, data=payload)
print(response)

if response.status_code == 200:
    # Extraer el token de autenticación del cuerpo de la respuesta
    token = response.text



# Leer el archivo Excel
# Obtener el path de la carpeta actual
path_actual = Path.cwd()
# Obtener el path de la carpeta anterior
carpeta_anterior = path_actual.parent
# Construir el path del archivo en la carpeta anterior (por ejemplo, archivo.csv)
archivo_path = carpeta_anterior / 'Registro de autorizados uno.xlsx'
df = pd.read_excel("Registro de autorizados uno.xlsx")

n_fila = 0
#ultima fila descargada del excel/ columna id
# Recorrer cada fila del DataFrame utilizando el bucle for
for index, row in df.iloc[n_fila:].iterrows():
    # Obtener información de columnas específicas
    CUPS = row['CUPS'] 
    NIF = row['DNI/NIE'] 
    n_supplie = int(row['n_suministro'])
    #nombre_apellidos = f"{row['Nombre']} {row['Apellidos']}"
    nombre_apellidos = row['Nombre y Apellido']
    print   
    datadis.datadis(token,NIF, CUPS, n_supplie, nombre_apellidos)