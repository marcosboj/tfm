import datadis
import requests
import pandas as pd
from pathlib import Path

url = 'https://datadis.es/nikola-auth/tokens/login'
payload = {
    'username': 'XXXXXXX',  
    'password': 'XXXXXXX'   
}
response = requests.post(url, data=payload)

if response.status_code == 200:
    token = response.text

df = pd.read_excel("Registro de autorizados.xlsx")

n_fila = 0
for index, row in df.iloc[n_fila:].iterrows():
    CUPS = row['CUPS'] 
    NIF = row['DNI/NIE'] 
    n_supplie = int(row['n_suministro'])
    nombre_apellidos = row['Nombre y Apellido']
    print   
    datadis.datadis(token,NIF, CUPS, n_supplie, nombre_apellidos)