"""
Descripción:
Descarga irradiancia forcasting.

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python meteogalicia.py, (archivo).nc
    Print(Forecasting irrandiancia)


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""

import netCDF4 as nc

# Cargar el archivo
dataset = nc.Dataset('wrf_arw_det_history_d01_20241017_0000.nc4.nc')

# Ver las variables disponibles
print(dataset.variables)

# Acceder a una variable (ej. swflx, irradiancia)
irradiancia = dataset.variables['swflx'][:]
#print(irradiancia)
punto_central = irradiancia[:, 1, 1]
print(punto_central)