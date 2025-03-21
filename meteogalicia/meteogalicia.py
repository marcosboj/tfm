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