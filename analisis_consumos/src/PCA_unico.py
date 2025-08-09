import os
import pandas as pd

# Directorio que contiene los CSV originales
input_dir = "resultados/pca_resumen"
# Nombre del CSV resultante
output_file = "resultados/merged_loadings_resumen.csv"

# Lista para acumular dataframes
df_list = []

# Recorremos todos los archivos del directorio
for filename in os.listdir(input_dir):
    # Solo procesamos los que terminen en .csv (o .CSV)
    if not filename.lower().endswith(".csv"):
        continue

    # Ruta completa al CSV
    filepath = os.path.join(input_dir, filename)
    # Nombre de estrategia = nombre de fichero sin extensión
    estrategia = os.path.splitext(filename)[0]

    # Leemos el CSV usando la primera columna como índice
    df = pd.read_csv(filepath, index_col=0)

    # Convertimos el índice en columna 'pc'
    df = df.reset_index().rename(columns={"index": "pc"})

    # Insertamos la columna 'estrategia' al principio
    df.insert(0, "estrategia", estrategia)

    df_list.append(df)

# Concatenamos todos los DataFrames y guardamos
if df_list:
    merged = pd.concat(df_list, ignore_index=True)
    merged.to_csv(output_file, index=False)
    print(f"Se han unido {len(df_list)} archivos en '{output_file}'")
else:
    print("No se encontró ningún archivo CSV en el directorio especificado.")
