"""
Descripción:
    "Desescalamiento" de los clusters

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python desescalar.py,resultados_clustering_marcos2.xlsx
    datos_desescalados_clustering.xlsx


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Cargar los datos del clustering
df_clustering = pd.read_excel('resultados_clustering_marcos2.xlsx')

# Lista de columnas a analizar (las mismas que se usaron para el preprocesamiento)
columnas_a_analizar = ['CP', 'Adultos_H', 'Adultos_M', 'Adultos_Otro', 'Edad_0_5',
                       'Edad_6_15', 'Edad_16_17', 'Edad_18_30', 'Edad_31_40', 'Edad_41_55', 'Edad_56_70',
                       'Edad_mas_70', 'Ocupados', 'Parados', 'Estudiantes', 'Jubilados', 'Tareas hogar',
                       'Ingresos']

# Filtrar el DataFrame para incluir solo estas columnas (y la columna 'Cluster')
df_filtrado = df_clustering[columnas_a_analizar + ['Cluster']]

# Identificar columnas para cada tipo de escalador
minmax_features = ['CP']
standard_features = [col for col in columnas_a_analizar if col not in minmax_features]

# Crear el transformador de columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('minmax', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', MinMaxScaler())
        ]), minmax_features),
        ('standard', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), standard_features)
    ])

# Ajustar el preprocesador a los datos originales (sin el clustering)
df_original = pd.read_excel('casos.xlsx')[columnas_a_analizar]
preprocessor.fit(df_original)

# Obtener el escalador usado en el preprocesamiento
minmax_scaler = preprocessor.named_transformers_['minmax'].named_steps['scaler']
standard_scaler = preprocessor.named_transformers_['standard'].named_steps['scaler']

# Aplicar la transformación inversa a los datos
preprocessed_data = preprocessor.transform(df_filtrado[columnas_a_analizar])
original_minmax_data = minmax_scaler.inverse_transform(preprocessed_data[:, :len(minmax_features)])
original_standard_data = standard_scaler.inverse_transform(preprocessed_data[:, len(minmax_features):])

# Combinar los datos desescalados
original_data = pd.DataFrame(
    data=np.hstack((original_minmax_data, original_standard_data)),
    columns=minmax_features + standard_features
)

# Agregar la columna 'Cluster' al DataFrame desescalado
original_data['Cluster'] = df_filtrado['Cluster'].values

# Guardar el DataFrame resultante en un archivo Excel nuevo, con cada clúster en una hoja separada
with pd.ExcelWriter('datos_desescalados_clustering.xlsx') as writer:
    for cluster in original_data['Cluster'].unique():
        cluster_data = original_data[original_data['Cluster'] == cluster]
        cluster_data.to_excel(writer, sheet_name=f'Cluster_{cluster}', index=False)

print("Datos desescalados y guardados en 'datos_desescalados_clustering.xlsx'")
