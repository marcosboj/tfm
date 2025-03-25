"""
Descripción:
    Modelo Kmeans para analizar casos herramienta ENERSOC (ECODES)

Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python kmeans_2_todas_var.py,casos_IASS.xlsx
    resultados_cluster.csv
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_excel( "casos_IASS.xlsx")

#elimino columnas que seguro no me van a hacer falta
columnas_a_eliminar = ['Proyecto','Id','Fecha', 'Admin', 'Usuario', 'Incidencia', 'Entregado',
                       'Atención', 'CP','Cód. Identif. Usuario',
                       'Atención', 'CP','Cód. Identif. Usuario',
                        'Fecha factura revisada', 'Financiador', 'Compañía recomendada', 'Contrato recomendado',
                        '¿Aplica bono social?', 'Tipo bono social que aplica']
df = df.drop(columns=columnas_a_eliminar)


#Eliminar 0 en las que no tiene sentido
df['IMV'] = df['IMV'].replace({0: 'no', 1: 'si'})

# Identificar columnas numéricas y categóricas automáticamente
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

# Limpieza de columnas categóricas
for col in categorical_features:
    df[col] = df[col].astype(str).fillna('desconocido')  # Convertir a str y reemplazar NaN

# Preprocesador para limpiar, escalar y codificar
preprocessor = ColumnTransformer(
    transformers=[
        # Procesar columnas numéricas
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Imputar valores faltantes
            ('scaler', StandardScaler())  # Escalar
        ]), numeric_features),

        # Procesar columnas categóricas
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputar valores faltantes
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # Codificar categóricas
        ]), categorical_features)
    ]
)

# Construir el pipeline completo con K-Means
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=6, random_state=42, n_init='auto'))
])

# Ajustar el pipeline
pipeline.fit(df)

# Asignar etiquetas de clusters al DataFrame
df['cluster'] = pipeline.named_steps['kmeans'].labels_

print("\nClusters asignados:")
print(df)
# Guardar el DataFrame con los clusters en un archivo CSV
output_path = "resultado_clusters.csv"  # Cambia el nombre del archivo si lo deseas
df.to_csv(output_path, index=False, encoding='utf-8')