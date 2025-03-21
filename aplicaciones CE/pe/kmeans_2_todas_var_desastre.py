# importar excel de casos
import pandas as pd
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


#quitar los valores que no son numeros de las columnas que deberian
# Limpiar las columnas especificadas
columnas_a_limpiar = ['Ingresos', 'Superficie', 'Consumo medio', 'Consumo factura revisada',
                      'Importe']
for col in columnas_a_limpiar:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convierte a numérico; reemplaza texto con NaN

## Columnas con ","
# Columnas que deseas convertir
columnas_a_convertir = ['Ahorro por cambio de potencia', 'Ahorro por mantenimiento', 'Ahorro por bono social']
# Reemplazar comas por puntos y convertir a float
for col in columnas_a_convertir:
    df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

import numpy as np
# Llenar valores faltantes con la media de cada columna
df = df.apply(lambda col: col.fillna(col.mean()) if col.dtype in [np.float64, np.int64] else col)
# Llenar valores faltantes con el modo para columnas categóricas
df = df.apply(lambda col: col.fillna(col.mode()[0]) if col.dtype == 'object' else col)
print(df.head())
### ESCALAR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Identificar columnas numéricas y categóricas automáticamente
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
print(numeric_features)
print(categorical_features)

# Validar y limpiar columnas numéricas
for col in numeric_features:
    # Reemplazar caracteres no numéricos y convertir a float
    df[col] = df[col].replace({',': ''}, regex=True)  # Eliminar comas
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir a numérico (NaN si no es convertible)
# Imputar valores faltantes en columnas numéricas con la media
for col in numeric_features:
    if df[col].isnull().any():  # Si hay NaN, imputar
        df[col].fillna(df[col].mean(), inplace=True)
# Validar y limpiar columnas categóricas
for col in categorical_features:
    df[col] = df[col].astype(str).fillna('desconocido')  # Asegurar tipo str y reemplazar NaN con 'desconocido'


print("Datos limpios:")
print(df)

# Preprocesador para escalar y codificar
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Imputar valores faltantes con la media
            ('scaler', StandardScaler())  # Escalar
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputar categóricas con la moda
            ('onehot', OneHotEncoder(drop='first'))  # Codificar categóricas
        ]), categorical_features)
    ]
)

# Construir el pipeline con K-Means
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=8, random_state=42,n_init='auto'))
])

# Ajustar el pipeline
pipeline.fit(df)

# Asignar etiquetas de clusters al DataFrame
df['cluster'] = pipeline.named_steps['kmeans'].labels_

print("\nClusters asignados:")
print(df)

