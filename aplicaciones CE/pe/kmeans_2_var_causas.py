import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_excel( "casos_IASS.xlsx")
'''
columnas_a_analizar = ['Provincia', 'Población', 'Adultos_H', 'Adultos_M', 'Adultos_Otro', 'Edad_0_5',
                       'Edad_6_15','Edad_16_17','Edad_18_30','Edad_31_40','Edad_41_55','Edad_56_70',
                       'Edad_mas_70','Ocupados','Parados','Estudiantes','Jubilados','Pension incapacidad',
                       'Pensión viudedaz','Tareas hogar','Ingresos', 'Familia numerosa',
                       'Pensión mínima','IMV','Situaciones especiales','Superficie','Calificación - antigüedad',
                       'Propiedad', 'Calefacción_Sistema','Calefacción_Suministro', 'Cocina eléctrica','Frigorifico escarcha',
                       'Frigorífico cierra','Entiendes factura','Mercado','Potencia P1','Potencia P2','Bono social','Al corriente pago',
                       'Consumo medio','Consumo factura revisada','Importe']
'''
columnas_a_analizar = ['Fecha','Adultos_H', 'Adultos_M', 'Adultos_Otro', 'Edad_0_5',
                       'Edad_6_15','Edad_16_17','Edad_18_30','Edad_31_40','Edad_41_55','Edad_56_70',
                       'Edad_mas_70','Ocupados','Parados','Estudiantes','Jubilados','Pension incapacidad',
                       'Pensión viudedaz','Tareas hogar','Ingresos', 'Familia numerosa',
                       'Pensión mínima','IMV','Situaciones especiales','Superficie','Calificación - antigüedad',
                       'Propiedad', 'Calefacción_Sistema','Calefacción_Suministro', 'Cocina eléctrica','Frigorifico escarcha',
                       'Frigorífico cierra','Entiendes factura','Mercado',
                       'Consumo medio','Consumo factura revisada','Importe']
#elimino las columnas
df = df.loc[:, columnas_a_analizar]
##############3
# Convertir la columna 'Fecha' a tipo datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')
# Definir la fecha límite
fecha_limite = pd.to_datetime('01/01/2023', format='%d/%m/%Y')
# Filtrar el DataFrame eliminando las filas con fechas anteriores
df = df[df['Fecha'] >= fecha_limite]
# Eliminar la columna 'Fecha'
df = df.drop(columns=['Fecha'])
####################3
df['Importe'] = pd.to_numeric(df['Importe'].astype(str).str.replace(',', '.'), errors='coerce')

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


# Guardar el DataFrame con los clusters en un archivo CSV
output_path = "resultado_clusters_acotado.csv"  # Cambia el nombre del archivo si lo deseas
df.to_csv(output_path, index=False, encoding='utf-8')
######################################################################################################################
# Centroides escalados
centroids = pipeline.named_steps['kmeans'].cluster_centers_

# Recuperar escalador para desescalar numéricas
scaler = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']

# Separar las columnas numéricas y categóricas
num_features = pipeline.named_steps['preprocessor'].named_transformers_['num'].feature_names_in_
cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].feature_names_in_

# Desescalar las columnas numéricas de los centroides
num_centroids = scaler.inverse_transform(centroids[:, :len(num_features)])
num_centroids_df = pd.DataFrame(num_centroids, columns=num_features)

# Procesar los valores de las categóricas
cat_centroids = centroids[:, len(num_features):]
cat_centroids_df = pd.DataFrame(cat_centroids, columns=pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())

# Combinar numéricas y categóricas
centroid_df = pd.concat([num_centroids_df, cat_centroids_df], axis=1)


centroid_df.to_csv("centroides_desescalados.csv", index=False, encoding='utf-8')
######################################################################################################################
# Transformar los datos con el pipeline
X_transformed = pipeline.named_steps['preprocessor'].transform(df)

# Recuperar los transformadores del preprocesador
scaler = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
onehot_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']

# Desescalar columnas numéricas
numeric_columns_descaled = scaler.inverse_transform(X_transformed[:, :len(scaler.mean_)])
numeric_df_descaled = pd.DataFrame(numeric_columns_descaled, columns=numeric_features)

# Decodificar columnas categóricas
categorical_columns_encoded = X_transformed[:, len(scaler.mean_):]  # Seleccionar columnas categóricas codificadas
categorical_columns_decoded = onehot_encoder.inverse_transform(categorical_columns_encoded)
categorical_df_decoded = pd.DataFrame(categorical_columns_decoded, columns=categorical_features)

# Combinar los datos desescalados y decodificados
X_descaled = pd.concat([numeric_df_descaled, categorical_df_decoded], axis=1)
X_descaled['cluster'] = df['cluster'].reset_index(drop=True)


# Exportar a CSV
output_path = "datos_desescalados.csv"
X_descaled.to_csv(output_path, index=False, encoding='utf-8')