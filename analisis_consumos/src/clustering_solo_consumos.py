"""
Descripción:
    Clustering de las variables de consumo obtenidas. No tiene en cuenta las de las encuestas.
    Algunas de las variables no las estoy utilizando para que tenga cada CUPS el mismo numero de variables.
    Hay casos en los que no tiene el mismo número porque por ejemplo una CUPS que tiene info solo de unos meses no puede tener el máx, media o suma de un mes en particular.


Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
Fecha: 24/03/2025

Desarrollo: Marcos Boj Pérez, Nora Barroso.

Código: Marcos Boj Pérez

Ejecución: python clustering_solo_consumos.py, datos_combinados.csv
    Cluster (pruebas varias)
    


Este software se proporciona "tal cual", sin ninguna garantía expresa o implícita.
This software is provided ""as-is,"" without any express or implied warranty.
"""


import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


path_actual = Path.cwd()
archivo_path = path_actual / "fichreos_consumo_y_potencias/Oliver/viviendas/consumos/datos_combinados.csv"
df_completo = pd.read_csv(archivo_path)


df_consumo = df_completo.iloc[:, :190]
df_consumo_sin_semanas = df_completo.iloc[:, :52]
df_consumo_sin_semanas = df_consumo_sin_semanas.drop([2, 6], axis=0)  # elimino datos que se van mucho



# --- Transformaciones Cíclicas ---
df_consumo_sin_semanas['hora_sin'] = np.sin(2 * np.pi * df_consumo_sin_semanas.index / 24)
df_consumo_sin_semanas['hora_cos'] = np.cos(2 * np.pi * df_consumo_sin_semanas.index / 24)

df_consumo_sin_semanas['dia_sin'] = np.sin(2 * np.pi * df_consumo_sin_semanas.index / 7)
df_consumo_sin_semanas['dia_cos'] = np.cos(2 * np.pi * df_consumo_sin_semanas.index / 7)

df_consumo_sin_semanas['mes_sin'] = np.sin(2 * np.pi * df_consumo_sin_semanas.index / 12)
df_consumo_sin_semanas['mes_cos'] = np.cos(2 * np.pi * df_consumo_sin_semanas.index / 12)

identificadores = df_consumo_sin_semanas.iloc[:, :2]  # Extrae la primera columna como Serie
df_sin_id = df_consumo_sin_semanas.iloc[:, 2:]  # Elimina la primera columna para clustering
print(identificadores)

#df_sin_id.dropna(inplace=True)
df_sin_id.fillna(df_sin_id.mean(), inplace=True)  # Rellenar con la media
#df_sin_id.dropna(axis=1, inplace=True)
#print(df_sin_id)


# --- Normalización de Datos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sin_id)
df_scaled = pd.DataFrame(X_scaled, columns=df_sin_id.columns, index=df_sin_id.index)
#print(df_scaled)




# --- Aplicar Clustering ---
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Convertimos clusters a Series antes de concatenarlo
df_scaled['cluster'] = pd.Series(clusters, index=df_scaled.index)


df_final = pd.concat([identificadores, df_scaled], axis=1)
#print(df_final)
ruta_completa_archivo = path_actual/ "fichreos_consumo_y_potencias/Oliver/viviendas/consumos/clustering_solo_consumos.csv"
df_final.to_csv(ruta_completa_archivo, index=False)


# --- Evaluación con Silhouette Score ---
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# --- Visualización con PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_final['pca_1'] = X_pca[:, 0]
df_final['pca_2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(df_final['pca_1'], df_final['pca_2'], c=df_final['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clustering de Viviendas Basado en Consumo')
plt.colorbar(label="Cluster")
plt.show()

# Probar diferentes valores de K
inertia = []
silhouette_scores = []
K_range = range(2, 10)  # Probar entre 2 y 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    
    from sklearn.metrics import silhouette_score
    silhouette_scores.append(silhouette_score(X_scaled, clusters))


# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='-')
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inertia (Suma de Distancias al Centroide)")
plt.title("Método del Codo para Seleccionar K")
plt.show()

# Graficar el Silhouette Score para diferentes K
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score en función de K")
plt.show()


from sklearn.ensemble import RandomForestClassifier

# Preparar datos
X = df_scaled.drop(columns=['cluster', ])  # Variables explicativas
y = df_scaled['cluster']  # Cluster como variable objetivo

# Entrenar un Random Forest para predecir clusters
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Obtener la importancia de las características
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)
#ruta_importances = path_actual/ "fichreos_consumo_y_potencias/Oliver/viviendas/consumos/importances.csv"
#importances.to_csv(ruta_importances, index=False)
pca_importance = pd.DataFrame(pca.components_, columns=X.columns, index=['PCA1', 'PCA2']).T
#ruta_pca_importance = path_actual/ "fichreos_consumo_y_potencias/Oliver/viviendas/consumos/pca_importance.csv"
#pca_importance.to_csv(ruta_pca_importance, index=False)
print(pca_importance.sort_values(by='PCA1', ascending=False))

# Calcular la media de cada característica por cluster
feature_importance = df_scaled.groupby('cluster').mean().std()

# Ordenar de mayor a menor importancia
feature_importance_sorted = feature_importance.sort_values(ascending=False)
print(feature_importance_sorted)
ruta_importances = path_actual/ "fichreos_consumo_y_potencias/Oliver/viviendas/consumos/feature_importance_sorted.csv"
df_feature_importance_sorted = feature_importance_sorted.to_frame()
df_feature_importance_sorted.to_csv(ruta_importances, index=True)