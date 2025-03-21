import pandas as pd
####################RESUMEN VARIABLES
# Leer el archivo CSV con los clusters
df = pd.read_csv('resultado_clusters_acotado.csv')

# Identificar columnas numéricas y categóricas
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Calcular valores medios para columnas numéricas por cluster
numeric_summary = df.groupby('cluster')[numeric_features].mean()

# Calcular el modo y porcentaje para columnas categóricas por cluster
categorical_summary = {}
for col in categorical_features:
    summary = df.groupby('cluster')[col].apply(
        lambda x: x.mode()[0] if not x.mode().empty else "Sin datos"  # Manejo de grupos vacíos
    )
    percentages = df.groupby('cluster')[col].apply(
        lambda x: (x.value_counts(normalize=True).iloc[0] * 100) if not x.value_counts().empty else 0
    )
    # Combinar el modo y porcentaje en un DataFrame
    categorical_summary[col] = pd.DataFrame({
        f"{col}_Most_Frequent": summary,
        f"{col}_Percentage": percentages
    })

# Combinar todas las columnas categóricas en un DataFrame
categorical_summary_df = pd.concat(categorical_summary.values(), axis=1)

# Combinar los resúmenes numéricos y categóricos en un único DataFrame
final_summary = pd.concat([numeric_summary, categorical_summary_df], axis=1)

# Mostrar el resumen final
print("Resumen combinado de clusters:")
print(final_summary)

# Exportar el resumen combinado a un archivo CSV
final_summary.to_csv('resumen_clusters_combinado.csv', index=True, encoding='utf-8')
print("\nResumen de clusters guardado como 'resumen_clusters_combinado.csv'.")

####################ARBOL
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Separar las variables independientes (X) y dependientes (y)
X = df.drop(columns=['cluster'])
y = df['cluster']

# Identificar columnas categóricas
categorical_features = X.select_dtypes(include=['object']).columns

# Crear un pipeline para preprocesar los datos y manejar valores faltantes
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputar valores categóricos
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # Codificar categóricas
        ]), categorical_features),
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=['int64', 'float64']).columns)  # Imputar numéricas
    ]
)

# Crear un pipeline completo con preprocesamiento y modelo
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo con el pipeline
pipeline.fit(X_train, y_train)

# Evaluar el modelo
y_pred = pipeline.predict(X_test)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Visualizar el árbol de decisión
# Configurar el tamaño del gráfico para que sea más grande
plt.figure(figsize=(20, 15))  # Ajusta las dimensiones según lo necesario

# Dibujar el árbol
plot_tree(
    pipeline.named_steps['classifier'], 
    feature_names=pipeline.named_steps['preprocessor'].get_feature_names_out(),
    class_names=[str(c) for c in y.unique()],
    filled=True
)

# Guardar como imagen
output_path = "arbol_decision.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Guardar como PNG con alta resolución
print(f"El árbol de decisión se ha guardado como {output_path}")


import pandas as pd
import matplotlib.pyplot as plt

# Obtener importancia de las características
feature_importances = pd.DataFrame({
    'Feature': pipeline.named_steps['preprocessor'].get_feature_names_out(),
    'Importance': pipeline.named_steps['classifier'].feature_importances_
}).sort_values(by='Importance', ascending=False)

# Mostrar las 10 características más importantes
print("Importancia de las variables:")
print(feature_importances.head(10))

# Visualizar las importancias de las variables
plt.figure(figsize=(10, 6))
feature_importances.head(10).plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title("Importancia de las Variables en el Árbol de Decisión")
plt.ylabel("Importancia")
plt.xlabel("Características")
plt.tight_layout()
# Guardar como imagen
output_path_2 = "importancia_variables.png"
plt.savefig(output_path_2, dpi=300, bbox_inches="tight")  # Guardar como PNG con alta resolución

