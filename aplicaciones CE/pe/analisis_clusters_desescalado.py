import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('datos_desescalados.csv')

# Identificar columnas numéricas y categóricas
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Calcular la media para las columnas numéricas
numeric_summary = df.groupby('cluster')[numeric_features].mean()

# Calcular el modo y el porcentaje para las columnas categóricas
categorical_summary = {}
for col in categorical_features:
    mode = df.groupby('cluster')[col].apply(lambda x: x.mode()[0] if not x.mode().empty else "Sin datos")
    percentage = df.groupby('cluster')[col].apply(
        lambda x: (x.value_counts(normalize=True).iloc[0] * 100) if not x.value_counts().empty else 0
    )
    categorical_summary[col] = pd.DataFrame({
        f"{col}_Most_Frequent": mode,
        f"{col}_Percentage": percentage
    })

# Unificar las estadísticas de las variables categóricas en un DataFrame
categorical_summary_combined = pd.concat(categorical_summary.values(), axis=1)

# Combinar las estadísticas numéricas y categóricas en un único DataFrame
final_summary = pd.concat([numeric_summary, categorical_summary_combined], axis=1)

# Exportar a un solo archivo CSV
output_path = "resumen_clusters_unificado.csv"
final_summary.to_csv(output_path, index=True, encoding='utf-8')


