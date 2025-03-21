import pandas as pd

# Cargar el archivo Excel con los resultados del clustering
file_path = '/mnt/data/resultados_clustering4.xlsx'
data = pd.read_excel(file_path)

# Calcular estadísticas descriptivas para cada clúster
cluster_stats = data.groupby('Cluster').describe()

# Mostrar las estadísticas descriptivas de cada clúster
import ace_tools as tools; tools.display_dataframe_to_user(name="Cluster Statistics", dataframe=cluster_stats)

cluster_stats