import pandas as pd
import os

# 1. Leer CSV
csv_path = 'data/viviendas/consumos/FJAG_Consumos_2023-01_2024-12.csv'
df = pd.read_csv(csv_path, sep=';', dtype={'consumptionKWh': float})

# 2. Calcular umbral: percentil 98.2% de consumptionKWh
PERCENTIL = 0.9825
threshold = df['consumptionKWh'].quantile(PERCENTIL)
print(f"Umbral (p{PERCENTIL*100:.1f}): {threshold:.3f} kWh")

# 3. Crear columna temporal cleanConsumptionKWh
df['cleanConsumptionKWh'] = df['consumptionKWh'].where(
    df['consumptionKWh'] <= threshold,
    df['consumptionKWh'] - threshold
)

# 4. Reemplazar consumptionKWh por la versión limpia
df['consumptionKWh'] = df['cleanConsumptionKWh']

# 5. Eliminar las columnas auxiliares
df = df.drop(columns=['cleanConsumptionKWh'])

# 6. Guardar CSV sobrescribiendo o en nuevo archivo
output_dir = 'data/viviendas/consumos'
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'FJAG_Consumos_2023-01_2024-12_sinPicos_982.csv')
df.to_csv(out_path, sep=';', index=False)

print(f"✔️ CSV actualizado guardado en: {out_path}")

