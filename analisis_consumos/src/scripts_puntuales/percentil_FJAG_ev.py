import pandas as pd

# 1. Carga de datos
csv_path = 'data/viviendas/consumos/FJAG_Consumos_2023-01_2024-12.csv'
df = pd.read_csv(csv_path, sep=';', dtype={'consumptionKWh': float})

# 2. Percentiles del 98.0% al 100.0% de 0.1 en 0.1
percentiles = [i/1000 for i in range(980, 1001)]

# 3. Calcular cuantiles
q = df['consumptionKWh'].quantile(percentiles)

# 4. Imprimir resultados y contar casos por encima de cada cuantil
total = len(df)
for p, val in q.items():
    count = (df['consumptionKWh'] >= val).sum()
    pct_cases = count / total * 100
    print(f"{p*100:.1f}th percentile: {val:.3f} kWh  │  "
          f"count ≥ threshold: {count} ({pct_cases:.2f} %)")
