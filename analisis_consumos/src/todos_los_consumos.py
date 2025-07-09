# concat_consumos.py
import pandas as pd, glob, os

carpeta = "data/viviendas/consumos"
archivos = glob.glob(os.path.join(carpeta, "*.csv"))
dfs = []
for f in archivos:
    df = pd.read_csv(f, sep=";")
    df["hogar"] = os.path.splitext(os.path.basename(f))[0]
    dfs.append(df)
pd.concat(dfs, ignore_index=True).to_csv(
    os.path.join(carpeta, "todos_los_consumos.csv"),
    index=False
)
