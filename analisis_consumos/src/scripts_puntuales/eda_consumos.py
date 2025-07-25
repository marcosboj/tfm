#!/usr/bin/env python3
# eda_consumos.py

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from pandas.plotting import autocorrelation_plot

def crear_carpetas(project_root: Path):
    """
    Crea las carpetas `results/csv/` y `results/plots/` dentro de project_root.
    """
    out_csv = project_root / "resultados/eda" / "csv"
    out_plots = project_root / "resultados/eda" / "plots"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)
    return out_csv, out_plots

def cargar_todos_consumos(carpeta: Path, sep: str = ';') -> pd.DataFrame:
    """
    Lee todos los CSV de la carpeta y devuelve un DataFrame concatenado,
    con una columna 'hogar' extraída del nombre de archivo.
    """
    archivos = sorted([f for f in os.listdir(carpeta) if f.endswith('.csv')])
    dfs = []
    for fn in archivos:
        ruta = carpeta / fn
        df = pd.read_csv(ruta, sep=sep)
        hogar = fn.split('.')[0]
        df['hogar'] = hogar
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def preparar_timestamp(df: pd.DataFrame,
                       date_col: str = 'date',
                       time_col: str = 'time') -> pd.DataFrame:
    """
    Corrige '24:00', une date+time en datetime y añade columnas útiles:
    date_only, year, month, season, day_type.
    """
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = df[time_col].replace('24:00', '00:00')
    df['timestamp'] = pd.to_datetime(
        df[date_col] + ' ' + df[time_col],
        format='%d/%m/%Y %H:%M',
        errors='coerce'
    )
    df['date_only'] = df['timestamp'].dt.date
    df['year']      = df['timestamp'].dt.year
    df['month']     = df['timestamp'].dt.month
    df['season']    = df['month'].map(lambda m:
                        'invierno'  if m in (12,1,2) else
                        'primavera' if m in (3,4,5) else
                        'verano'    if m in (6,7,8) else
                        'otoño')
    df['day_type']  = df['timestamp'].dt.weekday.map(
                        lambda wd: 'fin de semana' if wd >= 5 else 'entre semana')
    return df

def estadisticas_diarias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por hogar y fecha, calculando media, std, min, max y suma diaria.
    """
    grp = df.groupby(['hogar','date_only'])['consumptionKWh']
    agg = grp.agg(['mean','std','min','max','sum']).reset_index()
    agg = agg.rename(columns={
        'mean':'consumo_medio',
        'std':'consumo_std',
        'min':'consumo_min',
        'max':'consumo_max',
        'sum':'consumo_total'
    })
    return agg

def calcular_metricas_hogar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula coeficiente de variación, skewness y kurtosis por hogar.
    """
    grp = df.groupby('hogar')['consumptionKWh']
    cv   = grp.std() / grp.mean()
    skew = grp.apply(lambda x: ss.skew(x, nan_policy='omit'))
    kurt = grp.apply(lambda x: ss.kurtosis(x, nan_policy='omit'))
    dfm = pd.DataFrame({
        'hogar':    cv.index,
        'cv':       cv.values,
        'skewness': skew.values,
        'kurtosis': kurt.values
    })
    return dfm

def plot_hist_consumo_total_por_hogar(df_stats: pd.DataFrame, out_folder: Path):
    """
    Histograma del consumo total diario por hogar y guarda PNG.
    """
    plt.figure()
    plt.hist(df_stats['consumo_total'], bins=30)
    plt.xlabel('Consumo total diario (kWh)')
    plt.ylabel('Frecuencia')
    plt.title('Histograma: consumo total diario por hogar')
    plt.tight_layout()
    plt.savefig(out_folder / "hist_consumo_total_diario.png", dpi=300)
    plt.close()

def plot_boxplots_por_grupo(df: pd.DataFrame, columna: str, out_folder: Path):
    """
    Boxplot de consumptionKWh agrupado por la columna dada y guarda PNG.
    """
    plt.figure()
    df.boxplot(column='consumptionKWh', by=columna)
    plt.suptitle('')
    plt.xlabel(columna)
    plt.ylabel('consumptionKWh')
    plt.title(f'Boxplot consumo por {columna}')
    plt.tight_layout()
    nombre = f"boxplot_consumo_por_{columna}.png"
    plt.savefig(out_folder / nombre, dpi=300)
    plt.close()

def decompose_and_save(df: pd.DataFrame,
                       hogar: str,
                       out_folder: Path):
    """
    Descompone la serie diaria de un hogar:
      - si hay >= 365 días usa STL (estacionalidad anual)
      - elif >= 14 días usa seasonal_decompose(period=7) (estacionalidad semanal)
      - else omite la descomposición
    Guarda los componentes trend, seasonal y resid en PNG.
    """
    print(f"[EDA] Vivienda: {hogar}")
    serie = (df[df['hogar'] == hogar]
             .set_index('timestamp')
             .resample('D')['consumptionKWh']
             .sum()
             .interpolate())
    n = len(serie)
    print(f"  → {n} días de datos")

    if n >= 365:
        print("    * STL (period=365)")
        stl = STL(serie, period=365, robust=True).fit()
        comps = {'trend': stl.trend, 'seasonal': stl.seasonal, 'resid': stl.resid}
    elif n >= 14:
        print("    * seasonal_decompose (period=7)")
        dec = seasonal_decompose(serie, model='additive', period=7)
        comps = {'trend': dec.trend, 'seasonal': dec.seasonal, 'resid': dec.resid}
    else:
        print("    ! Muy pocos datos (<14 días): salto descomposición")
        return

    for name, comp in comps.items():
        plt.figure()
        comp.plot(title=f"{hogar} – {name}")
        plt.tight_layout()
        plt.savefig(out_folder / f"{hogar}_{name}.png", dpi=300)
        plt.close()

def plot_autocorr(df: pd.DataFrame, hogar: str, out_folder: Path):
    """
    Gráfico de autocorrelación de la serie diaria de un hogar y guarda PNG.
    """
    print(f"[EDA] Autocorrelación: {hogar}")
    serie = (df[df['hogar'] == hogar]
             .set_index('timestamp')
             .resample('D')['consumptionKWh']
             .sum()
             .interpolate())
    plt.figure()
    autocorrelation_plot(serie)
    plt.title(f"Autocorrelación: {hogar}")
    plt.tight_layout()
    plt.savefig(out_folder / f"autocorr_{hogar}.png", dpi=300)
    plt.close()

def main():
    # 1. Definir rutas y crear carpetas de salida
    project_root = Path.cwd()
    carpeta_data = project_root / "data" / "viviendas" / "consumos"
    out_csv, out_plots = crear_carpetas(project_root)

    # 2. Carga y preparación
    df_raw = cargar_todos_consumos(carpeta_data)
    df     = preparar_timestamp(df_raw)

    # 3. Estadísticas diarias
    df_diario = estadisticas_diarias(df)
    df_diario.to_csv(out_csv / "estadisticas_diarias.csv", index=False)

    # 4. Métricas por hogar
    df_metas = calcular_metricas_hogar(df)
    df_metas.to_csv(out_csv / "metadatos_hogar.csv", index=False)

    # 5. Gráficos generales
    plot_hist_consumo_total_por_hogar(df_diario, out_plots)
    for col in ['month', 'year', 'season', 'day_type']:
        plot_boxplots_por_grupo(df, col, out_plots)

    # 6. Descomposición y autocorrelación por hogar
    for hogar in df['hogar'].unique():
        try:
            decompose_and_save(df, hogar, out_plots)
            plot_autocorr(df, hogar, out_plots)
        except Exception as e:
            print(f"[ERROR] {hogar}: {type(e).__name__}: {e}")

    print("EDA completado. CSVs en 'results/csv/' y gráficos en 'results/plots/'.")
    
if __name__ == "__main__":
    main()

