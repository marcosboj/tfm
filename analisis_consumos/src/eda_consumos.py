#!/usr/bin/env python3
# eda_consumos.py

import os
import math
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as ss

from statsmodels.tsa.seasonal import seasonal_decompose, STL
from pandas.plotting import autocorrelation_plot

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ————————————————— Ajustes globales de Matplotlib —————————————————
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['figure.dpi']     = 150
mpl.rcParams['savefig.dpi']    = 150
mpl.rcParams['font.size']      = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['legend.fontsize']= 9
mpl.rcParams['xtick.labelsize']= 9
mpl.rcParams['ytick.labelsize']= 9
mpl.rcParams['figure.constrained_layout.use'] = True

# ————————————————— Input / Output —————————————————

def crear_carpetas(project_root: Path):
    out_csv   = project_root / "resultados" / "eda" / "csv"
    out_plots = project_root / "resultados" / "eda" / "plots"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)
    return out_csv, out_plots

def cargar_todos_consumos(carpeta: Path, sep: str = ';') -> pd.DataFrame:
    archivos = sorted([f for f in os.listdir(carpeta) if f.endswith('.csv')])
    dfs = []
    for fn in archivos:
        df = pd.read_csv(carpeta / fn, sep=sep)
        df['hogar'] = fn.split('_')[0]
        dfs.append(df)
    # 1. concatenamos todo
    full_df = pd.concat(dfs, ignore_index=True)

    ########################################
    # 1) Crea la serie de fechas datetime (para el cálculo interno, NO crea columna)
    _dates = pd.to_datetime(full_df['date'], dayfirst=True)

    # 2) Sustituye “24:00:00” por “00:00” sólo para parsear
    _times = full_df['time'].replace({'24:00:00': '00:00'})

    # 3) Construye el timestamp local, sumando 1 día si time era “24:00:00”
    _full_local = (
        pd.to_datetime(
            _dates.dt.strftime('%Y-%m-%d') + ' ' + _times,
            format='%Y-%m-%d %H:%M',
            errors='coerce'
        )
        + pd.to_timedelta(full_df['time'].eq('24:00:00').astype(int), unit='d')
    )

    # 4) Localiza en Europe/Madrid y convierte a UTC, guardando en la misma columna
    full_df['timestamp'] = (
        _full_local
        .dt.tz_localize('Europe/Madrid', ambiguous=False, nonexistent='shift_forward')
        .dt.tz_convert('UTC')
    )
    ########################################
    # 3. eliminamos filas sin timestamp válido
    n_nan = full_df['timestamp'].isna().sum()
    if n_nan:
        print(f"Atención: eliminando {n_nan} filas sin timestamp válido")
        full_df = full_df.dropna(subset=['timestamp'])


    # 5. Filtrar por rango de fechas deseado (UTC)
    start_date = pd.Timestamp("2024-07-01 00:00", tz="UTC")
    end_date = pd.Timestamp("2025-06-30 23:00", tz="UTC")
    full_df = full_df[(full_df['timestamp'] >= start_date) & (full_df['timestamp'] <= end_date)]
    print(f"Filtrado por fecha: quedan {len(full_df)} filas entre {start_date.date()} y {end_date.date()}.")

        # 4. detectamos duplicados reales (mismo hogar y timestamp)
    mask_dup = full_df.duplicated(subset=['hogar','timestamp'], keep=False)
    df_dups = full_df[mask_dup].sort_values(['hogar','timestamp'])
    if not df_dups.empty:
        # Mostrar un resumen de los duplicados antes de eliminarlos
        print("=== Estos son los registros duplicados (se eliminará uno de cada par) ===")
        print(df_dups[['hogar','timestamp','consumptionKWh']].head(50))
        print(f"Total filas marcadas como duplicadas: {len(df_dups)}")
        # Mostrar cuántos pares únicos (hogar, timestamp) hay
        unique_pairs = df_dups[['hogar','timestamp']].drop_duplicates()
        print(f"Total pares (hogar, timestamp) duplicados: {len(unique_pairs)}")

        # Ahora sí, eliminamos duplicados dejando solo la primera ocurrencia
        full_df = full_df.drop_duplicates(subset=['hogar','timestamp'], keep='first')
        print(f"Tras eliminación, quedan {len(full_df)} filas en total.")


    return full_df

# ————————————————— Preprocesado —————————————————

def preparar_timestamp(df: pd.DataFrame,project_root: Path) -> pd.DataFrame:
    df = df.copy()
    '''
    df['time'] = df['time'].replace('24:00:00', '00:00')
    df['timestamp'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        format='%d/%m/%Y %H:%M',
        errors='coerce'
    )
    '''
    df['date_only']  = df['timestamp'].dt.date
    df['year']       = df['timestamp'].dt.year
    df['month']      = df['timestamp'].dt.month
    df['dayofyear']  = df['timestamp'].dt.dayofyear
    df['dayofweek']  = df['timestamp'].dt.weekday
    df['hour']       = df['timestamp'].dt.hour
    df['month_year'] = df['timestamp'].dt.strftime('%Y-%m')
    df['season']     = df['month'].map(lambda m:
                         'invierno' if m in (12,1,2) else
                         'primavera' if m in (3,4,5) else
                         'verano' if m in (6,7,8) else
                         'otoño')
    
    festivos = (
        pd.read_csv(project_root/'data'/'festivos_zgz.csv', parse_dates=['fecha'])
          ['fecha']
          .dt.date
          .tolist()
    )
    # ahora: festivo si está en CSV ó es sábado/domingo; en otro caso, laboral
    df['day_type'] = np.where(
        df['date_only'].isin(festivos) | df['dayofweek'].isin([5, 6]),
        'festivo',
        'laborable'
    )
    return df

# ————————————————— Estadísticas y métricas —————————————————



def estadisticas_diarias(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(['hogar','date_only'])['consumptionKWh']
    agg = g.agg(['mean','std','min','max','sum']).reset_index()
    return agg.rename(columns={
        'mean':'consumo_medio','std':'consumo_std',
        'min':'consumo_min','max':'consumo_max','sum':'consumo_total'
    })


def descriptive_by_house(
    df: pd.DataFrame,
    temporal_col: str = None
) -> pd.DataFrame:
    """
    Si temporal_col es None: descriptivos globales por hogar.
    Si no: descriptivos por hogar y por cada categoría de temporal_col.
    """
    if temporal_col is None:
        g = df.groupby('hogar')['consumptionKWh']
        dfh = pd.DataFrame({
            'hogar':  g.mean().index,
            'mean':   g.mean().values,
            'median': g.median().values,
            'min':    g.min().values,
            'max':    g.max().values,
            'std':    g.std().values,
            'sum':    g.sum().values,
            'p25':    g.quantile(0.25).values,
            'p50':    g.quantile(0.50).values,
            'p75':    g.quantile(0.75).values,
            'skew':   g.apply(lambda x: ss.skew(x, nan_policy='omit')).values,
            'kurt':   g.apply(lambda x: ss.kurtosis(x, nan_policy='omit')).values,
            'n_obs':  g.count().values
        })
        dfh['cv'] = dfh['std'] / dfh['mean']
        return dfh.reset_index(drop=True)

    # con temporal_col
    g = df.groupby(['hogar', temporal_col])['consumptionKWh']
    dfh = pd.DataFrame({
        'hogar':      g.mean().index.get_level_values('hogar'),
        temporal_col: g.mean().index.get_level_values(temporal_col),
        'mean':       g.mean().values,
        'median':     g.median().values,
        'min':        g.min().values,
        'max':        g.max().values,
        'std':        g.std().values,
        'sum':        g.sum().values,
        'p25':        g.quantile(0.25).values,
        'p50':        g.quantile(0.50).values,
        'p75':        g.quantile(0.75).values,
        'skew':       g.apply(lambda x: ss.skew(x, nan_policy='omit')).values,
        'kurt':       g.apply(lambda x: ss.kurtosis(x, nan_policy='omit')).values,
        'n_obs':      g.count().values
    })
    dfh['cv'] = dfh['std'] / dfh['mean']
    return dfh.reset_index(drop=True)

# ————————————————— Outliers —————————————————

def detect_outliers_daily(df: pd.DataFrame, out_folder: Path, threshold: float = 3.0) -> pd.DataFrame:
    d = estadisticas_diarias(df)
    d['z'] = d.groupby('hogar')['consumo_total'] \
              .transform(lambda x: (x - x.mean()) / x.std())
    out = d[d['z'].abs() > threshold]
    out.to_csv(out_folder/"outliers_diarios.csv", index=False)
    return out

def detect_outliers_weekly(df: pd.DataFrame, out_folder: Path, threshold: float = 3.0) -> pd.DataFrame:
    d = df.dropna(subset=['timestamp']).copy()
    d['week'] = (
        d['timestamp']
        .dt.tz_convert(None)           # dejamos el datetime naive
        .dt.to_period('W')             # creamos PeriodArray semanal
        .dt.to_timestamp()            # volvemos a Timestamp (naive)
        .dt.date                       # sólo la fecha
    )
    w = d.groupby(['hogar','week'])['consumptionKWh'] \
         .sum().reset_index(name='weekly_sum')
    w['z'] = w.groupby('hogar')['weekly_sum'] \
               .transform(lambda x: (x - x.mean()) / x.std())
    out = w[w['z'].abs() > threshold]
    out.to_csv(out_folder/"outliers_semanales.csv", index=False)
    return out

# ————————————————— Descomposición + Autocorrelación —————————————————

def decompose_and_autocorr(df: pd.DataFrame, hogar: str, out_folder: Path):
    serie = (
        df[df['hogar'] == hogar]
        .set_index('timestamp')['consumptionKWh']
        .resample('D').sum()
        .interpolate()
    )
    n = len(serie)
    if n < 14:
        return
    if n >= 365:
        dec = STL(serie, period=365, robust=True).fit()
    else:
        dec = seasonal_decompose(serie, model='additive', period=7)

    fig, axs = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True)
    dec.trend.plot(ax=axs[0], title='Trend')
    dec.seasonal.plot(ax=axs[1], title='Seasonal')
    dec.resid.plot(ax=axs[2], title='Residual')
    autocorrelation_plot(serie, ax=axs[3])
    axs[3].set_title('Autocorrelación')
    axs[0].set_ylabel('kWh')
    axs[1].set_ylabel('kWh')
    axs[2].set_ylabel('kWh')
    axs[3].set_xlabel('Lag')
    axs[3].set_ylabel('Correlation')
    fig.suptitle(f'{hogar} – Descomposición & Autocorrelación', y=0.95)
    fig.savefig(out_folder/f"{hogar}_decompose_autocorr.png", dpi=300)
    plt.close(fig)

# ————————————————— Matrices de histogramas y boxplots por hogar —————————————————

def plot_hist_matrix_por_hogar(df: pd.DataFrame, out_folder: Path, bins=30):
    hogares = df['hogar'].unique()
    n = len(hogares); cols = math.ceil(math.sqrt(n)); rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False)
    for ax, hogar in zip(axes.flatten(), hogares):
        data = df[df['hogar']==hogar]['consumptionKWh']
        ax.hist(data, bins=bins)
        ax.set_title(hogar, fontsize=9)
        ax.set_xlabel('kWh', fontsize=7)
        ax.set_ylabel('freq', fontsize=7)
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    fig.suptitle('Histogramas por hogar')
    fig.savefig(out_folder/"histogramas_por_hogar.png", dpi=300)
    plt.close(fig)

def plot_boxplot_matrix_por_hogar(df: pd.DataFrame, group_col: str, out_folder: Path):
    df2 = df.copy()
    if group_col == 'month_year':
        df2['month_year'] = df2['timestamp'].dt.strftime('%Y-%m')
    hogares = df2['hogar'].unique()
    n = len(hogares); cols = math.ceil(math.sqrt(n)); rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False)
    for ax, hogar in zip(axes.flatten(), hogares):
        dfh = df2[df2['hogar']==hogar]
        dfh.boxplot(column='consumptionKWh', by=group_col, ax=ax, showfliers=False)
        ax.set_title(hogar, fontsize=8)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(axis='x', labelrotation=45)
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    fig.suptitle(f'Boxplots por hogar vs {group_col}')
    fig.savefig(out_folder/f"boxplots_por_hogar_{group_col}.png", dpi=300)
    plt.close(fig)

# ————————————————— Líneas de media por grupo —————————————————

def plot_matrix_line_por_hogar(df: pd.DataFrame, group_col: str, out_folder: Path):
    hogares = df['hogar'].unique()
    n = len(hogares); cols = math.ceil(math.sqrt(n)); rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False)
    for ax, hogar in zip(axes.flatten(), hogares):
        d = df[df['hogar']==hogar]
        m = d.groupby(group_col)['consumptionKWh'].mean()
        ax.plot(m.index, m.values)
        ax.set_title(hogar, fontsize=9)
        ax.set_xlabel(group_col, fontsize=7)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel('kWh', fontsize=7)
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    fig.suptitle(f'Media consumo vs {group_col}')
    fig.savefig(out_folder/f"matrix_line_por_{group_col}.png", dpi=300)
    plt.close(fig)

# ————————————————— Consumo medio por día del año (todas viviendas) —————————————————

def plot_all_homes_dayofyear_per_year(df: pd.DataFrame, out_folder: Path):
    df['date'] = df['timestamp'].dt.floor('D')
    pivot = df.groupby(['dayofyear','hogar'])['consumptionKWh'].mean().unstack()
    plt.figure()
    for hogar in pivot.columns:
        plt.plot(pivot.index, pivot[hogar], alpha=0.6, label=hogar)
    plt.xlim(1,365)
    plt.xlabel('Día del año')
    plt.ylabel('kWh')
    plt.title(f'Consumo medio por día del año')
    plt.legend(fontsize=6, ncol=3)
    plt.savefig(out_folder/f"all_homes_dayofyear.png", dpi=300)
    plt.close()

# ————————————————— Heatmaps mes vs hora matriz —————————————————

def plot_heatmap_matrix_month_hour(df: pd.DataFrame, out_folder: Path, suffix: str=""):
    hogares = df['hogar'].unique()
    n = len(hogares); cols = math.ceil(math.sqrt(n)); rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False)
    im = None
    for ax, hogar in zip(axes.flatten(), hogares):
        d = df[df['hogar']==hogar]
        pivot = d.groupby(['month','hour'])['consumptionKWh'].mean().unstack().fillna(0)
        im = ax.imshow(pivot.values, aspect='auto', origin='lower')
        ax.set_title(hogar, fontsize=9)
        ax.set_xlabel('Hora', fontsize=7)
        ax.set_ylabel('Mes', fontsize=7)
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    fig.suptitle('Heatmap mes vs hora por hogar')
    fig.colorbar(im, ax=axes, fraction=0.02)
    fname = f"heatmap_matrix_month_hour{('_'+suffix) if suffix else ''}.png"
    fig.savefig(out_folder/fname, dpi=300)
    plt.close(fig)

# ————————————————— Main —————————————————

def main():
    root = Path.cwd()
    data_dir = root/"data"/"viviendas"/"consumos"
    out_csv, out_plots = crear_carpetas(root)

    df = cargar_todos_consumos(data_dir)
    df = preparar_timestamp(df, root)

    estadisticas_diarias(df).to_csv(out_csv/"estadisticas_diarias.csv", index=False)

    # descriptivos globales
    descriptive_by_house(df).to_csv(out_csv/"descriptivos_por_hogar.csv", index=False)

    # en lugar de los boxplots, llamas a descriptive_by_house con cada temporal
    for col in ['dayofweek','day_type','season','month_year']:
        descriptive_by_house(df, temporal_col=col) \
            .to_csv(out_csv/f"descriptivos_por_hogar_{col}.csv", index=False)

    # Histogramas y boxplots globales por hogar
    plot_hist_matrix_por_hogar(df, out_plots, bins=40)
    for col in ['dayofweek','day_type','season','month_year']:
        plot_boxplot_matrix_por_hogar(df, col, out_plots)

    # Tendencias/patrones lineales
    for col in ['hour','dayofweek','month_year','season','day_type']:
        plot_matrix_line_por_hogar(df, col, out_plots)

    # Descomposición + Autocorrelación por hogar
    for hogar in df['hogar'].unique():
        decompose_and_autocorr(df, hogar, out_plots)

    # Consumo medio día-año por año
    plot_all_homes_dayofyear_per_year(df, out_plots)

    # Heatmaps mes vs hora global y por año
    plot_heatmap_matrix_month_hour(df, out_plots)


    # Outliers
    detect_outliers_daily(df, out_csv)
    detect_outliers_weekly(df, out_csv)

    print("EDA completo. CSV en resultados/eda/csv/, plots en resultados/eda/plots/")

if __name__ == "__main__":
    main()