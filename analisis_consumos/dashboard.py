import os
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ———————————— Carga y preprocesado cacheado ————————————

# Por esto:
BASE = Path(__file__).parent
DATOS_CARPETA = BASE / "data" / "viviendas" / "consumos"

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Lee todos los CSV, añade columnas temporales y devuelve un DataFrame."""
    dfs = []
    for fn in sorted(os.listdir(DATOS_CARPETA)):
        if fn.endswith(".csv"):
            df = pd.read_csv(DATOS_CARPETA / fn, sep=";")
            df["hogar"] = fn[:-4]
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # arregla horas
    df["time"] = df["time"].replace("24:00", "00:00")
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )
    df["date_only"]  = df["timestamp"].dt.date
    df["year"]       = df["timestamp"].dt.year
    df["month"]      = df["timestamp"].dt.month
    df["dayofyear"]  = df["timestamp"].dt.dayofyear
    df["dayofweek"]  = df["timestamp"].dt.weekday
    df["hour"]       = df["timestamp"].dt.hour
    df["month_year"] = df["timestamp"].dt.to_period("M").astype(str)
    df["season"]     = df["month"].map(lambda m:
                         "invierno"   if m in (12,1,2) else
                         "primavera"  if m in (3,4,5) else
                         "verano"     if m in (6,7,8) else
                         "otoño")
    df["day_type"]   = df["dayofweek"].map(lambda wd:
                         "fin de semana" if wd >= 5 else "entre semana")
    return df

@st.cache_data
def pivot_global_month_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot global mes vs hora (media consumo)."""
    return df.pivot_table(
        index="month",
        columns="hour",
        values="consumptionKWh",
        aggfunc="mean"
    ).fillna(0)

@st.cache_data
def pivot_per_home_month_hour(df: pd.DataFrame, hogares: list) -> dict:
    """
    Devuelve un dict {hogar: pivot_df} con pivot mes vs hora
    para cada hogar de la lista.
    """
    out = {}
    for h in hogares:
        d = df[df["hogar"] == h]
        out[h] = d.pivot_table(
            index="month",
            columns="hour",
            values="consumptionKWh",
            aggfunc="mean"
        ).fillna(0)
    return out

# ———————————— Carga inicial ————————————

df = load_data()
hogares = df["hogar"].unique().tolist()

# ———————————— Interfaz Streamlit ————————————

st.title("🔎 Dashboard EDA Consumos")

# Selector de hogar para análisis individual
hogar_sel = st.sidebar.selectbox("🏠 Selecciona Hogar", hogares)

# Dimensiones para boxplots individuales
group_opt = st.sidebar.multiselect(
    "🔖 Agrupar por (individual)",
    ["hour","dayofweek","month","year","month_year","season","day_type"],
    default=["hour","season"]
)

# Slider para small multiples globales
max_homes = st.sidebar.slider(
    "Número de hogares en small multiples",
    min_value=4, max_value=len(hogares), value=min(16, len(hogares))
)
sel_homes = hogares[:max_homes]

# Pestañas: ahora tres
tab_global, tab_hogar, tab_series = st.tabs([
    "Comparación Global",
    "Análisis por Hogar",
    "Curvas de Consumo"
])

# ———————————— PESTAÑA GLOBAL ————————————
with tab_global:
    st.subheader("🔀 Selecciona el tipo de gráfico global")
    graf_global = st.radio(
        "¿Qué quieres ver?",
        ["Histograma", "Boxplots", "Heatmap global", "Small multiples"],
        index=0,
        horizontal=True
    )

    if graf_global == "Histograma":
        with st.spinner("Generando histograma…"):
            fig = px.histogram(
                df[df["hogar"].isin(sel_homes)],
                x="consumptionKWh", nbins=40,
                facet_col="hogar", facet_col_wrap=4,
                title="Histograma consumos horarios"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif graf_global == "Boxplots":
        with st.spinner("Generando boxplots…"):
            for grp in ["dayofweek","day_type","hour","month","season","year","month_year"]:
                st.write(f"**Boxplot vs {grp}**")
                fig = px.box(
                    df[df["hogar"].isin(sel_homes)],
                    x=grp, y="consumptionKWh",
                    facet_col="hogar", facet_col_wrap=4
                )
                st.plotly_chart(fig, use_container_width=True)

    elif graf_global == "Heatmap global":
        with st.spinner("Generando heatmap global…"):
            pivot_all = pivot_global_month_hour(df)
            fig = px.imshow(
                pivot_all,
                labels={"x":"Hora","y":"Mes","color":"kWh"},
                title="Heatmap global mes vs hora",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif graf_global == "Small multiples":
        with st.spinner("Generando small multiples…"):
            pivots = pivot_per_home_month_hour(df, sel_homes)
            cols = int(math.sqrt(len(sel_homes)))
            rows = math.ceil(len(sel_homes) / cols)
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=sel_homes,
                horizontal_spacing=0.02, vertical_spacing=0.05
            )
            for i, h in enumerate(sel_homes):
                r = i//cols +1; c = i%cols +1
                hm = go.Heatmap(
                    z=pivots[h].values,
                    x=pivots[h].columns,
                    y=pivots[h].index,
                    coloraxis="coloraxis",
                    showscale=(i==0)
                )
                fig.add_trace(hm, row=r, col=c)
                fig.update_xaxes(showticklabels=False, row=r, col=c)
                fig.update_yaxes(showticklabels=False, row=r, col=c)
            fig.update_layout(
                coloraxis=dict(colorbar=dict(title="kWh")),
                height=rows*200, width=cols*250,
                title="Small multiples: mes vs hora"
            )
            st.plotly_chart(fig, use_container_width=True)

# ———————————— PESTAÑA INDIVIDUAL ————————————

with tab_hogar:
    st.subheader(f"📊 Boxplots de {hogar_sel}")
    df_h = df[df["hogar"] == hogar_sel]
    # incluye siempre dayofweek y day_type
    dims = list(dict.fromkeys(group_opt + ["dayofweek", "day_type"]))
    for grp in dims:
        fig = px.box(
            df_h, x=grp, y="consumptionKWh", points="outliers",
            title=f"{hogar_sel}: consumo vs {grp}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"🌡️ Heatmap mes vs hora – {hogar_sel}")
    pivot_h = pivot_per_home_month_hour(df, [hogar_sel])[hogar_sel]
    fig_hm = px.imshow(
        pivot_h,
        labels={"x":"Hora","y":"Mes","color":"kWh"},
        title=f"{hogar_sel}: consumo medio mes/hora",
        aspect="auto"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    
# ———————————— PESTAÑA SERIES DE CONSUMO ————————————
with tab_series:
    st.subheader("📈 Curvas de consumo por vivienda")

    # --- Calcular min/max de forma segura ---
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        st.error("No hay datos de fecha válidos en el DataFrame.")
        st.stop()
    min_date = min_ts.date()
    max_date = max_ts.date()

    # 1) Rango de fechas
    fecha_inicio, fecha_fin = st.date_input(
        "Selecciona rango de fechas",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Aseguramos que sean dos fechas
    if isinstance(fecha_inicio, tuple) or isinstance(fecha_inicio, list):
        fecha_inicio, fecha_fin = fecha_inicio

    # 2) Selección de viviendas
    viviendas_sel = st.multiselect(
        "Selecciona viviendas",
        options=hogares,
        default=[hogar_sel]  # o [] si prefieres vacío
    )

    # 3) Filtrado y gráfico
    if len(viviendas_sel) > 0:
        df_f = df[
            (df["hogar"].isin(viviendas_sel)) &
            (df["timestamp"].dt.date >= fecha_inicio) &
            (df["timestamp"].dt.date <= fecha_fin)
        ].copy()

        if df_f.empty:
            st.warning("No hay datos en ese rango para las viviendas seleccionadas.")
        else:
            # Si tus datos son horarios, puedes agrupar por día:
            # df_plot = df_f.groupby(
            #     ["hogar", df_f["timestamp"].dt.date]
            # )["consumptionKWh"].sum().reset_index(name="kWh")
            # fig = px.line(df_plot, x="timestamp", y="kWh", color="hogar",
            #               labels={"timestamp":"Fecha", "kWh":"Consumo (kWh)"})

            # O directamente la serie horaria:
            fig = px.line(
                df_f,
                x="timestamp",
                y="consumptionKWh",
                color="hogar",
                labels={
                    "timestamp":"Fecha y hora",
                    "consumptionKWh":"Consumo (kWh)",
                    "hogar":"Vivienda"
                },
                title=f"Consumo horario [{fecha_inicio} → {fecha_fin}]"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona al menos una vivienda para mostrar la curva.")
