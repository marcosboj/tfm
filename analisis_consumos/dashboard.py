import os, math
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ——————————— Funciones cacheadas ———————————

BASE = Path(__file__).parent
DATOS_CARPETA = BASE / "data" / "viviendas" / "consumos"

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    dfs = []
    for fn in sorted(os.listdir(DATOS_CARPETA)):
        if fn.endswith(".csv"):
            df = pd.read_csv(DATOS_CARPETA / fn, sep=";")
            df["hogar"] = fn[:-4]
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # --- Preprocesado de fechas/hora ---
    df["time"] = df["time"].replace("24:00","00:00")
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    # --- Columnas temporales necesarias para los boxplots ---
    df["year"]       = df["timestamp"].dt.year
    df["month"]      = df["timestamp"].dt.month
    df["month_year"] = df["timestamp"].dt.to_period("M").astype(str)

    df["dayofyear"]  = df["timestamp"].dt.dayofyear
    df["dayofweek"]  = df["timestamp"].dt.weekday
    df["hour"]       = df["timestamp"].dt.hour

    # season y day_type
    df["season"] = df["month"].map(
        lambda m: "invierno"   if m in (12,1,2) else
                  "primavera"  if m in (3,4,5) else
                  "verano"     if m in (6,7,8) else
                  "otoño"
    )
    df["day_type"] = df["dayofweek"].map(
        lambda wd: "fin de semana" if wd >= 5 else "entre semana"
    )

    return df


@st.cache_data
def pivot_global_month_hour(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        "consumptionKWh", index="month", columns="hour", aggfunc="mean"
    ).fillna(0)

@st.cache_data
def pivot_per_home_month_hour(df: pd.DataFrame, hogares: list) -> dict:
    out = {}
    for h in hogares:
        d = df[df["hogar"]==h]
        out[h] = d.pivot_table(
            "consumptionKWh", index="month", columns="hour", aggfunc="mean"
        ).fillna(0)
    return out

# ——————————— Carga inicial ———————————

df = load_data()
hogares = df["hogar"].unique().tolist()
sel_homes = hogares  # siempre todas

# ——————————— UI ———————————

st.title("🔎 Dashboard EDA Consumos")

# pestañas
tab_global, tab_hogar, tab_series = st.tabs([
    "Comparación Global",
    "Análisis por Hogar",
    "Curvas de Consumo"
])

# ——————————— PESTAÑA GLOBAL ———————————

with tab_global:
    st.subheader("🔀 Elige gráfico global")
    tipo = st.radio(
        "", ["Histograma","Boxplots","Heatmap global","Small multiples"],
        horizontal=True
    )

    if tipo == "Histograma":
        with st.spinner("Generando histograma…"):
            fig = px.histogram(
                df[df["hogar"].isin(sel_homes)],
                x="consumptionKWh", nbins=40,
                facet_col="hogar", facet_col_wrap=4,
                title="Histograma consumos horarios"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Boxplots":
        with st.spinner("Generando boxplots…"):
            for grp in ["dayofweek","day_type","hour","month","season","year","month_year"]:
                exp = st.expander(f"Boxplot vs {grp}", expanded=False)
                with exp:
                    fig = px.box(
                        df[df["hogar"].isin(sel_homes)],
                        x=grp, y="consumptionKWh",
                        facet_col="hogar", facet_col_wrap=4,
                        title=f"Consumo vs {grp}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Heatmap global":
        with st.spinner("Generando heatmap global…"):
            pivot_all = pivot_global_month_hour(df)
            fig = px.imshow(
                pivot_all,
                labels={"x":"Hora","y":"Mes","color":"kWh"},
                title="Heatmap mes vs hora (global)",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:  # Small multiples
        with st.spinner("Generando small multiples…"):
            pivots = pivot_per_home_month_hour(df, sel_homes)
            cols = int(math.sqrt(len(sel_homes)))
            rows = math.ceil(len(sel_homes)/cols)
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=sel_homes,
                horizontal_spacing=0.02, vertical_spacing=0.05
            )
            for i,h in enumerate(sel_homes):
                r = i//cols+1; c = i%cols+1
                hm = go.Heatmap(
                    z=pivots[h].values,
                    x=pivots[h].columns, y=pivots[h].index,
                    coloraxis="coloraxis", showscale=(i==0)
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

# ——————————— PESTAÑA ANÁLISIS POR HOGAR ———————————

with tab_hogar:
    st.subheader("🏠 Selecciona Hogar")
    hogar_sel = st.selectbox("Hogar", hogares)

    st.subheader(f"📊 Boxplots de {hogar_sel}")
    dfh = df[df["hogar"]==hogar_sel]
    # siempre dayofweek y day_type
    for grp in ["dayofweek","day_type","hour","month","season","year","month_year"]:
        exp = st.expander(f"Boxplot vs {grp}", expanded=False)
        with exp:
            fig = px.box(
                dfh, x=grp, y="consumptionKWh", points="outliers",
                title=f"{hogar_sel}: consumo vs {grp}"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"🌡️ Heatmap mes vs hora – {hogar_sel}")
    with st.spinner("Generando heatmap individual…"):
        pivot_h = pivot_per_home_month_hour(df, [hogar_sel])[hogar_sel]
        fig_hm = px.imshow(
            pivot_h,
            labels={"x":"Hora","y":"Mes","color":"kWh"},
            title=f"{hogar_sel}: consumo medio mes vs hora",
            aspect="auto"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

# ——————————— PESTAÑA CURVAS DE CONSUMO ———————————

with tab_series:
    st.subheader("📈 Curvas de consumo por vivienda")

    # rango de fechas seguro
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        st.error("No hay datos de fecha válidos.")
        st.stop()
    min_date, max_date = min_ts.date(), max_ts.date()

    # selector de fechas
    fecha_inicio, fecha_fin = st.date_input(
        "Rango de fechas", [min_date, max_date],
        min_value=min_date, max_value=max_date
    )
    # selector de viviendas
    viviendas_sel = st.multiselect(
        "Viviendas", hogares,
        default=[hogar_sel] if 'hogar_sel' in locals() else hogares[:3]
    )

    # filtrar y graficar
    if viviendas_sel:
        df_f = df[
            (df["hogar"].isin(viviendas_sel)) &
            (df["timestamp"].dt.date >= fecha_inicio) &
            (df["timestamp"].dt.date <= fecha_fin)
        ]
        if df_f.empty:
            st.warning("No hay datos para ese filtro.")
        else:
            fig = px.line(
                df_f, x="timestamp", y="consumptionKWh", color="hogar",
                title=f"Consumo horario [{fecha_inicio} → {fecha_fin}]"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona al menos una vivienda.")
