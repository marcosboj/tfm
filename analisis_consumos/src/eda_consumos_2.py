import os, math
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ——————————— Funciones cacheadas ———————————

DATOS_CARPETA = Path("data/viviendas/consumos")

@st.cache_data(show_spinner=False)
def load_raw_consumos() -> pd.DataFrame:
    """Carga y preprocesa todos los CSV sólo una vez."""
    dfs = []
    for fn in sorted(os.listdir(DATOS_CARPETA)):
        if fn.endswith(".csv"):
            df = pd.read_csv(DATOS_CARPETA/fn, sep=";")
            df["hogar"] = fn[:-4]
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["time"] = df["time"].replace("24:00","00:00")
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d/%m/%Y %H:%M", errors="coerce"
    )
    df["month"]     = df["timestamp"].dt.month
    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.weekday
    df["day_type"]  = df["dayofweek"].map(lambda wd: "finde" if wd>=5 else "lab")
    return df

@st.cache_data
def pivot_global_month_hour(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table("consumptionKWh", index="month", columns="hour", aggfunc="mean").fillna(0)

@st.cache_data
def pivot_per_home_month_hour(df: pd.DataFrame, hogares: list) -> dict:
    out = {}
    for h in hogares:
        d = df[df["hogar"]==h]
        out[h] = d.pivot_table("consumptionKWh", index="month", columns="hour", aggfunc="mean").fillna(0)
    return out

# ——————————— Carga inicial + selectores ———————————

df = load_raw_consumos()
hogares = df["hogar"].unique().tolist()

st.sidebar.title("⚙️ Configuración")
hogar_sel = st.sidebar.selectbox("🏠 Hogar (individual)", hogares)
max_homes = st.sidebar.slider("Máximo hogares small multiples", 4, len(hogares), 12)

# Páginas
page = st.sidebar.radio("🔎 Sección", ["Global", "Individual"])

# ——————————— PÁGINA GLOBAL ———————————

if page == "Global":
    st.title("📊 Comparación Global")
    sel_homes = hogares[:max_homes]

    with st.spinner("📈 Generando histogramas…"):
        fig = px.histogram(
            df[df["hogar"].isin(sel_homes)],
            x="consumptionKWh", nbins=40,
            facet_col="hogar", facet_col_wrap=4,
            title="Histograma consumos horarios"
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.spinner("📦 Generando boxplots…"):
        for grp in ["dayofweek","day_type"]:
            fig = px.box(
                df[df["hogar"].isin(sel_homes)],
                x=grp, y="consumptionKWh",
                facet_col="hogar", facet_col_wrap=4,
                title=f"Boxplot vs {grp}"
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.spinner("🌡️ Generando heatmap global…"):
        pivot_all = pivot_global_month_hour(df)
        fig = px.imshow(
            pivot_all, aspect="auto",
            labels={"x":"Hora","y":"Mes","color":"kWh"},
            title="Consumo medio mes vs hora (global)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.spinner("🔳 Generando small multiples de heatmaps…"):
        pivots = pivot_per_home_month_hour(df, sel_homes)
        cols = int(math.sqrt(len(sel_homes)))
        rows = math.ceil(len(sel_homes) / cols)
        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=sel_homes,
                            horizontal_spacing=0.02,
                            vertical_spacing=0.05)
        for i,h in enumerate(sel_homes):
            r = i//cols+1; c = i%cols+1
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
        fig.update_layout(coloraxis=dict(colorbar=dict(title="kWh")),
                          height=rows*200, width=cols*250)
        st.plotly_chart(fig, use_container_width=True)

# ——————————— PÁGINA INDIVIDUAL ———————————

else:
    st.title(f"🏠 Análisis de {hogar_sel}")
    dfh = df[df["hogar"]==hogar_sel]

    with st.spinner("📦 Boxplots individuales…"):
        for grp in ["dayofweek","day_type"]:
            fig = px.box(
                dfh, x=grp, y="consumptionKWh",
                title=f"{hogar_sel}: consumo vs {grp}"
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.spinner("🌡️ Heatmap individual…"):
        pivot = pivot_per_home_month_hour(df, [hogar_sel])[hogar_sel]
        fig = px.imshow(
            pivot, aspect="auto",
            labels={"x":"Hora","y":"Mes","color":"kWh"},
            title=f"{hogar_sel}: consumo medio mes vs hora"
        )
        st.plotly_chart(fig, use_container_width=True)

    # …añade aquí más secciones cacheadas si lo deseas…

