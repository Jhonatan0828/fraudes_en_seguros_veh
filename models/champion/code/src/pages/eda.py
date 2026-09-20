"""
Página de EDA Interactivo — Análisis exploratorio con filtros dinámicos.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from ..config import TARGET_COLUMN
from ..mappings import EDA_FILTERS
from ..plotting import plot_age_distribution, plot_fraud_rate_by_category


def render(df: pd.DataFrame) -> None:
    """Renderiza la página de EDA interactivo."""
    st.header("📊 Análisis exploratorio interactivo")
    st.markdown(
        f"**{len(EDA_FILTERS)} filtros disponibles** — "
        "selecciona los valores que quieras incluir."
    )

    btn_col1, btn_col2, _ = st.columns([1, 1, 6])
    select_all = btn_col1.button("✅ Seleccionar todo")
    clear_all = btn_col2.button("❌ Limpiar todo")
    st.markdown("---")

    # Renderizar filtros en 3 columnas
    selections = {}
    cols = st.columns(3)
    for i, col_name in enumerate(EDA_FILTERS):
        opts = sorted(df[col_name].dropna().unique().tolist())
        if select_all:
            default = opts
        elif clear_all:
            default = []
        else:
            default = opts
        with cols[i % 3]:
            selections[col_name] = st.multiselect(
                col_name, opts, default=default, key=f"filter_{col_name}"
            )

    st.markdown("---")

    # Aplicar filtros
    mask = pd.Series([True] * len(df), index=df.index)
    for col_name, selected in selections.items():
        if selected:
            mask = mask & df[col_name].isin(selected)
    df_filtered = df[mask]

    if len(df_filtered) == 0:
        st.warning("No hay datos con los filtros seleccionados. Ajusta los filtros.")
        return

    # Métricas resumen
    m1, m2, m3 = st.columns(3)
    m1.metric("Filas filtradas", f"{len(df_filtered):,}")
    m2.metric("Tasa de fraude", f"{df_filtered[TARGET_COLUMN].mean()*100:.2f}%")
    m3.metric("% del total", f"{len(df_filtered)/len(df)*100:.1f}%")

    global_avg = df[TARGET_COLUMN].mean() * 100

    # Gráficos principales
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_age_distribution(df_filtered, TARGET_COLUMN),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plot_fraud_rate_by_category(
                df_filtered, "Make", TARGET_COLUMN,
                title="Tasa de fraude (%) por Make",
                global_avg=global_avg,
            ),
            use_container_width=True,
        )

    # Heatmap BasePolicy x Fault
    if df_filtered["BasePolicy"].nunique() > 0 and df_filtered["Fault"].nunique() > 0:
        st.subheader("Tasa de fraude por BasePolicy × Fault")
        pivot = df_filtered.pivot_table(
            index="BasePolicy", columns="Fault",
            values=TARGET_COLUMN, aggfunc="mean",
        ).round(4) * 100
        fig = px.imshow(
            pivot, text_auto=".2f", color_continuous_scale="Reds",
            title="Tasa de fraude (%) — BasePolicy × Fault",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tasa de fraude por cada variable
    st.subheader("Tasa de fraude por variable")
    vars_to_plot = [c for c in EDA_FILTERS if df_filtered[c].nunique() > 1]
    for i in range(0, len(vars_to_plot), 2):
        row_cols = st.columns(2)
        for j, col_name in enumerate(vars_to_plot[i:i + 2]):
            tasa = (
                df_filtered.groupby(col_name)[TARGET_COLUMN]
                .mean().sort_values(ascending=False) * 100
            ).reset_index()
            tasa.columns = [col_name, "Tasa fraude (%)"]
            fig = px.bar(
                tasa, x=col_name, y="Tasa fraude (%)",
                title=f"Tasa por {col_name}",
                color="Tasa fraude (%)", color_continuous_scale="Reds",
            )
            fig.add_hline(
                y=global_avg, line_dash="dash", line_color="gray",
                annotation_text="Promedio",
            )
            with row_cols[j]:
                st.plotly_chart(fig, use_container_width=True)
