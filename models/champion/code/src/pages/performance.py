"""
Página de Desempeño — Comparativa de los 4 modelos entrenados.
"""
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from ..config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from ..metrics import compute_classification_metrics, get_best_model_name
from ..plotting import (
    plot_confusion_matrix, plot_metrics_comparison,
    plot_radar_comparison, plot_roc_pr_curves,
)


def _highlight_best_row(row, best_name):
    """Resalta en verde la fila del mejor modelo."""
    if row["Modelo"] == best_name:
        return ["background-color: #C6EFCE; color: #006100; font-weight: bold"] * len(row)
    return [""] * len(row)


def _highlight_max_per_column(s):
    """Resalta en amarillo el valor máximo de cada columna numérica."""
    if s.dtype.kind in "biufc":
        is_max = s == s.max()
        return [
            "background-color: #FFEB9C; font-weight: bold" if v else ""
            for v in is_max
        ]
    return [""] * len(s)


def render(df: pd.DataFrame, models: dict, selected_model_name: str) -> None:
    """Renderiza la página de comparación de modelos."""
    st.header("📈 Desempeño y comparación de los 4 modelos")
    st.markdown(
        "Métricas calculadas sobre el mismo conjunto de prueba "
        "(20% del dataset, estratificado)."
    )

    # Split de prueba
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    metrics_df, curves = compute_classification_metrics(models, X_te, y_te)
    best_name = get_best_model_name(metrics_df, by="PR-AUC")
    best_pr_auc = metrics_df.loc[metrics_df["Modelo"] == best_name, "PR-AUC"].values[0]

    st.success(
        f"⭐ **Mejor modelo (por PR-AUC):** {best_name} "
        f"con PR-AUC = {best_pr_auc:.4f}"
    )

    # ============ Tabla de métricas con resaltado ============
    st.subheader("📊 Tabla comparativa de métricas")

    styled = (
        metrics_df.style
        .apply(_highlight_best_row, best_name=best_name, axis=1)
        .apply(
            _highlight_max_per_column,
            subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        )
        .format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1": "{:.4f}",
            "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "🟩 Verde = mejor modelo global (por PR-AUC)  |  "
        "🟨 Amarillo = mejor valor por columna"
    )

    # ============ Visualizaciones ============
    st.subheader("📊 Comparación visual de indicadores")
    st.plotly_chart(plot_metrics_comparison(metrics_df), use_container_width=True)

    st.subheader("🎯 Vista radar — perfil de cada modelo")
    st.plotly_chart(
        plot_radar_comparison(metrics_df, best_name), use_container_width=True
    )

    st.subheader("📈 Curvas ROC y Precision-Recall")
    st.plotly_chart(plot_roc_pr_curves(curves, best_name), use_container_width=True)

    # ============ Matrices de confusión ============
    st.subheader("🧮 Matrices de confusión")
    st.caption(
        "**Convenciones:**  "
        "**VP** = Verdadero Positivo · **VN** = Verdadero Negativo · "
        "**FP** = Falso Positivo · **FN** = Falso Negativo"
    )
    cm_cols = st.columns(len(curves))
    for i, (name, c) in enumerate(curves.items()):
        with cm_cols[i]:
            suffix = " ⭐" if name == best_name else ""
            st.plotly_chart(
                plot_confusion_matrix(c["cm"], f"{name}{suffix}"),
                use_container_width=True,
            )
