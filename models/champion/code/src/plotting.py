"""
Funciones para generar gráficos de Plotly reutilizables.
"""
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COLOR_FRAUD, COLOR_LEGITIMATE, MODEL_COLORS


def plot_target_distribution(df: pd.DataFrame, target_col: str = "FraudFound_P") -> go.Figure:
    """Gráfico de torta con la distribución de la variable objetivo."""
    counts = df[target_col].value_counts()
    fig = px.pie(
        values=counts.values,
        names=["Legítimo", "Fraude"],
        title="Distribución de la variable objetivo",
        color_discrete_sequence=[COLOR_LEGITIMATE, COLOR_FRAUD],
        hole=0.5,
    )
    fig.update_traces(textinfo="percent+label+value")
    fig.update_layout(
        height=350,
        margin=dict(t=80, b=20, l=20, r=20),
        title_y=0.98,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )
    return fig


def plot_age_distribution(df: pd.DataFrame, target_col: str = "FraudFound_P") -> go.Figure:
    """Histograma de edad por clase de fraude."""
    return px.histogram(
        df, x="Age", color=target_col,
        barmode="overlay", nbins=40, opacity=0.7,
        color_discrete_map={0: COLOR_LEGITIMATE, 1: COLOR_FRAUD},
        title="Distribución por Age",
    )


def plot_fraud_rate_by_category(
    df: pd.DataFrame,
    column: str,
    target_col: str = "FraudFound_P",
    title: str = None,
    color_scale: str = "Reds",
    global_avg: float = None,
) -> go.Figure:
    """Barras con la tasa de fraude por categoría de una variable."""
    rate = df.groupby(column)[target_col].mean().sort_values(ascending=False) * 100
    fig = px.bar(
        x=rate.index, y=rate.values,
        title=title or f"Tasa de fraude (%) por {column}",
        labels={"x": column, "y": "Tasa fraude (%)"},
        color=rate.values, color_continuous_scale=color_scale,
    )
    if global_avg is not None:
        fig.add_hline(
            y=global_avg, line_dash="dash", line_color="gray",
            annotation_text="Promedio global",
        )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """Barras agrupadas comparando métricas de los 4 modelos."""
    metrics_long = metrics_df.melt(
        id_vars="Modelo", var_name="Métrica", value_name="Valor",
    )
    fig = px.bar(
        metrics_long, x="Métrica", y="Valor", color="Modelo",
        barmode="group", text_auto=".3f",
        title="Métricas comparadas — los 4 modelos",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=500, yaxis_range=[0, 1])
    return fig


def plot_radar_comparison(metrics_df: pd.DataFrame, best_model_name: str) -> go.Figure:
    """Gráfico radar de los 4 modelos sobre las 6 métricas."""
    radar_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    fig = go.Figure()

    for i, row in metrics_df.iterrows():
        suffix = " ⭐" if row["Modelo"] == best_model_name else ""
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in radar_metrics],
            theta=radar_metrics,
            fill="toself",
            name=row["Modelo"] + suffix,
            line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)]),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=500,
        title="Perfil de desempeño por modelo",
    )
    return fig


def plot_roc_pr_curves(curves: Dict[str, Dict], best_model_name: str) -> go.Figure:
    """Curvas ROC y PR superpuestas para los 4 modelos."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Curva ROC", "Curva Precision-Recall"),
    )

    for i, (name, c) in enumerate(curves.items()):
        suffix = " ⭐" if name == best_model_name else ""
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=c["fpr"], y=c["tpr"], mode="lines",
                name=f"{name}{suffix} (AUC={c['roc_auc']:.3f})",
                legendgroup=name, line=dict(color=color, width=3),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=c["recall"], y=c["precision"], mode="lines",
                name=f"{name} (AP={c['pr_auc']:.3f})",
                legendgroup=name, showlegend=False,
                line=dict(color=color, width=3),
            ),
            row=1, col=2,
        )

    # Diagonal de referencia ROC
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"), showlegend=False,
        ),
        row=1, col=1,
    )

    fig.update_xaxes(title_text="FPR", row=1, col=1)
    fig.update_yaxes(title_text="TPR", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_layout(height=500, title_text="ROC y PR — los 4 modelos")
    return fig


def plot_confusion_matrix(cm, title: str) -> go.Figure:
    """Matriz de confusión con etiquetas VP/VN/FP/FN."""
    labels_text = [
        [f"VN: {cm[0, 0]}", f"FP: {cm[0, 1]}"],
        [f"FN: {cm[1, 0]}", f"VP: {cm[1, 1]}"],
    ]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred. Legítimo", "Pred. Fraude"],
        y=["Real Legítimo", "Real Fraude"],
        text=labels_text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "black"},
        colorscale="Blues",
        showscale=False,
    ))
    fig.update_layout(
        title=title, height=380,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_risk_gauge(probability: float, model_name: str) -> go.Figure:
    """Indicador tipo gauge mostrando el riesgo de fraude."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": f"Riesgo de fraude (%) — {model_name}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLOR_FRAUD if probability > 0.5 else COLOR_LEGITIMATE},
            "steps": [
                {"range": [0, 30], "color": "#A8E6A1"},
                {"range": [30, 60], "color": "#FFE066"},
                {"range": [60, 100], "color": "#FFB3B3"},
            ],
        },
    ))
    return fig
