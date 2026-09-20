"""
Página de Inicio — Resumen ejecutivo del proyecto.
"""
import pandas as pd
import streamlit as st

from ..mappings import TOP_PREDICTORS
from ..metrics import compute_classification_metrics
from ..plotting import plot_target_distribution


def render(df: pd.DataFrame, models: dict, metadata: dict) -> None:
    """Renderiza la página de Inicio con resumen ejecutivo."""
    from sklearn.model_selection import train_test_split

    from ..config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE

    st.header("📋 Resumen Ejecutivo del Proyecto")

    # Cálculo de métricas para los KPIs
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    metrics_df, _ = compute_classification_metrics(models, X_te, y_te)

    best_model_name = metadata["best_model_name"]
    best_row = metrics_df[metrics_df["Modelo"] == best_model_name].iloc[0] \
        if best_model_name in metrics_df["Modelo"].values else metrics_df.iloc[0]

    # ============ KPIs ============
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Reclamaciones", f"{len(df):,}")
    k2.metric("Tasa de fraude", f"{df[TARGET_COLUMN].mean()*100:.2f}%")
    k3.metric("Variables", df.shape[1] - 1)
    k4.metric("Modelos", len(models))
    k5.metric("Mejor PR-AUC", f"{best_row['PR-AUC']:.3f}")
    k6.metric("Mejor Recall", f"{best_row['Recall']:.3f}")

    st.markdown("---")

    # ============ Resumen / Problemática / Contexto ============
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🎯 Resumen del Proyecto")
        st.markdown("""
        Análisis y modelado predictivo para la **detección automática de fraude**
        en reclamaciones de seguros vehiculares, utilizando el dataset
        *Vehicle Insurance Claim Fraud Detection* (Oracle/Kaggle) con
        15,420 registros del período 1994-1996.

        Se aplicaron técnicas de **EDA**, **preprocesamiento con
        Pipelines de Scikit-Learn** y se entrenaron **4 modelos de clasificación**
        comparándolos sobre métricas adecuadas para el desbalance de clases.
        """)

        st.subheader("⚠️ Problemática")
        st.markdown("""
        El fraude en seguros vehiculares representa **pérdidas multimillonarias**
        para la industria aseguradora a nivel global y encarece indirectamente
        las primas de los asegurados honestos.

        La detección manual es **lenta, costosa y propensa a errores humanos**,
        lo que hace inviable revisar el 100% de las reclamaciones que recibe
        una compañía aseguradora.
        """)

    with col_b:
        st.subheader("💼 Contexto de Negocio")
        st.markdown("""
        La aseguradora necesita un **sistema de soporte a la decisión** que asigne
        una *probabilidad de fraude* a cada reclamación nueva, permitiendo a los
        analistas **priorizar la revisión** de los casos de mayor riesgo.

        Un modelo predictivo bien calibrado:
        - **Reduce costos operativos** (menos casos a revisar manualmente)
        - **Aumenta la tasa de detección** sin generar fricción con clientes legítimos
        - **Genera trazabilidad** para auditorías y cumplimiento regulatorio
        """)

        st.plotly_chart(plot_target_distribution(df, TARGET_COLUMN), use_container_width=True)

    st.markdown("---")

    # ============ Metodología ============
    st.subheader("🔬 Metodología aplicada")
    cols = st.columns(5)
    steps = [
        ("1️⃣ Comprensión", "Definición del problema, exploración inicial del dataset y diccionario de variables."),
        ("2️⃣ Preprocesamiento", "Tratamiento de datos faltantes, codificación ordinal y exclusión de variables (Year, AgeOfPolicyHolder, PolicyNumber)."),
        ("3️⃣ EDA", "Análisis univariado y bivariado, correlaciones, tasas de fraude por categoría."),
        ("4️⃣ Modelado", "Pipelines con StandardScaler, OneHotEncoder y 4 modelos comparados."),
        ("5️⃣ Interpretación", "Coeficientes, importancias, permutation importance y SHAP values."),
    ]
    for col, (title, caption) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(caption)

    st.markdown("---")

    # ============ Mejor modelo ============
    st.subheader("🏆 Mejor modelo seleccionado")
    mod_col1, mod_col2 = st.columns([1, 2])

    with mod_col1:
        st.markdown(f"### 🥇 {best_model_name}")
        st.markdown("""
        **Modelo seleccionado** tras comparar 4 modelos
        candidatos sobre el conjunto de prueba.

        **Justificación técnica:**
        El modelo fue seleccionado tras una comparación
        sistemática contra Logistic Regression, Random Forest
        y Logistic Regression + SMOTE, evaluados sobre el mismo
        conjunto de prueba con métricas apropiadas para datos
        desbalanceados. El modelo obtuvo el mayor PR-AUC,
        métrica más fiable que accuracy o ROC-AUC en
        escenarios con clases asimétricas como el nuestro.

        **Métricas en el set de prueba:**
        """)

        c1, c2 = st.columns(2)
        c1.metric("PR-AUC", f"{best_row['PR-AUC']:.4f}")
        c2.metric("ROC-AUC", f"{best_row['ROC-AUC']:.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Recall", f"{best_row['Recall']:.4f}")
        c4.metric("F1-Score", f"{best_row['F1']:.4f}")

    with mod_col2:
        import plotly.express as px
        best_metrics = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
            "Valor": [
                best_row["Accuracy"], best_row["Precision"], best_row["Recall"],
                best_row["F1"], best_row["ROC-AUC"], best_row["PR-AUC"],
            ],
        })
        fig = px.bar(
            best_metrics, x="Métrica", y="Valor",
            text_auto=".3f",
            title=f"Métricas del modelo ganador — {best_model_name}",
            color="Valor", color_continuous_scale="Greens",
        )
        fig.update_layout(height=380, yaxis_range=[0, 1], coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 Para ver la comparativa con los otros 3 modelos candidatos, "
        "ve a la página **📈 Desempeño del modelo**."
    )

    st.markdown("---")

    # ============ Top variables predictoras ============
    st.subheader("⭐ Top variables predictoras")
    st.dataframe(pd.DataFrame(TOP_PREDICTORS), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ============ Aplicación práctica ============
    st.subheader("💡 Aplicación práctica del modelo")
    app_cols = st.columns(3)
    with app_cols[0]:
        st.info(
            "**🟢 Probabilidad < 30%**\n\n"
            "**Riesgo bajo.** Procesamiento automático sin revisión adicional. "
            "Reduce carga operativa al equipo."
        )
    with app_cols[1]:
        st.warning(
            "**🟡 Probabilidad 30% – 60%**\n\n"
            "**Riesgo medio.** Revisión estándar por analista. "
            "Solicitar documentación complementaria si aplica."
        )
    with app_cols[2]:
        st.error(
            "**🔴 Probabilidad > 60%**\n\n"
            "**Riesgo alto.** Investigación profunda. "
            "Asignar a analista senior y considerar peritaje adicional."
        )

    st.markdown("---")

    # ============ Navegación ============
    st.subheader("🧭 Cómo navegar esta aplicación")
    nav_cols = st.columns(3)
    with nav_cols[0]:
        st.markdown(
            "**📊 EDA Interactivo**\n\n"
            "Explora el dataset filtrando por las 15 variables categóricas. "
            "Visualiza tasas de fraude por cualquier segmento."
        )
    with nav_cols[1]:
        st.markdown(
            "**🎯 Predicción**\n\n"
            "Ingresa los datos de una reclamación y obtén la probabilidad "
            "de fraude comparada entre los 4 modelos."
        )
    with nav_cols[2]:
        st.markdown(
            "**📈 Desempeño**\n\n"
            "Tabla comparativa de métricas con el mejor modelo resaltado, "
            "curvas ROC/PR y matrices de confusión."
        )
