"""
🔍 App de Detección de Fraude en Seguros Vehiculares — Punto de entrada.

Ejecutar:  streamlit run app.py

Universidad de Medellín — Proyecto académico de Ciencia de Datos.
"""
import streamlit as st

from src.data_loader import load_artifacts, load_clean_dataset
from src.pages import eda, home, performance, prediction


def main() -> None:
    # Configuración de la página
    st.set_page_config(
        page_title="Fraud Detection",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Detección de Fraude en Seguros Vehiculares")
    st.markdown(
        "Aplicación académica — Universidad de Medellín — "
        "Aprendizaje Automático en la Nube | "
        "Profesora: Maria Camila Durango Barrera | "
        "Estudiantes: Andres Felipe Londoño Ocampo · Jhonatan Caro Atehortúa · "
        "Paulina Perez Ramirez · Victor Manuel Galeano Alvarez"
    )

    # Carga de artefactos (cacheada)
    @st.cache_resource
    def _load_artifacts():
        return load_artifacts()

    @st.cache_data
    def _load_data():
        return load_clean_dataset()

    try:
        models, metadata = _load_artifacts()
        df = _load_data()
    except FileNotFoundError as e:
        st.error(
            f"⚠️ {e}\n\n"
            "El dataset procesado lo genera el pipeline (`python -m flows.training_flow`) "
            "y los artefactos de esta app el notebook `notebooks/fraud_detection_analysis.ipynb`."
        )
        st.stop()

    all_model_names = list(models.keys())
    best_model_name = metadata["best_model_name"]

    # Sidebar
    st.sidebar.title("📋 Menú")
    page = st.sidebar.radio("Navegar a:", [
        "🏠 Inicio",
        "📊 EDA Interactivo",
        "🎯 Predicción",
        "📈 Desempeño del modelo",
    ])
    st.sidebar.markdown("---")

    st.sidebar.subheader("⚙️ Modelo activo")
    selected_model_name = st.sidebar.selectbox(
        "Selecciona un modelo:",
        all_model_names,
        index=(
            all_model_names.index(best_model_name)
            if best_model_name in all_model_names else 0
        ),
    )
    if selected_model_name == best_model_name:
        st.sidebar.success(f"⭐ Mejor modelo: {selected_model_name}")
    else:
        st.sidebar.info(f"Modelo: {selected_model_name}")

    # Ruteo de páginas
    if page == "🏠 Inicio":
        home.render(df, models, metadata)
    elif page == "📊 EDA Interactivo":
        eda.render(df)
    elif page == "🎯 Predicción":
        prediction.render(df, models, selected_model_name, best_model_name)
    elif page == "📈 Desempeño del modelo":
        performance.render(df, models, selected_model_name)


if __name__ == "__main__":
    main()
