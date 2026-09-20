"""
Página de Predicción — Formulario para predecir fraude en una reclamación.
"""
import pandas as pd
import streamlit as st

from ..mappings import ORDINAL_MAPPINGS
from ..plotting import plot_risk_gauge
from ..prediction import (
    build_input_dataframe, detect_risk_factors, predict_with_all_models,
)


def render(
    df: pd.DataFrame,
    models: dict,
    selected_model_name: str,
    best_model_name: str,
) -> None:
    """Renderiza la página de predicción individual."""
    selected_model = models[selected_model_name]

    st.header(f"🎯 Predicción individual — Modelo: {selected_model_name}")
    st.markdown("Ingresa los datos de la reclamación:")

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            month = st.selectbox("Month", [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ])
            day_of_week = st.selectbox("DayOfWeek", [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ])
            make = st.selectbox("Make", sorted(df["Make"].unique()))
            accident_area = st.selectbox("AccidentArea", ["Urban", "Rural"])
            sex = st.selectbox("Sex", ["Male", "Female"])
            marital_status = st.selectbox(
                "MaritalStatus", ["Single", "Married", "Widow", "Divorced"]
            )
            age = st.slider("Age", 16, 80, 35)

        with c2:
            fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
            policy_type = st.selectbox("PolicyType", sorted(df["PolicyType"].unique()))
            vehicle_category = st.selectbox(
                "VehicleCategory", ["Sport", "Sedan", "Utility"]
            )
            vehicle_price = st.selectbox(
                "VehiclePrice", list(ORDINAL_MAPPINGS["VehiclePrice"].keys())
            )
            base_policy = st.selectbox(
                "BasePolicy", ["Liability", "Collision", "All Perils"]
            )
            deductible = st.select_slider(
                "Deductible ($)", options=[300, 400, 500, 700], value=400
            )
            driver_rating = st.slider("DriverRating", 1, 4, 2)

        with c3:
            days_policy_accident = st.selectbox(
                "Days_Policy_Accident",
                list(ORDINAL_MAPPINGS["Days_Policy_Accident"].keys())
            )
            days_policy_claim = st.selectbox(
                "Days_Policy_Claim",
                list(ORDINAL_MAPPINGS["Days_Policy_Claim"].keys())
            )
            past_number_of_claims = st.selectbox(
                "PastNumberOfClaims",
                list(ORDINAL_MAPPINGS["PastNumberOfClaims"].keys())
            )
            age_of_vehicle = st.selectbox(
                "AgeOfVehicle", list(ORDINAL_MAPPINGS["AgeOfVehicle"].keys())
            )
            police_report_filed = st.selectbox("PoliceReportFiled", ["Yes", "No"])
            witness_present = st.selectbox("WitnessPresent", ["Yes", "No"])

        c4, c5 = st.columns(2)
        with c4:
            agent_type = st.selectbox("AgentType", ["Internal", "External"])
            number_of_suppliments = st.selectbox(
                "NumberOfSuppliments",
                list(ORDINAL_MAPPINGS["NumberOfSuppliments"].keys())
            )
        with c5:
            address_change_claim = st.selectbox(
                "AddressChange_Claim",
                list(ORDINAL_MAPPINGS["AddressChange_Claim"].keys())
            )
            number_of_cars = st.selectbox(
                "NumberOfCars", list(ORDINAL_MAPPINGS["NumberOfCars"].keys())
            )

        submitted = st.form_submit_button("🚀 Predecir", type="primary")

    if submitted:
        form_data = {
            "Month": month, "DayOfWeek": day_of_week, "Make": make,
            "AccidentArea": accident_area, "DayOfWeekClaimed": day_of_week,
            "MonthClaimed": month, "Sex": sex, "MaritalStatus": marital_status,
            "Age": age, "Fault": fault, "PolicyType": policy_type,
            "VehicleCategory": vehicle_category, "VehiclePrice": vehicle_price,
            "Deductible": deductible, "DriverRating": driver_rating,
            "Days_Policy_Accident": days_policy_accident,
            "Days_Policy_Claim": days_policy_claim,
            "PastNumberOfClaims": past_number_of_claims,
            "AgeOfVehicle": age_of_vehicle,
            "PoliceReportFiled": police_report_filed,
            "WitnessPresent": witness_present, "AgentType": agent_type,
            "NumberOfSuppliments": number_of_suppliments,
            "AddressChange_Claim": address_change_claim,
            "NumberOfCars": number_of_cars, "BasePolicy": base_policy,
        }

        x_input = build_input_dataframe(form_data)

        try:
            prob = selected_model.predict_proba(x_input)[0, 1]
            pred = int(prob >= 0.5)

            cA, cB = st.columns(2)
            cA.metric("Probabilidad de fraude", f"{prob*100:.2f}%")
            cB.metric("Predicción", "🚨 FRAUDE" if pred == 1 else "✅ Legítima")

            st.plotly_chart(
                plot_risk_gauge(prob, selected_model_name),
                use_container_width=True,
            )

            # Comparativa con todos los modelos
            st.subheader("🔍 Comparativa con los 4 modelos")
            comp_data = predict_with_all_models(models, x_input, best_model_name)
            st.dataframe(
                pd.DataFrame(comp_data), use_container_width=True, hide_index=True
            )

            # Factores de riesgo
            risk_factors = detect_risk_factors(form_data)
            if risk_factors:
                msg = "**Factores de riesgo detectados:**\n- " + "\n- ".join(risk_factors)
                st.warning(msg)

        except Exception as e:
            st.error(f"Error en la predicción: {e}")
