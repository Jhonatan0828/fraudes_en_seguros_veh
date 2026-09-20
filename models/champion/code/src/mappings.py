"""
Mapeos de variables ordinales y configuración de filtros.
Centraliza las traducciones de strings a valores numéricos.
"""

# --------------------------------------------------------------
# Mapeos ordinales para codificación
# Deben coincidir con los usados durante el entrenamiento
# --------------------------------------------------------------
ORDINAL_MAPPINGS = {
    "VehiclePrice": {
        "less than 20000": 0, "20000 to 29000": 1, "30000 to 39000": 2,
        "40000 to 59000": 3, "60000 to 69000": 4, "more than 69000": 5,
    },
    "Days_Policy_Accident": {
        "none": 0, "1 to 7": 1, "8 to 15": 2, "15 to 30": 3, "more than 30": 4,
    },
    "Days_Policy_Claim": {
        "none": 0, "8 to 15": 1, "15 to 30": 2, "more than 30": 3,
    },
    "PastNumberOfClaims": {
        "none": 0, "1": 1, "2 to 4": 2, "more than 4": 3,
    },
    "AgeOfVehicle": {
        "new": 0, "2 years": 1, "3 years": 2, "4 years": 3,
        "5 years": 4, "6 years": 5, "7 years": 6, "more than 7": 7,
    },
    "NumberOfSuppliments": {
        "none": 0, "1 to 2": 1, "3 to 5": 2, "more than 5": 3,
    },
    "AddressChange_Claim": {
        "no change": 0, "under 6 months": 1, "1 year": 2,
        "2 to 3 years": 3, "4 to 8 years": 4,
    },
    "NumberOfCars": {
        "1 vehicle": 1, "2 vehicles": 2, "3 to 4": 3,
        "5 to 8": 4, "more than 8": 5,
    },
}

# --------------------------------------------------------------
# Variables disponibles para filtros en el EDA interactivo
# --------------------------------------------------------------
EDA_FILTERS = [
    "Month", "DayOfWeek", "MonthClaimed", "DayOfWeekClaimed",
    "Make", "AccidentArea", "Sex", "MaritalStatus", "Fault",
    "PolicyType", "VehicleCategory", "PoliceReportFiled",
    "WitnessPresent", "AgentType", "BasePolicy",
]

# --------------------------------------------------------------
# Top variables predictoras (interpretación de negocio)
# --------------------------------------------------------------
TOP_PREDICTORS = [
    {
        "Variable": "Fault", "Tipo": "Categórica", "Importancia": "Muy alta",
        "Interpretación": "Responsable del siniestro (Policy Holder = mayor riesgo)",
    },
    {
        "Variable": "BasePolicy", "Tipo": "Categórica", "Importancia": "Muy alta",
        "Interpretación": "Tipo de póliza base (All Perils/Collision = mayor riesgo)",
    },
    {
        "Variable": "PastNumberOfClaims", "Tipo": "Ordinal", "Importancia": "Alta",
        "Interpretación": "Reclamaciones previas del asegurado",
    },
    {
        "Variable": "AddressChange_Claim", "Tipo": "Ordinal", "Importancia": "Alta",
        "Interpretación": "Cambio de dirección reciente antes de la reclamación",
    },
    {
        "Variable": "PolicyType", "Tipo": "Categórica", "Importancia": "Media-alta",
        "Interpretación": "Combinación de tipo de póliza y categoría",
    },
    {
        "Variable": "VehiclePrice", "Tipo": "Ordinal", "Importancia": "Media",
        "Interpretación": "Rango de precio del vehículo",
    },
    {
        "Variable": "AgeOfVehicle", "Tipo": "Ordinal", "Importancia": "Media",
        "Interpretación": "Antigüedad del vehículo",
    },
    {
        "Variable": "Deductible", "Tipo": "Continua", "Importancia": "Media",
        "Interpretación": "Monto del deducible de la póliza",
    },
]
