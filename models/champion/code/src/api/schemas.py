"""
Contratos de entrada y salida de la API.

Las variables ordinales aceptan tanto la etiqueta original del dataset
("more than 69000") como su código numérico (5): el pipeline del modelo
normaliza ambas formas.
"""
from __future__ import annotations

from typing import List, Literal, Union

from pydantic import BaseModel, Field

OrdinalValue = Union[str, int]

CLAIM_EXAMPLE = {
    "Month": "Dec", "WeekOfMonth": 5, "DayOfWeek": "Wednesday",
    "Make": "Honda", "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday",
    "MonthClaimed": "Jan", "WeekOfMonthClaimed": 1, "Sex": "Female",
    "MaritalStatus": "Single", "Age": 21, "Fault": "Policy Holder",
    "PolicyType": "Sport - Liability", "VehicleCategory": "Sport",
    "VehiclePrice": "more than 69000", "RepNumber": 12, "Deductible": 300,
    "DriverRating": 1, "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
    "AgeOfVehicle": "3 years", "PoliceReportFiled": "No", "WitnessPresent": "No",
    "AgentType": "External", "NumberOfSuppliments": "none",
    "AddressChange_Claim": "1 year", "NumberOfCars": "3 to 4",
    "BasePolicy": "Liability",
}


class ClaimRequest(BaseModel):
    """Una reclamación de seguro vehicular a evaluar."""

    model_config = {"json_schema_extra": {"example": CLAIM_EXAMPLE}}

    # Circunstancias del siniestro
    Month: str = Field(description="Mes del accidente, p. ej. 'Dec'")
    DayOfWeek: str = Field(description="Día de la semana del accidente")
    AccidentArea: Literal["Urban", "Rural"]
    Fault: Literal["Policy Holder", "Third Party"]

    # Circunstancias de la reclamación
    MonthClaimed: str = Field(description="Mes de la reclamación")
    DayOfWeekClaimed: str = Field(description="Día de la semana de la reclamación")
    Days_Policy_Accident: OrdinalValue = Field(description="Días entre póliza y accidente")
    Days_Policy_Claim: OrdinalValue = Field(description="Días entre póliza y reclamación")
    PoliceReportFiled: Literal["Yes", "No"]
    WitnessPresent: Literal["Yes", "No"]
    NumberOfSuppliments: OrdinalValue = Field(description="Suplementos de la reclamación")

    # Asegurado
    Age: int = Field(ge=0, le=120, description="Edad del asegurado")
    Sex: Literal["Male", "Female"]
    MaritalStatus: str = Field(description="Estado civil: Single, Married, Divorced, Widow")
    PastNumberOfClaims: OrdinalValue = Field(description="Reclamaciones previas")
    AddressChange_Claim: OrdinalValue = Field(description="Antigüedad del cambio de dirección")
    NumberOfCars: OrdinalValue = Field(description="Vehículos asegurados")

    # Vehículo
    Make: str = Field(description="Marca del vehículo")
    VehicleCategory: Literal["Sedan", "Sport", "Utility"]
    VehiclePrice: OrdinalValue = Field(description="Rango de precio del vehículo")
    AgeOfVehicle: OrdinalValue = Field(description="Antigüedad del vehículo")

    # Póliza
    PolicyType: str = Field(description="Tipo de póliza, p. ej. 'Sport - Liability'")
    BasePolicy: Literal["Liability", "Collision", "All Perils"]
    Deductible: int = Field(ge=0, description="Deducible de la póliza")
    DriverRating: int = Field(ge=1, le=4, description="Calificación del conductor")
    AgentType: Literal["External", "Internal"]

    # Campos operativos con valor por defecto (medianas del dataset)
    WeekOfMonth: int = Field(default=3, ge=1, le=5)
    WeekOfMonthClaimed: int = Field(default=3, ge=1, le=5)
    RepNumber: int = Field(default=12, ge=1, le=16)


class BatchClaimRequest(BaseModel):
    """Lote de reclamaciones para puntuación masiva."""

    claims: List[ClaimRequest] = Field(min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    """Resultado de la evaluación de una reclamación."""

    fraud_probability: float = Field(description="Probabilidad estimada de fraude")
    is_fraud: bool = Field(description="Decisión al umbral del modelo campeón")
    decision_threshold: float = Field(description="Umbral aplicado")


class BatchPredictionResponse(BaseModel):
    """Resultado de una puntuación por lotes."""

    predictions: List[PredictionResponse]
    n_claims: int
    n_flagged: int


class ModelInfoResponse(BaseModel):
    """Identidad y desempeño del modelo que está sirviendo la API."""

    model_config = {"protected_namespaces": ()}

    registered_model: str
    version: Union[int, None] = None
    model_name: Union[str, None] = None
    run_id: Union[str, None] = None
    decision_threshold: float
    selection_metric: Union[str, None] = None
    selection_score: Union[str, None] = None
    promoted_at: Union[str, None] = None


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: Literal["ok"]
    model_loaded: bool
    registered_model: str
    version: Union[int, None] = None
