from pathlib import Path
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --------------------------------------------------------------
# Rutas del proyecto (Usando pathlib + .env)
# --------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

# Si el .env tiene una ruta, la usamos; si no, usamos la ruta por defecto
RAW_DATASET = Path(os.getenv("RAW_DATA_PATH", ROOT_DIR / "data/raw/fraud_oracle.csv"))
CLEAN_DATASET = Path(os.getenv("PROCESSED_DATA_PATH", ROOT_DIR / "data/processed/fraud_clean.csv"))

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
BEST_MODEL_PATH = MODELS_DIR / "best_fraud_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.pkl"

# Export local del modelo campeón (artefacto que se empaqueta en la imagen Docker)
CHAMPION_EXPORT_DIR = MODELS_DIR / "champion"
CHAMPION_METADATA_FILE = "champion_metadata.json"

# --------------------------------------------------------------
# Constantes del modelado
# --------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "FraudFound_P"

# Variables a excluir del modelado
COLUMNS_TO_DROP = ["PolicyNumber", "Year", "AgeOfPolicyHolder"]

# Variables categóricas nominales (One-Hot Encoding)
NOMINAL_COLUMNS = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed",
    "MonthClaimed", "Sex", "MaritalStatus", "Fault", "PolicyType",
    "VehicleCategory", "PoliceReportFiled", "WitnessPresent",
    "AgentType", "BasePolicy",
]

# Variables numéricas y ordinales ya codificadas (escalado estándar)
NUMERIC_COLUMNS = [
    "WeekOfMonth", "WeekOfMonthClaimed", "Age", "RepNumber", "Deductible",
    "DriverRating", "VehiclePrice", "Days_Policy_Accident", "Days_Policy_Claim",
    "PastNumberOfClaims", "AgeOfVehicle", "NumberOfSuppliments",
    "AddressChange_Claim", "NumberOfCars",
]

# Columnas que debe traer el CSV crudo para considerarse válido
RAW_REQUIRED_COLUMNS = sorted(
    NOMINAL_COLUMNS + NUMERIC_COLUMNS + COLUMNS_TO_DROP + [TARGET_COLUMN]
)

# Valores centinela que en este dataset representan datos faltantes
SENTINEL_CATEGORICAL_COLUMNS = ["MonthClaimed", "DayOfWeekClaimed"]
SENTINEL_VALUE = "0"

# --------------------------------------------------------------
# Orquestación: MLflow
# --------------------------------------------------------------
# El backend sqlite es obligatorio para poder usar el Model Registry.
# Las rutas son relativas: los flows se ejecutan desde la raíz del repositorio.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraude-seguros-vehiculares")

# Ubicación de los artefactos. Por defecto ./mlruns; se puede reubicar a una
# ruta corta en Windows, donde el límite de 260 caracteres rompe la escritura
# de artefactos si el repositorio está en un directorio profundo.
MLFLOW_ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION") or None

REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "fraud-detection-classifier")
CHAMPION_ALIAS = "champion"

# Métrica de decisión del modelo candidato y umbral mínimo para promoverlo.
# PR-AUC porque la clase positiva (fraude) es el 5.99% del dataset.
SELECTION_METRIC = "pr_auc"
MIN_SELECTION_METRIC = float(os.getenv("MIN_PR_AUC", "0.20"))

# --------------------------------------------------------------
# Optimización de hiperparámetros
# --------------------------------------------------------------
TUNING_ENABLED = os.getenv("TUNING_ENABLED", "true").lower() == "true"
TUNING_N_ITER = int(os.getenv("TUNING_N_ITER", "12"))
TUNING_CV_FOLDS = int(os.getenv("TUNING_CV_FOLDS", "3"))

# --------------------------------------------------------------
# Servicio de predicción (API)
# --------------------------------------------------------------
MODEL_URI = os.getenv("MODEL_URI", str(CHAMPION_EXPORT_DIR))
DEFAULT_DECISION_THRESHOLD = 0.5

# --------------------------------------------------------------
# Colores y temas
# --------------------------------------------------------------
COLOR_LEGITIMATE = "#2E86AB"
COLOR_FRAUD = "#E63946"
MODEL_COLORS = ["#2E86AB", "#E63946", "#06A77D", "#F4A261"]
