"""
Tests del pipeline de Machine Learning.
Ejecutar:  pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNS_TO_DROP, NOMINAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from src.pipeline.evaluation import compute_metrics, find_optimal_threshold
from src.pipeline.features import compute_scale_pos_weight, stratified_split
from src.pipeline.ingestion import validate_dataset
from src.pipeline.preprocessing import (
    ColumnDropper, OrdinalMapper, SentinelImputer, build_cleaning_pipeline,
)
from src.pipeline.registry import select_candidate
from src.pipeline.training import build_model_pipelines

RAW_ROW = {
    "Month": "Dec", "WeekOfMonth": 5, "DayOfWeek": "Wednesday", "Make": "Honda",
    "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday", "MonthClaimed": "Jan",
    "WeekOfMonthClaimed": 1, "Sex": "Female", "MaritalStatus": "Single", "Age": 21,
    "Fault": "Policy Holder", "PolicyType": "Sport - Liability",
    "VehicleCategory": "Sport", "VehiclePrice": "more than 69000",
    "FraudFound_P": 0, "PolicyNumber": 1, "RepNumber": 12, "Deductible": 300,
    "DriverRating": 1, "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
    "AgeOfVehicle": "3 years", "AgeOfPolicyHolder": "26 to 30",
    "PoliceReportFiled": "No", "WitnessPresent": "No", "AgentType": "External",
    "NumberOfSuppliments": "none", "AddressChange_Claim": "1 year",
    "NumberOfCars": "3 to 4", "Year": 1994, "BasePolicy": "Liability",
}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Dataset crudo sintético con la estructura del original."""
    rows = []
    for index in range(200):
        row = dict(RAW_ROW)
        row["PolicyNumber"] = index + 1
        row["Age"] = 0 if index % 10 == 0 else 25 + index % 40
        row["FraudFound_P"] = 1 if index % 5 == 0 else 0
        row["MonthClaimed"] = "0" if index % 20 == 0 else "Jan"
        row["Fault"] = "Third Party" if index % 3 else "Policy Holder"
        rows.append(row)
    return pd.DataFrame(rows)


# -------- Adquisición y validación --------

def test_validate_dataset_returns_quality_report(raw_df):
    report = validate_dataset(raw_df)
    assert report["n_rows"] == 200
    assert report["sentinel_values"]["Age"] == 20
    assert report["fraud_rate"] == pytest.approx(0.20)


def test_validate_dataset_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="columnas obligatorias"):
        validate_dataset(raw_df.drop(columns=["BasePolicy"]))


def test_validate_dataset_rejects_non_binary_target(raw_df):
    corrupted = raw_df.copy()
    corrupted.loc[0, TARGET_COLUMN] = 7
    with pytest.raises(ValueError, match="binaria"):
        validate_dataset(corrupted)


# -------- Procesamiento --------

def test_sentinel_imputer_replaces_hidden_missing_values(raw_df):
    imputer = SentinelImputer().fit(raw_df)
    result = imputer.transform(raw_df)
    assert (result["Age"] == 0).sum() == 0
    assert (result["MonthClaimed"] == "0").sum() == 0


def test_sentinel_imputer_learns_from_training_data_only(raw_df):
    """La mediana se toma del set de ajuste, no del que se transforma."""
    train = raw_df[raw_df["Age"] > 0].copy()
    imputer = SentinelImputer().fit(train)
    assert imputer.age_median_ == int(train["Age"].median())


def test_ordinal_mapper_converts_labels_to_numbers(raw_df):
    result = OrdinalMapper().fit(raw_df).transform(raw_df)
    assert result["VehiclePrice"].iloc[0] == 5
    assert result["PastNumberOfClaims"].iloc[0] == 0


def test_ordinal_mapper_is_idempotent(raw_df):
    """Aplicarlo sobre datos ya codificados no los destruye."""
    mapper = OrdinalMapper().fit(raw_df)
    once = mapper.transform(raw_df)
    twice = mapper.transform(once)
    pd.testing.assert_series_equal(once["VehiclePrice"], twice["VehiclePrice"])


def test_column_dropper_ignores_absent_columns(raw_df):
    result = ColumnDropper().fit(raw_df).transform(raw_df.drop(columns=["Year"]))
    assert not set(COLUMNS_TO_DROP) & set(result.columns)


def test_cleaning_pipeline_output_matches_model_contract(raw_df):
    features = raw_df.drop(columns=[TARGET_COLUMN])
    result = build_cleaning_pipeline().fit(features).transform(features)
    assert set(NUMERIC_COLUMNS + NOMINAL_COLUMNS) <= set(result.columns)
    assert not set(COLUMNS_TO_DROP) & set(result.columns)


# -------- Feature engineering --------

def test_stratified_split_preserves_fraud_rate(raw_df):
    X, y = raw_df.drop(columns=[TARGET_COLUMN]), raw_df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    assert len(X_train) + len(X_test) == len(raw_df)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)


def test_compute_scale_pos_weight(raw_df):
    weight = compute_scale_pos_weight(raw_df[TARGET_COLUMN])
    assert weight == pytest.approx(4.0)


# -------- Entrenamiento --------

def test_build_model_pipelines_returns_four_independent_candidates():
    models = build_model_pipelines(scale_pos_weight=15.0)
    assert len(models) == 4
    preprocessors = [model.named_steps["prep"] for model in models.values()]
    assert len({id(prep) for prep in preprocessors}) == 4


# -------- Evaluación --------

def test_find_optimal_threshold_beats_default_on_imbalanced_scores():
    y_true = pd.Series([0] * 90 + [1] * 10)
    y_proba = np.concatenate([np.linspace(0.01, 0.30, 90), np.linspace(0.25, 0.45, 10)])
    threshold, f1 = find_optimal_threshold(y_true, y_proba)
    assert 0 < threshold < 1
    assert f1 >= compute_metrics(y_true, y_proba, threshold=0.5)["f1"]


def test_compute_metrics_reports_confusion_matrix():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.8, 0.9, 0.2])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1


# -------- Selección del candidato --------

def _run(name, pr_auc):
    return {"model_name": name, "run_id": name, "model_uri": name,
            "metrics": {"pr_auc": pr_auc}}


def test_select_candidate_picks_best_pr_auc():
    runs = [_run("a", 0.21), _run("b", 0.35), _run("c", 0.28)]
    assert select_candidate(runs)["model_name"] == "b"


def test_select_candidate_blocks_promotion_below_minimum():
    with pytest.raises(ValueError, match="mínimo"):
        select_candidate([_run("a", 0.05)], minimum=0.20)
