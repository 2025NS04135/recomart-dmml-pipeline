import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, get_run_logger, task

Path("logs").mkdir(exist_ok=True)
alert_logger = logging.getLogger("PipelineAlerts")
if not alert_logger.handlers:
    fh = logging.FileHandler("logs/pipeline_alerts.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    alert_logger.addHandler(fh)
    alert_logger.setLevel(logging.INFO)


def run_module_step(module_name: str, step_label: str) -> dict:
    """Execute a pipeline Python module as a monitored subprocess and capture telemetry."""
    logger = get_run_logger()
    start = time.perf_counter()
    logger.info(f"[{step_label}] Starting module: {module_name}")

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        capture_output=True,
        text=True,
    )
    elapsed = round(time.perf_counter() - start, 2)

    if result.returncode != 0:
        err_msg = (
            f"ALERT | Stage '{step_label}' ({module_name}) FAILED after {elapsed}s "
            f"(exit_code={result.returncode}): {result.stderr.strip()}"
        )
        alert_logger.critical(err_msg)
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    logger.info(f"[{step_label}] Completed successfully in {elapsed}s")
    return {
        "stage": step_label,
        "module": module_name,
        "status": "COMPLETED",
        "duration_sec": elapsed,
    }


@task(name="1_Ingest_Raw_Sources", retries=2, retry_delay_seconds=5)
def task_ingest() -> list[dict]:
    """Ingest REST API product catalog and clickstream interactions into partitioned raw storage."""
    ingestion_scripts = sorted(
        p.stem for p in Path("src/ingestion").glob("*.py") if p.stem != "__init__"
    )
    if not ingestion_scripts:
        raise FileNotFoundError("No ingestion scripts found in src/ingestion/")
    return [
        run_module_step(f"src.ingestion.{script}", f"Ingestion ({script})")
        for script in ingestion_scripts
    ]


@task(name="2_Validate_Data_Quality", retries=1, retry_delay_seconds=3)
def task_validate(upstream_dep: list[dict]) -> dict:
    """Run automated schema, null, duplicate, range, and referential integrity checks."""
    return run_module_step("src.validation.validate_data", "Data Validation")


@task(name="3_Prepare_And_EDA", retries=1, retry_delay_seconds=3)
def task_prepare(upstream_dep: dict) -> dict:
    """Clean anomalies, encode/scale attributes, save Parquet datasets, and generate EDA figures."""
    return run_module_step("src.preparation.prepare_data", "Data Preparation & EDA")


@task(name="4_Engineer_Warehouse_Features", retries=1, retry_delay_seconds=3)
def task_warehouse_features(upstream_dep: dict) -> dict:
    """Compute user, item, and cross-affinity features and load SQLite data warehouse."""
    return run_module_step("src.features.engineer_features", "Feature Engineering & Warehouse")


@task(name="5_Materialize_Feature_Store", retries=1, retry_delay_seconds=3)
def task_feature_store(upstream_dep: dict) -> dict:
    """Register metadata and materialize v1.0 feature views into Feature Store registry."""
    return run_module_step("src.features.feature_store", "Feature Store Materialization")


@task(name="6_Record_Data_Lineage", retries=1, retry_delay_seconds=3)
def task_lineage(upstream_dep: dict) -> dict:
    """Update end-to-end lineage manifest linking sources, transformations, and DVC hashes."""
    return run_module_step("src.features.record_lineage", "Data Lineage Audit")


@task(name="7_Train_Evaluate_MLflow", retries=1, retry_delay_seconds=5)
def task_train_evaluate(upstream_dep: dict) -> dict:
    """Train SVD, Random Forest, and Gradient Boosting recommenders and log metrics to MLflow."""
    return run_module_step("src.models.train_evaluate", "Model Training & MLflow Tracking")


@flow(name="RecoMart_EndToEnd_DMML_Pipeline", log_prints=True)
def recomart_pipeline_flow() -> dict:
    """End-to-end orchestrated Data Management & Machine Learning DAG for RecoMart."""
    flow_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    t1 = task_ingest()
    t2 = task_validate(t1)
    t3 = task_prepare(t2)
    t4 = task_warehouse_features(t3)
    t5 = task_feature_store(t4)
    t6 = task_lineage(t5)
    t7 = task_train_evaluate(t6)

    total_duration = round(time.perf_counter() - flow_start, 2)
    stages = [*t1, t2, t3, t4, t5, t6, t7]

    eval_summary = json.loads(Path("reports/model_evaluation.json").read_text(encoding="utf-8"))
    run_report = {
        "flow_name": "RecoMart_EndToEnd_DMML_Pipeline",
        "status": "SUCCEEDED",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_duration_sec": total_duration,
        "stages": stages,
        "champion_model": eval_summary["champion_model"],
        "model_artifact": eval_summary["model_artifact"],
    }

    Path("reports/orchestration_run_report.json").write_text(
        json.dumps(run_report, indent=2), encoding="utf-8"
    )
    alert_logger.info(
        f"SUCCESS | Flow 'RecoMart_EndToEnd_DMML_Pipeline' completed in {total_duration}s | "
        f"Champion={eval_summary['champion_model']}"
    )
    return run_report


if __name__ == "__main__":
    summary = recomart_pipeline_flow()
    print(json.dumps(summary, indent=2))
