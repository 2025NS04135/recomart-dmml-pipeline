# Task 10: End-to-End Pipeline Orchestration & Monitoring

## 1. Orchestration DAG (`src/orchestration/pipeline_flow.py`)
RecoMart uses **Prefect** (`@flow` and `@task` decorators) to orchestrate all seven pipeline stages as a directed acyclic graph (DAG) with strict dependency ordering:

```text
[1_Ingest_Raw_Sources] ──▶ [2_Validate_Data_Quality] ──▶ [3_Prepare_And_EDA]
                                                                 │
                                                                 ▼
[7_Train_Evaluate_MLflow] ◀── [6_Record_Data_Lineage] ◀── [5_Materialize_Feature_Store] ◀── [4_Engineer_Warehouse_Features]