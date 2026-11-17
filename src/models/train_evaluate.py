import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.features.feature_store import RecoMartFeatureStore

Path("logs").mkdir(exist_ok=True)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
logger = logging.getLogger("ModelTraining")

FEATURE_COLS = [
    "user_activity_count",
    "user_purchase_count",
    "user_avg_score",
    "user_category_diversity",
    "recency_hours",
    "item_popularity_count",
    "item_avg_score",
    "item_score_std",
    "conversion_rate",
    "category_affinity_score",
    "co_occurrence_boost",
]


def compute_ranking_metrics(
    test_df: pd.DataFrame,
    pred_col: str = "pred_score",
    k: int = 5,
    relevance_threshold: float = 3.0,
) -> dict:
    """Compute mean Precision@K, Recall@K, and NDCG@K across users in the test split."""
    precisions, recalls, ndcgs = [], [], []

    for _, group in test_df.groupby("user_id"):
        relevant_items = set(group.loc[group["target_score"] >= relevance_threshold, "item_id"])
        if not relevant_items:
            continue

        ranked = group.sort_values(by=pred_col, ascending=False).head(k)
        hits = [1 if item in relevant_items else 0 for item in ranked["item_id"]]

        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(relevant_items))

        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(hits))
        ideal_hits = [1] * min(len(relevant_items), k)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"precision_at_{k}": round(float(np.mean(precisions)), 4),
        f"recall_at_{k}": round(float(np.mean(recalls)), 4),
        f"ndcg_at_{k}": round(float(np.mean(ndcgs)), 4),
    }


def predict_svd_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, n_components: int = 12) -> np.ndarray:
    """Collaborative Filtering baseline using mean-centered TruncatedSVD matrix factorization."""
    pivot = train_df.pivot(index="user_id", columns="item_id", values="target_score")
    user_means = pivot.mean(axis=1)
    centered = pivot.sub(user_means, axis=0).fillna(0.0)

    k_latent = min(n_components, min(centered.shape) - 1)
    svd = TruncatedSVD(n_components=k_latent, random_state=42)
    u_factors = svd.fit_transform(centered)
    reconstructed = np.dot(u_factors, svd.components_) + user_means.values[:, None]
    pred_matrix = pd.DataFrame(reconstructed, index=pivot.index, columns=pivot.columns)

    global_mean = float(train_df["target_score"].mean())
    preds = []
    for row in test_df.itertuples():
        if row.user_id in pred_matrix.index and row.item_id in pred_matrix.columns:
            val = float(pred_matrix.loc[row.user_id, row.item_id])
        elif row.user_id in user_means.index:
            val = float(user_means.loc[row.user_id])
        else:
            val = global_mean
        preds.append(np.clip(val, 1.0, 5.0))
    return np.array(preds)


def train_and_evaluate(feature_version: str = "v1.0", top_k: int = 5) -> dict:
    """Train candidate recommendation models, log to MLflow, and save the best artifact."""
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Configure local MLflow tracking URI safely on Windows
    tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RecoMart_Recommendation_Models")

    # 1. Retrieve offline point-in-time training dataset from Feature Store
    fs = RecoMartFeatureStore()
    df = fs.get_historical_features(version=feature_version)

    # Stratified per-user split (80/20) so every user appears in both train and test sets
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["user_id"]
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df["target_score"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target_score"]

    results = []
    best_model = None
    best_model_name = ""
    best_score = -1.0

    # --- Candidate 1: Collaborative Filtering (TruncatedSVD) ---
    with mlflow.start_run(run_name="CollaborativeFiltering_SVD"):
        mlflow.log_params(
            {
                "model_family": "CollaborativeFiltering_SVD",
                "n_components": 12,
                "feature_version": feature_version,
                "data_tag": "data-v1.0",
            }
        )
        svd_preds = predict_svd_baseline(train_df, test_df, n_components=12)
        test_df["pred_score"] = svd_preds

        rmse = float(np.sqrt(mean_squared_error(y_test, svd_preds)))
        mae = float(mean_absolute_error(y_test, svd_preds))
        rank_metrics = compute_ranking_metrics(test_df, "pred_score", k=top_k)

        metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), **rank_metrics}
        mlflow.log_metrics(metrics)
        results.append({"model": "CollaborativeFiltering_SVD", **metrics})

    # --- Candidate 2 & 3: Feature-Store Hybrid Recommenders ---
    hybrid_candidates = {
        "Hybrid_RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42
        ),
        "Hybrid_GradientBoosting": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }

    for name, model in hybrid_candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model_family": name,
                    "feature_version": feature_version,
                    "num_features": len(FEATURE_COLS),
                    "data_tag": "data-v1.0",
                    **model.get_params(),
                }
            )
            model.fit(X_train, y_train)
            preds = np.clip(model.predict(X_test), 1.0, 5.0)
            test_df["pred_score"] = preds

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            rank_metrics = compute_ranking_metrics(test_df, "pred_score", k=top_k)

            metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), **rank_metrics}
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            results.append({"model": name, **metrics})
            if metrics[f"ndcg_at_{top_k}"] > best_score:
                best_score = metrics[f"ndcg_at_{top_k}"]
                best_model = model
                best_model_name = name

    # 2. Persist the winning model artifact
    model_path = Path("models/best_recommender.joblib")
    joblib.dump({"model_name": best_model_name, "model": best_model, "features": FEATURE_COLS}, model_path)

    # 3. Generate Live Top-5 Sample Recommendations for User U001 via Online Feature Store
    online_candidates = fs.get_online_features(user_id="U001", version=feature_version)
    online_candidates["predicted_score"] = np.clip(
        best_model.predict(online_candidates[FEATURE_COLS]), 1.0, 5.0
    ).round(4)

    with sqlite3.connect("warehouse/recomart_dw.db") as conn:
        items_meta = pd.read_sql_query("SELECT item_id, title, category, price FROM dim_items", conn)

    top_recs = (
        online_candidates.merge(items_meta, on="item_id", how="left")
        .sort_values(by="predicted_score", ascending=False)
        .head(top_k)[["item_id", "title", "category", "price", "predicted_score"]]
    )

    # 4. Write Human-Readable & JSON Model Evaluation Reports
    report_lines = [
        "# Task 9: Recommendation Model Evaluation & Selection Report",
        f"- **Generated At (UTC):** `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Feature Store Version:** `{feature_version}` (`{len(df)}` total user-item pairs | `{len(train_df)}` train / `{len(test_df)}` test)",
        f"- **Selected Champion Model:** `{best_model_name}` (Saved to `{model_path}`)",
        "",
        "## 1. Benchmark Comparison (MLflow Experiment: `RecoMart_Recommendation_Models`)",
        f"| Model Name | RMSE (Lower is Better) | MAE (Lower is Better) | Precision@{top_k} | Recall@{top_k} | NDCG@{top_k} |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        report_lines.append(
            f"| `{r['model']}` | {r['rmse']:.4f} | {r['mae']:.4f} | "
            f"{r[f'precision_at_{top_k}']:.4f} | {r[f'recall_at_{top_k}']:.4f} | **{r[f'ndcg_at_{top_k}']:.4f}** |"
        )

    report_lines.extend(
        [
            "",
            f"## 2. Sample Online Top-{top_k} Recommendations for User `U001`",
            "| Rank | Item ID | Product Title | Category | Price ($) | Predicted Score |",
            "| :---: | :---: | :--- | :--- | :---: | :---: |",
        ]
    )
    for rank, row in enumerate(top_recs.itertuples(), start=1):
        report_lines.append(
            f"| {rank} | `{row.item_id}` | {row.title} | `{row.category}` | {row.price:.2f} | **{row.predicted_score:.4f}** |"
        )

    md_path = Path("reports/model_evaluation_report.md")
    md_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary = {
        "champion_model": best_model_name,
        "model_artifact": str(model_path),
        "benchmarks": results,
        "sample_recommendations_U001": top_recs.to_dict(orient="records"),
    }
    Path("reports/model_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Model training complete. Champion={best_model_name} (NDCG@{top_k}={best_score:.4f}).")
    return summary


if __name__ == "__main__":
    train_and_evaluate()