import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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
logger = logging.getLogger("FeatureEngineering")


def build_warehouse_features() -> dict:
    """Compute user, item, and user-item affinity features and load into SQLite warehouse."""
    Path("warehouse").mkdir(exist_ok=True)
    df_inter = pd.read_parquet("data/processed/interactions_clean.parquet")
    df_prod = pd.read_parquet("data/processed/products_clean.parquet")
    now = datetime.now(timezone.utc)

    # Enrich interactions with product category for behavioral aggregation
    merged = df_inter.merge(df_prod[["item_id", "category"]], on="item_id", how="left")

    # 1. User-Level Behavioral Features
    user_feats = (
        merged.groupby("user_id")
        .agg(
            user_activity_count=("event_id", "count"),
            user_purchase_count=("event_type", lambda s: int((s == "purchase").sum())),
            user_avg_score=("interaction_score", "mean"),
            user_category_diversity=("category", "nunique"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
    )
    user_feats["recency_hours"] = ((now - user_feats["last_seen"]).dt.total_seconds() / 3600.0).round(2)
    user_feats["user_avg_score"] = user_feats["user_avg_score"].round(4)
    user_feats["feature_timestamp"] = now.isoformat()
    user_feats = user_feats.drop(columns=["last_seen"])

    # 2. Item-Level Popularity & Conversion Features
    item_feats = (
        merged.groupby("item_id")
        .agg(
            item_popularity_count=("event_id", "count"),
            item_avg_score=("interaction_score", "mean"),
            item_score_std=("interaction_score", "std"),
            purchases=("event_type", lambda s: int((s == "purchase").sum())),
        )
        .reset_index()
    )
    item_feats["item_avg_score"] = item_feats["item_avg_score"].round(4)
    item_feats["item_score_std"] = item_feats["item_score_std"].fillna(0.0).round(4)
    item_feats["conversion_rate"] = (item_feats["purchases"] / item_feats["item_popularity_count"]).round(4)
    item_feats["feature_timestamp"] = now.isoformat()
    item_feats = item_feats.drop(columns=["purchases"])

    # 3. User-Item Category Affinity & Co-Occurrence Cross-Features
    user_cat = merged.groupby(["user_id", "category"]).size().reset_index(name="cat_clicks")
    user_tot = merged.groupby("user_id").size().reset_index(name="tot_clicks")
    user_cat = user_cat.merge(user_tot, on="user_id")
    user_cat["category_affinity_score"] = user_cat["cat_clicks"] / user_cat["tot_clicks"]

    cross = pd.MultiIndex.from_product(
        [user_feats["user_id"].unique(), df_prod["item_id"].unique()], names=["user_id", "item_id"]
    ).to_frame(index=False)
    cross = cross.merge(df_prod[["item_id", "category"]], on="item_id", how="left")
    cross = cross.merge(
        user_cat[["user_id", "category", "category_affinity_score"]],
        on=["user_id", "category"],
        how="left",
    )
    cross["category_affinity_score"] = cross["category_affinity_score"].fillna(0.0).round(4)
    cross["co_occurrence_boost"] = np.log1p(cross["category_affinity_score"] * 10.0).round(4)
    affinity_feats = cross[["user_id", "item_id", "category_affinity_score", "co_occurrence_boost"]]

    # 4. Execute Schema DDL & Load Tables into SQLite Warehouse
    db_path = Path("warehouse/recomart_dw.db")
    conn = sqlite3.connect(db_path)
    schema_sql = Path("warehouse/schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema_sql)

    dim_cols = [
        "item_id",
        "title",
        "category",
        "category_encoded",
        "price",
        "price_normalized",
        "api_rating_rate",
        "api_rating_count",
    ]
    df_prod[dim_cols].to_sql("dim_items", conn, if_exists="replace", index=False)

    fact_cols = [
        "event_id",
        "user_id",
        "item_id",
        "event_type",
        "event_type_encoded",
        "rating",
        "interaction_score",
        "timestamp",
        "timestamp_normalized",
    ]
    df_inter_sql = df_inter[fact_cols].copy()
    df_inter_sql["timestamp"] = df_inter_sql["timestamp"].astype(str)
    df_inter_sql.to_sql("fact_interactions", conn, if_exists="replace", index=False)

    user_feats.to_sql("user_features", conn, if_exists="replace", index=False)
    item_feats.to_sql("item_features", conn, if_exists="replace", index=False)
    affinity_feats.to_sql("user_item_affinity", conn, if_exists="replace", index=False)
    conn.close()

    summary = {
        "warehouse_db": str(db_path),
        "dim_items_rows": int(len(df_prod)),
        "fact_interactions_rows": int(len(df_inter_sql)),
        "user_features_rows": int(len(user_feats)),
        "item_features_rows": int(len(item_feats)),
        "user_item_affinity_rows": int(len(affinity_feats)),
    }
    logger.info(f"Warehouse feature tables populated successfully: {summary}")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_warehouse_features(), indent=2))