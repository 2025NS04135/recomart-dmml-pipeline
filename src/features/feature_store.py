import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

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
logger = logging.getLogger("FeatureStore")


class RecoMartFeatureStore:
    """Versioned Feature Store & Metadata Registry for offline training and online inference."""

    def __init__(
        self,
        registry_db: str = "feature_repo/registry.db",
        warehouse_db: str = "warehouse/recomart_dw.db",
    ):
        Path(registry_db).parent.mkdir(parents=True, exist_ok=True)
        self.registry_db = registry_db
        self.warehouse_db = warehouse_db
        self._init_registry()

    def _init_registry(self) -> None:
        with sqlite3.connect(self.registry_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    feature_view TEXT NOT NULL,
                    version TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    source_table TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    transformation_desc TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (feature_view, version)
                )
                """
            )

    def register_and_materialize(self, version: str = "v1.0") -> dict:
        """Register feature view metadata and materialize versioned snapshots from the warehouse."""
        now = datetime.now(timezone.utc).isoformat()
        specs = [
            (
                "user_behavioral_features",
                version,
                "user_id",
                "user_features",
                [
                    "user_activity_count",
                    "user_purchase_count",
                    "user_avg_score",
                    "user_category_diversity",
                    "recency_hours",
                ],
                "Aggregated clickstream frequency, purchase counts, mean interaction score, category diversity, and recency.",
                now,
            ),
            (
                "item_popularity_features",
                version,
                "item_id",
                "item_features",
                [
                    "item_popularity_count",
                    "item_avg_score",
                    "item_score_std",
                    "conversion_rate",
                ],
                "Item-level engagement volume, preference score mean/std, and view-to-purchase conversion rate.",
                now,
            ),
            (
                "user_item_cross_features",
                version,
                "user_id,item_id",
                "user_item_affinity",
                [
                    "category_affinity_score",
                    "co_occurrence_boost",
                ],
                "Normalized user category preference share and log-scaled co-occurrence boost.",
                now,
            ),
        ]

        row_counts = {}
        with sqlite3.connect(self.registry_db) as reg_conn, sqlite3.connect(self.warehouse_db) as wh_conn:
            for fv, ver, ent, src, feats, desc, ts in specs:
                reg_conn.execute(
                    "INSERT OR REPLACE INTO feature_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (fv, ver, ent, src, json.dumps(feats), desc, ts),
                )
                df = pd.read_sql_query(f"SELECT * FROM {src}", wh_conn)
                snapshot_table = f"{fv}_{ver.replace('.', '_')}"
                df.to_sql(snapshot_table, reg_conn, if_exists="replace", index=False)
                row_counts[snapshot_table] = len(df)

        logger.info(f"Registered & materialized 3 feature views at version={version}: {row_counts}")
        return {"version": version, "materialized_tables": row_counts}

    def export_metadata_doc(self, out_path: str = "docs/04_feature_store_catalog.md") -> Path:
        """Generate Markdown documentation of all registered feature views and lineage."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.registry_db) as conn:
            meta_df = pd.read_sql_query("SELECT * FROM feature_metadata ORDER BY version, feature_view", conn)

        lines = [
            "# Task 7: Feature Store Metadata & Lineage Catalog",
            f"- **Registry Backend:** `{self.registry_db}`",
            f"- **Warehouse Source:** `{self.warehouse_db}`",
            "",
            "## Registered Feature Views",
            "| Feature View | Version | Entity Key | Source Table | Features | Transformation Logic |",
            "| :--- | :---: | :--- | :--- | :--- | :--- |",
        ]
        for row in meta_df.itertuples():
            feats = ", ".join(f"`{f}`" for f in json.loads(row.features_json))
            lines.append(
                f"| `{row.feature_view}` | `{row.version}` | `{row.entity}` | `{row.source_table}` | {feats} | {row.transformation_desc} |"
            )

        target = Path(out_path)
        target.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Exported feature metadata catalog to {target}")
        return target

    def get_historical_features(self, version: str = "v1.0") -> pd.DataFrame:
        """Offline Retrieval: Point-in-time training dataset joining targets with versioned feature views."""
        ver_suffix = version.replace(".", "_")
        with sqlite3.connect(self.warehouse_db) as wh_conn:
            interactions = pd.read_sql_query(
                "SELECT user_id, item_id, AVG(interaction_score) AS target_score "
                "FROM fact_interactions GROUP BY user_id, item_id",
                wh_conn,
            )
        with sqlite3.connect(self.registry_db) as reg_conn:
            u_df = pd.read_sql_query(f"SELECT * FROM user_behavioral_features_{ver_suffix}", reg_conn)
            i_df = pd.read_sql_query(f"SELECT * FROM item_popularity_features_{ver_suffix}", reg_conn)
            c_df = pd.read_sql_query(f"SELECT * FROM user_item_cross_features_{ver_suffix}", reg_conn)

        training_df = (
            interactions.merge(u_df.drop(columns=["feature_timestamp"]), on="user_id", how="left")
            .merge(i_df.drop(columns=["feature_timestamp"]), on="item_id", how="left")
            .merge(c_df, on=["user_id", "item_id"], how="left")
        )
        return training_df

    def get_online_features(self, user_id: str, version: str = "v1.0") -> pd.DataFrame:
        """Online Retrieval: Serve candidate feature vectors for a single user at inference time."""
        ver_suffix = version.replace(".", "_")
        with sqlite3.connect(self.registry_db) as reg_conn:
            u_df = pd.read_sql_query(
                f"SELECT * FROM user_behavioral_features_{ver_suffix} WHERE user_id = ?",
                reg_conn,
                params=(user_id,),
            )
            i_df = pd.read_sql_query(f"SELECT * FROM item_popularity_features_{ver_suffix}", reg_conn)
            c_df = pd.read_sql_query(
                f"SELECT * FROM user_item_cross_features_{ver_suffix} WHERE user_id = ?",
                reg_conn,
                params=(user_id,),
            )

        candidates = c_df.merge(u_df.drop(columns=["feature_timestamp"]), on="user_id", how="inner").merge(
            i_df.drop(columns=["feature_timestamp"]), on="item_id", how="inner"
        )
        return candidates


if __name__ == "__main__":
    fs = RecoMartFeatureStore()
    mat_summary = fs.register_and_materialize(version="v1.0")
    doc_path = fs.export_metadata_doc()

    print("=== 1. MATERIALIZATION SUMMARY ===")
    print(json.dumps(mat_summary, indent=2))

    hist_df = fs.get_historical_features(version="v1.0")
    print(f"\n=== 2. OFFLINE TRAINING RETRIEVAL (v1.0) | Shape: {hist_df.shape} ===")
    print(hist_df.head(3).to_string(index=False))

    online_df = fs.get_online_features(user_id="U001", version="v1.0")
    print(f"\n=== 3. ONLINE INFERENCE RETRIEVAL FOR U001 (v1.0) | Candidates: {len(online_df)} ===")
    print(online_df.head(5).to_string(index=False))