import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure logging works both when run standalone and when imported by Prefect
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
logger = logging.getLogger("ValidationPipeline")


def get_latest_file(base_pattern: str) -> Path:
    """Locate the most recent partitioned file matching the glob pattern."""
    files = sorted(Path("data/raw").glob(base_pattern))
    if not files:
        raise FileNotFoundError(f"No raw files matched pattern: {base_pattern}")
    return files[-1]


def validate_datasets() -> dict:
    """Execute automated profiling, schema, range, and integrity checks on raw data."""
    Path("reports").mkdir(exist_ok=True)

    prod_file = get_latest_file("catalog_api/products/**/products.json")
    inter_file = get_latest_file("clickstream/interactions/**/interactions.csv")
    logger.info(f"Validating raw partitions: products={prod_file}, interactions={inter_file}")

    df_prod = pd.read_json(prod_file)
    df_inter = pd.read_csv(inter_file)

    # 1. Schema Validation
    expected_inter_cols = {"event_id", "user_id", "item_id", "event_type", "rating", "timestamp"}
    expected_prod_cols = {
        "item_id",
        "title",
        "price",
        "description",
        "category",
        "api_rating_rate",
        "api_rating_count",
    }
    missing_inter_cols = sorted(list(expected_inter_cols - set(df_inter.columns)))
    missing_prod_cols = sorted(list(expected_prod_cols - set(df_prod.columns)))

    # 2. Completeness (Missing Values)
    null_inter = {k: int(v) for k, v in df_inter.isnull().sum().to_dict().items()}
    null_prod = {k: int(v) for k, v in df_prod.isnull().sum().to_dict().items()}

    # 3. Uniqueness (Duplicates)
    dup_events = int(df_inter.duplicated(subset=["event_id"]).sum())
    dup_items = int(df_prod.duplicated(subset=["item_id"]).sum())

    # 4. Range & Format Checks
    explicit_ratings = df_inter["rating"].dropna()
    invalid_ratings = int(((explicit_ratings < 1.0) | (explicit_ratings > 5.0)).sum())
    invalid_prices = int((df_prod["price"] <= 0).sum())
    valid_event_types = {"view", "add_to_cart", "purchase", "rate"}
    invalid_event_types = int((~df_inter["event_type"].isin(valid_event_types)).sum())

    parsed_ts = pd.to_datetime(df_inter["timestamp"], errors="coerce", utc=True)
    invalid_timestamps = int(parsed_ts.isna().sum())

    # 5. Referential Integrity (Foreign Key check)
    orphan_items = int((~df_inter["item_id"].isin(df_prod["item_id"])).sum())

    # Determine overall validation gate status
    if missing_inter_cols or missing_prod_cols or invalid_prices > 0 or invalid_timestamps > 0:
        status = "FAILED"
    elif dup_events > 0 or invalid_ratings > 0 or orphan_items > 0:
        status = "PASSED_WITH_WARNINGS"
    else:
        status = "PASSED"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "sources": {"products": str(prod_file), "interactions": str(inter_file)},
        "row_counts": {"products": int(len(df_prod)), "interactions": int(len(df_inter))},
        "schema_check": {
            "missing_interaction_cols": missing_inter_cols,
            "missing_product_cols": missing_prod_cols,
        },
        "anomalies": {
            "duplicate_event_ids": dup_events,
            "duplicate_item_ids": dup_items,
            "out_of_range_ratings_1_to_5": invalid_ratings,
            "non_positive_prices": invalid_prices,
            "invalid_event_types": invalid_event_types,
            "invalid_timestamps": invalid_timestamps,
            "orphan_item_references": orphan_items,
            "null_counts_interactions": null_inter,
            "null_counts_products": null_prod,
        },
    }

    # Save machine-readable JSON report
    json_path = Path("reports/data_quality_report.json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save human-readable Markdown report
    md_lines = [
        "# RecoMart Data Quality & Profiling Report",
        f"- **Generated At (UTC):** `{report['generated_at']}`",
        f"- **Validation Gate Status:** `{status}`",
        f"- **Product Catalog Rows:** `{len(df_prod)}` | **Interaction Log Rows:** `{len(df_inter)}`",
        "",
        "## 1. Schema & Completeness Summary",
        f"- **Missing Interaction Columns:** `{missing_inter_cols}`",
        f"- **Missing Product Columns:** `{missing_prod_cols}`",
        f"- **Null Values in Interactions:** `{null_inter}`",
        f"- **Null Values in Products:** `{null_prod}`",
        "",
        "## 2. Quality Rules & Anomaly Audit",
        "| Validation Dimension | Rule Checked | Issues Found | Pipeline Action |",
        "| :--- | :--- | :---: | :--- |",
        f"| Uniqueness | Unique `event_id` in clickstream | **{dup_events}** | Deduplicate in Preparation stage |",
        f"| Uniqueness | Unique `item_id` in catalog | **{dup_items}** | Deduplicate catalog by `item_id` |",
        f"| Range Check | Explicit `rating` within `[1.0, 5.0]` | **{invalid_ratings}** | Filter out invalid rating rows |",
        f"| Range Check | Product `price > 0` | **{invalid_prices}** | Fail pipeline if violated |",
        f"| Domain Check | `event_type` in `{sorted(list(valid_event_types))}` | **{invalid_event_types}** | Filter invalid event types |",
        f"| Format Check | Valid ISO-8601 UTC `timestamp` | **{invalid_timestamps}** | Parse & enforce UTC datetime |",
        f"| Referential Integrity | `item_id` exists in `products.json` | **{orphan_items}** | Inner-join against valid catalog |",
    ]
    md_path = Path("reports/data_quality_report.md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info(f"Validation completed with status={status}. Reports saved to {md_path} and {json_path}.")

    if status == "FAILED":
        raise ValueError("Critical data validation failure! Pipeline halted.")

    return report


if __name__ == "__main__":
    validate_datasets()