import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.validation.validate_data import get_latest_file

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
logger = logging.getLogger("PreparationPipeline")


def prepare_and_eda() -> dict:
    """Clean raw data, encode/normalize features, generate EDA plots, and save Parquet files."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    prod_file = get_latest_file("catalog_api/products/**/products.json")
    inter_file = get_latest_file("clickstream/interactions/**/interactions.csv")

    df_prod = pd.read_json(prod_file)
    df_inter = pd.read_csv(inter_file)
    raw_inter_count = len(df_inter)

    # 1. Data Cleaning: Deduplicate, Enforce Rating Bounds, & Referential Integrity
    df_prod = df_prod.drop_duplicates(subset=["item_id"]).copy()
    df_inter = df_inter.drop_duplicates(subset=["event_id"]).copy()
    df_inter = df_inter[
        df_inter["rating"].isna() | ((df_inter["rating"] >= 1.0) & (df_inter["rating"] <= 5.0))
    ].copy()
    df_inter = df_inter[df_inter["item_id"].isin(df_prod["item_id"])].copy()

    # 2. Handle Missing Explicit Ratings via Implicit Feedback Mapping
    implicit_weights = {"view": 1.5, "add_to_cart": 3.0, "purchase": 4.5, "rate": 3.5}
    df_inter["interaction_score"] = df_inter["rating"].fillna(df_inter["event_type"].map(implicit_weights))
    df_inter["timestamp"] = pd.to_datetime(df_inter["timestamp"], utc=True)

    # 3. Categorical Encoding & Numerical Normalization
    cat_encoder = LabelEncoder()
    df_prod["category_encoded"] = cat_encoder.fit_transform(df_prod["category"])

    event_encoder = LabelEncoder()
    df_inter["event_type_encoded"] = event_encoder.fit_transform(df_inter["event_type"])

    price_scaler = MinMaxScaler()
    df_prod["price_normalized"] = price_scaler.fit_transform(df_prod[["price"]])

    ts_scaler = MinMaxScaler()
    epoch_seconds = df_inter["timestamp"].astype("int64") // 10**9
    df_inter["timestamp_normalized"] = ts_scaler.fit_transform(epoch_seconds.to_frame())

    # 4. Exploratory Data Analysis (EDA) & Visualizations
    sns.set_theme(style="whitegrid")

    # Figure 1: Clickstream Event Counts & Unified Interaction Score Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(
        data=df_inter,
        x="event_type",
        hue="event_type",
        order=["view", "add_to_cart", "purchase", "rate"],
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Clickstream Event Type Distribution")
    axes[0].set_xlabel("Event Type")
    axes[0].set_ylabel("Event Count")

    sns.histplot(df_inter["interaction_score"], bins=10, kde=True, color="#2b5c8f", ax=axes[1])
    axes[1].set_title("Unified Interaction Score Distribution (1.0 - 5.0)")
    axes[1].set_xlabel("Interaction Score")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/figures/01_interaction_distributions.png", dpi=150)
    plt.close()

    # Figure 2: Top-15 Most Popular Items (Long-Tail Skew)
    top_items = df_inter["item_id"].value_counts().head(15).reset_index()
    top_items.columns = ["item_id", "interaction_count"]
    plt.figure(figsize=(10, 4.5))
    sns.barplot(
        data=top_items,
        x="item_id",
        y="interaction_count",
        hue="item_id",
        palette="mako",
        legend=False,
    )
    plt.title("Top-15 Most Popular Products (Long-Tail Interaction Skew)")
    plt.xlabel("Product Item ID")
    plt.ylabel("Total Interactions")
    plt.tight_layout()
    plt.savefig("reports/figures/02_item_popularity.png", dpi=150)
    plt.close()

    # Figure 3: User-Item Interaction Sparsity Heatmap
    full_matrix = df_inter.pivot_table(
        index="user_id", columns="item_id", values="interaction_score", aggfunc="mean"
    )
    total_cells = df_inter["user_id"].nunique() * df_prod["item_id"].nunique()
    observed_pairs = int(full_matrix.notna().sum().sum())
    sparsity = 1.0 - (observed_pairs / total_cells)

    plt.figure(figsize=(11, 6))
    sample_matrix = full_matrix.iloc[:35, :35].fillna(0)
    sns.heatmap(sample_matrix, cmap="YlGnBu", cbar_kws={"label": "Interaction Score"})
    plt.title(f"User-Item Matrix Sample (35x35) | Global Matrix Sparsity: {sparsity:.2%}")
    plt.xlabel("Item ID")
    plt.ylabel("User ID")
    plt.tight_layout()
    plt.savefig("reports/figures/03_sparsity_heatmap.png", dpi=150)
    plt.close()

    # 5. Export Prepared Datasets & EDA Metrics Summary
    inter_out = Path("data/processed/interactions_clean.parquet")
    prod_out = Path("data/processed/products_clean.parquet")
    df_inter.to_parquet(inter_out, index=False)
    df_prod.to_parquet(prod_out, index=False)

    summary = {
        "raw_interaction_rows": int(raw_inter_count),
        "clean_interaction_rows": int(len(df_inter)),
        "rows_removed": int(raw_inter_count - len(df_inter)),
        "clean_product_rows": int(len(df_prod)),
        "unique_users": int(df_inter["user_id"].nunique()),
        "unique_interacted_items": int(df_inter["item_id"].nunique()),
        "observed_user_item_pairs": observed_pairs,
        "matrix_sparsity": round(float(sparsity), 4),
        "mean_interaction_score": round(float(df_inter["interaction_score"].mean()), 4),
        "event_breakdown": {k: int(v) for k, v in df_inter["event_type"].value_counts().to_dict().items()},
    }
    Path("reports/eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        f"Data preparation complete: {raw_inter_count} -> {len(df_inter)} clean rows | "
        f"Sparsity={sparsity:.2%} | Saved Parquet files & 3 EDA plots."
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare_and_eda(), indent=2))