# Task 1: Problem Formulation — RecoMart Recommendation Pipeline

## 1. Business Problem & Objectives
RecoMart, a growing e-commerce platform, currently displays static product listings that result in low click-through rates (CTR) and missed cross-selling opportunities. 
- **Primary Objective:** Design and implement an automated, reproducible end-to-end data management and machine learning pipeline that ingests multi-source user and catalog data, enforces data quality, engineers behavioral features, and trains personalized Top-K recommendation models.
- **Business Impact:** Improve product discovery, user engagement, and order conversion rates through fresh, data-driven recommendations.

## 2. Key Data Sources & Attributes
To model RecoMart's ecosystem, the pipeline integrates two primary data sources:

1. **User Clickstream & Interaction Logs (Batch CSV / Simulated Event Stream)**
   - `event_id` (String, Primary Key): Unique identifier for each user action.
   - `user_id` (String): Anonymized customer identifier (e.g., `U001`–`U080`).
   - `item_id` (Integer, Foreign Key): Product identifier linking to the catalog.
   - `event_type` (Categorical): Interaction funnel stage (`view`, `add_to_cart`, `purchase`, `rate`).
   - `rating` (Float, Nullable): Explicit user satisfaction rating on a `1.0–5.0` scale.
   - `timestamp` (ISO-8601 UTC): Event occurrence timestamp.

2. **Product Catalog & Popularity Metadata (DummyJSON REST API: `https://dummyjson.com/products?limit=0`)**
   - `item_id` (Integer, Primary Key): Unique product ID (`1`–`194`).
   - `title` (String): Product name used for content-based text representations.
   - `category` (Categorical): Product taxonomy label (e.g., `beauty`, `laptops`, `smartphones`).
   - `price` (Float): Retail price in USD (`> 0`).
   - `description` (String): Product specification text.
   - `api_rating_rate` (Float): Aggregated external product rating (`0.0–5.0`).
   - `api_rating_count` (Integer): Total external customer review count.

## 3. Expected Pipeline Outputs
1. **Validated & Cleaned Datasets:** Deduplicated, schema-verified Parquet files (`interactions_clean.parquet`, `products_clean.parquet`) ready for Exploratory Data Analysis (EDA).
2. **Engineered Features & Feature Store:** Analytical warehouse tables (`user_features`, `item_features`, `user_item_affinity`) materialized into a versioned Feature Store for offline training and online Top-K inference.
3. **Trained Models & Inference Interface:** Collaborative Filtering (Truncated SVD) and Content-Based Filtering (TF-IDF + Cosine Similarity) models tracked in MLflow, exposing a Top-K recommendation generator for any `user_id`.

## 4. Evaluation Metrics
Model performance is evaluated on a held-out 20% test split using both rating accuracy and ranking metrics at cutoff $K = 5$:
- **RMSE (Root Mean Squared Error):** Measures predicted vs. actual preference score accuracy.
- **Precision@5:** Fraction of Top-5 recommended items that are relevant ($\text{score} \ge 3.0$).
- **Recall@5:** Fraction of all relevant test items captured within the Top-5 recommendations.
- **NDCG@5 (Normalized Discounted Cumulative Gain):** Evaluates ranking quality by rewarding relevant items placed higher in the Top-5 list.