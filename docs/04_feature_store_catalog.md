# Task 7: Feature Store Metadata & Lineage Catalog
- **Registry Backend:** `feature_repo/registry.db`
- **Warehouse Source:** `warehouse/recomart_dw.db`

## Registered Feature Views
| Feature View | Version | Entity Key | Source Table | Features | Transformation Logic |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `item_popularity_features` | `v1.0` | `item_id` | `item_features` | `item_popularity_count`, `item_avg_score`, `item_score_std`, `conversion_rate` | Item-level engagement volume, preference score mean/std, and view-to-purchase conversion rate. |
| `user_behavioral_features` | `v1.0` | `user_id` | `user_features` | `user_activity_count`, `user_purchase_count`, `user_avg_score`, `user_category_diversity`, `recency_hours` | Aggregated clickstream frequency, purchase counts, mean interaction score, category diversity, and recency. |
| `user_item_cross_features` | `v1.0` | `user_id,item_id` | `user_item_affinity` | `category_affinity_score`, `co_occurrence_boost` | Normalized user category preference share and log-scaled co-occurrence boost. |