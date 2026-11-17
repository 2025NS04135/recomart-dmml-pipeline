# RecoMart Data Quality & Profiling Report
- **Generated At (UTC):** `2026-11-17T14:37:31.067997+00:00`
- **Validation Gate Status:** `PASSED_WITH_WARNINGS`
- **Product Catalog Rows:** `194` | **Interaction Log Rows:** `2502`

## 1. Schema & Completeness Summary
- **Missing Interaction Columns:** `[]`
- **Missing Product Columns:** `[]`
- **Null Values in Interactions:** `{'event_id': 0, 'user_id': 0, 'item_id': 0, 'event_type': 0, 'rating': 1899, 'timestamp': 0}`
- **Null Values in Products:** `{'item_id': 0, 'title': 0, 'price': 0, 'description': 0, 'category': 0, 'api_rating_rate': 0, 'api_rating_count': 0, 'ingested_at': 0}`

## 2. Quality Rules & Anomaly Audit
| Validation Dimension | Rule Checked | Issues Found | Pipeline Action |
| :--- | :--- | :---: | :--- |
| Uniqueness | Unique `event_id` in clickstream | **1** | Deduplicate in Preparation stage |
| Uniqueness | Unique `item_id` in catalog | **0** | Deduplicate catalog by `item_id` |
| Range Check | Explicit `rating` within `[1.0, 5.0]` | **1** | Filter out invalid rating rows |
| Range Check | Product `price > 0` | **0** | Fail pipeline if violated |
| Domain Check | `event_type` in `['add_to_cart', 'purchase', 'rate', 'view']` | **0** | Filter invalid event types |
| Format Check | Valid ISO-8601 UTC `timestamp` | **0** | Parse & enforce UTC datetime |
| Referential Integrity | `item_id` exists in `products.json` | **0** | Inner-join against valid catalog |