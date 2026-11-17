# Task 9: Recommendation Model Evaluation & Selection Report
- **Generated At (UTC):** `2026-11-17T14:38:45.342227+00:00`
- **Feature Store Version:** `v1.0` (`2174` total user-item pairs | `1739` train / `435` test)
- **Selected Champion Model:** `Hybrid_RandomForest` (Saved to `models\best_recommender.joblib`)

## 1. Benchmark Comparison (MLflow Experiment: `RecoMart_Recommendation_Models`)
| Model Name | RMSE (Lower is Better) | MAE (Lower is Better) | Precision@5 | Recall@5 | NDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `CollaborativeFiltering_SVD` | 1.1495 | 0.9614 | 0.3927 | 0.9898 | **0.7657** |
| `Hybrid_RandomForest` | 1.1101 | 0.9102 | 0.3927 | 0.9898 | **0.7853** |
| `Hybrid_GradientBoosting` | 1.1105 | 0.9024 | 0.3951 | 0.9959 | **0.7844** |

## 2. Sample Online Top-5 Recommendations for User `U001`
| Rank | Item ID | Product Title | Category | Price ($) | Predicted Score |
| :---: | :---: | :--- | :--- | :---: | :---: |
| 1 | `46` | Plant Pot | `home-decoration` | 14.99 | **3.2692** |
| 2 | `145` | Cricket Wicket | `sports-accessories` | 29.99 | **3.1926** |
| 3 | `84` | Gigabyte Aorus Men Tshirt | `mens-shirts` | 24.99 | **3.0583** |
| 4 | `68` | Pan | `kitchen-accessories` | 24.99 | **3.0062** |
| 5 | `153` | Volleyball | `sports-accessories` | 11.99 | **2.9947** |