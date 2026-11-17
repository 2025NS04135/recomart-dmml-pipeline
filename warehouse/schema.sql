CREATE TABLE IF NOT EXISTS dim_items (
    item_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    category_encoded INTEGER,
    price REAL,
    price_normalized REAL,
    api_rating_rate REAL,
    api_rating_count INTEGER
);

CREATE TABLE IF NOT EXISTS fact_interactions (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_type_encoded INTEGER,
    rating REAL,
    interaction_score REAL NOT NULL,
    timestamp TEXT NOT NULL,
    timestamp_normalized REAL,
    FOREIGN KEY (item_id) REFERENCES dim_items(item_id)
);

CREATE TABLE IF NOT EXISTS user_features (
    user_id TEXT PRIMARY KEY,
    user_activity_count INTEGER,
    user_purchase_count INTEGER,
    user_avg_score REAL,
    user_category_diversity INTEGER,
    recency_hours REAL,
    feature_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS item_features (
    item_id INTEGER PRIMARY KEY,
    item_popularity_count INTEGER,
    item_avg_score REAL,
    item_score_std REAL,
    conversion_rate REAL,
    feature_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS user_item_affinity (
    user_id TEXT,
    item_id INTEGER,
    category_affinity_score REAL,
    co_occurrence_boost REAL,
    PRIMARY KEY (user_id, item_id)
);