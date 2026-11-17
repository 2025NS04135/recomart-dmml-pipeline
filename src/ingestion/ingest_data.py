import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure audit logging to both file and console
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("IngestionPipeline")


def get_retrying_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """Create an HTTP session with automatic retry and exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_partition_path(base_dir: str, source: str, data_type: str, ts: datetime) -> Path:
    """Build Hive-style partitioned directory path: source/type/year=YYYY/month=MM/day=DD/hour=HH."""
    path = (
        Path(base_dir)
        / source
        / data_type
        / f"year={ts.strftime('%Y')}"
        / f"month={ts.strftime('%m')}"
        / f"day={ts.strftime('%d')}"
        / f"hour={ts.strftime('%H')}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def ingest_product_catalog(ts: datetime) -> Path:
    """Fetch product catalog metadata from DummyJSON REST API and store as partitioned JSON."""
    url = "https://dummyjson.com/products?limit=0"
    session = get_retrying_session()
    logger.info(f"Starting product catalog ingestion from REST API: {url}")

    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
        products = payload.get("products", [])
        logger.info(f"API fetch succeeded: retrieved {len(products)} products.")
    except Exception as exc:
        logger.error(f"API fetch failed ({exc}). Activating offline fallback generator.")
        categories = ["smartphones", "laptops", "fragrances", "skincare", "groceries", "home-decoration"]
        products = [
            {
                "id": i,
                "title": f"RecoMart Item {i}",
                "price": round(random.uniform(15.0, 899.0), 2),
                "description": f"Standard catalog item {i} in {categories[i % len(categories)]}",
                "category": categories[i % len(categories)],
                "rating": round(random.uniform(2.5, 5.0), 2),
                "reviews": [{"rating": 4}] * random.randint(5, 50),
            }
            for i in range(1, 195)
        ]

    flat_products = [
        {
            "item_id": int(p["id"]),
            "title": p["title"],
            "price": float(p["price"]),
            "description": p.get("description", ""),
            "category": p.get("category", "uncategorized"),
            "api_rating_rate": float(p.get("rating", np.nan)),
            "api_rating_count": len(p.get("reviews", [])),
            "ingested_at": ts.isoformat(),
        }
        for p in products
    ]

    out_dir = get_partition_path("data/raw", "catalog_api", "products", ts)
    out_file = out_dir / "products.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(flat_products, f, indent=2)

    logger.info(f"Saved {len(flat_products)} product records to {out_file}")
    return out_file


def ingest_clickstream_interactions(ts: datetime, num_events: int = 2500) -> Path:
    """Simulate batch user clickstream & transaction CSV logs and store in partitioned data lake."""
    logger.info(f"Starting clickstream batch ingestion ({num_events} events)...")
    rng = np.random.default_rng(seed=42)

    users = [f"U{str(i).zfill(3)}" for i in range(1, 101)]  # 100 users: U001 - U100
    items = np.arange(1, 195)  # 194 items matching DummyJSON catalog
    # Zipf-like popularity weights so popular items receive realistically more traffic
    weights = 1.0 / (items ** 0.65)
    weights /= weights.sum()

    event_types = ["view", "add_to_cart", "purchase", "rate"]
    event_probs = [0.55, 0.20, 0.15, 0.10]

    records = []
    for idx in range(num_events):
        u = str(rng.choice(users))
        i = int(rng.choice(items, p=weights))
        ev = str(rng.choice(event_types, p=event_probs))
        rating = (
            float(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], p=[0.05, 0.08, 0.17, 0.35, 0.35]))
            if ev in ("rate", "purchase")
            else np.nan
        )
        event_time = ts - timedelta(minutes=int(rng.integers(0, 20160)))  # Past 14 days
        records.append(
            {
                "event_id": f"EVT-{100000 + idx}",
                "user_id": u,
                "item_id": i,
                "event_type": ev,
                "rating": rating,
                "timestamp": event_time.isoformat(),
            }
        )

    # Inject 2 controlled anomalies so Step 3 (Data Validation) demonstrates anomaly detection
    records.append(records[0].copy())  # Anomaly 1: Duplicate event_id
    records.append(
        {
            "event_id": "EVT-999999",
            "user_id": "U001",
            "item_id": 5,
            "event_type": "rate",
            "rating": 7.5,  # Anomaly 2: Out-of-range rating (> 5.0)
            "timestamp": ts.isoformat(),
        }
    )

    df = pd.DataFrame(records)
    out_dir = get_partition_path("data/raw", "clickstream", "interactions", ts)
    out_file = out_dir / "interactions.csv"
    df.to_csv(out_file, index=False)

    logger.info(f"Saved {len(df)} interaction rows to {out_file}")
    return out_file


def run_ingestion() -> dict:
    now = datetime.now(timezone.utc)
    prod_path = ingest_product_catalog(now)
    inter_path = ingest_clickstream_interactions(now)
    logger.info("Ingestion stage completed successfully.")
    return {
        "products_path": str(prod_path),
        "interactions_path": str(inter_path),
        "ingested_at": now.isoformat(),
    }


if __name__ == "__main__":
    run_ingestion()