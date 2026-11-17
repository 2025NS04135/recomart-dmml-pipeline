# Task 3: Raw Data Lake Storage Architecture

## 1. Storage Layer Overview
RecoMart uses a local file-based data lake (`data/raw/`) designed to mimic cloud object storage (such as AWS S3 or Google Cloud Storage). All ingested datasets are stored in immutable, append-only partitions organized by **Source System -> Entity Type -> UTC Ingestion Timestamp**.

## 2. Partitioned Folder Structure
```text
data/raw/
├── catalog_api/
│   └── products/
│       └── year=2026/
│           └── month=11/
│               └── day=15/
│                   └── hour=10/
│                       └── products.json
└── clickstream/
    └── interactions/
        └── year=2026/
            └── month=11/
                └── day=15/
                    └── hour=10/
                        └── interactions.csv