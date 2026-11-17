# Task 8: Data Versioning & Lineage Workflow

## 1. Versioning Architecture (Git + DVC)
RecoMart separates lightweight source code tracking from heavy binary dataset versioning by pairing **Git** with **DVC (Data Version Control)**:
- **Git:** Versions Python ETL modules, SQL warehouse schemas (`warehouse/schema.sql`), `.dvc` cryptographic pointer files, documentation, and `reports/lineage_manifest.json`.
- **DVC:** Versions raw data lake partitions (`data/raw`), cleaned Parquet datasets (`data/processed`), the SQLite analytical warehouse (`warehouse/recomart_dw.db`), and the materialized Feature Store (`feature_repo/registry.db`) using MD5 content hashes stored in `.dvc/cache`.

## 2. Repository Structure Showing Dataset Versions
```text
recomart_dmml_pipeline/
├── .dvc/                         # DVC local cache & configuration
├── data/
│   ├── raw.dvc                   # MD5 pointer to raw partitioned data lake (v1.0)
│   └── processed.dvc             # MD5 pointer to cleaned Parquet datasets (v1.0)
├── warehouse/
│   ├── schema.sql                # Git-tracked relational star schema DDL
│   └── recomart_dw.db.dvc        # MD5 pointer to SQLite analytical warehouse (v1.0)
├── feature_repo/
│   └── registry.db.dvc           # MD5 pointer to materialized Feature Store (v1.0)
└── reports/
    └── lineage_manifest.json     # End-to-end provenance, timestamps, and MD5 hashes