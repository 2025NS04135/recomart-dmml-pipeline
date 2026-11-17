import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("DataLineage")


def parse_dvc_md5(dvc_file: str) -> str:
    """Extract the MD5 content hash from a .dvc metadata pointer file."""
    path = Path(dvc_file)
    if not path.exists():
        return "not_tracked_yet"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- md5:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def generate_lineage_manifest(dataset_version: str = "data-v1.0") -> dict:
    """Create a structured lineage manifest linking raw sources to engineered feature views."""
    val_report = json.loads(Path("reports/data_quality_report.json").read_text(encoding="utf-8"))
    eda_report = json.loads(Path("reports/eda_summary.json").read_text(encoding="utf-8"))

    manifest = {
        "dataset_version_tag": dataset_version,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_sources": {
            "product_catalog": {
                "source_type": "REST API (DummyJSON)",
                "raw_partition": val_report["sources"]["products"],
                "record_count": val_report["row_counts"]["products"],
            },
            "user_interactions": {
                "source_type": "Batch Clickstream CSV",
                "raw_partition": val_report["sources"]["interactions"],
                "raw_record_count": val_report["row_counts"]["interactions"],
                "clean_record_count": eda_report["clean_interaction_rows"],
            },
        },
        "applied_transformations": [
            "1_Ingestion: Hive-partitioned raw storage by source/type/YYYY/MM/DD/HH",
            "2_Validation: Schema, null, duplicate, [1.0-5.0] rating range, and FK integrity checks",
            "3_Preparation: Deduplication, implicit feedback mapping, LabelEncoding, MinMaxScaling",
            "4_Feature_Engineering: User behavioral stats, item conversion rates, user-item affinity",
            "5_Feature_Store: Materialized v1.0 offline & online feature views in registry.db",
        ],
        "validation_status": val_report["status"],
        "dvc_tracked_artifacts": {
            "data/raw.dvc": parse_dvc_md5("data/raw.dvc"),
            "data/processed.dvc": parse_dvc_md5("data/processed.dvc"),
            "warehouse/recomart_dw.db.dvc": parse_dvc_md5("warehouse/recomart_dw.db.dvc"),
            "feature_repo/registry.db.dvc": parse_dvc_md5("feature_repo/registry.db.dvc"),
        },
    }

    out_path = Path("reports/lineage_manifest.json")
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(generate_lineage_manifest(), indent=2))