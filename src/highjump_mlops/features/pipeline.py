import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from highjump_mlops.config import FEATURES_PATH, RAW_RESULTS_PATH
from highjump_mlops.features.engineering import build_features


FEATURE_VERSION_FILENAME = "highjump_features.parquet"
FEATURE_VERSIONS_DIR = FEATURES_PATH.parent / "versions"
LATEST_FEATURE_VERSION_PATH = FEATURES_PATH.parent / "latest_feature_version.json"


def create_feature_version_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def feature_version_path(version_id: str) -> Path:
    return FEATURE_VERSIONS_DIR / version_id / FEATURE_VERSION_FILENAME


def save_feature_store_version(features: pd.DataFrame, source_path: Path) -> Path:
    version_id = create_feature_version_id()
    versioned_path = feature_version_path(version_id)

    versioned_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(versioned_path, index=False)

    latest_metadata: dict[str, Any] = {
        "version_id": version_id,
        "latest_path": str(FEATURES_PATH),
        "versioned_path": str(versioned_path),
        "source_path": str(source_path),
        "row_count": int(len(features)),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    LATEST_FEATURE_VERSION_PATH.write_text(
        json.dumps(latest_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return versioned_path


def main() -> None:
    if not RAW_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Raw result file not found: {RAW_RESULTS_PATH}. "
            "Fetch the raw data inside Docker first:\n\n"
            "make fetch"
        )

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = pd.read_parquet(RAW_RESULTS_PATH)
    features = build_features(results)

    features.to_parquet(FEATURES_PATH, index=False)
    versioned_path = save_feature_store_version(features, RAW_RESULTS_PATH)

    print(f"Saved {len(features)} latest feature rows to {FEATURES_PATH}", flush=True)
    print(f"Saved versioned feature store file to {versioned_path}", flush=True)
    print(f"Saved latest feature version metadata to {LATEST_FEATURE_VERSION_PATH}", flush=True)


if __name__ == "__main__":
    main()
