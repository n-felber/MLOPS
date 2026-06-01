import os
from pathlib import Path

from google.cloud import storage

from highjump_mlops.config import FEATURES_PATH, MODEL_PATH, RAW_RESULTS_PATH


DEFAULT_GCS_ARTIFACT_PREFIX = "latest"

ARTIFACTS: dict[str, Path] = {
    "data/raw/highjump_results.parquet": RAW_RESULTS_PATH,
    "data/features/highjump_features.parquet": FEATURES_PATH,
    "models/highjump_model.joblib": MODEL_PATH,
    "models/latest_mlflow_run.txt": MODEL_PATH.parent / "latest_mlflow_run.txt",
}

REQUIRED_INFERENCE_ARTIFACTS: list[str] = [
    "data/features/highjump_features.parquet",
    "models/highjump_model.joblib",
]

_download_done: bool = False


def get_bucket_name() -> str | None:
    bucket_name = os.getenv("GCS_BUCKET_NAME", "").strip()
    return bucket_name or None


def get_artifact_prefix() -> str:
    return os.getenv("GCS_ARTIFACT_PREFIX", DEFAULT_GCS_ARTIFACT_PREFIX).strip("/")


def cloud_blob_name(relative_path: str, prefix: str | None = None) -> str:
    effective_prefix = get_artifact_prefix() if prefix is None else prefix.strip("/")

    if not effective_prefix:
        return relative_path

    return f"{effective_prefix}/{relative_path}"


def upload_file(bucket: storage.Bucket, local_path: Path, relative_path: str, prefix: str | None = None) -> None:
    if not local_path.exists():
        print(f"Skipping missing artifact: {local_path}", flush=True)
        return

    blob_name = cloud_blob_name(relative_path, prefix)
    bucket.blob(blob_name).upload_from_filename(str(local_path))

    print(f"Uploaded {local_path} to gs://{bucket.name}/{blob_name}", flush=True)


def download_file(bucket: storage.Bucket, local_path: Path, relative_path: str, prefix: str | None = None) -> None:
    blob_name = cloud_blob_name(relative_path, prefix)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"Missing cloud artifact: gs://{bucket.name}/{blob_name}")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))

    print(f"Downloaded gs://{bucket.name}/{blob_name} to {local_path}", flush=True)


def upload_artifacts(bucket_name: str | None = None, prefix: str | None = None) -> None:
    effective_bucket_name = bucket_name or get_bucket_name()

    if not effective_bucket_name:
        raise ValueError("GCS_BUCKET_NAME is required to upload cloud artifacts.")

    client = storage.Client()
    bucket = client.bucket(effective_bucket_name)

    for relative_path, local_path in ARTIFACTS.items():
        upload_file(bucket, local_path, relative_path, prefix)


def download_artifacts(bucket_name: str | None = None, prefix: str | None = None, required_only: bool = False) -> None:
    effective_bucket_name = bucket_name or get_bucket_name()

    if not effective_bucket_name:
        raise ValueError("GCS_BUCKET_NAME is required to download cloud artifacts.")

    client = storage.Client()
    bucket = client.bucket(effective_bucket_name)

    artifact_paths = (
        REQUIRED_INFERENCE_ARTIFACTS if required_only else list(ARTIFACTS.keys())
    )

    for relative_path in artifact_paths:
        download_file(bucket, ARTIFACTS[relative_path], relative_path, prefix)


def ensure_cloud_artifacts_available() -> None:
    global _download_done

    if _download_done:
        return

    bucket_name = get_bucket_name()

    if not bucket_name:
        return

    missing_required_artifacts = [
        relative_path
        for relative_path in REQUIRED_INFERENCE_ARTIFACTS
        if not ARTIFACTS[relative_path].exists()
    ]

    if missing_required_artifacts:
        download_artifacts(bucket_name=bucket_name, required_only=True)

    _download_done = True


def upload_artifacts_cli() -> None:
    upload_artifacts()


def download_artifacts_cli() -> None:
    download_artifacts(required_only=True)
