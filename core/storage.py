"""
Durable storage for contributions.

The Contribute tab writes to `contributions/` on the disk of whatever machine
is running the app. On a field laptop that disk persists. On Streamlit
Community Cloud it does not: a reboot, or any push to the deployed branch,
re-clones the repository and everything submitted since is gone. An app that
accepts a photograph, thanks the contributor, and then loses it is worse than
one with no Contribute tab, because the contributor believes the job is done.

So when object storage is configured, every submission is written there as well
as locally, and a submission that cannot be stored durably is refused rather
than accepted into a directory that will not survive the week.

CONFIGURATION
-------------
Any S3-compatible service — AWS S3, Cloudflare R2, Backblaze B2, Wasabi,
MinIO. Set these in the environment, in a gitignored `.env`, or in the
deployment's secrets:

    CHELONID_S3_BUCKET       required; enabling this switches durable mode on
    CHELONID_S3_ACCESS_KEY   required
    CHELONID_S3_SECRET_KEY   required
    CHELONID_S3_ENDPOINT     required for anything that is not AWS S3
    CHELONID_S3_REGION       optional, defaults to us-east-1
    CHELONID_S3_PREFIX       optional key prefix, e.g. "chelonid/"

With no bucket set the app behaves exactly as it did before: local writes only,
and the Contribute tab says so rather than implying a durability it does not
have.

WHAT IS STORED
--------------
    <prefix>images/<species_id>_<digest>.jpg     the scrubbed photograph
    <prefix>records/<record_id>.json             its metadata

Two objects rather than one so that `training/pull_contributions.py` can list
metadata without downloading photographs. EXIF is already stripped before
anything reaches this module; nothing here re-encodes or inspects pixels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from config import env_value
from core import github_storage

logger = logging.getLogger(__name__)

BUCKET_ENV = "CHELONID_S3_BUCKET"
ACCESS_KEY_ENV = "CHELONID_S3_ACCESS_KEY"
SECRET_KEY_ENV = "CHELONID_S3_SECRET_KEY"
ENDPOINT_ENV = "CHELONID_S3_ENDPOINT"
REGION_ENV = "CHELONID_S3_REGION"
PREFIX_ENV = "CHELONID_S3_PREFIX"

IMAGE_PREFIX = "images/"
RECORD_PREFIX = "records/"


class StorageError(Exception):
    """A durable write was expected and did not happen."""


@dataclass(frozen=True)
class Settings:
    bucket: str
    access_key: str
    secret_key: str
    endpoint: str | None
    region: str
    prefix: str

    def key_for(self, name: str) -> str:
        return f"{self.prefix}{name}"


def settings() -> Settings | None:
    """Configuration, or None when durable storage is switched off.

    A bucket named without credentials is a misconfiguration rather than a
    choice to run locally, so it is reported instead of being silently ignored
    — that mistake would otherwise look identical to working correctly right up
    until the first reboot.
    """
    bucket = env_value(BUCKET_ENV)
    if not bucket:
        return None

    access_key = env_value(ACCESS_KEY_ENV)
    secret_key = env_value(SECRET_KEY_ENV)
    if not access_key or not secret_key:
        raise StorageError(
            f"{BUCKET_ENV} is set but {ACCESS_KEY_ENV} and {SECRET_KEY_ENV} are "
            f"not. Set both, or unset {BUCKET_ENV} to store contributions "
            f"locally only."
        )

    prefix = (env_value(PREFIX_ENV) or "").lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    return Settings(
        bucket=bucket,
        access_key=access_key,
        secret_key=secret_key,
        endpoint=env_value(ENDPOINT_ENV),
        region=env_value(REGION_ENV) or "us-east-1",
        prefix=prefix,
    )


def configured() -> bool:
    """Whether submissions will be stored durably. Never raises."""
    if github_storage.configured():
        return True
    try:
        return settings() is not None
    except StorageError as exc:
        logger.error("Durable storage is misconfigured: %s", exc)
        return False


def _github_first(action, *args) -> str | None:
    """Run `action` on the GitHub backend when it is the configured one.

    Returns None when GitHub is not in use, so the caller falls through to S3.
    GitHub wins if both are set: it is the more deliberate configuration, and
    silently writing somewhere the maintainer did not intend is worse than
    ignoring one of two answers.
    """
    if not github_storage.configured():
        return None
    try:
        both = settings() is not None
    except StorageError:
        # S3 is half-configured. That is worth knowing, but it is not this
        # submission's problem: GitHub is the backend in force.
        both = False
    if both:
        logger.warning(
            "Both GitHub and S3 contribution storage are configured. Using "
            "GitHub; unset %s to silence this.", REPO_HINT,
        )
    try:
        return action(*args)
    except github_storage.GitHubStorageError as exc:
        raise StorageError(str(exc)) from exc


REPO_HINT = github_storage.REPO_ENV


def client(config: Settings | None = None):
    """A boto3 S3 client. Imported lazily so the app runs without boto3."""
    config = config or settings()
    if config is None:
        raise StorageError("Durable storage is not configured.")
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise StorageError(
            "boto3 is not installed, so contributions cannot be stored "
            "durably. Install it, or unset "
            f"{BUCKET_ENV} to accept local-only storage."
        ) from exc

    return boto3.client(
        "s3",
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        endpoint_url=config.endpoint,
        region_name=config.region,
    )


def put_image(filename: str, data: bytes, *, s3=None, config: Settings | None = None) -> str:
    """Store a photograph. Returns its key. Raises StorageError on failure."""
    committed = _github_first(github_storage.put_image, filename, data)
    if committed is not None:
        return committed
    return _put(f"{IMAGE_PREFIX}{filename}", data, "image/jpeg", s3=s3, config=config)


def put_record(record: dict[str, Any], *, s3=None, config: Settings | None = None) -> str:
    """Store a contribution record. Returns its key."""
    committed = _github_first(github_storage.put_record, record)
    if committed is not None:
        return committed
    body = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    return _put(f"{RECORD_PREFIX}{record['id']}.json", body, "application/json",
                s3=s3, config=config)


def _put(name: str, data: bytes, content_type: str, *, s3, config: Settings | None) -> str:
    config = config or settings()
    if config is None:
        raise StorageError("Durable storage is not configured.")
    s3 = s3 or client(config)
    key = config.key_for(name)
    try:
        s3.put_object(Bucket=config.bucket, Key=key, Body=data, ContentType=content_type)
    except Exception as exc:
        # Deliberately broad: botocore raises a wide family, and the caller's
        # only decision is whether the write happened.
        raise StorageError(f"Could not store {key}: {exc}") from exc
    return key


def list_records(*, s3=None, config: Settings | None = None) -> list[str]:
    """Keys of every stored contribution record."""
    config = config or settings()
    if config is None:
        raise StorageError("Durable storage is not configured.")
    s3 = s3 or client(config)
    prefix = config.key_for(RECORD_PREFIX)
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": config.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3.list_objects_v2(**kwargs)
        keys.extend(item["Key"] for item in response.get("Contents", []))
        if not response.get("IsTruncated"):
            return keys
        token = response.get("NextContinuationToken")


def fetch(key: str, *, s3=None, config: Settings | None = None) -> bytes:
    config = config or settings()
    if config is None:
        raise StorageError("Durable storage is not configured.")
    s3 = s3 or client(config)
    try:
        return s3.get_object(Bucket=config.bucket, Key=key)["Body"].read()
    except Exception as exc:
        raise StorageError(f"Could not fetch {key}: {exc}") from exc


def describe() -> str:
    """One line for the UI. Never raises, never prints a credential."""
    if github_storage.configured():
        return f"GitHub {github_storage.describe()}"
    try:
        config = settings()
    except StorageError as exc:
        return f"misconfigured — {exc}"
    if config is None:
        github_state = github_storage.describe()
        if github_state.startswith("misconfigured"):
            return github_state
        return "local only"
    where = config.endpoint or "AWS S3"
    return f"{config.bucket} at {where}"
