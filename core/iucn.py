"""
IUCN Red List API v4 client.

Endpoints used (from https://api.iucnredlist.org/api-docs/v4/openapi.yaml):

    GET /api/v4/taxa/scientific_name?genus_name=&species_name=
        Summary of latest and historic assessments; gives assessment_ids.
    GET /api/v4/assessment/{assessment_id}
        Full assessment: category, criteria, population trend, habitats,
        threats, use and trade, conservation actions, locations, citation.
    GET /api/v4/information/red_list_version
        Which Red List version the API is currently serving.

Three things about this client are deliberate.

**It caches to disk, and that is not an optimisation.** IUCN's own terms ask
API users to cache between Red List releases rather than re-query, and warn
that extraction-style usage can get a token rate-limited or revoked. The cache
is keyed by Red List version, so it invalidates itself when IUCN publishes an
update instead of going stale silently.

**It never raises into the application.** A field laptop in Churna has no
connectivity. Every failure path returns None and logs; the app falls back to
the curated local database, which is the authoritative source for this tool
anyway.

**It does not write to the species database.** IUCN supplies conservation
status; it does not supply WPA schedules, diagnostic characters, or the
confusion pairs that make this tool useful. Sync writes to a separate cache and
reports divergences for a human to review. See scripts/sync_iucn.py.

Token: set the IUCN_API_TOKEN environment variable. Request one at
https://api.iucnredlist.org/users/sign_up. Never commit it.

Citation required when using this data:
IUCN. IUCN Red List of Threatened Species. https://www.iucnredlist.org
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DATA_DIR

logger = logging.getLogger(__name__)

API_BASE = "https://api.iucnredlist.org/api/v4"
CACHE_PATH = DATA_DIR / "iucn_cache.json"
TOKEN_ENV = "IUCN_API_TOKEN"

REQUEST_TIMEOUT = 20        # seconds
MIN_REQUEST_INTERVAL = 1.1  # seconds between calls; be a good citizen
MAX_RETRIES = 3

SIGNUP_URL = "https://api.iucnredlist.org/users/sign_up"


class IUCNError(Exception):
    """Raised only by explicit CLI paths. The app-facing methods return None."""


def get_token() -> str | None:
    """
    Read the token from the environment, or from a local .env file.

    .env is gitignored. Storing a token in the repository is how tokens end up
    revoked, so it is never written by this code — only read.
    """
    token = os.environ.get(TOKEN_ENV, "").strip()
    if token:
        return token

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == TOKEN_ENV:
                    return value.strip().strip("'\"")
        except OSError as exc:
            logger.warning("Could not read .env: %s", exc)
    return None


class IUCNClient:
    """Cached, rate-limited, failure-tolerant client for the Red List API v4."""

    def __init__(self, token: str | None = None, cache_path: Path = CACHE_PATH):
        self.token = token or get_token()
        self.cache_path = Path(cache_path)
        self._cache = self._load_cache()
        self._last_request = 0.0

    # -- availability --------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.token)

    @property
    def red_list_version(self) -> str | None:
        return self._cache.get("_meta", {}).get("red_list_version")

    @property
    def cached_species(self) -> list[str]:
        return [k for k in self._cache if not k.startswith("_")]

    # -- cache ---------------------------------------------------------
    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {"_meta": {}}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("IUCN cache unreadable (%s); starting fresh.", exc)
            return {"_meta": {}}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self.cache_path)
        except OSError as exc:
            logger.error("Could not write IUCN cache: %s", exc)

    def cached(self, scientific_name: str) -> dict[str, Any] | None:
        """Cache lookup only. Works offline; this is what the app calls."""
        return self._cache.get(scientific_name)

    # -- transport -----------------------------------------------------
    def _request(self, path: str, params: dict[str, str] | None = None) -> Any | None:
        if not self.token:
            logger.info("No IUCN token configured; skipping request to %s", path)
            return None

        url = f"{API_BASE}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        for attempt in range(1, MAX_RETRIES + 1):
            elapsed = time.monotonic() - self._last_request
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)

            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/json",
                    "User-Agent": "Chelonid-ID/1.0 (conservation tooling)",
                },
            )
            try:
                self._last_request = time.monotonic()
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    logger.error(
                        "IUCN rejected the token (401). Check %s is set correctly. "
                        "Request or cycle a token at %s", TOKEN_ENV, SIGNUP_URL
                    )
                    return None
                if exc.code == 404:
                    logger.info("IUCN has no record for %s", url)
                    return None
                if exc.code == 429:
                    wait = min(60, 2 ** attempt * 5)
                    logger.warning("Rate limited by IUCN; waiting %ds", wait)
                    time.sleep(wait)
                    continue
                logger.warning("IUCN HTTP %d on attempt %d", exc.code, attempt)

            except (urllib.error.URLError, TimeoutError) as exc:
                logger.warning(
                    "IUCN unreachable on attempt %d (%s). Offline is normal in "
                    "the field; the local database still works.", attempt, exc
                )
            except json.JSONDecodeError as exc:
                logger.error("IUCN returned unparseable JSON: %s", exc)
                return None

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

        return None

    # -- endpoints -----------------------------------------------------
    def fetch_red_list_version(self) -> str | None:
        payload = self._request("/information/red_list_version")
        return payload.get("red_list_version") if isinstance(payload, dict) else None

    def fetch_species(self, scientific_name: str) -> dict[str, Any] | None:
        """
        Fetch and cache the latest global assessment for a binomial.

        Returns a flattened summary, or None on any failure.
        """
        parts = scientific_name.strip().split()
        if len(parts) < 2:
            logger.warning("Not a binomial: %r", scientific_name)
            return None
        genus, species = parts[0], parts[1]
        infra = parts[2] if len(parts) > 2 else None

        params = {"genus_name": genus, "species_name": species}
        if infra:
            params["infra_name"] = infra

        taxon = self._request("/taxa/scientific_name", params)
        if not isinstance(taxon, dict):
            return None

        assessments = taxon.get("assessments") or []
        if not assessments:
            logger.info("No assessments returned for %s", scientific_name)
            return None

        # Prefer the latest global assessment; fall back to whatever is latest.
        latest = next(
            (a for a in assessments
             if a.get("latest") and (a.get("scopes") or [{}])[0].get("code") == "1"),
            None,
        ) or next((a for a in assessments if a.get("latest")), assessments[0])

        assessment_id = latest.get("assessment_id")
        detail = self._request(f"/assessment/{assessment_id}") if assessment_id else None

        summary = self._flatten(scientific_name, taxon, latest, detail)
        self._cache[scientific_name] = summary
        self._cache.setdefault("_meta", {})["last_sync"] = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")
        self._save_cache()
        return summary

    # -- shaping -------------------------------------------------------
    @staticmethod
    def _flatten(
        scientific_name: str,
        taxon: dict[str, Any],
        latest: dict[str, Any],
        detail: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Reduce the v4 response to the fields this tool uses.

        The v4 schema declares many fields as free-form objects, so every
        access here is defensive — a shape change upstream should degrade a
        field to None, not crash a sync.
        """
        def dig(obj: Any, *keys: str, default: Any = None) -> Any:
            for key in keys:
                if not isinstance(obj, dict):
                    return default
                obj = obj.get(key)
            return obj if obj is not None else default

        detail = detail or {}
        out: dict[str, Any] = {
            "scientific_name": scientific_name,
            "sis_id": taxon.get("sis_id") or dig(taxon, "taxon", "sis_id"),
            "assessment_id": latest.get("assessment_id"),
            "category": dig(detail, "red_list_category", "code")
            or latest.get("red_list_category_code"),
            "category_name": dig(detail, "red_list_category", "description", "en"),
            "criteria": detail.get("criteria"),
            "year_published": latest.get("year_published"),
            "assessment_date": detail.get("assessment_date"),
            "population_trend": dig(detail, "population_trend", "description", "en"),
            "citation": dig(detail, "citation", "value"),
            "url": dig(detail, "url"),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        def names(block: Any, *path: str) -> list[str]:
            if not isinstance(block, list):
                return []
            collected = []
            for item in block:
                value = item
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                if isinstance(value, str):
                    collected.append(value)
            return sorted(set(collected))

        out["habitats"] = names(detail.get("habitats"), "description", "en")
        out["threats"] = names(detail.get("threats"), "description", "en")
        out["use_and_trade"] = names(detail.get("use_and_trade"), "description", "en")
        out["conservation_actions"] = names(
            detail.get("conservation_actions"), "description", "en"
        )
        out["countries"] = names(detail.get("locations"), "description", "en")

        # Historic assessments are useful context: an uplisting from EN to CR
        # is a signal a field officer should see.
        history = []
        for a in taxon.get("assessments") or []:
            if a.get("year_published"):
                history.append({
                    "year": a.get("year_published"),
                    "category": a.get("red_list_category_code"),
                    "latest": bool(a.get("latest")),
                })
        out["history"] = sorted(history, key=lambda h: str(h["year"]))
        return out


def compare_with_local(local_status: str, iucn: dict[str, Any] | None) -> dict[str, Any]:
    """
    Compare the curated status against IUCN's.

    Divergence is information, not an error. The local database may be ahead of
    a cached fetch, or IUCN may have reassessed since the database was compiled.
    Either way a human decides, not this function.
    """
    if not iucn or not iucn.get("category"):
        return {"status": "no_data", "message": "No IUCN record cached."}

    remote = iucn["category"]
    if remote == local_status:
        return {
            "status": "match",
            "message": f"Local and IUCN agree: {remote}.",
            "iucn_category": remote,
        }
    return {
        "status": "divergent",
        "message": (
            f"Local database says {local_status}; IUCN's latest assessment "
            f"({iucn.get('year_published', 'year unknown')}) says {remote}. "
            "Review before changing the database — the local value may be "
            "correct and more recent than this cache."
        ),
        "iucn_category": remote,
        "local_category": local_status,
    }
