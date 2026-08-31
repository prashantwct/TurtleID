"""
Durable contribution storage in a GitHub repository.

Photographs and their records are committed through the GitHub Contents API,
one file per submission, so a hosted deployment stops losing contributions on
restart and review happens through the same pull requests and history as the
rest of the project.

    CHELONID_GITHUB_TOKEN    required; a fine-grained token with Contents:write
                             on the target repository and nothing else
    CHELONID_GITHUB_REPO     required; "owner/repo"
    CHELONID_GITHUB_BRANCH   optional, defaults to main
    CHELONID_GITHUB_PATH     optional, defaults to submissions/

READ THIS BEFORE POINTING IT AT A PUBLIC REPOSITORY
---------------------------------------------------
Everything written here is as visible as the repository is, and a commit cannot
be taken back: deleting a file later removes it from the working tree, not from
the history, and anything already cloned or mirrored is gone for good. On a
public repository that means every contributed photograph, the contributor's
name, and the state, are published permanently the moment they are submitted.

The scrubbing this project does elsewhere still runs — EXIF GPS is stripped and
typed coordinates are redacted before anything reaches this module — but it
cannot help with locality that is *in the frame*: a recognisable bank, a
signboard, a vehicle number plate. For a Schedule I species that is the exact
disclosure the rest of the codebase is built to avoid.

None of which makes publishing wrong; it makes it a decision. It is the
maintainer's to take deliberately, with contributors told plainly what happens
to what they send, which is why `app.py` refuses to submit through this backend
without an explicit acknowledgement. A private repository gives the identical
workflow with none of the exposure, and is the safer default where either would
do.

WHAT IS STORED
--------------
    <path>images/<species_id>_<digest>.jpg     the scrubbed photograph
    <path>records/<record_id>.json             its metadata

One file per submission, so two people contributing at the same moment write to
different paths and never collide. Commit messages carry the record id and the
species only — never the contributor, the state, or any note.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import env_value

logger = logging.getLogger(__name__)

TOKEN_ENV = "CHELONID_GITHUB_TOKEN"
REPO_ENV = "CHELONID_GITHUB_REPO"
BRANCH_ENV = "CHELONID_GITHUB_BRANCH"
PATH_ENV = "CHELONID_GITHUB_PATH"

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
DEFAULT_PATH = "submissions/"
DEFAULT_BRANCH = "main"

IMAGE_PREFIX = "images/"
RECORD_PREFIX = "records/"

TIMEOUT_SECONDS = 30

# The visibility probe runs while a page renders, so it fails fast. A commit
# can afford to wait; a status line cannot.
VISIBILITY_TIMEOUT_SECONDS = 5


class GitHubStorageError(Exception):
    """A commit that was expected to happen did not."""


@dataclass(frozen=True)
class Settings:
    token: str
    repo: str
    branch: str
    path: str

    def path_for(self, name: str) -> str:
        return f"{self.path}{name}"

    @property
    def public_description(self) -> str:
        """Safe to render in the UI. Contains no credential."""
        return f"{self.repo} ({self.branch}:{self.path})"


def settings() -> Settings | None:
    """Configuration, or None when this backend is switched off.

    A repository named without a token is a misconfiguration rather than a
    choice to store locally, so it is reported instead of silently ignored —
    that mistake otherwise looks identical to working correctly, right up until
    the first restart loses everything.
    """
    repo = env_value(REPO_ENV)
    if not repo:
        return None

    if repo.count("/") != 1 or not all(part.strip() for part in repo.split("/")):
        raise GitHubStorageError(
            f"{REPO_ENV} must be \"owner/repo\", not {repo!r}."
        )

    token = env_value(TOKEN_ENV)
    if not token:
        raise GitHubStorageError(
            f"{REPO_ENV} is set but {TOKEN_ENV} is not. Set a fine-grained token "
            f"with Contents:write on {repo}, or unset {REPO_ENV} to fall back to "
            f"local-only storage."
        )

    path = (env_value(PATH_ENV) or DEFAULT_PATH).lstrip("/")
    if path and not path.endswith("/"):
        path += "/"

    return Settings(
        token=token,
        repo=repo,
        branch=env_value(BRANCH_ENV) or DEFAULT_BRANCH,
        path=path,
    )


def configured() -> bool:
    """Whether submissions will be committed. Never raises."""
    try:
        return settings() is not None
    except GitHubStorageError as exc:
        logger.error("GitHub contribution storage is misconfigured: %s", exc)
        return False


def _commit(path: str, data: bytes, message: str, config: Settings) -> str:
    """PUT one file through the Contents API. Returns the path committed."""
    url = f"{API_ROOT}/repos/{config.repo}/contents/{path}"
    body = json.dumps({
        "message": message,
        "content": base64.b64encode(data).decode("ascii"),
        "branch": config.branch,
    }).encode("utf-8")

    request = urllib.request.Request(url, data=body, method="PUT")
    request.add_header("Authorization", f"Bearer {config.token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", API_VERSION)
    request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status not in (200, 201):
                raise GitHubStorageError(
                    f"GitHub returned {response.status} for {path}."
                )
    except urllib.error.HTTPError as exc:
        detail = _explain(exc, config, path)
        raise GitHubStorageError(detail) from exc
    except urllib.error.URLError as exc:
        raise GitHubStorageError(
            f"Could not reach GitHub to store {path}: {exc.reason}"
        ) from exc
    return path


def _explain(exc: urllib.error.HTTPError, config: Settings, path: str) -> str:
    """Turn an API error into something a maintainer can act on.

    The response body is read but never echoed verbatim: it can repeat request
    headers, and one of those is the token.
    """
    if exc.code in (401, 403):
        return (
            f"GitHub refused the token ({exc.code}). It needs Contents:write on "
            f"{config.repo}; a fine-grained token also has to list that "
            f"repository explicitly, and an expired one fails the same way."
        )
    if exc.code == 404:
        return (
            f"GitHub reports {config.repo} as not found (404). Either the name "
            f"is wrong or the token cannot see it — a token with no access to a "
            f"private repository gets 404 rather than 403."
        )
    if exc.code == 409:
        return (
            f"Branch {config.branch!r} does not exist in {config.repo}, or the "
            f"repository is empty."
        )
    if exc.code == 422:
        return (
            f"{path} already exists in {config.repo}. This photograph has been "
            f"contributed before."
        )
    if exc.code == 429:
        return "GitHub is rate-limiting this token. Try again shortly."
    return f"GitHub returned {exc.code} storing {path}."


def put_image(filename: str, data: bytes, *, config: Settings | None = None) -> str:
    """Commit a photograph. Returns its path. Raises on failure."""
    config = config or settings()
    if config is None:
        raise GitHubStorageError("GitHub contribution storage is not configured.")
    path = config.path_for(f"{IMAGE_PREFIX}{filename}")
    # The filename already carries the species and a content digest. Nothing
    # about the contributor or the locality belongs in a commit message.
    return _commit(path, data, f"Contribution: {filename}", config)


def put_record(record: dict[str, Any], *, config: Settings | None = None) -> str:
    """Commit a contribution record. Returns its path."""
    config = config or settings()
    if config is None:
        raise GitHubStorageError("GitHub contribution storage is not configured.")
    path = config.path_for(f"{RECORD_PREFIX}{record['id']}.json")
    body = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    return _commit(path, body, f"Contribution record: {record['id']}", config)


_VISIBILITY_CACHE: dict[str, bool | None] = {}


def is_public(refresh: bool = False) -> bool | None:
    """Whether the target repository is publicly readable.

    Asked unauthenticated on purpose: a repository that answers a request
    carrying no token is one anybody can read. None means the question could
    not be answered, which the UI must treat as "assume public" rather than as
    reassurance.

    Cached per repository, because `describe()` feeds a sidebar that Streamlit
    re-renders on every interaction and this is a network call.
    """
    try:
        config = settings()
    except GitHubStorageError:
        return None
    if config is None:
        return None
    if not refresh and config.repo in _VISIBILITY_CACHE:
        return _VISIBILITY_CACHE[config.repo]

    request = urllib.request.Request(f"{API_ROOT}/repos/{config.repo}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", API_VERSION)
    verdict: bool | None
    try:
        with urllib.request.urlopen(request, timeout=VISIBILITY_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
            verdict = not payload.get("private", True)
    except urllib.error.HTTPError as exc:
        # Invisible to an anonymous request, therefore not public.
        verdict = False if exc.code == 404 else None
    except (urllib.error.URLError, ValueError, OSError):
        verdict = None

    # An unanswered question is not cached: it is usually a transient network
    # failure, and "unknown" is the answer that makes the UI shout loudest.
    if verdict is not None:
        _VISIBILITY_CACHE[config.repo] = verdict
    return verdict


def describe() -> str:
    """One line for the UI. Never raises, never prints a credential."""
    try:
        config = settings()
    except GitHubStorageError as exc:
        return f"misconfigured — {exc}"
    if config is None:
        return "not configured"
    visibility = {True: "PUBLIC", False: "private", None: "visibility unknown"}[is_public()]
    return f"{config.public_description} — {visibility}"
