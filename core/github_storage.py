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


# Prefixes GitHub issues today. An unrecognised one is reported, never refused:
# the list changes, and refusing a valid token would be the worse failure.
TOKEN_PREFIXES = ("ghp_", "github_pat_", "gho_", "ghu_", "ghs_", "ghr_")


def normalise_token(raw: str) -> str:
    """Strip what a copy-and-paste adds around a token.

    `config.env_value` removes surrounding quotes from a value in `.env` but
    not from an environment variable, and a deployment's secrets arrive as
    environment variables. A token carrying its own quote marks is sent
    verbatim and comes back 401 — indistinguishable, from the error alone,
    from a revoked one.
    """
    token = raw.strip()
    for quote in ('"', "'"):
        if len(token) >= 2 and token.startswith(quote) and token.endswith(quote):
            token = token[1:-1].strip()
    return token


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
    token = normalise_token(token)
    if not token:
        raise GitHubStorageError(
            f"{TOKEN_ENV} is set but contains nothing usable once quotes and "
            f"whitespace are removed."
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
    if exc.code == 401:
        return (
            f"GitHub does not recognise the token (401 Bad credentials). This is "
            f"the token value itself, not its permissions: it has been revoked, "
            f"has expired, or was copied incompletely. Issue a new one and paste "
            f"the whole value — GitHub shows it once, at creation. "
            f"Run `python -m scripts.check_github_storage` to confirm."
        )
    if exc.code == 403:
        return (
            f"GitHub refused the token (403 Forbidden). It authenticates, but it "
            f"lacks permission: it needs Contents:write on {config.repo}, and a "
            f"fine-grained token has to list that repository explicitly."
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


def check() -> list[tuple[str, bool | None, str]]:
    """Diagnose the configuration without writing anything.

    Returns (label, ok, detail) rows. `ok` is None where a question could not
    be answered. Nothing is committed: a write test would leave a stray file in
    the repository, and the three questions below — is the token recognised, can
    it see the repository, can it push — separate every failure this backend
    produces.
    """
    rows: list[tuple[str, bool | None, str]] = []

    try:
        config = settings()
    except GitHubStorageError as exc:
        return [("Configuration", False, str(exc))]
    if config is None:
        return [("Configuration", False,
                 f"{REPO_ENV} is not set, so contributions stay on local disk.")]

    rows.append(("Configuration", True,
                 f"{config.repo}, branch {config.branch}, path {config.path}"))

    prefix = next((p for p in TOKEN_PREFIXES if config.token.startswith(p)), None)
    rows.append((
        "Token shape",
        prefix is not None,
        f"{len(config.token)} characters, recognised prefix {prefix!r}" if prefix
        else f"{len(config.token)} characters, no recognised GitHub prefix — "
             f"check the whole value was pasted, and that no quotes crept in",
    ))

    identity = _get("/user", config)
    if identity is None:
        rows.append(("Token authenticates", False,
                     "GitHub returned 401. The token is revoked, expired, or "
                     "incomplete. Issue a new one — permissions are not the issue."))
        return rows
    rows.append(("Token authenticates", True,
                 f"acting as {identity.get('login', 'unknown')}"))

    repo = _get(f"/repos/{config.repo}", config)
    if repo is None:
        rows.append(("Repository visible", False,
                     f"{config.repo} is not visible to this token. Either the "
                     f"name is wrong, or a fine-grained token does not list it "
                     f"under Repository access."))
        return rows
    rows.append(("Repository visible", True,
                 "public" if not repo.get("private", True) else "private"))

    can_push = bool(repo.get("permissions", {}).get("push"))
    rows.append((
        "Write permission", can_push,
        "Contents:write present" if can_push else
        "read-only. A fine-grained token needs Repository permissions -> "
        "Contents: Read and write; a classic token needs the 'repo' scope.",
    ))

    branch = _get(f"/repos/{config.repo}/branches/{config.branch}", config)
    rows.append((
        "Branch exists", branch is not None,
        f"{config.branch} found" if branch is not None
        else f"{config.branch!r} not found in {config.repo}",
    ))
    return rows


def _get(path: str, config: Settings) -> dict | None:
    """One authenticated GET. None on any failure; the caller says what that means."""
    request = urllib.request.Request(f"{API_ROOT}{path}")
    request.add_header("Authorization", f"Bearer {config.token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", API_VERSION)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, OSError):
        return None


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
