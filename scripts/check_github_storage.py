#!/usr/bin/env python3
"""
Check GitHub contribution storage without submitting a photograph.

    python -m scripts.check_github_storage

A failed submission tells a contributor what went wrong, which is right for
them and slow for whoever has to fix it: change a setting, reboot the app,
upload a photograph, read the message, repeat. This asks the same questions
directly.

Nothing is written. A write test would leave a stray file in the repository,
and the questions below separate every failure this backend produces: is the
token one GitHub recognises, can it see the repository, may it push, does the
branch exist.

The token itself is never printed — only its length and which prefix it
carries, which is what distinguishes a truncated paste from a revoked token.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import github_storage  # noqa: E402

MARK = {True: "PASS", False: "FAIL", None: "????"}


def main() -> int:
    rows = github_storage.check()
    width = max(len(label) for label, _, _ in rows)

    print()
    for label, ok, detail in rows:
        print(f"  {MARK[ok]}  {label:<{width}}  {detail}")
    print()

    if all(ok for _, ok, _ in rows):
        print("Contributions will be committed. Reboot the app if you have just "
              "changed a secret — they are read at startup.")
        return 0

    print("Fix the first FAIL above, then run this again. After changing a "
          "secret, reboot the app: secrets are read once, at startup.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
