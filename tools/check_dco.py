"""Check every commit in a git revision range for a DCO sign-off."""

from __future__ import annotations

import re
import subprocess
import sys

SIGN_OFF = re.compile(r"(?mi)^Signed-off-by: [^<>\n]+ <[^<>\s]+@[^<>\s]+>$")
REVISION_RANGE = re.compile(r"[0-9a-fA-F]{40}\.\.[0-9a-fA-F]{40}\Z")


def has_valid_sign_off(body: str) -> bool:
    """Use Git's end-of-message trailer block semantics for DCO sign-offs."""
    parsed = subprocess.run(
        ["git", "interpret-trailers", "--parse"],
        input=body,
        check=True,
        capture_output=True,
        text=True,
    )
    return any(SIGN_OFF.fullmatch(line) for line in parsed.stdout.splitlines())


def unsigned_commits(revision_range: str) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--format=%H%x00%B%x00", revision_range],
        check=True,
        capture_output=True,
        text=True,
    )
    records = result.stdout.split("\x00")
    return [
        commit
        for commit, body in zip(records[0::2], records[1::2], strict=False)
        if commit and not has_valid_sign_off(body)
    ]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or REVISION_RANGE.fullmatch(args[0]) is None:
        print(
            "usage: python tools/check_dco.py BASE..HEAD (exact SHA40..SHA40 required)",
            file=sys.stderr,
        )
        return 2
    missing = unsigned_commits(args[0])
    if missing:
        print("commits missing a valid Signed-off-by trailer:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        return 1
    print("DCO sign-off present on every commit in the range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
