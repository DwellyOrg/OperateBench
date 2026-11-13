"""Pack input boundary; limits belong to the existing producer formats."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from operatebench.sdk import OperationPackError


class InputError(OperationPackError, ValueError):
    """Named caller-input refusal, distinct from internal execution errors."""


def read_bytes(path: str | Path, limit: int) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise InputError("input must be a regular file")
            with os.fdopen(fd, "rb") as stream:
                fd = -1  # stream owns the descriptor, including on read failure
                raw = stream.read(limit + 1)
        finally:
            if fd != -1:
                os.close(fd)
    except OSError as exc:
        raise InputError("input file read refused") from exc
    if len(raw) > limit:
        raise InputError("oversize input")
    return raw


def unique(pairs: Any) -> Any:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError("duplicate key")
        result[key] = value
    return result


def read_json(path: str | Path, limit: int) -> Any:
    raw = read_bytes(path, limit)
    try:
        return json.loads(raw, object_pairs_hook=unique)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InputError("invalid input JSON") from exc
