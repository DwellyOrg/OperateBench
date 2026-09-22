# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Small bounded descriptor reads for this isolated operation pack."""

from __future__ import annotations

import os


def read_bounded(descriptor: int, maximum_bytes: int) -> bytes:
    """Read through short reads, stopping at EOF or ``maximum_bytes``."""
    chunks: list[bytes] = []
    remaining = maximum_bytes
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
