# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Shared validation predicates for operation-pack ownership metadata."""

from __future__ import annotations

from operatebench._contribution_structure import (
    canonical_owner_display_name_key,
    is_operation_identity,
    is_pack_version,
    is_safe_display_name,
    is_safe_owner_display_name,
)

__all__ = [
    "canonical_owner_display_name_key",
    "is_operation_identity",
    "is_pack_version",
    "is_safe_display_name",
    "is_safe_owner_display_name",
]
