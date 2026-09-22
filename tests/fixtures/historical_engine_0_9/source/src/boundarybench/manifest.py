"""Semantic manifests: a committed fingerprint for a shipped Cube.

A construct card is a versioned artefact. Its meaning — not its YAML formatting —
is pinned in a sibling ``<card>.manifest.json`` recording a canonical hash of the
parsed card plus the content digest of every compiled variant. Reflowing the file
is free; changing a pressure cue, a distractor value or a rule id is an explicit,
reviewed artefact revision that must update the manifest in the same commit.

Everything a caller can get wrong here is untrusted input: a hand-edited JSON
file, a card that has moved on, a manifest borrowed from another Cube. Each of
those is a named :class:`ManifestError`, never a raw ``KeyError``, ``TypeError``
or JSON traceback.

Versioning rule: ``manifest_version`` tracks the *manifest file format*, not the
card. It changes only when the recorded fields change shape, and this build
accepts exactly :data:`MANIFEST_VERSION`. A card revision changes the recorded
hashes, never the version.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boundarybench.compiler import Cube, compile_cube
from boundarybench.freezing import to_json
from boundarybench.jsonsafe import (
    JsonSafetyError,
    NestingDepthError,
    canonical_json_bytes,
    ensure_json_safe,
    ensure_raw_json_depth,
)
from boundarybench.loader import load_card
from boundarybench.schema import ConstructCard

#: The manifest file format this build reads and writes.
MANIFEST_VERSION = 1

#: A Cube always compiles to four cells plus two probes.
EXPECTED_VARIANTS = 6

_REQUIRED_FIELDS: tuple[str, ...] = (
    "manifest_version",
    "card",
    "cube_id",
    "card_fingerprint_sha256",
    "variants",
)
#: Prose for reviewers. Recorded, never interpreted.
_OPTIONAL_FIELDS: tuple[str, ...] = ("note",)
_VARIANT_FIELDS: tuple[str, ...] = ("variant_id", "content_digest")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Base class for every semantic-manifest failure."""


class ManifestFormatError(ManifestError):
    """The manifest file itself is unreadable or malformed."""


class ManifestMismatchError(ManifestError):
    """The manifest is well-formed but no longer describes its card."""


# -- the canonical fingerprint ----------------------------------------------


def card_fingerprint(card: ConstructCard) -> str:
    """Canonical SHA-256 over a card's detached, JSON-safe parsed content.

    ``to_json`` detaches the frozen structure into plain built-ins, so nothing
    reaches back into the compiled card. The canonical encoding, and the proof
    that the payload can survive it, both come from :mod:`boundarybench.jsonsafe`.
    A card containing a lone surrogate is an adversarial input and receives a
    named manifest failure before the encoder is asked for bytes.
    """
    try:
        return hashlib.sha256(
            canonical_json_bytes(to_json(card.raw), f"card {card.cube_id}")
        ).hexdigest()
    except JsonSafetyError as exc:
        raise ManifestFormatError(
            f"card {card.cube_id} cannot be fingerprinted: {exc}"
        ) from exc


# -- the parsed manifest -----------------------------------------------------


@dataclass(frozen=True)
class VariantRecord:
    """One pinned variant: its stable id and its content digest."""

    variant_id: str
    content_digest: str

    def as_dict(self) -> dict[str, str]:
        return {"variant_id": self.variant_id, "content_digest": self.content_digest}


@dataclass(frozen=True)
class SemanticManifest:
    manifest_version: int
    card: str
    cube_id: str
    card_fingerprint_sha256: str
    variants: tuple[VariantRecord, ...]
    note: str | None = None

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(record.variant_id for record in self.variants)

    @property
    def content_digests(self) -> tuple[str, ...]:
        return tuple(record.content_digest for record in self.variants)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "card": self.card,
            "cube_id": self.cube_id,
            "card_fingerprint_sha256": self.card_fingerprint_sha256,
            "variants": [record.as_dict() for record in self.variants],
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


# -- strict JSON loading -----------------------------------------------------


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Stock ``json`` keeps the last duplicate; a pinned artefact cannot.

    A manifest whose ``cube_id`` is stated twice would verify against whichever
    copy the parser happened to keep, which is exactly the silent ambiguity the
    fingerprint exists to remove.
    """
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise ManifestFormatError(
                f"duplicate JSON key {key!r}; a semantic manifest must state each "
                "key exactly once"
            )
        seen.add(key)
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise ManifestFormatError(
        f"{name} is not a finite JSON value; a semantic manifest carries only "
        "strings, integers and lists"
    )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # The bytes arrived but are not text. Narrowly a decoding failure, so it
        # is reported like any other malformed manifest rather than escaping as
        # a UnicodeDecodeError traceback.
        raise ManifestFormatError(
            f"{path} is not valid UTF-8: a semantic manifest must be a UTF-8 "
            f"text file ({exc.reason} at byte {exc.start})"
        ) from exc
    except OSError as exc:
        raise ManifestFormatError(f"cannot read semantic manifest {path}: {exc}") from exc
    try:
        ensure_raw_json_depth(text, str(path))
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except NestingDepthError as exc:
        raise ManifestFormatError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise ManifestFormatError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestFormatError(
            f"{path}: a semantic manifest must be a JSON object, got "
            f"{type(payload).__name__}"
        )
    try:
        ensure_json_safe(payload, f"{path}: semantic manifest")
    except JsonSafetyError as exc:
        raise ManifestFormatError(str(exc)) from exc
    return payload


def _text(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ManifestFormatError(f"{context}: {key!r} must be a non-empty string")
    return value


def _digest(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ManifestFormatError(
            f"{context}: {key!r} must be a lowercase 64-character SHA-256 hex "
            f"digest, got {value!r}"
        )
    return value


def _parse_variants(raw: Mapping[str, Any], context: str) -> tuple[VariantRecord, ...]:
    section = raw["variants"]
    if isinstance(section, (str, bytes)) or not isinstance(section, Sequence):
        raise ManifestFormatError(f"{context}: 'variants' must be a list")
    if len(section) != EXPECTED_VARIANTS:
        raise ManifestFormatError(
            f"{context}: 'variants' must record exactly {EXPECTED_VARIANTS} entries "
            f"(four cells and two probes), got {len(section)}"
        )
    records: list[VariantRecord] = []
    seen: set[str] = set()
    for index, entry in enumerate(section):
        entry_context = f"{context}: variants[{index}]"
        if not isinstance(entry, Mapping):
            raise ManifestFormatError(f"{entry_context} must be a JSON object")
        missing = [field for field in _VARIANT_FIELDS if field not in entry]
        if missing:
            raise ManifestFormatError(f"{entry_context} is missing {missing}")
        unknown = sorted(set(entry) - set(_VARIANT_FIELDS))
        if unknown:
            raise ManifestFormatError(f"{entry_context} has unknown fields {unknown}")
        variant_id = _text(entry, "variant_id", entry_context)
        if variant_id in seen:
            raise ManifestFormatError(
                f"{entry_context}: duplicate variant id {variant_id!r}"
            )
        seen.add(variant_id)
        records.append(
            VariantRecord(
                variant_id=variant_id,
                content_digest=_digest(entry, "content_digest", entry_context),
            )
        )
    return tuple(records)


def load_manifest(path: str | Path) -> SemanticManifest:
    """Read a committed semantic manifest under the public JSON depth limit."""
    return _load_manifest(Path(path))


def _load_manifest(path: str | Path) -> SemanticManifest:
    """Read and strictly validate a semantic manifest file."""
    path = Path(path)
    raw = _read_json_object(path)
    context = str(path)

    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ManifestFormatError(f"{context}: missing required field(s) {missing}")
    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS) - set(_OPTIONAL_FIELDS))
    if unknown:
        raise ManifestFormatError(
            f"{context}: unknown field(s) {unknown}; allowed: "
            f"{sorted(_REQUIRED_FIELDS + _OPTIONAL_FIELDS)}"
        )

    version = raw["manifest_version"]
    # bool is an int subclass and 1.0 == 1, so the type check has to be exact.
    if type(version) is not int or version != MANIFEST_VERSION:
        raise ManifestFormatError(
            f"{context}: unsupported manifest_version {version!r}; this build "
            f"reads {MANIFEST_VERSION}"
        )

    card = _text(raw, "card", context)
    if Path(card).name != card or card in {".", ".."}:
        raise ManifestFormatError(
            f"{context}: 'card' must be a bare filename beside the manifest, got {card!r}"
        )

    # Present-but-null is a statement, not an omission. Reading ``"note": null``
    # as "no note" would be the one manifest field quietly coerced rather than
    # refused, so presence is tested before the value is.
    note: str | None = None
    if "note" in raw:
        if not isinstance(raw["note"], str):
            raise ManifestFormatError(f"{context}: 'note' must be a string when present")
        note = raw["note"]

    return SemanticManifest(
        manifest_version=version,
        card=card,
        cube_id=_text(raw, "cube_id", context),
        card_fingerprint_sha256=_digest(raw, "card_fingerprint_sha256", context),
        variants=_parse_variants(raw, context),
        note=note,
    )


# -- verification ------------------------------------------------------------


@dataclass(frozen=True)
class ManifestVerification:
    """A card, its compiled Cube and the manifest they were checked against."""

    card_path: Path
    manifest_path: Path
    cube: Cube
    manifest: SemanticManifest
    card_fingerprint_sha256: str

    def as_dict(self) -> dict[str, Any]:
        card = self.cube.card
        return {
            "verified": True,
            "card": self.card_path.name,
            "manifest": self.manifest_path.name,
            "cube_id": self.cube.cube_id,
            "cube_type": card.cube_type,
            "fact_location": card.fact_location,
            "privacy_status": card.privacy_status,
            "manifest_version": self.manifest.manifest_version,
            "card_fingerprint_sha256": self.card_fingerprint_sha256,
            "variant_count": len(self.cube.variants),
            "variants": [
                {"variant_id": v.variant_id, "content_digest": v.content_digest}
                for v in self.cube.variants
            ],
        }


def verify_cube(
    cube: Cube, manifest: SemanticManifest, *, card_name: str, context: str
) -> str:
    """Check a compiled Cube against an already-parsed manifest.

    Returns the card fingerprint that matched, so a caller that needs it for a
    collection-level digest does not have to recompute it.
    """
    if manifest.card != card_name:
        raise ManifestMismatchError(
            f"{context}: manifest names card {manifest.card!r}, but was checked "
            f"against {card_name!r}"
        )
    if manifest.cube_id != cube.cube_id:
        raise ManifestMismatchError(
            f"{context}: manifest records cube_id {manifest.cube_id!r}, but the "
            f"card compiles to {cube.cube_id!r}"
        )

    fingerprint = card_fingerprint(cube.card)
    if manifest.card_fingerprint_sha256 != fingerprint:
        raise ManifestMismatchError(
            f"{context}: stale card fingerprint; manifest records "
            f"{manifest.card_fingerprint_sha256}, card is {fingerprint}. A change "
            "to the card's meaning requires an explicit, reviewed manifest revision."
        )

    compiled_ids = tuple(v.variant_id for v in cube.variants)
    if manifest.variant_ids != compiled_ids:
        raise ManifestMismatchError(
            f"{context}: manifest variant ids do not match the compiled Cube in "
            f"order; manifest has {list(manifest.variant_ids)}, Cube has "
            f"{list(compiled_ids)}"
        )

    for variant, record in zip(cube.variants, manifest.variants, strict=True):
        if variant.content_digest != record.content_digest:
            raise ManifestMismatchError(
                f"{context}: stale content digest for {variant.variant_id}; "
                f"manifest records {record.content_digest}, Cube compiles to "
                f"{variant.content_digest}"
            )
    return fingerprint


def verify_manifest(
    card_path: str | Path, manifest_path: str | Path
) -> ManifestVerification:
    """Compile a card and prove its semantic manifest still describes it."""
    card_path = Path(card_path)
    manifest_path = Path(manifest_path)
    cube = compile_cube(load_card(card_path))
    manifest = load_manifest(manifest_path)
    fingerprint = verify_cube(
        cube, manifest, card_name=card_path.name, context=str(manifest_path)
    )
    return ManifestVerification(
        card_path=card_path,
        manifest_path=manifest_path,
        cube=cube,
        manifest=manifest,
        card_fingerprint_sha256=fingerprint,
    )
