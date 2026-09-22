"""YAML entry point for construct cards and suite manifests.

Stock PyYAML resolves a duplicate mapping key by silently keeping the last one.
In a benchmark artefact that is a correctness hazard rather than a convenience,
so the loader below refuses duplicates at any nesting depth while keeping
``SafeLoader``'s safe construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from boundarybench.schema import ConstructCard, SchemaError, ensure_json_safe


class DuplicateKeyError(SchemaError):
    """Raised when a YAML mapping declares the same key twice."""


class _StrictSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that rejects duplicate mapping keys."""


def _construct_mapping(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> Any:
    # A generator, like PyYAML's own ``construct_yaml_map``: the empty dict is
    # registered with the loader before it is populated, so a self-referential
    # alias resolves to this same (still-mutating) object instead of tripping
    # PyYAML's "unconstructable recursive node" guard. The resulting cycle is
    # then rejected cleanly downstream by ``ensure_json_safe``.
    mapping: dict[Any, Any] = {}
    yield mapping
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            mark = key_node.start_mark
            raise DuplicateKeyError(
                f"mapping key at line {mark.line + 1}, column {mark.column + 1} "
                f"is not a usable key: {exc}"
            ) from exc
        if duplicate:
            mark = key_node.start_mark
            raise DuplicateKeyError(
                f"duplicate mapping key {key!r} at line {mark.line + 1}, "
                f"column {mark.column + 1}; a construct card must state each "
                "key exactly once"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def read_yaml_mapping(path: Path, *, what: str) -> dict[str, Any]:
    """Read a YAML file that must be a mapping, rejecting duplicate keys.

    Shared by construct cards and suite manifests: both are hand-authored,
    reviewed artefacts where a silently discarded duplicate key would change
    meaning without changing the review.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # Narrowly a decoding failure, not a read failure: the bytes arrived but
        # are not text. Caught here so a binary or mis-encoded artefact is a
        # named domain error like any other malformed input, rather than a
        # UnicodeDecodeError traceback out of a CI gate.
        raise SchemaError(
            f"{path} is not valid UTF-8: a {what} must be a UTF-8 text file "
            f"({exc.reason} at byte {exc.start})"
        ) from exc
    except OSError as exc:
        raise SchemaError(f"cannot read {what} {path}: {exc}") from exc
    try:
        raw = yaml.load(text, Loader=_StrictSafeLoader)
    except DuplicateKeyError as exc:
        raise SchemaError(f"{path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SchemaError(f"{path} is not valid YAML: {exc}") from exc
    except RecursionError as exc:
        # PyYAML composes nested collections recursively, so a deeply nested
        # document exhausts the stack. RecursionError is not a YAMLError and not
        # a ValueError, so without this it reached the operator as a traceback.
        raise SchemaError(
            f"{path} is nested too deeply to read; a {what} whose structure "
            "exceeds this build's parser depth is refused rather than parsed"
        ) from exc
    if raw is None:
        raise SchemaError(f"{path} is empty")
    if not isinstance(raw, dict):
        raise SchemaError(
            f"{path}: a {what} must be a YAML mapping, got {type(raw).__name__}"
        )
    # The whole document, not just the fields a later validator happens to look
    # at. A YAML escape such as ``"\uD800"`` parses into a string UTF-8 cannot
    # encode, and every artefact here ends up inside a canonical UTF-8 digest;
    # a cycle from an anchor and a document nested past the serialisable depth
    # are the same class of problem. All three are named here, at the door.
    ensure_json_safe(raw, f"{path}: {what}")
    return raw


def load_card(path: str | Path) -> ConstructCard:
    """Read and validate a construct card from a YAML file."""
    path = Path(path)
    return ConstructCard.from_dict(read_yaml_mapping(path, what="construct card"))
