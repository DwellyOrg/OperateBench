"""YAML entry point for construct cards and suite manifests.

Stock PyYAML resolves a duplicate mapping key by silently keeping the last one.
In a benchmark artefact that is a correctness hazard rather than a convenience,
so the loader below refuses duplicates at any nesting depth while keeping
``SafeLoader``'s safe construction.

It refuses aliases outright for the same reason, one class of hazard further on.
An anchor and an alias make the document a graph while the reviewed text still
reads as a tree, and a walk that recurses over that graph visits a shared
subgraph once per path *through* it rather than once. The JSON-safety walk this
reader calls below is such a walk, and so is anything that reads the structure
after it, so a few hundred bytes of ``[*previous, *previous]`` cost seconds to
minutes before a single field has been looked at. A benchmark contract is
hand-authored and reviewed, so it states its structure literally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent

from boundarybench.schema import ConstructCard, SchemaError, ensure_json_safe


class DuplicateKeyError(SchemaError):
    """Raised when a YAML mapping declares the same key twice."""


class AliasError(SchemaError):
    """Raised when a YAML document refers to an anchor instead of restating it."""


class _StrictSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that rejects duplicate mapping keys."""


def _refuse_aliases(text: str) -> None:
    """Refuse a document that refers to an anchor instead of restating it.

    A pass over the event stream, before anything is composed: an alias that has
    already expanded has already cost what this is here to prevent, so the
    refusal cannot be downstream of the expansion. The stream is flat and linear
    in the file, which is what makes the refusal cheap on an adversarial document
    and cheap on a well-formed one. Same shape as the manifest reader in
    :mod:`operatebench.sdk.company_scaffold`, which refuses the same feature.
    """
    for event in yaml.parse(text):
        if isinstance(event, AliasEvent):
            mark = event.start_mark
            raise AliasError(
                f"alias *{event.anchor} at line {mark.line + 1}, column "
                f"{mark.column + 1}: an alias makes the document a graph while "
                "the reviewed text still reads as a tree, and a shared subgraph "
                "costs every later traversal one visit per path through it. "
                "State the structure literally instead"
            )


def _construct_mapping(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> Any:
    # A generator, like PyYAML's own ``construct_yaml_map``: the empty dict is
    # registered with the loader before it is populated. Nothing can refer back
    # to it, since aliases are refused before the document is composed, but the
    # shape stays the same as PyYAML's own mapping constructor: this is a
    # duplicate-key check and not also a change of construction order.
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
        _refuse_aliases(text)
        raw = yaml.load(text, Loader=_StrictSafeLoader)
    except (AliasError, DuplicateKeyError) as exc:
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
    # encode, and every artefact here ends up inside a canonical UTF-8 digest; a
    # document nested past the serialisable depth is the same class of problem.
    # Both are named here, at the door. Cycles cannot arrive through this reader
    # at all any more — they need an alias — but this walk is the shared one and
    # still answers for them on behalf of callers that build values some other
    # way.
    ensure_json_safe(raw, f"{path}: {what}")
    return raw


def load_card(path: str | Path) -> ConstructCard:
    """Read and validate a construct card from a YAML file."""
    path = Path(path)
    return ConstructCard.from_dict(read_yaml_mapping(path, what="construct card"))
