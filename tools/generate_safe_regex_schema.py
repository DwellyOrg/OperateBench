"""Generate the published Card v1 authored safe-regex lexical grammar.

The generated regular expression is intentionally built from named grammar
fragments.  Do not hand-edit its serialized copy in card-defs-v1.schema.json.
"""

from __future__ import annotations

import json
from pathlib import Path

PATTERN_MAX_CHARACTERS = 256
PATTERN_MAX_UTF8_BYTES = 256
INPUT_MAX_CHARACTERS = 4096
REPEAT_MAXIMUM = 4096

_PRINTABLE = tuple(chr(codepoint) for codepoint in range(0x20, 0x7F))
_OUTSIDE_SPECIAL = frozenset(r"^$[\()|.*+?{}]")
_CLASS_SYNTAX = frozenset("[]\\-")


def _hex_escape(character: str) -> str:
    return rf"\x{ord(character):02x}"


def _character_class(characters: tuple[str, ...]) -> str:
    """Return an ASCII-only regex class without context-sensitive punctuation."""
    return "[" + "".join(_hex_escape(character) for character in characters) + "]"


def _alternation(parts: list[str] | tuple[str, ...]) -> str:
    return "(?:" + "|".join(parts) + ")"


def _approved_escape() -> str:
    escaped_non_alphanumeric = tuple(
        character for character in _PRINTABLE if not character.isalnum()
    )
    return rf"\\(?:[dDsSwW]|{_character_class(escaped_non_alphanumeric)})"


def _published_ranges() -> str:
    """Ranges retained by the conservative published authored grammar."""
    return _alternation(
        [
            r"\x30\x2d\x39",  # 0-9
            r"\x41\x2d\x5a",  # A-Z
            r"\x61\x2d\x7a",  # a-z
            r"\x20\x2d\x7e",  # full printable ASCII
        ]
    )


def _character_class_atom() -> str:
    """A conservative, auditable subset of runtime-valid character classes.

    Literal class atoms are complete. The portable 0-9, A-Z, a-z, and full
    printable-ASCII ranges are supported. Other raw ranges and escaped range
    endpoints remain runtime compatibility forms rather than authored forms.
    """
    plain = tuple(character for character in _PRINTABLE if character not in _CLASS_SYNTAX)
    first_plain = tuple(character for character in plain if character != "^")
    escape = _approved_escape()
    first_item = _alternation(
        [escape, _character_class(first_plain), _published_ranges()]
    )
    item = _alternation([escape, _character_class(plain), _published_ranges()])
    # A raw caret is special only as the first byte after '['. Once a leading
    # hyphen or the negation marker has been consumed it is an ordinary atom.
    hyphens_only = r"\x2d(?:\x2d)?"
    unnegated_body = _alternation(
        [
            hyphens_only,
            rf"{first_item}(?:{item})*(?:\x2d)?",
            rf"\x2d{item}(?:{item})*(?:\x2d)?",
        ]
    )
    negated_body = _alternation(
        [
            hyphens_only,
            rf"{item}(?:{item})*(?:\x2d)?",
            rf"\x2d{item}(?:{item})*(?:\x2d)?",
        ]
    )
    return rf"\[(?:\^{negated_body}|{unnegated_body})\]"


def _atom() -> str:
    literal = _character_class(
        tuple(character for character in _PRINTABLE if character not in _OUTSIDE_SPECIAL)
    )
    return _alternation([literal, _approved_escape(), _character_class_atom()])


def _bounded_number(maximum: int = REPEAT_MAXIMUM) -> str:
    if maximum != 4096:
        raise ValueError("the compact decimal grammar is pinned to 4096")
    return r"(?:0|[1-9][0-9]{0,2}|[1-3][0-9]{3}|40(?:[0-8][0-9]|9[0-6]))"


def _bounded_number_below_maximum() -> str:
    return r"(?:0|[1-9][0-9]{0,2}|[1-3][0-9]{3}|40(?:[0-8][0-9]|9[0-5]))"


def _canonical_number_of_width(width: int) -> str:
    if width == 1:
        return "[0-9]"
    return rf"[1-9][0-9]{{{width - 1}}}"


def _strictly_increasing_bounded_pair() -> str:
    """A sound, representative subset of min/max pairs.

    Different decimal widths are ordered without arithmetic correlation. Same-
    width pairs are published for one-digit bounds; arbitrary minima are also
    supported at the 4096 ceiling and arbitrary maxima from zero.
    """
    positive_number = r"(?:[1-9][0-9]{0,2}|[1-3][0-9]{3}|40(?:[0-8][0-9]|9[0-6]))"
    below_maximum = _bounded_number_below_maximum()
    branches = [rf"0,{positive_number}", rf"{below_maximum},4096"]
    for minimum_width in range(1, 5):
        minimum = _canonical_number_of_width(minimum_width)
        for maximum_width in range(minimum_width + 1, 5):
            maximum = rf"[1-9][0-9]{{{maximum_width - 1}}}"
            branches.append(minimum + "," + maximum)
    for low in range(0, 9):
        highs = _character_class(tuple(str(high) for high in range(low + 1, 10)))
        branches.append(f"{low},{highs}")
    number = _bounded_number()
    return rf"(?={number},{number}\}}){_alternation(branches)}"


def _fixed_quantifier() -> str:
    number = _bounded_number()
    # {4096,} has equal effective bounds in the runtime parser.
    return _alternation([rf"\{{{number}\}}", r"\{4096,\}"])


def _variable_quantifier() -> str:
    open_range = rf"\{{{_bounded_number_below_maximum()},\}}"
    bounded_range = rf"\{{{_strictly_increasing_bounded_pair()}\}}"
    return _alternation([r"[?*+]", open_range, bounded_range])


def safe_regex_lexical_pattern() -> str:
    """Return the anchored regular language accepted by the published schema.

    Units are split into unquantified atoms (U), fixed-width quantified atoms
    (F), and variable-width quantified atoms (V). The two alternatives encode:
    no adjacent F/V units, and zero or exactly one V unit respectively.
    """
    atom = _atom()
    fixed = atom + _fixed_quantifier()
    variable = atom + _variable_quantifier()
    no_variable = rf"(?:{atom})*(?:{fixed}(?:{atom})+)*(?:{fixed})?"
    before_variable = rf"(?:{atom})*(?:{fixed}(?:{atom})+)*"
    after_variable = rf"(?:(?:{atom})+{fixed})*(?:{atom})*"
    body = _alternation([no_variable, before_variable + variable + after_variable])
    # The final negative lookahead makes '$' an absolute end assertion in Python
    # as well as ECMAScript, rather than accepting before a final newline.
    return rf"^\^?{body}\$?$(?![\s\S])"


def authored_pattern_schema() -> dict[str, object]:
    return {
        "description": (
            "Card v1 authored safe-regex lexical grammar generated by "
            "tools/generate_safe_regex_schema.py; see the registry's published "
            "authored-schema profile for its conservative portable forms."
        ),
        "type": "string",
        "maxLength": PATTERN_MAX_CHARACTERS,
        "pattern": safe_regex_lexical_pattern(),
    }


def _write_schema(repo_root: Path) -> None:
    docs_path = repo_root / "docs/schemas/card-defs-v1.schema.json"
    schema = json.loads(docs_path.read_text(encoding="utf-8"))
    schema["$defs"]["safeRegexPattern"] = authored_pattern_schema()
    reference = {"$ref": "#/$defs/safeRegexPattern"}
    field_properties = schema["$defs"]["fieldShape"]["oneOf"][0]["properties"]
    field_pattern = field_properties["pattern"]
    subset_pattern = schema["$defs"]["jsonSchemaSubset"]["oneOf"][3]["properties"][
        "pattern"
    ]
    field_pattern["oneOf"][0] = reference
    subset_pattern["oneOf"][0] = reference
    serialized = json.dumps(schema, indent=2, ensure_ascii=True) + "\n"
    docs_path.write_text(serialized, encoding="utf-8")
    packaged_path = (
        repo_root / "src/operatebench/resources/cards/card-defs-v1.schema.json"
    )
    packaged_path.write_text(serialized, encoding="utf-8")
    registry_text = (repo_root / "docs/schemas/card-registry-v1.json").read_text(
        encoding="utf-8"
    )
    (repo_root / "src/operatebench/resources/cards/card-registry-v1.json").write_text(
        registry_text, encoding="utf-8"
    )


if __name__ == "__main__":
    _write_schema(Path(__file__).resolve().parents[1])
