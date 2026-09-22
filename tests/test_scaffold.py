"""The versioned scaffold artefact and its strict loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from boundarybench.scaffold import (
    ScaffoldDigestError,
    ScaffoldFormatError,
    ScaffoldLeakageError,
    load_scaffold,
    scaffold_digest,
)
from tests.scaffolds import fixture_digest, scaffold_payload, write_scaffold


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_scaffold_digest_refuses_a_number_json_cannot_carry(value: float) -> None:
    """A pin over a value JSON cannot represent is a pin nothing can re-derive.

    Python spells these as bare ``NaN``/``Infinity`` tokens, which the loader's
    own parser refuses on the way back in. Hashing one would mint an identity
    for a document this build would never accept.
    """
    with pytest.raises(ValueError) as excinfo:
        scaffold_digest({**scaffold_payload(), "schema_version": value})

    assert "is not a finite number" in str(excinfo.value)


def test_a_loaded_scaffold_remembers_the_file_it_came_from(tmp_path: Path) -> None:
    """Which artefact was executed is provenance, and it travels on the object.

    A run is only reproducible if the scaffold it ran under can be pointed at.
    The digest says *what* was loaded; the path is the only field that says
    where the operator got it.
    """
    directory = tmp_path / "artefacts"
    directory.mkdir()
    path = write_scaffold(directory / "scaffold.json", scaffold_payload())

    scaffold = load_scaffold(path)

    assert scaffold.path == path
    assert scaffold.path.read_text(encoding="utf-8")


def test_loads_the_exact_prompt_and_action_schemas(tmp_path: Path) -> None:
    path = write_scaffold(tmp_path / "scaffold.json", scaffold_payload())

    scaffold = load_scaffold(path)

    assert scaffold.scaffold_id == "fixture_scaffold"
    assert scaffold.scaffold_version == "0.1.0-dev.1"
    assert scaffold.schema_version == 1
    assert scaffold.system_prompt == scaffold_payload()["system_prompt"]
    assert [action.name for action in scaffold.actions] == [
        "read_records",
        "ask_user",
        "complete_case",
    ]
    terminal = scaffold.terminal_action
    assert terminal.name == "complete_case"
    assert terminal.terminal is True
    assert [parameter.name for parameter in terminal.parameters] == [
        "disposition",
        "primary_reason_code",
        "secondary_reason_codes",
        "evidence_refs",
    ]
    assert terminal.parameters[3].type == "array_of_string"
    assert terminal.parameters[3].required is True


def test_stale_pinned_digest_is_refused(tmp_path: Path) -> None:
    """A prompt edited without re-pinning must fail closed, not load quietly."""
    payload = scaffold_payload()
    stale = fixture_digest(payload)
    payload["system_prompt"] = payload["system_prompt"] + " Be brief."
    path = write_scaffold(tmp_path / "scaffold.json", payload, digest=stale)

    with pytest.raises(ScaffoldDigestError) as excinfo:
        load_scaffold(path)

    assert excinfo.value.declared == stale
    assert excinfo.value.computed == fixture_digest(payload)
    assert "re-pin" in str(excinfo.value)


def test_digest_covers_everything_except_the_pin_itself(tmp_path: Path) -> None:
    """Non-ASCII prose and key order must not move the digest; content must."""
    payload = scaffold_payload()
    payload["system_prompt"] = "Vérifiez la politique — puis décidez."
    reordered = dict(reversed(list(payload.items())))
    path = write_scaffold(tmp_path / "scaffold.json", reordered)

    scaffold = load_scaffold(path)

    assert scaffold.content_digest == fixture_digest(payload)


def test_the_loaded_scaffold_reprojects_to_the_document_it_was_read_from(
    tmp_path: Path,
) -> None:
    """``as_dict`` is what a run manifest and a turn request are built from.

    If the projection dropped or renamed anything, the digest recorded against a
    run would not be the digest of what the adapter was actually shown.
    """
    payload = scaffold_payload()
    path = write_scaffold(tmp_path / "scaffold.json", payload)

    scaffold = load_scaffold(path)
    projected = scaffold.as_dict()

    assert projected == {**payload, "content_digest": scaffold.content_digest}
    assert fixture_digest(projected) == scaffold.content_digest


def test_a_scaffold_that_is_not_readable_utf8_text_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "scaffold.json"
    path.write_bytes(b'{"scaffold_id": "\xff\xfe"}')

    with pytest.raises(ScaffoldFormatError) as excinfo:
        load_scaffold(path)

    assert "not valid UTF-8" in str(excinfo.value)


def test_a_scaffold_path_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    """A directory, a missing file: an OS-level failure names the artefact."""
    with pytest.raises(ScaffoldFormatError) as excinfo:
        load_scaffold(tmp_path)

    assert "cannot read scaffold" in str(excinfo.value)


def _without(key: str) -> dict[str, object]:
    payload = scaffold_payload()
    payload.pop(key)
    return payload


def _with(key: str, value: object) -> dict[str, object]:
    payload = scaffold_payload()
    payload[key] = value
    return payload


def _action(index: int, **overrides: object) -> dict[str, object]:
    payload = scaffold_payload()
    actions = payload["actions"]
    assert isinstance(actions, list)
    actions[index] = {**actions[index], **overrides}
    return payload


def _parameter(**overrides: object) -> dict[str, object]:
    payload = scaffold_payload()
    actions = payload["actions"]
    assert isinstance(actions, list)
    parameters = list(actions[1]["parameters"])
    parameters[0] = {**parameters[0], **overrides}
    actions[1] = {**actions[1], "parameters": parameters}
    return payload


def _duplicate_parameter() -> dict[str, object]:
    """An action that declares the same argument twice cannot be dispatched."""
    payload = scaffold_payload()
    actions = payload["actions"]
    assert isinstance(actions, list)
    parameters = list(actions[1]["parameters"])
    actions[1] = {**actions[1], "parameters": [*parameters, dict(parameters[0])]}
    return payload


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_without("system_prompt"), "missing required field(s) ['system_prompt']"),
        (_with("extra", "x"), "unknown field(s) ['extra']"),
        # bool is an int subclass, so `schema_version: true` must not read as 1.
        (_with("schema_version", True), "unsupported schema_version True"),
        (_with("schema_version", 2), "unsupported schema_version 2"),
        (_with("scaffold_id", "Not An Id"), "scaffold_id 'Not An Id'"),
        (_with("scaffold_version", "0.1.0"), "explicit prerelease"),
        (_with("system_prompt", 7), "'system_prompt' must be a non-empty string"),
        (_with("system_prompt", "   "), "'system_prompt' must be a non-empty string"),
        (_with("content_digest", "abc"), "'content_digest' must be a lowercase"),
        (_with("actions", {}), "'actions' must be a list"),
        (_with("actions", []), "'actions' must not be empty"),
        (_action(0, terminal="no"), "'terminal' must be true or false"),
        (_action(0, name=None), "'name' must be a non-empty string"),
        (_action(0, parameters={}), "'parameters' must be a list"),
        (_action(0, extra=1), "unknown field(s) ['extra']"),
        (_action(2, terminal=False), "exactly one terminal action"),
        (_action(0, terminal=True), "exactly one terminal action"),
        (_action(0, name="ask_user"), "duplicate action name 'ask_user'"),
        (_parameter(type="widget"), "unknown parameter type 'widget'"),
        (_parameter(required="yes"), "'required' must be true or false"),
        (_parameter(name=""), "'name' must be a non-empty string"),
        (_with("actions", ["read_records"]), "actions[0] must be a JSON object"),
        (_action(1, parameters=["fact"]), "parameters[0] must be a JSON object"),
        (_duplicate_parameter(), "duplicate parameter name 'fact'"),
    ],
)
def test_malformed_scaffolds_are_refused(
    tmp_path: Path, payload: dict[str, object], expected: str
) -> None:
    path = write_scaffold(tmp_path / "scaffold.json", payload)

    with pytest.raises(ScaffoldFormatError) as excinfo:
        load_scaffold(path)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"scaffold_id": "a", "scaffold_id": "b"}', "duplicate JSON key 'scaffold_id'"),
        ('{"system_prompt": NaN}', "not a finite JSON value"),
        ('{"system_prompt": Infinity}', "not a finite JSON value"),
        ("[]", "must be a JSON object"),
        ("{", "is not valid JSON"),
    ],
)
def test_unparseable_scaffold_documents_are_refused(
    tmp_path: Path, text: str, expected: str
) -> None:
    path = tmp_path / "scaffold.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ScaffoldFormatError) as excinfo:
        load_scaffold(path)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_with("system_prompt", "Answer S0_P1 as recorded."), "'S0_P1'"),
        (_with("system_prompt", "Policy arm P0 always applies."), "'P0'"),
        (_with("system_prompt", "State S1 is the live one."), "'S1'"),
        (
            _with("system_prompt", "See variant digest " + "a" * 64 + "."),
            "looks like a pinned digest",
        ),
        (
            _action(2, description="Give the expected_disposition."),
            "expected_disposition",
        ),
        (_action(2, description="Cite the required_evidence_ids."), "required_evidence"),
        (_action(2, description="Read the answer key first."), "answer key"),
        (_parameter(name="expected_primary_reason"), "expected_primary_reason"),
    ],
)
def test_scaffold_may_not_leak_the_answer_key(
    tmp_path: Path, payload: dict[str, object], expected: str
) -> None:
    """The scaffold is what the agent sees; grading ground truth is not."""
    path = write_scaffold(tmp_path / "scaffold.json", payload)

    with pytest.raises(ScaffoldLeakageError) as excinfo:
        load_scaffold(path)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    "payload",
    [
        _with("answer_key", "ACT"),
        _action(2, ground_truth="ACT"),
    ],
)
def test_a_grading_side_field_is_diagnosed_as_leakage_not_as_a_typo(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """A field named for the answer key is a leak, whatever else is wrong with it.

    Both diagnoses reject the document, but they ask for different responses:
    an unknown field is a typo to fix, while a field named ``answer_key`` means
    someone tried to put the grading contract in the agent-facing artefact, and
    every run made with it has to be discarded. The stricter reading has to win.
    """
    path = write_scaffold(tmp_path / "scaffold.json", payload)

    with pytest.raises(ScaffoldLeakageError) as excinfo:
        load_scaffold(path)

    assert "belongs to the grading contract" in str(excinfo.value)
