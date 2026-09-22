from __future__ import annotations

import json

import pytest

from boundarybench.cli import main
from tests.conftest import EXAMPLE_CARD, minimal_card


@pytest.fixture()
def broken_card(tmp_path):
    import yaml

    card = minimal_card()
    card["disposition_table"]["S1_P0"] = "ACT"
    card["pressure_probes"]["reassurance"]["base_cell"] = "S0_P1"
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(card), encoding="utf-8")
    return path


@pytest.fixture()
def malformed_card(tmp_path):
    path = tmp_path / "malformed.yaml"
    path.write_text("schema_version: 1\ncube_id: nope\n", encoding="utf-8")
    return path


# -- validate --------------------------------------------------------------


def test_validate_accepts_the_example_card(capsys):
    assert main(["validate", str(EXAMPLE_CARD)]) == 0
    out = capsys.readouterr().out
    assert "lettings_maintenance_authority_v1" in out
    assert "OK" in out


def test_validate_surfaces_the_synthetic_disclaimer(capsys):
    main(["validate", str(EXAMPLE_CARD)])
    assert "not official Dwelly policy" in capsys.readouterr().out


def test_validate_returns_non_zero_on_a_contract_failure(capsys, broken_card):
    assert main(["validate", str(broken_card)]) == 1
    assert "S1_P0" in capsys.readouterr().err


def test_validate_returns_non_zero_on_a_malformed_card(capsys, malformed_card):
    assert main(["validate", str(malformed_card)]) == 1
    assert capsys.readouterr().err.strip()


def test_validate_returns_non_zero_on_a_missing_file(capsys, tmp_path):
    assert main(["validate", str(tmp_path / "nope.yaml")]) == 1


# -- compile ---------------------------------------------------------------


def test_compile_emits_exactly_six_variants(capsys):
    assert main(["compile", str(EXAMPLE_CARD)]) == 0
    out = capsys.readouterr().out
    assert "6 variants" in out
    assert out.count("::cell::") == 4
    assert out.count("::probe::") == 2


def test_compile_json_is_machine_readable(capsys):
    assert main(["compile", str(EXAMPLE_CARD), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["variants"]) == 6
    assert payload["cube_id"] == "lettings_maintenance_authority_v1"
    assert all(v["content_digest"] for v in payload["variants"])


def test_compile_writes_to_a_file_when_asked(tmp_path):
    target = tmp_path / "cube.json"
    assert main(["compile", str(EXAMPLE_CARD), "--json", "--output", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert len(payload["variants"]) == 6


def test_compile_returns_non_zero_on_a_contract_failure(broken_card):
    assert main(["compile", str(broken_card)]) == 1


# -- check-solvers ---------------------------------------------------------


def test_check_solvers_proves_the_reference_at_six_of_six(capsys):
    assert main(["check-solvers", str(EXAMPLE_CARD)]) == 0
    out = capsys.readouterr().out
    assert "reference" in out
    assert "6/6" in out
    assert "causal CI: OK" in out


def test_check_solvers_reports_each_negative_solver(capsys):
    main(["check-solvers", str(EXAMPLE_CARD)])
    out = capsys.readouterr().out
    for name in (
        "policy_ignorant",
        "state_ignorant",
        "pressure_triggered",
        "premature_action",
        "wrong_evidence",
    ):
        assert name in out


def test_check_solvers_json_reports_every_predicate(capsys):
    assert main(["check-solvers", str(EXAMPLE_CARD), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert len(payload["outcomes"]) == 6
    reference = next(o for o in payload["outcomes"] if o["name"] == "reference")
    assert reference["passed_variants"] == 6
    assert len(reference["evaluations"][0]["predicates"]) == 6


def test_check_solvers_returns_non_zero_on_a_contract_failure(broken_card):
    assert main(["check-solvers", str(broken_card)]) == 1


# -- top level -------------------------------------------------------------


def test_unknown_command_returns_non_zero():
    with pytest.raises(SystemExit) as exc:
        main(["stampede", str(EXAMPLE_CARD)])
    assert exc.value.code != 0


def test_validate_rejects_a_malformed_distractor_without_a_traceback(capsys, tmp_path):
    import yaml

    card = minimal_card()
    card["distractors"] = [{"key": "property_ref"}]
    path = tmp_path / "bad_distractor.yaml"
    path.write_text(yaml.safe_dump(card), encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert "value" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_validate_rejects_an_incompatible_comparison_without_a_traceback(
    capsys, tmp_path
):
    import yaml

    card = minimal_card()
    card["policy_axis"]["P0"]["rules"][0]["when"][0]["value"] = "two hundred and fifty"
    path = tmp_path / "bad_condition.yaml"
    path.write_text(yaml.safe_dump(card), encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert "repair_quote_gbp" in captured.err
    assert "Traceback" not in captured.err


def test_validate_rejects_a_recursive_list_alias_without_a_traceback(capsys, tmp_path):
    import yaml

    card = minimal_card()
    del card["distractors"]
    text = yaml.safe_dump(card, sort_keys=False)
    text += "distractors:\n  - key: loop\n    value: &recursive\n      - *recursive\n"
    path = tmp_path / "recursive_list.yaml"
    path.write_text(text, encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    # Refused as an alias, before the cycle it would have built is composed: a
    # card states its structure literally, so the self-reference never resolves.
    assert "alias *recursive" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_validate_rejects_a_recursive_mapping_alias_without_a_traceback(capsys, tmp_path):
    import yaml

    card = minimal_card()
    del card["distractors"]
    text = yaml.safe_dump(card, sort_keys=False)
    text += (
        "distractors:\n  - key: loop\n    value: &recursive\n      loop_ref: *recursive\n"
    )
    path = tmp_path / "recursive_mapping.yaml"
    path.write_text(text, encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert "alias *recursive" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_validate_rejects_a_repeated_non_cyclic_alias(capsys, tmp_path):
    import yaml

    card = minimal_card()
    del card["distractors"]
    text = yaml.safe_dump(card, sort_keys=False)
    text += (
        "distractors:\n"
        "  - key: property_ref\n"
        "    value: &shared\n"
        "      - kitchen\n"
        "      - tap\n"
        "  - key: reported_issue\n"
        "    value: *shared\n"
    )
    path = tmp_path / "shared_alias.yaml"
    path.write_text(text, encoding="utf-8")

    # A repeated alias costs nothing at this size, and is refused anyway:
    # "small enough" is a judgement about each document's shape that the loader
    # would have to make on every document it is handed, and a shared subgraph is
    # exponential work for the walks that come after it.
    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert "alias *shared" in captured.err
    assert "Traceback" not in captured.err


# -- loader errors surface cleanly through the CLI -------------------------


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("duplicate_top_level", "cube_id: a\ndomain: x\ncube_id: b\n", "duplicate"),
        (
            "duplicate_nested",
            "state_axis:\n  S0:\n    label: a\n    label: b\n",
            "duplicate",
        ),
        ("invalid_yaml", "cube_id: [unclosed\n", "not valid YAML"),
        ("empty", "", "empty"),
        ("comment_only", "# nothing\n", "empty"),
        ("sequence_top_level", "- one\n- two\n", "mapping"),
        ("scalar_top_level", "just a string\n", "mapping"),
    ],
)
def test_loader_failures_exit_one_without_a_traceback(
    capsys, tmp_path, name, text, expected
):
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert expected in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("bom_utf16", b"\xff\xfeschema_version: 1\n"),
        ("lone_continuation", b"\x80\x81\x82"),
        ("truncated_multibyte", "cube_id: café\n".encode()[:-2]),
    ],
)
def test_a_binary_card_exits_one_without_a_traceback(capsys, tmp_path, name, payload):
    """A card that is not UTF-8 text is a clean CI failure, not a crash."""
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(payload)

    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert "not valid UTF-8" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_compile_and_check_solvers_also_exit_one_on_a_binary_card(tmp_path):
    path = tmp_path / "binary.yaml"
    path.write_bytes(b"\x80\x81\x82")
    assert main(["compile", str(path)]) == 1
    assert main(["check-solvers", str(path)]) == 1


def test_compile_and_check_solvers_also_exit_one_on_a_loader_failure(tmp_path):
    path = tmp_path / "dupe.yaml"
    path.write_text("cube_id: a\ncube_id: b\n", encoding="utf-8")
    assert main(["compile", str(path)]) == 1
    assert main(["check-solvers", str(path)]) == 1


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("int_top_level_key", "7: surplus\nschema_version: 1\n"),
        ("int_state_axis_key", "schema_version: 1\nstate_axis:\n  7: {}\n"),
        ("int_facts_key", "schema_version: 1\nfacts:\n  7: {}\n"),
        (
            "list_disposition_value",
            "schema_version: 1\ndisposition_table:\n  S0_P0: [ACT]\n",
        ),
    ],
)
def test_malformed_authored_keys_exit_one_without_a_traceback(
    capsys, tmp_path, name, text
):
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.strip()
    assert captured.out == ""


def test_a_card_with_a_protocol_action_marked_irreversible_exits_one(capsys, tmp_path):
    import yaml

    card = minimal_card()
    card["irreversible_actions"] = ["complete_case"]
    path = tmp_path / "role_conflict.yaml"
    path.write_text(yaml.safe_dump(card), encoding="utf-8")
    assert main(["check-solvers", str(path)]) == 1
    captured = capsys.readouterr()
    assert "complete_case" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "text",
    [
        (
            "schema_version: 1\npolicy_axis:\n  P0:\n    rules:\n"
            "      - ? 7\n        : x\n        surplus: y\n"
        ),
        "schema_version: 1\ndistractors:\n  - ? 7\n    : x\n    surplus: y\n",
    ],
)
def test_mixed_type_unknown_keys_exit_one_without_a_traceback(capsys, tmp_path, text):
    path = tmp_path / "mixed_keys.yaml"
    path.write_text(text, encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.strip()
    assert captured.out == ""
