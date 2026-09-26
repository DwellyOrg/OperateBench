"""Focused gate for the fixed Mistral Small 2603 one-shot operator."""

from __future__ import annotations

import ast
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from operatebench.agents.pricing import RATE_SOURCE_OPERATOR, LifecyclePricingPolicy
from operatebench.execution_ledger import read_execution_ledger
from operatebench.providers.cost import request_input_token_bound
from tests.historical_canary_controls import historical_canary_gate_build  # noqa: F401
from tests.test_lifecycle_v1_anthropic_sonnet_canary_gate import (
    _security_ast_projection,
)
from tools import run_lifecycle_v1_anthropic_haiku_canary as haiku_canary
from tools import run_lifecycle_v1_anthropic_sonnet_canary as sonnet_canary
from tools import run_lifecycle_v1_mistral_small_2603_canary as canary

pytestmark = pytest.mark.usefixtures("historical_canary_gate_build")

# Literal review oracles: these are intentionally not derived from either module.
# The Sonnet import is solely a hardened-sibling parity oracle; it is never used
# as a Mistral request, identity, authorization, or execution dependency.
SHARED_TRUST_FUNCTIONS = frozenset(
    {
        "_audit_what_was_written",
        "_check_target_free",
        "_decode_preregistration",
        "_expected_preregistration",
        "_open_output_directory",
        "_open_preregistration",
        "_runtime_source_identity",
        "_scan_what_was_written",
        "_take_reservation",
        "check_live_authorization",
        "check_output_directory",
        "check_spec_identity",
        "check_worst_case",
        "controls_for",
        "open_live_transport",
        "refuse_live_environment",
        "reservation_name_for",
        "reserve_output_names",
        "run_live_cell",
        "scan_for_secrets",
        "wall_clock_deadline_at",
    }
)
SHARED_TRUST_CLASSES = frozenset(
    {
        "CanaryDeadlineExceeded",
        "CanaryFailure",
        "CanaryRefusal",
        "DeadlinedModelAgent",
        "OutputReservation",
        "_NetworkAttemptTransport",
    }
)
MISTRAL_SPECIFIC_FUNCTIONS = frozenset(
    {
        "_consume_preregistration_marker",
        "_emit",
        "_execute_one_cell",
        "_failure",
        "_run_live_cell",
        "build_parser",
        "check_fixed_canary_pricing_envelope",
        "check_pinned_controls",
        "consume_mistral_preregistration",
        "main",
        "mistral_preregistration_schema",
        "open_live_client",
        "open_offline_client",
        "run_offline_preflight",
    }
)
SONNET_ONLY_FUNCTIONS = frozenset(
    {"consume_sonnet_preregistration", "sonnet_preregistration_schema"}
)
MISTRAL_SPECIFIC_CLASSES = frozenset({"ProviderDispatchTelemetry", "ScriptedProvider"})

_FROZEN_OPERATOR_AST = {
    "haiku": {
        "module": "90b3f86e625cdb6954cc8f3a2efac655735b8752150bc979947613819bef64bb",
        "census": "b9264e1ce421cb48a0baabbdcf066904240914500179e46e484037977571237f",
        "imports": "69820bc2e24babc5268e91bb4dc79cbe2e40d9dab55fbbd0542391cb680cb6b5",
        "assignments": "5f7e0d9fdc5a6314eb753826d5aebf6110c6496c859a925520653acd4709c4d1",
        "classes": "d04ce42247b8854581857d3d5d2d1349604dbcf1b9f10ebe9913a44e64c192e0",
        "functions": "faf3831380975f0750e65cd867bbf50375b8e14df437ba03d43085b767749872",
    },
    "sonnet": {
        "module": "9195312c3b7f271ef21780ba78bd1d74f7dc15cadfa6316e5c359e0dfdb22341",
        "census": "1e6ec72e4b35da62612d57a297eff3cdcd0dfc180dd3cf78e58ef3261f57d9bb",
        "imports": "03227beee5c252d8e16ab4d21934026a64973eae5c041d3be415b671333d4418",
        "assignments": "624fdcf1dc243ad3d1161c51257dd50c55ebdfdd01510c6ce3659ff8f50ae4ad",
        "classes": "b52c541038122132eaed8e7c97165f0eea0fd8584b1e78ed39b85822eef5a22b",
        "functions": "82cf8e6717e4fbe09d3f539777a6e18e9c3fdeeb858ffa2a20085a1e5d1a418b",
    },
    "mistral": {
        "module": "be3fbb773a23d979518ca3442e3ebcf894bdf2fcada94a7c778224292e0f80d0",
        "census": "4062be30da3459eb86c471dc045e821fc4e755139887476cfdfc276ed7fee680",
        "imports": "332f314e525e4d38eb9820eca95f01657a365f420cd3226cdd11c7f934cd62ff",
        "assignments": "5f8170c41d19a462b1552d03975d9b72917a62bfd5dbefb7118eb87b78131d68",
        "classes": "1e4fe01a83a73eed5ac7da11524a96772ce3fb72d9fd9e5b44e138c417b9748f",
        "functions": "14c930c4ed0a3509bfab6234cd8cf7b9728ae108c1440d30a9ad9aa6addc528d",
    },
}
_FROZEN_TOP_LEVEL_COUNTS = {
    "haiku": {
        "AnnAssign": 2,
        "Assign": 46,
        "ClassDef": 6,
        "Expr": 1,
        "FunctionDef": 28,
        "If": 2,
        "Import": 10,
        "ImportFrom": 20,
    },
    "sonnet": {
        "AnnAssign": 3,
        "Assign": 53,
        "ClassDef": 8,
        "Expr": 1,
        "FunctionDef": 35,
        "If": 2,
        "Import": 12,
        "ImportFrom": 20,
    },
    "mistral": {
        "AnnAssign": 3,
        "Assign": 53,
        "ClassDef": 8,
        "Expr": 1,
        "FunctionDef": 35,
        "If": 2,
        "Import": 12,
        "ImportFrom": 22,
    },
}


def policy() -> LifecyclePricingPolicy:
    return LifecyclePricingPolicy(
        policy_id=canary.FIXED_CANARY_POLICY_ID,
        input_usd_per_mtok=Decimal("0.15"),
        output_usd_per_mtok=Decimal("0.60"),
        rate_source=RATE_SOURCE_OPERATOR,
    )


def prepare(directory: Path) -> None:
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)


def _definitions(module: Any, kind: type[ast.AST]) -> dict[str, ast.AST]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, kind) and hasattr(node, "name")
    }


def _operator_ast_fingerprints(module: Any) -> tuple[dict[str, str], dict[str, int]]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

    def digest(value: Any) -> str:
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    assignments: dict[str, Any] = {}
    census: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for index, node in enumerate(tree.body):
        node_kind = type(node).__name__
        counts[node_kind] = counts.get(node_kind, 0) + 1
        target: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(target, ast.Name):
            name = target.id
            assignments[name] = _security_ast_projection(node)
        else:
            name = f"@{index}"
        census.append((node_kind, name))

    imports = [
        _security_ast_projection(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    def definitions(kind: type[ast.FunctionDef] | type[ast.ClassDef]) -> dict[str, Any]:
        return {
            node.name: _security_ast_projection(node)
            for node in tree.body
            if isinstance(node, kind)
        }

    return (
        {
            "module": digest(_security_ast_projection(tree)),
            "census": digest(census),
            "imports": digest(imports),
            "assignments": digest(assignments),
            "classes": digest(definitions(ast.ClassDef)),
            "functions": digest(definitions(ast.FunctionDef)),
        },
        counts,
    )


def _source_identity() -> dict[str, str]:
    return {
        "merge_commit": "1" * 40,
        "git_tree": "2" * 40,
        "operator_sha256": hashlib.sha256(Path(canary.__file__).read_bytes()).hexdigest(),
    }


def _write_preregistration(
    tmp_path: Path,
    output: Path,
    *,
    nonce: str,
    mutate: tuple[str, Any] | None = None,
) -> Path:
    record = canary._expected_preregistration(
        output, policy(), cap=Decimal("0.33"), source_identity=_source_identity()
    )
    record["nonce"] = nonce
    if mutate is not None:
        record[mutate[0]] = mutate[1]
    parent = tmp_path / f"prereg-{nonce}"
    parent.mkdir(mode=0o700)
    path = parent / "mistral-live.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _authorise(
    tmp_path: Path, output: Path, monkeypatch: pytest.MonkeyPatch, *, nonce: str
) -> Path:
    path = _write_preregistration(tmp_path, output, nonce=nonce)
    monkeypatch.setenv(
        canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
    )
    monkeypatch.setenv(
        canary.CREDENTIAL_VARIABLE, "synthetic-mistral-key-not-a-credential"
    )
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(path))
    monkeypatch.setattr(canary, "_runtime_source_identity", _source_identity)
    return path


def test_fixed_identity_bounds_and_exact_worst_case() -> None:
    assert (
        canary.CANARY_PROVIDER,
        canary.CANARY_API,
        canary.CANARY_MODEL,
        canary.CANARY_PROFILE_ID,
    ) == (
        "mistral",
        "chat_completions",
        "mistral-small-2603",
        "mistralsmall2603_temperature_zero_v1",
    )
    assert canary.LIVE_AUTHORIZATION_VALUE == (
        "mistral:mistral-small-2603:V1:1:max-calls-50"
    )
    assert canary.CREDENTIAL_VARIABLE == "MISTRAL_API_KEY"
    assert canary.CANARY_OPERATION_VERSION == "0.6.0"
    assert canary.MAX_PROVIDER_CALLS == 50
    assert canary.EXPECTED_PROVIDER_CALLS == 46
    assert canary.LARGEST_REQUEST_TOKEN_BOUND == 27_331
    assert canary.EXPECTED_TOKEN_UPPER_BOUND == 1_445_642
    assert canary.TOKEN_HARD_CAP == 1_571_350
    assert policy().worst_case_usd(
        calls=50, input_token_bound=27_331, max_output_tokens=4096
    ) == Decimal("0.32786250")
    assert Decimal("0.33") == canary.FIXED_CANARY_COST_CAP_USD


def test_current_spec_and_preregistration_pin_maintenance_0_6(tmp_path: Path) -> None:
    output = tmp_path / "out"
    prepare(output)
    spec = canary.load_spec(canary.FIXTURE)

    canary.check_spec_identity(spec)
    assert spec.operation_version == "0.6.0"
    expected = canary._expected_preregistration(
        output, policy(), cap=Decimal("0.33"), source_identity=_source_identity()
    )
    info = output.stat()
    assert expected == {
        "schema": "operatebench.mistral-small-2603-live-preregistration.v1",
        "type": "mistral-small-2603-one-shot-live-authorization",
        "merge_commit": "1" * 40,
        "git_tree": "2" * 40,
        "operator_sha256": hashlib.sha256(Path(canary.__file__).read_bytes()).hexdigest(),
        "provider": "mistral",
        "api": "chat_completions",
        "model": "mistral-small-2603",
        "request_profile": "mistralsmall2603_temperature_zero_v1",
        "scenario_id": "V1",
        "agent_id": "mistral-small-2603",
        "operation_version": "0.6.0",
        "operatebench_version": "0.12.0",
        "episodes": 1,
        "expected_provider_calls": 46,
        "max_provider_calls": 50,
        "max_attempts_per_call": 1,
        "sdk_max_retries": 0,
        "max_output_tokens": 4096,
        "turn_deadline_seconds": 30,
        "wall_clock_deadline_seconds": 2700,
        "input_token_bound": 27_331,
        "expected_token_upper_bound": 1_445_642,
        "token_hard_cap": 1_571_350,
        "pricing_policy_id": "mistral_mistral_small_2603_official_2026_09_07_v1",
        "pricing_rate_source": "operator_supplied_pinned_rates",
        "pricing_recorded_on": "2026-09-07",
        "pricing_source_urls": list(canary.PRICING_SOURCE_URLS),
        "input_usd_per_mtok": "0.15",
        "output_usd_per_mtok": "0.6",
        "worst_case_usd": "0.3278625",
        "cost_cap_usd": "0.33",
        "output_directory_path": str(output.absolute()),
        "output_directory_dev": info.st_dev,
        "output_directory_ino": info.st_ino,
    }


def test_operator_prose_names_exact_native_api_identity() -> None:
    description = canary.build_parser().description
    operator_source = Path(canary.__file__).read_text(encoding="utf-8")
    telemetry_doc = canary.ProviderDispatchTelemetry.__doc__

    assert description is not None
    assert "mistral / Chat Completions / mistral-small-2603" in description
    for stale_identity in ("Messages", "Anthropic", "Sonnet", "Claude"):
        assert stale_identity not in description
    assert "messages.with_raw_response.create" not in operator_source
    assert telemetry_doc is not None
    assert "client.chat.complete" in telemetry_doc


def test_provider_neutral_trust_kernel_is_exact_ast_equal_to_sonnet() -> None:
    for name, module in (
        ("haiku", haiku_canary),
        ("sonnet", sonnet_canary),
        ("mistral", canary),
    ):
        fingerprints, counts = _operator_ast_fingerprints(module)
        assert fingerprints == _FROZEN_OPERATOR_AST[name]
        assert counts == _FROZEN_TOP_LEVEL_COUNTS[name]

    mistral_functions = _definitions(canary, ast.FunctionDef)
    sonnet_functions = _definitions(sonnet_canary, ast.FunctionDef)
    mistral_classes = _definitions(canary, ast.ClassDef)
    sonnet_classes = _definitions(sonnet_canary, ast.ClassDef)

    assert set(mistral_functions) == SHARED_TRUST_FUNCTIONS | MISTRAL_SPECIFIC_FUNCTIONS
    assert set(sonnet_functions) == (
        SHARED_TRUST_FUNCTIONS
        | (
            MISTRAL_SPECIFIC_FUNCTIONS
            - {"consume_mistral_preregistration", "mistral_preregistration_schema"}
        )
        | SONNET_ONLY_FUNCTIONS
    )
    assert set(mistral_classes) == SHARED_TRUST_CLASSES | MISTRAL_SPECIFIC_CLASSES
    assert set(sonnet_classes) == SHARED_TRUST_CLASSES | MISTRAL_SPECIFIC_CLASSES

    for name in SHARED_TRUST_FUNCTIONS:
        assert _security_ast_projection(
            mistral_functions[name]
        ) == _security_ast_projection(sonnet_functions[name]), name
    for name in SHARED_TRUST_CLASSES:
        assert _security_ast_projection(
            mistral_classes[name]
        ) == _security_ast_projection(sonnet_classes[name]), name

    common_functions = set(mistral_functions) & set(sonnet_functions)
    observed_function_differences = {
        name
        for name in common_functions
        if _security_ast_projection(mistral_functions[name])
        != _security_ast_projection(sonnet_functions[name])
    }
    assert observed_function_differences == MISTRAL_SPECIFIC_FUNCTIONS - {
        "consume_mistral_preregistration",
        "mistral_preregistration_schema",
    }
    observed_class_differences = {
        name
        for name in mistral_classes
        if _security_ast_projection(mistral_classes[name])
        != _security_ast_projection(sonnet_classes[name])
    }
    assert observed_class_differences == MISTRAL_SPECIFIC_CLASSES

    function = ast.parse("def f():\n    pass\n").body[0]
    baseline = _security_ast_projection(function)
    if "type_params" not in function._fields:
        function.__dict__["_fields"] = (*function._fields, "type_params")
    function.__dict__["type_params"] = []
    assert _security_ast_projection(function) == baseline
    function.__dict__["type_params"] = [ast.Name(id="T", ctx=ast.Load())]
    assert _security_ast_projection(function) != baseline

    function.__dict__["type_params"] = []
    function.__dict__["_fields"] = (*function._fields, "future_semantics")
    function.__dict__["future_semantics"] = []
    assert _security_ast_projection(function) != baseline

    name = ast.Name(id="value", ctx=ast.Load())
    name_baseline = _security_ast_projection(name)
    name.__dict__["_fields"] = (*name._fields, "type_params")
    name.__dict__["type_params"] = []
    assert _security_ast_projection(name) != name_baseline


def test_unique_mistral_authority_schema_and_marker_bytes_are_exact() -> None:
    assert canary.LIVE_AUTHORIZATION_VARIABLE == (
        "OPERATEBENCH_MISTRAL_SMALL_2603_LIVE_CELL_AUTHORIZATION"
    )
    assert canary.PREREGISTRATION_PATH_VARIABLE == (
        "OPERATEBENCH_MISTRAL_SMALL_2603_PREREGISTRATION_PATH"
    )
    assert canary.PREREGISTRATION_SCHEMA == (
        "operatebench.mistral-small-2603-live-preregistration.v1"
    )
    assert canary.PREREGISTRATION_TYPE == (
        "mistral-small-2603-one-shot-live-authorization"
    )
    assert canary.PREREGISTRATION_CONSUMED_SUFFIX == ".consumed-not-authorization"
    assert canary.PREREGISTRATION_CONSUMED_BYTES == (
        b"operatebench Mistral Small 2603 preregistration permanently consumed; "
        b"this is not authorization or run evidence.\n"
    )
    assert canary.RESERVATION_SUFFIX == (
        ".mistralsmall2603-canary-attempt-claim-not-evidence"
    )
    assert canary.RESERVATION_BYTES == (
        b"operatebench Mistral Small 2603 one-cell canary: this exact output name "
        b"was claimed by an attempted launch.\nThis permanent one-shot marker is "
        b"intentionally retained and is not evidence.\n"
    )
    forbidden = ("SONNET", "HAIKU", "ANTHROPIC", "CLAUDE")
    authorities = (
        canary.LIVE_AUTHORIZATION_VARIABLE,
        canary.LIVE_AUTHORIZATION_VALUE,
        canary.PREREGISTRATION_PATH_VARIABLE,
        canary.PREREGISTRATION_SCHEMA,
        canary.PREREGISTRATION_TYPE,
        canary.CREDENTIAL_VARIABLE,
    )
    assert all(token not in value.upper() for token in forbidden for value in authorities)


def test_offline_real_sdk_runs_46_exact_requests_and_freezes_native_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    observed: list[Any] = []

    class Observed(canary.ScriptedProvider):
        def __call__(self, request: Any) -> Any:
            observed.append(request)
            response = super().__call__(request)
            response.headers["set-cookie"] = "session=provider-secret"
            return response

    monkeypatch.setattr(canary, "ScriptedProvider", Observed)
    summary = canary.run_offline_preflight(
        directory=output, policy=policy(), cap=Decimal("0.33"), environ={}
    )

    assert summary["provider_calls"] == summary["attempts"] == 46
    assert summary["artifact_version"] == 8
    assert summary["replay"]["provider_calls"] == 0
    assert summary["budget"]["worst_case_usd"] == "0.3278625"
    assert Decimal(summary["budget"]["worst_case_usd"]) == Decimal("0.32786250")
    assert len(observed) == 46
    assert all("cookie" not in request.headers for request in observed)
    bodies = [json.loads(request.content) for request in observed]
    for request, body in zip(observed, bodies, strict=True):
        assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
        assert body["model"] == "mistral-small-2603"
        assert body["temperature"] == 0.0
        assert body["tool_choice"] == "required"
        assert body["parallel_tool_calls"] is False
        assert body["max_tokens"] == 4096
        assert body["stream"] is False
        assert not (
            {"reasoning_effort", "service_tier", "prompt_cache_key"} & body.keys()
        )
    bounds = [request_input_token_bound(body, "Mistral request") for body in bodies]
    assert len(bounds) == 46
    assert max(bounds) == 27_301
    assert max(bounds) <= canary.LARGEST_REQUEST_TOKEN_BOUND == 27_331


@pytest.mark.parametrize(
    "flag", ["--provider", "--model", "--scenario", "--episode", "--max-calls"]
)
def test_cli_has_no_cell_or_call_cap_selector(flag: str) -> None:
    with pytest.raises(SystemExit):
        canary.build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/no-run",
                "--cost-cap-usd",
                "0.33",
                "--input-usd-per-mtok",
                "0.15",
                "--output-usd-per-mtok",
                "0.60",
                flag,
                "changed",
            ]
        )


def test_wrong_provider_authority_and_one_cent_under_are_refused() -> None:
    with pytest.raises(canary.CanaryRefusal, match="live_authorization"):
        canary.check_live_authorization(
            {"OPERATEBENCH_SONNET_5_LIVE_CELL_AUTHORIZATION": "anything"}
        )
    with pytest.raises(canary.CanaryRefusal, match="fixed_canary_pricing"):
        canary.check_fixed_canary_pricing_envelope(policy(), cap=Decimal("0.32"))


def test_preregistration_and_reservation_identities_do_not_overlap_sonnet() -> None:
    assert "mistral-small-2603" in canary.PREREGISTRATION_SCHEMA
    assert "mistral-small-2603" in canary.PREREGISTRATION_TYPE
    assert b"Mistral Small 2603" in canary.PREREGISTRATION_CONSUMED_BYTES
    assert b"Mistral Small 2603" in canary.RESERVATION_BYTES
    assert "sonnet" not in canary.PREREGISTRATION_SCHEMA.lower()
    assert "sonnet" not in canary.PREREGISTRATION_TYPE.lower()


def test_environment_only_authorization_refuses_before_credential_or_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)

    class CountingEnvironment(dict[str, Any]):
        credential_reads = 0

        def get(self, key: str, default: Any = None) -> Any:
            if key == canary.CREDENTIAL_VARIABLE:
                self.credential_reads += 1
            return super().get(key, default)

        def pop(self, key: str, default: Any = None) -> Any:
            if key == canary.CREDENTIAL_VARIABLE:
                self.credential_reads += 1
            return super().pop(key, default)

    environ = CountingEnvironment()
    reached: list[str] = []
    monkeypatch.setattr(canary.os, "environ", environ)
    monkeypatch.setattr(
        canary, "open_live_transport", lambda: reached.append("transport")
    )
    monkeypatch.setattr(
        canary,
        "open_live_client",
        lambda *_args, **_kwargs: reached.append("client"),
    )

    with pytest.raises(canary.CanaryRefusal, match="live_authorization"):
        canary.run_live_cell(directory=output, policy=policy(), cap=Decimal("0.33"))
    assert environ.credential_reads == 0
    assert reached == []


def test_wrong_record_and_sequential_consumed_record_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    prepare(output)
    monkeypatch.setattr(canary, "_runtime_source_identity", _source_identity)

    wrong = _write_preregistration(
        tmp_path, output, nonce="a" * 64, mutate=("provider", "anthropic")
    )
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(wrong))
    with pytest.raises(canary.CanaryRefusal, match=canary.CODE_PREREGISTRATION_REFUSED):
        canary.consume_mistral_preregistration(output, policy(), cap=Decimal("0.33"))

    valid = _write_preregistration(tmp_path, output, nonce="b" * 64)
    monkeypatch.setenv(canary.PREREGISTRATION_PATH_VARIABLE, str(valid))
    identity = canary.consume_mistral_preregistration(
        output, policy(), cap=Decimal("0.33")
    )
    info = output.stat()
    assert identity == (info.st_dev, info.st_ino)
    with pytest.raises(canary.CanaryRefusal, match=canary.CODE_PREREGISTRATION_REFUSED):
        canary.consume_mistral_preregistration(output, policy(), cap=Decimal("0.33"))


def test_output_inode_mismatch_refuses_before_output_claims(tmp_path: Path) -> None:
    output = tmp_path / "out"
    prepare(output)
    info = output.stat()
    with (
        pytest.raises(canary.CanaryRefusal, match="output_directory_identity_changed"),
        canary.reserve_output_names(
            output, expected_identity=(info.st_dev, info.st_ino + 1)
        ),
    ):
        pytest.fail("mismatched output inode was admitted")
    assert list(output.iterdir()) == []


def test_first_transport_fault_is_one_dispatch_without_artifact_or_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "out"
    prepare(output)
    _authorise(tmp_path, output, monkeypatch, nonce="c" * 64)
    sdk_calls = 0

    def fail(request: httpx.Request) -> httpx.Response:
        nonlocal sdk_calls
        sdk_calls += 1
        raise httpx.ConnectError("synthetic transport failure", request=request)

    monkeypatch.setattr(canary, "open_live_transport", lambda: httpx.MockTransport(fail))
    status = canary.main(
        [
            "--live",
            "--output-dir",
            str(output),
            "--cost-cap-usd",
            "0.33",
            "--input-usd-per-mtok",
            "0.15",
            "--output-usd-per-mtok",
            "0.60",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert status != 0
    assert sdk_calls == summary["provider_calls_dispatched"] == 1
    assert summary["live_provider_call"] is True
    assert "artifact" not in summary
    assert "evaluator" not in summary
    ledger = read_execution_ledger(output / canary.LEDGER_NAME)
    assert ledger.totals.provider_calls == ledger.totals.attempts == 1
    assert ledger.calls[0].attempts[0].response_received is False
    assert not (output / canary.ARTIFACT_NAME).exists()
