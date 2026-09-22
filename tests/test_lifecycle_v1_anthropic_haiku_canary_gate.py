"""The one-cell canary: what it refuses, and what it does when it does not.

``tools/run_lifecycle_v1_anthropic_haiku_canary.py`` is not a provider CLI. It
runs exactly one pinned cell — anthropic / Messages /
``claude-haiku-4-5-20251001`` / V1 / one episode — and it
has no flag that could make it run another. Its default and always-available
path is an offline preflight over ``httpx.MockTransport``; its live path exists
but is gated behind a conjunction of conditions, every one of which this module
proves is load-bearing.

**No test here makes a live provider call.** The single function that would
construct a live client is monkeypatched in every test that reaches it, and the
gating tests assert it was never reached at all — not "was called with mock
arguments", but *never called*, because a gate that constructs the client and
then declines to use it has already handed a credential to an SDK.

The authorised-path test replaces that one function with a client over
``MockTransport``, which is what makes "the credential is removed from the
environment and the local reference is dropped in ``finally``" a tested contract
rather than an intention.
"""

from __future__ import annotations

import errno
import json
import os
import shlex
import stat
import subprocess
import sys
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import replace
from decimal import ROUND_FLOOR, Decimal, Inexact, Rounded, localcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from operatebench._write_once import _write_once_bytes_at, _WriteOnceFailure
from operatebench.agents.anthropic_messages import (
    LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION,
    anthropic_outcome_tools,
)
from operatebench.agents.model import MODEL_PROTOCOL_VERSION
from operatebench.artifact import ARTIFACT_VERSION, ArtifactError, read_artifact
from operatebench.execution_bundle import audit_execution_bundle_files
from operatebench.execution_ledger import (
    EXECUTION_LEDGER_VERSION,
    TERMINAL_SCORED,
    LedgerPathError,
    read_execution_ledger,
)
from operatebench.providers.anthropic_messages import TOOL_CHOICE
from operatebench.version import OPERATEBENCH_VERSION
from tools import run_lifecycle_v1_anthropic_haiku_canary as canary

FAKE_API_KEY = "invented-not-a-credential"
LIVE_KEY = "-".join(("sk", "ant", "api03", "0" * 90))

#: What an SDK, a client or a provider is entitled to put in an exception, and
#: what this build is therefore never allowed to render. Every fragment here is
#: invented in this file: no credential exists in this repository.
SECRET_PROSE = (
    f"Incorrect API key provided: {LIVE_KEY}. "
    f"Authorization: Bearer *** was rejected by the provider "
    "(x-api-key header echoed back in the response body)"
)

#: Fragments that must not appear in anything a run prints or writes.
FORBIDDEN_FRAGMENTS = (
    LIVE_KEY,
    "Bearer",
    "x-api-key",
    "Incorrect API key",
    "RuntimeError",
)

#: The two pinned deadlines, and a value for each that this build is not
#: authorised to run under.
DEADLINE_MUTATIONS = [
    ("TURN_DEADLINE_SECONDS", 31.0),
    ("WALL_CLOCK_DEADLINE_SECONDS", 2701.0),
]

#: Each term in the authorised fifty-call token and time envelope, changed by
#: itself.  The expected values deliberately do not derive from these names: a
#: test whose expected side moves with the subject cannot detect a widened build.
AUTHORISED_ENVELOPE_MUTATIONS = [
    ("LARGEST_REQUEST_TOKEN_BOUND", 25_650, 1_487_300),
    ("MAX_OUTPUT_TOKENS", 4_097, 1_487_300),
    ("TOKEN_HARD_CAP", 1_487_251, None),
    ("MAX_PROVIDER_CALLS", 51, 1_516_995),
    ("WALL_CLOCK_DEADLINE_SECONDS", 2_701.0, None),
]

# Every source term that defines the authorised cell. The replacement values
# and derivative updates are literal so this oracle cannot move with production.
EXACT_CELL_MUTATIONS = [
    (
        "CANARY_PROVIDER",
        "another-provider",
        {
            "LIVE_AUTHORIZATION_VALUE": (
                "another-provider:claude-haiku-4-5-20251001:V1:1:max-calls-50"
            )
        },
    ),
    ("CANARY_API", "chat_completions", {}),
    (
        "CANARY_MODEL",
        "claude-haiku-4-5-20251001-moved",
        {
            "HAIKU_4_5_MODEL": "claude-haiku-4-5-20251001-moved",
            "LIVE_AUTHORIZATION_VALUE": (
                "anthropic:claude-haiku-4-5-20251001-moved:V1:1:max-calls-50"
            ),
        },
    ),
    (
        "CANARY_PROFILE_ID",
        "gpt56luna_moved",
        {
            "HAIKU_4_5_PROFILE": replace(
                canary.HAIKU_4_5_PROFILE, profile_id="haiku45_moved"
            )
        },
    ),
    (
        "CANARY_SCENARIO_ID",
        "V2",
        {
            "LIVE_AUTHORIZATION_VALUE": (
                "anthropic:claude-haiku-4-5-20251001:V2:1:max-calls-50"
            )
        },
    ),
    ("CANARY_AGENT_ID", "another-agent", {}),
    (
        "CANARY_EPISODES",
        2,
        {
            "LIVE_AUTHORIZATION_VALUE": (
                "anthropic:claude-haiku-4-5-20251001:V1:2:max-calls-50"
            )
        },
    ),
    ("EXPECTED_PROVIDER_CALLS", 47, {}),
    (
        "LIVE_AUTHORIZATION_VALUE",
        "anthropic:claude-haiku-4-5-20251001:V1:1:max-calls-51",
        {},
    ),
    ("MAX_PROVIDER_CALLS", 51, {"TOKEN_HARD_CAP": 1_516_995}),
    (
        "MAX_ATTEMPTS_PER_CALL",
        2,
        {
            "LIFECYCLE_ANTHROPIC_RETRY_POLICY": replace(
                canary.LIFECYCLE_ANTHROPIC_RETRY_POLICY, max_attempts=2
            )
        },
    ),
    ("SDK_MAX_RETRIES", 1, {}),
    ("LARGEST_REQUEST_TOKEN_BOUND", 25_650, {"TOKEN_HARD_CAP": 1_487_300}),
    ("MAX_OUTPUT_TOKENS", 4_097, {"TOKEN_HARD_CAP": 1_487_300}),
    ("TOKEN_HARD_CAP", 1_487_251, {}),
    ("TURN_DEADLINE_SECONDS", 31.0, {}),
    ("WALL_CLOCK_DEADLINE_SECONDS", 2_701.0, {}),
    ("TOOL_CHOICE", {"type": "auto"}, {}),
    ("OPERATEBENCH_VERSION", "0.7.1", {}),
    ("CANARY_OPERATION_VERSION", "0.6.1", {}),
    ("MODEL_PROTOCOL_VERSION", "operatebench.model.v5", {}),
    (
        "LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION",
        "lifecycle_anthropic_messages_model_request_v3",
        {},
    ),
    ("ARTIFACT_VERSION", 9, {}),
    ("EXECUTION_LEDGER_VERSION", 4, {}),
]

EXACT_PRICING_CONSTANT_MUTATIONS = [
    ("FIXED_CANARY_POLICY_ID", "operator_pinned_v2"),
    ("RATE_SOURCE_OPERATOR", "another_operator_source"),
    ("FIXED_CANARY_RATE_SOURCE", "another_operator_source"),
    ("FIXED_CANARY_INPUT_USD_PER_MTOK", Decimal("1.01")),
    ("FIXED_CANARY_OUTPUT_USD_PER_MTOK", Decimal("5.01")),
    ("FIXED_CANARY_COST_CAP_USD", Decimal("2.38")),
]


def test_the_operator_script_is_directly_invocable() -> None:
    """The documented file-path entrypoint must import before any gate or client."""
    result = subprocess.run(
        [
            sys.executable,
            str(canary.REPO_ROOT / "tools/run_lifecycle_v1_anthropic_haiku_canary.py"),
            "--help",
        ],
        cwd=canary.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--offline-preflight" in result.stdout


#: Where a secret-bearing exception is injected: the two client-construction
#: seams, the run itself, and the write and read either side of the artefact.
INJECTION_POINTS = [
    "open_live_transport",
    "open_live_client",
    "run_episode",
    "write_artifact",
    "read_artifact",
]

RATE_ARGUMENTS = [
    "--cost-cap-usd",
    "2.32",
    "--input-usd-per-mtok",
    "1",
    "--output-usd-per-mtok",
    "5",
]

PERSISTENT_MARKER_NAMES = (
    ".execution_ledger.ndjson.canary-attempt-claim-not-evidence",
    ".episode_artifact.json.canary-attempt-claim-not-evidence",
    ".execution_ledger.partial.ndjson.canary-attempt-claim-not-evidence",
)
PERSISTENT_MARKER_BYTES = (
    b"operatebench one-cell canary: this exact output name was claimed by an "
    b"attempted launch.\nThis permanent one-shot marker is intentionally retained "
    b"and is not evidence.\n"
)


def expected_output_entries(*evidence_names: str) -> list[str]:
    # Beginning the central ledger also creates its NON-SCORED prefix sidecar.
    partial_names = (
        (Path(canary.LEDGER_NAME).with_suffix(".partial.ndjson").name,)
        if canary.LEDGER_NAME in evidence_names
        else ()
    )
    return sorted({*PERSISTENT_MARKER_NAMES, *evidence_names, *partial_names})


def assert_exact_persistent_markers(directory: Path) -> None:
    for name in PERSISTENT_MARKER_NAMES:
        marker = directory / name
        assert marker.read_bytes() == PERSISTENT_MARKER_BYTES
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600


PRICING_ARGUMENT_MUTATIONS = [
    ("--input-usd-per-mtok", "0.99"),
    ("--input-usd-per-mtok", "1.01"),
    ("--output-usd-per-mtok", "4.99"),
    ("--output-usd-per-mtok", "5.01"),
    ("--cost-cap-usd", "2.30"),
    ("--cost-cap-usd", "2.38"),
]

EXPECTED_PRICING_SOURCES = (
    "https://platform.claude.com/docs/en/models/overview",
    "https://platform.claude.com/docs/en/about-claude/pricing",
)


def documented_operator_commands() -> list[list[str]]:
    docstring = canary.__doc__ or ""
    logical_lines = docstring.replace("\\\n", " ").splitlines()
    return [
        shlex.split(line.strip())
        for line in logical_lines
        if line.strip().startswith("uv run python ")
    ]


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "canary"
    directory.mkdir(mode=0o700)
    return directory


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        canary.LIVE_AUTHORIZATION_VARIABLE,
        canary.CREDENTIAL_VARIABLE,
        *canary.REJECTED_VARIABLES,
    ):
        monkeypatch.delenv(name, raising=False)


class NeverBuilt(Exception):
    """Raised by the stand-in for the live client. Reaching it is the failure."""


class CredentialUnreadEnvironment(MutableMapping[str, str]):
    """Environment tripwire that permits deletion but refuses credential reads."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._data = dict(values)

    def __getitem__(self, key: str) -> str:
        if key == canary.CREDENTIAL_VARIABLE:
            raise AssertionError("credential read before pricing-envelope refusal")
        return self._data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class CredentialReadCountingEnvironment(MutableMapping[str, str]):
    """Environment stand-in that counts access to the live credential."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._data = dict(values)
        self.credential_reads = 0

    def __getitem__(self, key: str) -> str:
        if key == canary.CREDENTIAL_VARIABLE:
            self.credential_reads += 1
        return self._data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


@pytest.fixture()
def no_client(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Tripwire both functions a live run would have to go through.

    Two, not one, because they are two separate things a gate has to prevent:
    being in a position to send, and constructing a client around a credential.
    A path that did the second and stopped short of the first has already handed
    a key to an SDK.
    """
    reached: list[str] = []

    def no_transport() -> Any:
        reached.append("transport")
        raise NeverBuilt("a live transport was opened on a path that must refuse")

    def no_client_built(api_key: str, *, transport: Any) -> Any:
        reached.append("client")
        raise NeverBuilt("a live client was constructed on a path that must refuse")

    monkeypatch.setattr(canary, "open_live_transport", no_transport)
    monkeypatch.setattr(canary, "open_live_client", no_client_built)
    return reached


@pytest.fixture()
def no_construction(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Tripwire every way this tool could build a client, a transport or a cell.

    Wider than :func:`no_client`, because the properties proven with it are
    about the *offline* path too: a refusal that has already built a mock
    transport and an SDK client has done the construction work a preflight
    exists to make unnecessary, and a refusal that reached ``_execute_one_cell`` has
    dispatched. Reaching any of these is the failure.
    """
    reached: list[str] = []

    def trap(label: str) -> Any:
        def _trap(*args: Any, **kwargs: Any) -> Any:
            reached.append(label)
            raise NeverBuilt(f"{label} was reached on a path that must refuse")

        return _trap

    monkeypatch.setattr(canary.anthropic, "Anthropic", trap("sdk_client"))
    monkeypatch.setattr(httpx, "MockTransport", trap("mock_transport"))
    monkeypatch.setattr(canary, "WireCaptureTransport", trap("wire_capture"))
    monkeypatch.setattr(canary, "ScriptedProvider", trap("scripted_provider"))
    monkeypatch.setattr(canary, "open_live_transport", trap("live_transport"))
    monkeypatch.setattr(canary, "open_live_client", trap("live_client"))
    monkeypatch.setattr(canary, "_execute_one_cell", trap("execute_one_cell"))
    return reached


def offline_arguments(directory: Path) -> list[str]:
    return ["--offline-preflight", "--output-dir", str(directory), *RATE_ARGUMENTS]


def live_arguments(directory: Path) -> list[str]:
    return ["--live", "--output-dir", str(directory), *RATE_ARGUMENTS]


def authorise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
    )
    monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)


class Wire:
    def __init__(self) -> None:
        self.provider: canary.ScriptedProvider | None = None

    @property
    def bodies(self) -> list[bytes]:
        return [] if self.provider is None else self.provider.bodies


def mock_live_transport(wire: Wire, **kwargs: Any) -> Any:
    def factory() -> httpx.BaseTransport:
        fault_map = kwargs.get("faults", {})
        fault_at = next(iter(fault_map), 0)
        fault = fault_map.get(fault_at)
        wire.provider = canary.ScriptedProvider(
            max_calls=canary.MAX_PROVIDER_CALLS, fault=fault, fault_at=fault_at
        )
        return httpx.MockTransport(wire.provider)

    return factory


# -- the tool is one cell wide ------------------------------------------------


class TestTheToolIsOneCellWide:
    def test_direct_and_module_examples_pin_the_operator_rates(self) -> None:
        commands = documented_operator_commands()

        assert commands == [
            [
                "uv",
                "run",
                "python",
                "tools/run_lifecycle_v1_anthropic_haiku_canary.py",
                "--offline-preflight",
                "--output-dir",
                "DIR",
                *RATE_ARGUMENTS,
            ],
            [
                "uv",
                "run",
                "python",
                "-m",
                "tools.run_lifecycle_v1_anthropic_haiku_canary",
                "--offline-preflight",
                "--output-dir",
                "DIR",
                *RATE_ARGUMENTS,
            ],
        ]

    def test_it_is_pinned_to_one_provider_api_model_scenario_and_episode(self) -> None:
        assert canary.CANARY_PROVIDER == "anthropic"
        assert canary.CANARY_API == "messages"
        assert canary.CANARY_MODEL == "claude-haiku-4-5-20251001"
        assert canary.CANARY_PROFILE_ID.startswith("haiku45")
        assert canary.CANARY_SCENARIO_ID == "V1"
        assert canary.CANARY_EPISODES == 1

    def test_no_argument_selects_a_scenario_a_model_or_a_provider(self) -> None:
        options = {
            option
            for action in canary.build_parser()._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--scenario",
            "--scenario-id",
            "--model",
            "--provider",
            "--api",
            "--agent",
            "--episodes",
            "--base-url",
            "--max-provider-calls",
            "--pricing-policy-id",
        ):
            assert forbidden not in options

    def test_the_removed_pricing_policy_selector_is_rejected_as_stale_cli(self) -> None:
        with pytest.raises(SystemExit) as refusal:
            canary.build_parser().parse_args(
                [
                    *offline_arguments(Path("unused")),
                    "--pricing-policy-id",
                    "anthropic_haiku_operator_pinned_2026_09_06_v1",
                ]
            )
        assert refusal.value.code == 2

    def test_the_source_holds_no_loop_over_cells(self) -> None:
        source = Path(canary.__file__).read_text(encoding="utf-8")
        assert "for scenario" not in source
        assert "for cell" not in source
        assert "for model" not in source

    def test_the_pinned_controls_are_this_builds_fixed_ones(self) -> None:
        assert canary.MAX_PROVIDER_CALLS == 50
        assert canary.EXPECTED_PROVIDER_CALLS == 46
        assert canary.LARGEST_REQUEST_TOKEN_BOUND == 25_881
        assert canary.MAX_OUTPUT_TOKENS == 4_096
        assert canary.MAX_ATTEMPTS_PER_CALL == 1
        assert canary.SDK_MAX_RETRIES == 0
        assert canary.EXPECTED_TOKEN_UPPER_BOUND == 1_378_942
        assert canary.TOKEN_HARD_CAP == 1_498_850
        assert canary.TURN_DEADLINE_SECONDS == 30.0
        assert canary.WALL_CLOCK_DEADLINE_SECONDS == 2_700.0

    def test_the_runtime_interface_versions_are_pinned(self) -> None:
        spec = canary.load_spec(canary.FIXTURE)

        assert OPERATEBENCH_VERSION == "0.12.0"
        assert spec.operation_version == "0.6.0"
        assert MODEL_PROTOCOL_VERSION == "operatebench.model.v4"
        assert (
            LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION
            == "lifecycle_anthropic_messages_model_request_v4"
        )
        assert ARTIFACT_VERSION == 8
        assert EXECUTION_LEDGER_VERSION == 3

    def test_the_runtime_request_profile_and_settings_are_pinned(self) -> None:
        profile = canary.HAIKU_4_5_PROFILE

        assert canary.CANARY_PROFILE_ID == (
            "haiku45_temperature_zero_thinking_omitted_v1"
        )
        assert profile.temperature == 0.0
        assert profile.thinking is None
        assert profile.omitted_fields() == ("thinking", "top_k", "top_p")
        assert dict(TOOL_CHOICE) == {"type": "any", "disable_parallel_tool_use": True}

    def test_the_fixed_pricing_policy_source_is_pinned(self) -> None:
        assert canary.PRICING_SOURCE_URLS == EXPECTED_PRICING_SOURCES
        assert canary.PRICING_RETRIEVED_DATE == "2026-09-06"

    def test_the_live_authorization_names_the_fifty_call_envelope(self) -> None:
        assert (
            canary.LIVE_AUTHORIZATION_VALUE
            == "anthropic:claude-haiku-4-5-20251001:V1:1:max-calls-50"
        )

    def test_pricing_help_describes_numeric_not_lexical_equality(self) -> None:
        help_text = " ".join(canary.build_parser().format_help().lower().split())

        for amount in ("2.32", "1", "5"):
            assert f"must equal usd {amount} numerically" in help_text
        assert help_text.count("equivalent exact-decimal spellings are accepted") == 3
        assert "required exact decimal string" not in help_text
        assert "accepts only 0.50 usd" not in help_text

    def test_the_pessimistic_fifty_call_cost_fits_the_fixed_cap_at_stated_rates(
        self,
    ) -> None:
        policy = canary.LifecyclePricingPolicy(
            policy_id=canary.FIXED_CANARY_POLICY_ID,
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
        )
        worst_case = policy.worst_case_usd(
            calls=canary.MAX_PROVIDER_CALLS,
            input_token_bound=canary.LARGEST_REQUEST_TOKEN_BOUND,
            max_output_tokens=canary.MAX_OUTPUT_TOKENS,
        )

        assert worst_case == Decimal("2.31805")
        assert worst_case <= Decimal("2.32")

    def test_worst_case_and_emitted_budget_ignore_hostile_decimal_context(
        self, output_dir: Path, capsys: Any
    ) -> None:
        with localcontext() as hostile:
            hostile.prec = 1
            hostile.rounding = ROUND_FLOOR
            hostile.traps[Inexact] = True
            hostile.traps[Rounded] = True
            policy = canary.LifecyclePricingPolicy(
                policy_id=canary.FIXED_CANARY_POLICY_ID,
                input_usd_per_mtok=Decimal("1"),
                output_usd_per_mtok=Decimal("5"),
            )
            assert policy.worst_case_usd(
                calls=50, input_token_bound=25_881, max_output_tokens=4_096
            ) == Decimal("2.31805")
            assert canary.check_worst_case(policy, cap=Decimal("2.32")) == {
                "worst_case_usd": "2.31805",
                "cost_cap_usd": "2.32",
                "token_hard_cap": 1_498_850,
                "pricing_digest_sha256": policy.digest_sha256,
                "rate_source": "operator_supplied_pinned_rates",
                "pricing_source_urls": list(EXPECTED_PRICING_SOURCES),
                "pricing_retrieved_date": "2026-09-06",
            }
            assert canary.main(offline_arguments(output_dir)) == 0

        summary = json.loads(capsys.readouterr().out)
        assert summary["budget"]["worst_case_usd"] == "2.31805"
        assert summary["budget"]["cost_cap_usd"] == "2.32"
        assert summary["measured_cost_usd"] == "0.220846"
        assert Decimal(summary["ledger_audit"]["totals"]["reserved_usd"]) == (
            Decimal("1.832845")
            - Decimal(23 * 30)
            * canary.FIXED_CANARY_INPUT_USD_PER_MTOK
            / Decimal(1_000_000)
        )

    def test_the_fixed_canary_pricing_envelope_is_named_as_build_owned_controls(
        self,
    ) -> None:
        assert Decimal("1") == canary.FIXED_CANARY_INPUT_USD_PER_MTOK
        assert Decimal("5") == canary.FIXED_CANARY_OUTPUT_USD_PER_MTOK
        assert Decimal("2.32") == canary.FIXED_CANARY_COST_CAP_USD
        assert canary.FIXED_CANARY_RATE_SOURCE == canary.RATE_SOURCE_OPERATOR
        assert (
            canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED
            == "fixed_canary_pricing_envelope_required"
        )
        assert (
            canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED in canary.FAILURE_DETAILS
        )
        assert (
            canary.FIXED_CANARY_POLICY_ID
            == "anthropic_haiku_operator_pinned_2026_09_06_v1"
        )

    def test_episode_mutation_uses_the_exact_cell_code_without_stale_vocabulary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert "pinned_episode_count_changed" not in canary.FAILURE_DETAILS
        monkeypatch.setattr(canary, "CANARY_EPISODES", 2)

        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.check_pinned_controls()

        assert refusal.value.code == "pinned_exact_cell_envelope_changed"
        assert refusal.value.code in canary.FAILURE_DETAILS

    def test_a_direct_custom_policy_id_is_not_the_authorised_policy(self) -> None:
        policy = canary.LifecyclePricingPolicy(
            policy_id="caller_selected",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
            rate_source="operator_supplied_pinned_rates",
        )
        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.check_fixed_canary_pricing_envelope(policy, cap=Decimal("2.32"))
        assert refusal.value.code == canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED


class TestEveryExactAuthorizationTermIsLoadBearing:
    @pytest.mark.parametrize("name,mutated,derivatives", EXACT_CELL_MUTATIONS)
    def test_each_cell_term_refuses_with_literal_oracles(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output_dir: Path,
        no_construction: list[str],
        name: str,
        mutated: Any,
        derivatives: Mapping[str, Any],
    ) -> None:
        mapping_mutation = name == "LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION"
        if mapping_mutation:
            canary.check_pinned_controls()
            assert getattr(canary, name) != mutated
        monkeypatch.setattr(canary, name, mutated)
        for derivative, value in derivatives.items():
            monkeypatch.setattr(canary, derivative, value)
        if mapping_mutation:
            # Prove this mapping refuses independently of later launch gates.
            with pytest.raises(canary.CanaryRefusal) as mapping_refusal:
                canary.check_pinned_controls()
            assert mapping_refusal.value.code == "pinned_exact_cell_envelope_changed"
        environment = CredentialUnreadEnvironment(
            {
                canary.LIVE_AUTHORIZATION_VARIABLE: canary.LIVE_AUTHORIZATION_VALUE,
                canary.CREDENTIAL_VARIABLE: LIVE_KEY,
            }
        )
        monkeypatch.setattr(canary.os, "environ", environment)
        policy = canary.LifecyclePricingPolicy(
            policy_id="anthropic_haiku_operator_pinned_2026_09_06_v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
            rate_source="operator_supplied_pinned_rates",
        )
        with pytest.raises(canary.CanaryRefusal):
            canary.run_live_cell(directory=output_dir, policy=policy, cap=Decimal("2.32"))
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    @pytest.mark.parametrize("name,mutated", EXACT_PRICING_CONSTANT_MUTATIONS)
    def test_each_fixed_pricing_constant_refuses_even_when_policy_moves_with_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output_dir: Path,
        no_construction: list[str],
        name: str,
        mutated: Any,
    ) -> None:
        monkeypatch.setattr(canary, name, mutated)
        policy = canary.LifecyclePricingPolicy(
            policy_id="anthropic_haiku_operator_pinned_2026_09_06_v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
            rate_source="operator_supplied_pinned_rates",
        )
        object.__setattr__(policy, "policy_id", canary.FIXED_CANARY_POLICY_ID)
        object.__setattr__(policy, "rate_source", canary.FIXED_CANARY_RATE_SOURCE)
        object.__setattr__(
            policy,
            "input_usd_per_mtok",
            canary.FIXED_CANARY_INPUT_USD_PER_MTOK,
        )
        object.__setattr__(
            policy,
            "output_usd_per_mtok",
            canary.FIXED_CANARY_OUTPUT_USD_PER_MTOK,
        )
        assert (
            policy.policy_id,
            policy.rate_source,
            policy.input_usd_per_mtok,
            policy.output_usd_per_mtok,
        ) == (
            canary.FIXED_CANARY_POLICY_ID,
            canary.FIXED_CANARY_RATE_SOURCE,
            canary.FIXED_CANARY_INPUT_USD_PER_MTOK,
            canary.FIXED_CANARY_OUTPUT_USD_PER_MTOK,
        )
        environment = CredentialUnreadEnvironment(
            {
                canary.LIVE_AUTHORIZATION_VARIABLE: canary.LIVE_AUTHORIZATION_VALUE,
                canary.CREDENTIAL_VARIABLE: LIVE_KEY,
            }
        )
        monkeypatch.setattr(canary.os, "environ", environment)
        with pytest.raises(canary.CanaryRefusal):
            canary.run_live_cell(
                directory=output_dir,
                policy=policy,
                cap=canary.FIXED_CANARY_COST_CAP_USD,
            )
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    @pytest.mark.parametrize(
        "profile",
        [
            replace(
                canary.HAIKU_4_5_PROFILE,
                thinking={"type": "enabled", "budget_tokens": 1024},
            ),
            replace(canary.HAIKU_4_5_PROFILE, temperature=None),
            SimpleNamespace(
                profile_id="haiku45_temperature_zero_thinking_omitted_v1",
                temperature=0.0,
                thinking=None,
                omitted_fields=lambda: ("thinking",),
            ),
        ],
    )
    def test_each_request_profile_setting_is_independently_pinned(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output_dir: Path,
        no_construction: list[str],
        profile: Any,
    ) -> None:
        monkeypatch.setattr(canary, "HAIKU_4_5_PROFILE", profile)
        environment = CredentialUnreadEnvironment(
            {
                canary.LIVE_AUTHORIZATION_VARIABLE: canary.LIVE_AUTHORIZATION_VALUE,
                canary.CREDENTIAL_VARIABLE: LIVE_KEY,
            }
        )
        monkeypatch.setattr(canary.os, "environ", environment)
        with pytest.raises(canary.CanaryRefusal):
            canary.run_live_cell(
                directory=output_dir,
                policy=canary.LifecyclePricingPolicy(
                    policy_id="anthropic_haiku_operator_pinned_2026_09_06_v1",
                    input_usd_per_mtok=Decimal("1"),
                    output_usd_per_mtok=Decimal("5"),
                ),
                cap=Decimal("2.32"),
            )
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    def test_expected_calls_is_an_expectation_not_a_scientific_outcome_gate(self) -> None:
        source = Path(canary.__file__).read_text(encoding="utf-8")
        audit_body = source.split("def _audit_what_was_written", 1)[1].split(
            "def _scan_what_was_written", 1
        )[0]
        assert "EXPECTED_PROVIDER_CALLS" not in audit_body

    @pytest.mark.parametrize(
        "invalid", ["controls", "spec_version", "pricing", "directory_mode"]
    )
    def test_private_cell_execution_defensively_rechecks_its_pinned_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output_dir: Path,
        invalid: str,
    ) -> None:
        spec = canary.load_spec(canary.FIXTURE)
        policy = canary.LifecyclePricingPolicy(
            policy_id="anthropic_haiku_operator_pinned_2026_09_06_v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
        )
        directory = output_dir
        if invalid == "controls":
            monkeypatch.setattr(canary, "MAX_PROVIDER_CALLS", 51)
            monkeypatch.setattr(canary, "TOKEN_HARD_CAP", 1_516_995)
        elif invalid == "spec_version":
            object.__setattr__(spec, "operation_version", "another-version")
            object.__setattr__(spec, "spec_digest_sha256", spec.compute_digest())
        elif invalid == "pricing":
            object.__setattr__(policy, "policy_id", "caller-selected")
        else:
            directory = output_dir.parent / "loose"
            directory.mkdir(mode=0o755)
            directory.chmod(0o755)

        def dispatched(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("private defensive recheck reached dispatch")

        monkeypatch.setattr(canary, "check_client_endpoint", dispatched)
        monkeypatch.setattr(canary, "run_episode", dispatched)
        with pytest.raises(canary.CanaryRefusal):
            canary._execute_one_cell(
                spec,
                client=object(),
                directory=directory,
                policy=policy,
                cap=Decimal("2.32"),
                wire=canary.WireCaptureTransport(),
                started_at=0.0,
            )
        assert list(directory.iterdir()) == []


# -- the offline path ---------------------------------------------------------


class TestTheOfflinePreflight:
    @pytest.mark.parametrize("option,unauthorised", PRICING_ARGUMENT_MUTATIONS)
    def test_each_changed_rate_or_cap_refuses_before_offline_construction(
        self,
        output_dir: Path,
        no_construction: list[str],
        capsys: Any,
        option: str,
        unauthorised: str,
    ) -> None:
        arguments = offline_arguments(output_dir)
        arguments[arguments.index(option) + 1] = unauthorised

        assert canary.main(arguments) != 0

        summary = json.loads(capsys.readouterr().out)
        assert summary["failure"]["code"] == (
            canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED
        )
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    def test_a_changed_rate_source_refuses_before_offline_construction(
        self, output_dir: Path, no_construction: list[str]
    ) -> None:
        policy = canary.LifecyclePricingPolicy(
            policy_id=canary.FIXED_CANARY_POLICY_ID,
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
        )
        object.__setattr__(policy, "rate_source", "another_operator_source")

        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.run_offline_preflight(
                directory=output_dir,
                policy=policy,
                cap=Decimal("2.32"),
                environ={},
            )

        assert refusal.value.code == canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    def test_is_what_runs_when_nothing_asks_for_a_live_call(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        code = canary.main(
            ["--offline-preflight", "--output-dir", str(output_dir), *RATE_ARGUMENTS]
        )
        assert code == 0
        assert no_client == []
        summary = json.loads(capsys.readouterr().out)
        assert summary["ok"] is True
        assert summary["mode"] == "offline_preflight"
        assert summary["live_provider_call"] is False
        assert summary["credential_read"] is False

    def test_equivalent_decimal_spellings_keep_the_same_offline_envelope(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        code = canary.main(
            [
                "--offline-preflight",
                "--output-dir",
                str(output_dir),
                "--cost-cap-usd",
                "2.32",
                "--input-usd-per-mtok",
                "1.0",
                "--output-usd-per-mtok",
                "5.0",
            ]
        )

        assert code == 0
        assert no_client == []
        summary = json.loads(capsys.readouterr().out)
        assert summary["ok"] is True
        assert summary["mode"] == canary.MODE_OFFLINE
        assert summary["live_provider_call"] is False
        assert summary["budget"]["cost_cap_usd"] == "2.32"
        assert Decimal(summary["budget"]["worst_case_usd"]) == Decimal("2.31805")

    def test_writes_an_artefact_eight_a_ledger_and_an_audited_bundle(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        assert (
            canary.main(
                [
                    "--offline-preflight",
                    "--output-dir",
                    str(output_dir),
                    *RATE_ARGUMENTS,
                ]
            )
            == 0
        )
        summary = json.loads(capsys.readouterr().out)
        artifact_path = output_dir / summary["artifact"]
        ledger_path = output_dir / summary["ledger"]
        policy = canary.LifecyclePricingPolicy(
            policy_id=canary.FIXED_CANARY_POLICY_ID,
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
        )
        calculated_worst_case = policy.worst_case_usd(
            calls=canary.MAX_PROVIDER_CALLS,
            input_token_bound=canary.LARGEST_REQUEST_TOKEN_BOUND,
            max_output_tokens=canary.MAX_OUTPUT_TOKENS,
        )
        assert Decimal(summary["budget"]["worst_case_usd"]) == calculated_worst_case
        assert summary["budget"]["cost_cap_usd"] == "2.32"
        assert summary["budget"]["token_hard_cap"] == canary.TOKEN_HARD_CAP
        assert summary["artifact_version"] == 8
        assert read_artifact(artifact_path)["artifact_version"] == 8
        ledger = read_execution_ledger(ledger_path, require_complete=True)
        assert ledger.status == TERMINAL_SCORED
        assert ledger.totals.provider_calls == 46
        assert ledger.totals.attempts == 46
        assert ledger.header.controls.max_provider_calls == 50
        assert ledger.header.controls.expected_provider_calls == 46
        assert ledger.header.controls.max_attempts_per_call == 1
        assert ledger.header.controls.token_hard_cap == canary.MAX_PROVIDER_CALLS * (
            canary.LARGEST_REQUEST_TOKEN_BOUND + canary.MAX_OUTPUT_TOKENS
        )
        assert ledger.header.controls.wall_clock_deadline_seconds == 2700.0
        assert audit_execution_bundle_files(artifact_path, ledger_path).ok is True
        assert summary["bundle_audit"]["ok"] is True
        assert (
            summary["bundle_audit"]["execution_ledger_version"] == ledger.ledger_version
        )
        assert summary["ledger_audit"]["ledger_version"] == ledger.ledger_version == 3
        assert summary["replay"]["provider_calls"] == 0
        assert summary["replay"]["ok"] is True

    def test_reports_one_synthetic_cell_and_claims_no_score(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        canary.main(
            ["--offline-preflight", "--output-dir", str(output_dir), *RATE_ARGUMENTS]
        )
        summary = json.loads(capsys.readouterr().out)
        assert summary["n"] == 1
        assert summary["claim"] == canary.CANARY_CLAIM
        assert "execution-integrity" in canary.CANARY_CLAIM
        assert "not a model score" in canary.CANARY_CLAIM
        assert "leaderboard" in canary.CANARY_CLAIM
        assert "score" not in summary


# -- every gate, and nothing dispatched ---------------------------------------


class TestEveryGateRefusesBeforeAnythingIsConstructed:
    @pytest.mark.parametrize("option,unauthorised", PRICING_ARGUMENT_MUTATIONS)
    def test_each_changed_rate_or_cap_refuses_live_before_credential_read(
        self,
        output_dir: Path,
        no_construction: list[str],
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
        option: str,
        unauthorised: str,
    ) -> None:
        authorise(monkeypatch)
        arguments = live_arguments(output_dir)
        arguments[arguments.index(option) + 1] = unauthorised

        assert canary.main(arguments) != 0

        summary = json.loads(capsys.readouterr().out)
        assert summary["failure"]["code"] == (
            canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED
        )
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    @pytest.mark.parametrize("option,unauthorised", PRICING_ARGUMENT_MUTATIONS)
    def test_each_changed_rate_or_cap_reaches_no_live_credential_read(
        self,
        output_dir: Path,
        no_construction: list[str],
        monkeypatch: pytest.MonkeyPatch,
        option: str,
        unauthorised: str,
    ) -> None:
        policy = canary.LifecyclePricingPolicy(
            policy_id=canary.FIXED_CANARY_POLICY_ID,
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
        )
        cap = Decimal("2.32")
        if option == "--input-usd-per-mtok":
            object.__setattr__(policy, "input_usd_per_mtok", Decimal(unauthorised))
        elif option == "--output-usd-per-mtok":
            object.__setattr__(policy, "output_usd_per_mtok", Decimal(unauthorised))
        else:
            cap = Decimal(unauthorised)
        environment = CredentialUnreadEnvironment(
            {
                canary.LIVE_AUTHORIZATION_VARIABLE: canary.LIVE_AUTHORIZATION_VALUE,
                canary.CREDENTIAL_VARIABLE: LIVE_KEY,
            }
        )
        monkeypatch.setattr(canary.os, "environ", environment)

        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.run_live_cell(directory=output_dir, policy=policy, cap=cap)

        assert refusal.value.code == canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    def test_a_changed_rate_source_refuses_live_before_credential_read(
        self,
        output_dir: Path,
        no_construction: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        policy = canary.LifecyclePricingPolicy(
            policy_id=canary.FIXED_CANARY_POLICY_ID,
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("5"),
        )
        object.__setattr__(policy, "rate_source", "another_operator_source")
        environment = CredentialUnreadEnvironment(
            {
                canary.LIVE_AUTHORIZATION_VARIABLE: canary.LIVE_AUTHORIZATION_VALUE,
                canary.CREDENTIAL_VARIABLE: LIVE_KEY,
            }
        )
        monkeypatch.setattr(canary.os, "environ", environment)

        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.run_live_cell(directory=output_dir, policy=policy, cap=Decimal("2.32"))

        assert refusal.value.code == canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    @pytest.mark.parametrize(
        "name,mutated,formula_derived_cap", AUTHORISED_ENVELOPE_MUTATIONS
    )
    def test_exact_authorization_refuses_each_independently_changed_envelope_term(
        self,
        output_dir: Path,
        no_construction: list[str],
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        mutated: int | float,
        formula_derived_cap: int | None,
    ) -> None:
        environment = CredentialUnreadEnvironment(
            {
                canary.LIVE_AUTHORIZATION_VARIABLE: (
                    "anthropic:claude-haiku-4-5-20251001:V1:1:max-calls-50"
                ),
                canary.CREDENTIAL_VARIABLE: LIVE_KEY,
            }
        )
        monkeypatch.setattr(canary.os, "environ", environment)
        monkeypatch.setattr(canary, name, mutated)
        if formula_derived_cap is not None:
            monkeypatch.setattr(canary, "TOKEN_HARD_CAP", formula_derived_cap)

        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary._run_live_cell(
                directory=output_dir,
                policy=canary.LifecyclePricingPolicy(
                    policy_id=canary.FIXED_CANARY_POLICY_ID,
                    input_usd_per_mtok=Decimal("1"),
                    output_usd_per_mtok=Decimal("5"),
                ),
                cap=Decimal("2.32"),
            )

        assert refusal.value.code in {
            "pinned_call_bound_changed",
            "pinned_token_ceiling_changed",
            "pinned_wall_clock_deadline_changed",
        }
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    def test_the_stale_sixty_call_authorization_is_refused_before_credential_read(
        self,
    ) -> None:
        class Environment(Mapping[str, str]):
            def __getitem__(self, key: str) -> str:
                if key == canary.CREDENTIAL_VARIABLE:
                    raise AssertionError("the credential was read before authorization")
                if key == canary.LIVE_AUTHORIZATION_VARIABLE:
                    return "anthropic:claude-haiku-4-5-20251001:V1:1"
                raise KeyError(key)

            def __iter__(self) -> Iterator[str]:
                yield canary.LIVE_AUTHORIZATION_VARIABLE

            def __len__(self) -> int:
                return 1

        environment = Environment()
        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.check_live_authorization(environment)

        assert refusal.value.code == "live_authorization_absent_or_wrong"

    def test_the_fifty_call_authorization_is_accepted(self) -> None:
        canary.check_live_authorization(
            {
                canary.LIVE_AUTHORIZATION_VARIABLE: (
                    "anthropic:claude-haiku-4-5-20251001:V1:1:max-calls-50"
                ),
                canary.CREDENTIAL_VARIABLE: LIVE_KEY,
            }
        )

    def test_live_without_the_authorization_variable(
        self, output_dir: Path, no_client: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        assert (
            canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS]) != 0
        )
        assert no_client == []

    def test_live_with_the_wrong_authorization_value(
        self, output_dir: Path, no_client: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        for wrong in (
            "anthropic:claude-haiku-4-5-20251001:V1:1",
            "anthropic:claude-haiku-4-5-20251001:V1:1:max-calls-90",
            "anthropic:claude-haiku-4-5-20251001:V2:1",
            "anthropic:claude-haiku-4-5-20251001:V1:2",
            "openai:gpt-4:V1:1",
            "yes",
            "",
        ):
            monkeypatch.setenv(canary.LIVE_AUTHORIZATION_VARIABLE, wrong)
            assert (
                canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS])
                != 0
            )
        assert no_client == []

    def test_live_without_a_credential(
        self, output_dir: Path, no_client: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        assert (
            canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS]) != 0
        )
        assert no_client == []
        assert sorted(path.name for path in output_dir.iterdir()) == sorted(
            PERSISTENT_MARKER_NAMES
        )
        assert_exact_persistent_markers(output_dir)
        before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        assert canary.main(live_arguments(output_dir)) != 0
        assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before

    @pytest.mark.parametrize("name", sorted(canary.REJECTED_VARIABLES))
    def test_live_beside_an_endpoint_or_tenancy_override(
        self,
        output_dir: Path,
        no_client: list[str],
        monkeypatch: pytest.MonkeyPatch,
        name: str,
    ) -> None:
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        monkeypatch.setenv(name, "https://elsewhere.invalid/v1")
        assert (
            canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS]) != 0
        )
        assert no_client == []

    def test_live_without_operator_rates_or_a_cap(
        self, output_dir: Path, no_client: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        for incomplete in (
            [],
            ["--cost-cap-usd", "5.00"],
            ["--input-usd-per-mtok", "1.25", "--output-usd-per-mtok", "10.00"],
        ):
            with pytest.raises(SystemExit) as exit_code:
                canary.main(["--live", "--output-dir", str(output_dir), *incomplete])
            assert exit_code.value.code != 0
        assert no_client == []

    def test_live_whose_worst_case_does_not_fit_the_cap(
        self, output_dir: Path, no_client: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        assert (
            canary.main(
                [
                    "--live",
                    "--output-dir",
                    str(output_dir),
                    "--cost-cap-usd",
                    "2.30",
                    "--input-usd-per-mtok",
                    "1",
                    "--output-usd-per-mtok",
                    "5",
                ]
            )
            != 0
        )
        assert no_client == []

    def test_live_whose_output_directory_is_missing_or_world_readable(
        self, tmp_path: Path, no_client: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        loose = tmp_path / "loose"
        loose.mkdir(mode=0o755)
        for directory in (tmp_path / "absent", loose):
            assert (
                canary.main(["--live", "--output-dir", str(directory), *RATE_ARGUMENTS])
                != 0
            )
        assert no_client == []

    def test_a_refused_run_leaves_the_output_directory_untouched(
        self, output_dir: Path, no_client: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS])
        assert list(output_dir.iterdir()) == []
        assert no_client == []


# -- the authorised path, over a mock ----------------------------------------


class TestTheAuthorisedPathOverAMock:
    def _run(self, output_dir: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[int, Wire]:
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        wire = Wire()
        monkeypatch.setattr(canary, "open_live_transport", mock_live_transport(wire))
        code = canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS])
        return code, wire

    def test_runs_exactly_one_cell_and_writes_its_evidence(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        code, wire = self._run(output_dir, monkeypatch)
        assert code == 0
        assert len(wire.bodies) == canary.EXPECTED_PROVIDER_CALLS
        summary = json.loads(capsys.readouterr().out)
        assert summary["mode"] == "live"
        assert summary["n"] == 1
        assert summary["episodes"] == 1
        assert summary["provider_calls"] == canary.EXPECTED_PROVIDER_CALLS
        assert summary["bundle_audit"]["ok"] is True
        assert summary["replay"]["provider_calls"] == 0
        assert sorted(
            path.name for path in output_dir.iterdir()
        ) == expected_output_entries(*canary.OUTPUT_NAMES)
        assert_exact_persistent_markers(output_dir)
        expected_tools = anthropic_outcome_tools()
        for body in wire.bodies:
            request = json.loads(body)
            assert (
                body
                == json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode()
            )
            assert request["tools"] == expected_tools
            for tool in request["tools"]:
                schema = tool["input_schema"]
                assert schema["type"] == "object"
                assert not ({"oneOf", "anyOf", "allOf"} & schema.keys())
            assert not any("cache" in key.lower() for key in request)

    def test_one_byte_short_marker_write_completes_before_live_construction(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
    ) -> None:
        """Reproduce the reviewer probe through the real SDK and public entrypoint."""
        authorise(monkeypatch)
        environ = CredentialReadCountingEnvironment(os.environ)
        monkeypatch.setattr(canary.os, "environ", environ)
        wire = Wire()
        construction: list[str] = []
        real_open_client = canary.open_live_client

        def transport() -> httpx.BaseTransport:
            construction.append("transport")
            return mock_live_transport(wire)()

        def client(api_key: str, *, transport: Any) -> Any:
            construction.append("client")
            return real_open_client(api_key, transport=transport)

        real_write = os.write
        marker_writes = 0

        def one_byte_short_first_write(fd: int, data: Any) -> int:
            nonlocal marker_writes
            if marker_writes == 0:
                marker_writes += 1
                return real_write(fd, data[:-1])
            marker_writes += 1
            return real_write(fd, data)

        monkeypatch.setattr(canary, "open_live_transport", transport)
        monkeypatch.setattr(canary, "open_live_client", client)
        monkeypatch.setattr(os, "write", one_byte_short_first_write)

        code = canary.main(live_arguments(output_dir))

        assert code == 0, capsys.readouterr().out
        assert environ.credential_reads == 1
        assert construction == ["transport", "client"]
        assert len(wire.bodies) == canary.EXPECTED_PROVIDER_CALLS
        assert marker_writes > 1
        assert_exact_persistent_markers(output_dir)

    @pytest.mark.parametrize(
        "failure_at", ["zero", "after_prefix", "marker_fsync", "fstat", "dir_fsync"]
    )
    def test_marker_creation_failure_refuses_before_credential_client_or_dispatch(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
        failure_at: str,
    ) -> None:
        authorise(monkeypatch)
        environ = CredentialReadCountingEnvironment(os.environ)
        monkeypatch.setattr(canary.os, "environ", environ)
        reached: list[str] = []
        unlink_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def reached_trap(label: str) -> Any:
            def trap(*args: Any, **kwargs: Any) -> Any:
                reached.append(label)
                raise AssertionError(f"{label} reached after marker failure")

            return trap

        real_write = os.write
        real_fsync = os.fsync
        real_fstat = os.fstat
        write_calls = 0

        def failing_write(fd: int, data: Any) -> int:
            nonlocal write_calls
            write_calls += 1
            if failure_at == "zero":
                return 0
            if failure_at == "after_prefix" and write_calls == 1:
                return real_write(fd, data[:11])
            if failure_at == "after_prefix":
                raise OSError(errno.EIO, "untrusted public write failure")
            return real_write(fd, data)

        def failing_fsync(fd: int) -> None:
            info = real_fstat(fd)
            if failure_at == "marker_fsync" and stat.S_ISREG(info.st_mode):
                raise OSError(errno.EIO, "untrusted public fsync failure")
            if failure_at == "dir_fsync" and stat.S_ISDIR(info.st_mode):
                raise OSError(errno.EIO, "untrusted public directory fsync failure")
            real_fsync(fd)

        def failing_fstat(fd: int) -> os.stat_result:
            info = real_fstat(fd)
            if failure_at == "fstat" and stat.S_ISREG(info.st_mode):
                raise OSError(errno.EIO, "untrusted public fstat failure")
            return info

        def forbidden_unlink(*args: Any, **kwargs: Any) -> None:
            unlink_attempts.append((args, kwargs))
            raise AssertionError("permanent residue unlink attempted")

        monkeypatch.setattr(canary, "open_live_transport", reached_trap("transport"))
        monkeypatch.setattr(canary, "open_live_client", reached_trap("client"))
        monkeypatch.setattr(canary, "_execute_one_cell", reached_trap("dispatch"))
        monkeypatch.setattr(os, "write", failing_write)
        monkeypatch.setattr(os, "fsync", failing_fsync)
        monkeypatch.setattr(os, "fstat", failing_fstat)
        monkeypatch.setattr(os, "unlink", forbidden_unlink)

        assert canary.main(live_arguments(output_dir)) != 0

        failure = json.loads(capsys.readouterr().out)["failure"]
        assert failure["code"] == "output_not_reservable"
        assert failure["detail"] == canary.FAILURE_DETAILS["output_not_reservable"]
        assert environ.credential_reads == 0
        assert reached == []
        assert unlink_attempts == []
        expected = {
            "zero": b"",
            "after_prefix": PERSISTENT_MARKER_BYTES[:11],
            "marker_fsync": PERSISTENT_MARKER_BYTES,
            "fstat": PERSISTENT_MARKER_BYTES,
            "dir_fsync": PERSISTENT_MARKER_BYTES,
        }[failure_at]
        assert (output_dir / PERSISTENT_MARKER_NAMES[0]).read_bytes() == expected
        assert sorted(path.name for path in output_dir.iterdir()) == [
            PERSISTENT_MARKER_NAMES[0]
        ]

    def test_directory_swap_after_reservation_refuses_before_client_dispatch(
        self,
        output_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
    ) -> None:
        moved = tmp_path / "reserved-inode"
        attacker = b"attacker-owned replacement directory\n"
        original_reserve = canary.reserve_output_names

        @canary.contextmanager
        def swap_after_reservation(directory: Path) -> Iterator[Any]:
            with original_reserve(directory) as reservation:
                directory.rename(moved)
                directory.mkdir(mode=0o700)
                (directory / "attacker.txt").write_bytes(attacker)
                yield reservation

        monkeypatch.setattr(canary, "reserve_output_names", swap_after_reservation)
        code, wire = self._run(output_dir, monkeypatch)

        assert code != 0
        assert wire.bodies == []
        assert (output_dir / "attacker.txt").read_bytes() == attacker
        assert sorted(path.name for path in output_dir.iterdir()) == ["attacker.txt"]
        assert not (output_dir / canary.LEDGER_NAME).exists()
        assert not (output_dir / canary.ARTIFACT_NAME).exists()
        assert sorted(path.name for path in moved.iterdir()) == sorted(
            PERSISTENT_MARKER_NAMES
        )
        assert_exact_persistent_markers(moved)
        failure = json.loads(capsys.readouterr().out)["failure"]
        assert failure["code"] == "output_directory_identity_changed"

    def test_directory_swap_after_client_construction_refuses_before_ledger_or_dispatch(
        self,
        output_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
    ) -> None:
        moved = tmp_path / "reserved-after-client"
        attacker = b"attacker stays here\n"
        real_open_client = canary.open_live_client

        def swapping_open_client(api_key: str, *, transport: Any) -> Any:
            client = real_open_client(api_key, transport=transport)
            output_dir.rename(moved)
            output_dir.mkdir(mode=0o700)
            (output_dir / "attacker.txt").write_bytes(attacker)
            return client

        monkeypatch.setattr(canary, "open_live_client", swapping_open_client)
        code, wire = self._run(output_dir, monkeypatch)

        assert code != 0
        assert wire.bodies == []
        assert sorted(path.name for path in moved.iterdir()) == sorted(
            PERSISTENT_MARKER_NAMES
        )
        assert_exact_persistent_markers(moved)
        assert (output_dir / "attacker.txt").read_bytes() == attacker
        assert sorted(path.name for path in output_dir.iterdir()) == ["attacker.txt"]
        assert canary.CREDENTIAL_VARIABLE not in os.environ
        assert canary.LIVE_AUTHORIZATION_VARIABLE not in os.environ
        assert json.loads(capsys.readouterr().out)["failure"]["code"] == (
            "output_directory_identity_changed"
        )

    def test_directory_swap_during_provider_calls_keeps_all_evidence_anchored(
        self,
        output_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
    ) -> None:
        authorise(monkeypatch)
        moved = tmp_path / "reserved-mid-dispatch"
        attacker = b"attacker replacement bytes\n"
        wire = Wire()

        class SwappingProvider(canary.ScriptedProvider):
            def __call__(self, request: httpx.Request) -> httpx.Response:
                response = super().__call__(request)
                if len(self.bodies) == 4:
                    output_dir.rename(moved)
                    output_dir.mkdir(mode=0o700)
                    (output_dir / "attacker.txt").write_bytes(attacker)
                return response

        def transport() -> httpx.BaseTransport:
            wire.provider = SwappingProvider(max_calls=canary.MAX_PROVIDER_CALLS)
            return httpx.MockTransport(wire.provider)

        monkeypatch.setattr(canary, "open_live_transport", transport)
        assert canary.main(live_arguments(output_dir)) != 0

        assert len(wire.bodies) == canary.EXPECTED_PROVIDER_CALLS
        assert (output_dir / "attacker.txt").read_bytes() == attacker
        assert sorted(path.name for path in output_dir.iterdir()) == ["attacker.txt"]
        assert sorted(path.name for path in moved.iterdir()) == expected_output_entries(
            *canary.OUTPUT_NAMES
        )
        assert_exact_persistent_markers(moved)
        dir_fd = os.open(moved, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            ledger = read_execution_ledger(
                canary.LEDGER_NAME, require_complete=True, dir_fd=dir_fd
            )
            artifact = read_artifact(canary.ARTIFACT_NAME, dir_fd=dir_fd)
            bundle = audit_execution_bundle_files(
                canary.ARTIFACT_NAME,
                canary.LEDGER_NAME,
                artifact_dir_fd=dir_fd,
                ledger_dir_fd=dir_fd,
            )
        finally:
            os.close(dir_fd)
        assert ledger.totals.provider_calls == canary.EXPECTED_PROVIDER_CALLS
        assert artifact["artifact_version"] == ARTIFACT_VERSION
        assert bundle.ok is True
        assert json.loads(capsys.readouterr().out)["failure"]["code"] == (
            "output_directory_identity_changed"
        )

    @pytest.mark.parametrize(
        "seam", ["artifact", "final_namespace", "cleanup_final_namespace"]
    )
    def test_late_directory_swaps_never_report_success_or_touch_replacement(
        self,
        output_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
        seam: str,
    ) -> None:
        moved = tmp_path / f"reserved-{seam}"
        attacker = b"late attacker bytes\n"
        swapped = False

        def swap() -> None:
            nonlocal swapped
            if swapped:
                return
            swapped = True
            output_dir.rename(moved)
            output_dir.mkdir(mode=0o700)
            (output_dir / "attacker.txt").write_bytes(attacker)

        if seam == "artifact":
            real_write = canary.write_artifact

            def swapping_write(*args: Any, **kwargs: Any) -> Path:
                swap()
                return real_write(*args, **kwargs)

            monkeypatch.setattr(canary, "write_artifact", swapping_write)
        else:
            real_require = canary.OutputReservation.require_named_directory
            calls = 0

            def swapping_final_check(
                reservation: canary.OutputReservation, *, after_dispatch: bool = False
            ) -> None:
                nonlocal calls
                calls += 1
                if after_dispatch and (
                    seam == "final_namespace"
                    or (seam == "cleanup_final_namespace" and calls == 5)
                ):
                    swap()
                real_require(reservation, after_dispatch=after_dispatch)

            monkeypatch.setattr(
                canary.OutputReservation,
                "require_named_directory",
                swapping_final_check,
            )

        code, wire = self._run(output_dir, monkeypatch)

        assert code != 0
        assert len(wire.bodies) == canary.EXPECTED_PROVIDER_CALLS
        assert swapped
        assert (output_dir / "attacker.txt").read_bytes() == attacker
        assert sorted(path.name for path in output_dir.iterdir()) == ["attacker.txt"]
        assert sorted(path.name for path in moved.iterdir()) == expected_output_entries(
            *canary.OUTPUT_NAMES
        )
        assert_exact_persistent_markers(moved)
        audit = read_execution_ledger(moved / canary.LEDGER_NAME, require_complete=True)
        assert audit.totals.provider_calls == canary.EXPECTED_PROVIDER_CALLS
        assert audit_execution_bundle_files(
            moved / canary.ARTIFACT_NAME, moved / canary.LEDGER_NAME
        ).ok
        assert json.loads(capsys.readouterr().out)["failure"]["code"] == (
            "output_directory_identity_changed"
        )

    @pytest.mark.parametrize(
        "cache_field", ["cache_creation_input_tokens", "cache_read_input_tokens"]
    )
    def test_nonzero_cache_usage_fails_before_measured_settlement_or_safe_success(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
        cache_field: str,
    ) -> None:
        authorise(monkeypatch)
        wire = Wire()

        class CacheReportingProvider(canary.ScriptedProvider):
            def __call__(self, request: httpx.Request) -> httpx.Response:
                response = super().__call__(request)
                body = response.json()
                body["usage"][cache_field] = 1
                return httpx.Response(200, json=body, request=request)

        def transport() -> httpx.BaseTransport:
            wire.provider = CacheReportingProvider(max_calls=canary.MAX_PROVIDER_CALLS)
            return httpx.MockTransport(wire.provider)

        monkeypatch.setattr(canary, "open_live_transport", transport)
        assert canary.main(live_arguments(output_dir)) != 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["ok"] is False
        assert summary["failure"]["code"] == "provider_boundary_fault"
        assert sorted(
            path.name for path in output_dir.iterdir()
        ) == expected_output_entries(canary.LEDGER_NAME)
        ledger = read_execution_ledger(output_dir / canary.LEDGER_NAME)
        attempt = ledger.calls[0].attempts[0]
        assert attempt.outcome == "fault"
        assert attempt.fault == "provider_response_invalid"
        assert attempt.cost_measured_usd is None
        assert attempt.cost_settlement == "forfeited"

    def test_removes_the_authorization_and_the_credential_from_the_environment(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        code, _ = self._run(output_dir, monkeypatch)
        assert code == 0
        assert canary.CREDENTIAL_VARIABLE not in os.environ
        assert canary.LIVE_AUTHORIZATION_VARIABLE not in os.environ

    def test_clears_the_environment_even_when_the_cell_fails(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)

        def exploding() -> Any:
            raise RuntimeError("the transport failed after the credential was read")

        monkeypatch.setattr(canary, "open_live_transport", exploding)
        assert (
            canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS]) != 0
        )
        assert canary.CREDENTIAL_VARIABLE not in os.environ
        assert canary.LIVE_AUTHORIZATION_VARIABLE not in os.environ

    def test_never_writes_the_credential_to_any_output(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        self._run(output_dir, monkeypatch)
        printed = capsys.readouterr()
        assert LIVE_KEY not in printed.out
        assert LIVE_KEY not in printed.err
        for path in output_dir.iterdir():
            text = path.read_text(encoding="utf-8")
            assert LIVE_KEY not in text
            assert FAKE_API_KEY not in text
            assert "uthoriz" not in text

    def test_a_cell_past_its_wall_clock_is_abandoned_before_its_next_turn(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound is checked before a turn, not reported after the run."""
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        wire = Wire()
        monkeypatch.setattr(canary, "open_live_transport", mock_live_transport(wire))
        # The overrun is simulated at the seam that computes the run's deadline
        # instant, not by moving the pinned bound: the bound itself is one of
        # this build's fixed controls, and a run under a different one is
        # refused before a client exists (see TestThePinnedDeadlinesAreFailClosed).
        monkeypatch.setattr(
            canary, "wall_clock_deadline_at", lambda started_at: started_at
        )
        assert (
            canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS]) != 0
        )
        assert wire.bodies == []
        assert sorted(p.name for p in output_dir.iterdir()) == expected_output_entries(
            canary.LEDGER_NAME
        )
        audit = read_execution_ledger(output_dir / canary.LEDGER_NAME)
        assert audit.scored is False

    def test_a_faulted_cell_keeps_the_ledger_writes_no_artefact_and_exits_nonzero(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv(
            canary.LIVE_AUTHORIZATION_VARIABLE, canary.LIVE_AUTHORIZATION_VALUE
        )
        monkeypatch.setenv(canary.CREDENTIAL_VARIABLE, LIVE_KEY)
        wire = Wire()
        monkeypatch.setattr(
            canary,
            "open_live_transport",
            mock_live_transport(
                wire,
                faults={
                    4: httpx.Response(
                        400,
                        json={
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": (
                                    "Your organization has reached its spend limit."
                                ),
                            },
                        },
                        headers={
                            "request-id": "req_synthetic_canary_fault_29ac",
                            "x-synthetic-secret": "-".join(
                                ("synthetic", "secret", "canary", "29ac")
                            ),
                        },
                    )
                },
            ),
        )
        assert (
            canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS]) != 0
        )
        written = sorted(path.name for path in output_dir.iterdir())
        assert written == expected_output_entries(canary.LEDGER_NAME)
        audit = read_execution_ledger(output_dir / canary.LEDGER_NAME)
        assert audit.status == "excluded"
        assert audit.scored is False
        attempt = audit.calls[-1].attempts[-1]
        assert attempt.fault == "provider_spend_limit"
        assert attempt.http_status == 400
        assert attempt.response_received is True
        assert attempt.usage_reported is False
        assert attempt.response_model is None
        assert attempt.response_id_digest_sha256 is None
        assert attempt.response_stop_classification is None
        assert attempt.response_normalized_digest_sha256 is None
        summary_text = capsys.readouterr().out
        summary = json.loads(summary_text)
        assert summary["failure"]["code"] == "provider_boundary_fault"
        assert "score" not in summary
        assert "evaluator" not in summary
        assert "provider_spend_limit" not in summary_text
        assert "req_synthetic_canary_fault_29ac" not in summary_text
        assert "synthetic-secret-canary-29ac" not in summary_text


# -- the pinned deadlines are controls, not defaults --------------------------


class TestThePinnedDeadlinesAreFailClosed:
    """Both deadlines are fixed controls, and a build that moved one is refused.

    A per-turn deadline and a wall-clock bound are what make "fifty calls" a
    bound on *time* as well as on count, and they are two of the figures the
    ledger records as this run's controls. A build whose deadlines are not the
    pinned ones is not the build the authorisation was granted for, so it stops
    in :func:`check_pinned_controls` — which runs before a transport, a client
    or a request exists, on both paths.
    """

    @pytest.mark.parametrize("name,mutated", DEADLINE_MUTATIONS)
    def test_the_self_check_refuses_each_deadline_independently(
        self, monkeypatch: pytest.MonkeyPatch, name: str, mutated: float
    ) -> None:
        monkeypatch.setattr(canary, name, mutated)
        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.check_pinned_controls()
        assert refusal.value.code in canary.FAILURE_DETAILS

    @pytest.mark.parametrize("name,mutated", DEADLINE_MUTATIONS)
    def test_an_offline_preflight_under_a_moved_deadline_constructs_nothing(
        self,
        output_dir: Path,
        no_construction: list[str],
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        mutated: float,
    ) -> None:
        monkeypatch.setattr(canary, name, mutated)
        assert canary.main(offline_arguments(output_dir)) != 0
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    @pytest.mark.parametrize("name,mutated", DEADLINE_MUTATIONS)
    def test_an_authorised_live_run_under_a_moved_deadline_constructs_nothing(
        self,
        output_dir: Path,
        no_construction: list[str],
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        mutated: float,
    ) -> None:
        authorise(monkeypatch)
        monkeypatch.setattr(canary, name, mutated)
        assert canary.main(live_arguments(output_dir)) != 0
        assert no_construction == []
        assert list(output_dir.iterdir()) == []
        assert canary.CREDENTIAL_VARIABLE not in os.environ
        assert canary.LIVE_AUTHORIZATION_VARIABLE not in os.environ

    def test_the_pinned_deadlines_are_still_the_ones_the_ledger_records(self) -> None:
        assert canary.TURN_DEADLINE_SECONDS == 30.0
        assert canary.WALL_CLOCK_DEADLINE_SECONDS == 2700.0


# -- every output name is reserved before anything is dispatched --------------


class TestEveryOutputNameIsReservedBeforeAnythingIsDispatched:
    """Output that cannot be written is discovered before, not after, spending.

    An exclusive create at the end of a run is what stops two runs overwriting
    one another's evidence, but on its own it discovers the collision after
    every provider call has been paid for. Every name a run may write — whether
    it succeeds or faults — is therefore reserved up front, against the
    directory's own descriptor, before a client exists.

    Each reservation is a permanent, one-shot claim under a name no reader could
    mistake for evidence. It records only a local attempted claim of the output
    name; it does not authenticate who approved the run, the source revision, or
    any external controlled record. If local marker creation starts but cannot
    complete, the partial permanent residue is not a valid marker or evidence;
    it is retained and consumes the one-shot output root.
    """

    @pytest.mark.parametrize("interrupt", ["exception", "errno"])
    def test_interrupted_marker_write_retries_to_exact_completion(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        interrupt: str,
    ) -> None:
        real_write = os.write
        calls = 0

        def interrupted_once(fd: int, data: Any) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                if interrupt == "exception":
                    raise InterruptedError("untrusted interrupted prose")
                raise OSError(errno.EINTR, "untrusted EINTR prose")
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", interrupted_once)
        dirfd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        markerfd: int | None = None
        try:
            markerfd = canary._take_reservation(dirfd, canary.LEDGER_NAME)
        finally:
            if markerfd is not None:
                os.close(markerfd)
            os.close(dirfd)

        assert calls == 2
        marker = output_dir / PERSISTENT_MARKER_NAMES[0]
        assert marker.read_bytes() == PERSISTENT_MARKER_BYTES
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600

    def test_multiple_short_marker_writes_offer_only_the_remaining_bytes(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_write = os.write
        limits = iter((1, 7, 31, len(PERSISTENT_MARKER_BYTES)))
        offered: list[int] = []

        def repeatedly_short(fd: int, data: Any) -> int:
            offered.append(len(data))
            count = min(next(limits), len(data))
            return real_write(fd, data[:count])

        monkeypatch.setattr(os, "write", repeatedly_short)
        dirfd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        markerfd: int | None = None
        try:
            markerfd = canary._take_reservation(dirfd, canary.LEDGER_NAME)
        finally:
            if markerfd is not None:
                os.close(markerfd)
            os.close(dirfd)

        assert offered == [167, 166, 159, 128]
        assert (output_dir / PERSISTENT_MARKER_NAMES[0]).read_bytes() == (
            PERSISTENT_MARKER_BYTES
        )

    def test_marker_creation_orders_exact_write_validation_and_durability(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        real_write = os.write
        real_fsync = os.fsync
        real_fstat = os.fstat
        real_lseek = os.lseek
        real_read = os.read
        markerfd: int | None = None
        dirfd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)

        def recording_write(fd: int, data: Any) -> int:
            events.append(f"write:{len(data)}")
            count = min(50, len(data))
            return real_write(fd, data[:count])

        def recording_fsync(fd: int) -> None:
            events.append("dir_fsync" if fd == dirfd else "marker_fsync")
            real_fsync(fd)

        def recording_fstat(fd: int) -> os.stat_result:
            if fd != dirfd:
                events.append("fstat")
            return real_fstat(fd)

        def recording_lseek(fd: int, offset: int, whence: int) -> int:
            events.append("lseek")
            return real_lseek(fd, offset, whence)

        def recording_read(fd: int, count: int) -> bytes:
            events.append(f"read:{count}")
            return real_read(fd, count)

        monkeypatch.setattr(os, "write", recording_write)
        monkeypatch.setattr(os, "fsync", recording_fsync)
        monkeypatch.setattr(os, "fstat", recording_fstat)
        monkeypatch.setattr(os, "lseek", recording_lseek)
        monkeypatch.setattr(os, "read", recording_read)
        try:
            markerfd = canary._take_reservation(dirfd, canary.LEDGER_NAME)
        finally:
            if markerfd is not None:
                os.close(markerfd)
            os.close(dirfd)

        assert events == [
            "write:167",
            "write:117",
            "write:67",
            "write:17",
            "marker_fsync",
            "fstat",
            "lseek",
            "read:167",
            "read:1",
            "dir_fsync",
        ]
        marker = output_dir / PERSISTENT_MARKER_NAMES[0]
        assert marker.read_bytes() == PERSISTENT_MARKER_BYTES
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600

    @pytest.mark.parametrize(
        "failure", ["zero", "after_prefix", "marker_fsync", "dir_fsync"]
    )
    def test_unrecoverable_marker_creation_closes_once_and_preserves_residue(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: str,
    ) -> None:
        real_open = os.open
        real_write = os.write
        real_fsync = os.fsync
        real_close = os.close
        markerfd: int | None = None
        marker_close_attempts = 0
        write_calls = 0
        unlink_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def recording_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            nonlocal markerfd
            fd = real_open(path, flags, *args, **kwargs)
            if path == PERSISTENT_MARKER_NAMES[0]:
                markerfd = fd
            return fd

        def failing_write(fd: int, data: Any) -> int:
            nonlocal write_calls
            write_calls += 1
            if failure == "zero":
                return 0
            if failure == "after_prefix" and write_calls == 1:
                return real_write(fd, data[:13])
            if failure == "after_prefix":
                raise OSError(errno.EIO, "untrusted write failure prose")
            return real_write(fd, data)

        def failing_fsync(fd: int) -> None:
            if failure == "marker_fsync" and fd == markerfd:
                raise OSError(errno.EIO, "untrusted fsync failure prose")
            if failure == "dir_fsync" and fd == dirfd:
                raise OSError(errno.EIO, "untrusted directory fsync failure prose")
            real_fsync(fd)

        def recording_close(fd: int) -> None:
            nonlocal marker_close_attempts
            if fd == markerfd:
                marker_close_attempts += 1
            real_close(fd)

        def forbidden_unlink(*args: Any, **kwargs: Any) -> None:
            unlink_attempts.append((args, kwargs))
            raise AssertionError("permanent residue unlink attempted")

        monkeypatch.setattr(os, "open", recording_open)
        monkeypatch.setattr(os, "write", failing_write)
        monkeypatch.setattr(os, "fsync", failing_fsync)
        monkeypatch.setattr(os, "close", recording_close)
        monkeypatch.setattr(os, "unlink", forbidden_unlink)
        dirfd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        descriptors_before = len(list(Path("/proc/self/fd").iterdir()))
        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary._take_reservation(dirfd, canary.LEDGER_NAME)
        descriptors_after = len(list(Path("/proc/self/fd").iterdir()))
        os.close(dirfd)

        expected = {
            "zero": b"",
            "after_prefix": PERSISTENT_MARKER_BYTES[:13],
            "marker_fsync": PERSISTENT_MARKER_BYTES,
            "dir_fsync": PERSISTENT_MARKER_BYTES,
        }[failure]
        assert refusal.value.code == "output_not_reservable"
        assert marker_close_attempts == 1
        assert descriptors_after == descriptors_before
        assert unlink_attempts == []
        assert (output_dir / PERSISTENT_MARKER_NAMES[0]).read_bytes() == expected

    def test_exact_byte_validation_refuses_same_size_mutation_and_never_unlinks(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_fsync = os.fsync
        mutated = False
        unlink_attempts: list[Any] = []

        def mutate_after_marker_fsync(fd: int) -> None:
            nonlocal mutated
            real_fsync(fd)
            info = os.fstat(fd)
            if not mutated and stat.S_ISREG(info.st_mode):
                os.pwrite(fd, b"X", 0)
                mutated = True

        def forbidden_unlink(*args: Any, **kwargs: Any) -> None:
            unlink_attempts.append((args, kwargs))
            raise AssertionError("mutated permanent residue unlink attempted")

        monkeypatch.setattr(os, "fsync", mutate_after_marker_fsync)
        monkeypatch.setattr(os, "unlink", forbidden_unlink)
        dirfd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(canary.CanaryRefusal) as refusal:
                canary._take_reservation(dirfd, canary.LEDGER_NAME)
        finally:
            os.close(dirfd)

        marker = output_dir / PERSISTENT_MARKER_NAMES[0]
        assert refusal.value.code == "output_not_reservable"
        assert marker.read_bytes() == b"X" + PERSISTENT_MARKER_BYTES[1:]
        assert unlink_attempts == []

    def test_second_marker_failure_retains_first_exact_and_second_partial(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_write = os.write
        marker_number = 0
        writes_in_marker = 0
        unlink_attempts: list[Any] = []

        def fail_second_marker(fd: int, data: Any) -> int:
            nonlocal marker_number, writes_in_marker
            if writes_in_marker == 0:
                marker_number += 1
            writes_in_marker += 1
            if marker_number == 1:
                writes_in_marker = 0
                return real_write(fd, data)
            if writes_in_marker == 1:
                return real_write(fd, data[:19])
            raise OSError(errno.EIO, "untrusted second marker failure")

        def forbidden_unlink(*args: Any, **kwargs: Any) -> None:
            unlink_attempts.append((args, kwargs))
            raise AssertionError("permanent residue unlink attempted")

        monkeypatch.setattr(os, "write", fail_second_marker)
        monkeypatch.setattr(os, "unlink", forbidden_unlink)

        with (
            pytest.raises(canary.CanaryRefusal) as refusal,
            canary.reserve_output_names(output_dir),
        ):
            raise AssertionError("reservation must not succeed")

        assert refusal.value.code == "output_not_reservable"
        assert unlink_attempts == []
        assert (output_dir / PERSISTENT_MARKER_NAMES[0]).read_bytes() == (
            PERSISTENT_MARKER_BYTES
        )
        assert (output_dir / PERSISTENT_MARKER_NAMES[1]).read_bytes() == (
            PERSISTENT_MARKER_BYTES[:19]
        )

    def test_offline_success_retains_exact_owner_only_non_evidence_markers(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        assert canary.main(offline_arguments(output_dir)) == 0
        capsys.readouterr()
        assert sorted(p.name for p in output_dir.iterdir()) == expected_output_entries(
            *canary.OUTPUT_NAMES
        )
        assert_exact_persistent_markers(output_dir)

    def test_the_bundle_audit_and_the_summary_are_reported_and_never_filed(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        """The scored bundle pair remains distinct from the non-scored sidecar.

        The audit over the pair and the run summary are computed and printed.
        Neither becomes a file. All three output names, including the partial
        sidecar, have permanent claim markers ignored by readers and bundle audit.
        """
        assert canary.main(offline_arguments(output_dir)) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["bundle_audit"]["ok"] is True
        assert summary["ledger_audit"]["status"] == "scored"
        assert sorted(p.name for p in output_dir.iterdir()) == expected_output_entries(
            *canary.OUTPUT_NAMES
        )

    def test_a_reservation_cannot_be_mistaken_for_evidence(self) -> None:
        for name in canary.OUTPUT_NAMES:
            reservation = canary.reservation_name_for(name)
            assert reservation not in canary.OUTPUT_NAMES
            assert reservation.startswith(".")
            assert not reservation.endswith((".json", ".ndjson"))
            assert "not-evidence" in reservation

    def test_safe_scanner_requires_exact_fixed_marker_bytes(
        self, output_dir: Path
    ) -> None:
        marker = output_dir / PERSISTENT_MARKER_NAMES[0]
        with canary.reserve_output_names(output_dir) as reservation:
            marker.write_bytes(b"attacker changed a non-evidence marker\n")
            with pytest.raises(canary.CanaryFailure) as failure:
                canary._scan_what_was_written(output_dir, reservation=reservation)
        assert failure.value.code == "output_reservation_marker_changed"
        assert marker.read_bytes() == b"attacker changed a non-evidence marker\n"

    def test_marker_and_directory_descriptors_close_exactly_once(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker_identities: list[tuple[int, int]] = []
        reserved_names: list[str] = []
        held_directory_fd: int | None = None
        marker_close_counts: dict[tuple[int, int], int] = {}
        held_directory_close_count = 0
        real_take = canary._take_reservation
        real_close = os.close

        def recording_take(dirfd: int, name: str) -> int:
            fd = real_take(dirfd, name)
            info = os.fstat(fd)
            marker_identities.append((info.st_dev, info.st_ino))
            reserved_names.append(name)
            return fd

        def recording_close(fd: int) -> None:
            nonlocal held_directory_close_count
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            if identity in marker_identities:
                marker_close_counts[identity] = marker_close_counts.get(identity, 0) + 1
            if fd == held_directory_fd:
                held_directory_close_count += 1
            real_close(fd)

        monkeypatch.setattr(canary, "_take_reservation", recording_take)
        monkeypatch.setattr(os, "close", recording_close)
        with canary.reserve_output_names(output_dir) as reservation:
            held_directory_fd = reservation.dir_fd
        assert reserved_names == [
            "execution_ledger.ndjson",
            "episode_artifact.json",
            "execution_ledger.partial.ndjson",
        ]
        assert len(marker_identities) == len(set(marker_identities)) == 3
        expected_identities = {
            ((info := (output_dir / name).stat()).st_dev, info.st_ino)
            for name in PERSISTENT_MARKER_NAMES
        }
        assert set(marker_identities) == expected_identities
        assert marker_close_counts == dict.fromkeys(marker_identities, 1)
        assert held_directory_close_count == 1
        assert_exact_persistent_markers(output_dir)

    def test_marker_close_error_is_fixed_and_never_retried_or_unlinked(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target_identity: tuple[int, int] | None = None
        close_attempts = 0
        unlink_attempts: list[Any] = []
        real_take = canary._take_reservation
        real_close = os.close

        def recording_take(dirfd: int, name: str) -> int:
            nonlocal target_identity
            fd = real_take(dirfd, name)
            if target_identity is None:
                info = os.fstat(fd)
                target_identity = (info.st_dev, info.st_ino)
            return fd

        def failing_close(fd: int) -> None:
            nonlocal close_attempts
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            real_close(fd)
            if identity == target_identity:
                close_attempts += 1
                raise OSError("hostile close details")

        def forbidden_unlink(*args: Any, **kwargs: Any) -> None:
            unlink_attempts.append((args, kwargs))
            raise AssertionError("permanent marker unlink attempted")

        monkeypatch.setattr(canary, "_take_reservation", recording_take)
        monkeypatch.setattr(os, "close", failing_close)
        monkeypatch.setattr(os, "unlink", forbidden_unlink)
        with (
            pytest.raises(canary.CanaryFailure) as failure,
            canary.reserve_output_names(output_dir),
        ):
            pass
        assert failure.value.code == "output_reservation_cleanup_failed"
        assert close_attempts == 1
        assert unlink_attempts == []
        assert_exact_persistent_markers(output_dir)

    @pytest.mark.parametrize("name", ["execution_ledger.ndjson", "episode_artifact.json"])
    def test_a_pre_existing_output_refuses_at_zero_dispatch(
        self, output_dir: Path, no_construction: list[str], name: str
    ) -> None:
        target = output_dir / name
        target.write_text("older evidence", encoding="utf-8")
        assert canary.main(offline_arguments(output_dir)) != 0
        assert no_construction == []
        assert target.read_text(encoding="utf-8") == "older evidence"
        assert sorted(p.name for p in output_dir.iterdir()) == [name]

    @pytest.mark.parametrize("name", ["execution_ledger.ndjson", "episode_artifact.json"])
    def test_a_pre_existing_output_refuses_an_authorised_live_run_too(
        self,
        output_dir: Path,
        no_construction: list[str],
        monkeypatch: pytest.MonkeyPatch,
        name: str,
    ) -> None:
        authorise(monkeypatch)
        (output_dir / name).write_text("older evidence", encoding="utf-8")
        assert canary.main(live_arguments(output_dir)) != 0
        assert no_construction == []
        assert canary.CREDENTIAL_VARIABLE not in os.environ
        assert canary.LIVE_AUTHORIZATION_VARIABLE not in os.environ

    @pytest.mark.parametrize("name", ["execution_ledger.ndjson", "episode_artifact.json"])
    def test_a_symlink_at_an_output_name_is_refused_and_never_followed(
        self, output_dir: Path, tmp_path: Path, no_construction: list[str], name: str
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        (output_dir / name).symlink_to(elsewhere)
        assert canary.main(offline_arguments(output_dir)) != 0
        assert no_construction == []
        assert not elsewhere.exists()
        assert (output_dir / name).is_symlink()

    def test_another_runs_reservation_refuses_this_one_at_zero_dispatch(
        self, output_dir: Path, no_construction: list[str]
    ) -> None:
        held = output_dir / canary.reservation_name_for(canary.ARTIFACT_NAME)
        held.write_text("held by a run that is still going", encoding="utf-8")
        assert canary.main(offline_arguments(output_dir)) != 0
        assert no_construction == []
        # The refusing run does not tidy up a reservation it does not own.
        assert held.read_text(encoding="utf-8") == "held by a run that is still going"
        assert not (output_dir / canary.ARTIFACT_NAME).exists()
        assert not (output_dir / canary.LEDGER_NAME).exists()
        first_claim = output_dir / PERSISTENT_MARKER_NAMES[0]
        assert first_claim.read_bytes() == PERSISTENT_MARKER_BYTES
        assert stat.S_IMODE(first_claim.stat().st_mode) == 0o600
        before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
        assert canary.main(offline_arguments(output_dir)) != 0
        assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before

    def test_an_occupied_output_refuses_before_acquiring_any_marker(
        self, output_dir: Path, no_construction: list[str]
    ) -> None:
        (output_dir / canary.ARTIFACT_NAME).write_text("older evidence", encoding="utf-8")
        assert canary.main(offline_arguments(output_dir)) != 0
        assert no_construction == []
        assert sorted(p.name for p in output_dir.iterdir()) == [canary.ARTIFACT_NAME]

    def test_an_output_directory_that_is_not_0700_refuses_offline_too(
        self, tmp_path: Path, no_construction: list[str]
    ) -> None:
        loose = tmp_path / "loose"
        loose.mkdir(mode=0o755)
        assert canary.main(offline_arguments(loose)) != 0
        assert no_construction == []
        assert list(loose.iterdir()) == []

    def test_an_output_directory_reached_through_a_link_refuses_offline(
        self, tmp_path: Path, output_dir: Path, no_construction: list[str]
    ) -> None:
        link = tmp_path / "link"
        link.symlink_to(output_dir)
        assert canary.main(offline_arguments(link)) != 0
        assert no_construction == []
        assert list(output_dir.iterdir()) == []

    def test_a_faulted_run_leaves_the_ledger_and_all_three_permanent_markers(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        authorise(monkeypatch)
        wire = Wire()
        monkeypatch.setattr(
            canary,
            "open_live_transport",
            mock_live_transport(
                wire, faults={4: httpx.Response(500, json={"error": {"m": "mock"}})}
            ),
        )
        assert canary.main(live_arguments(output_dir)) != 0
        assert sorted(p.name for p in output_dir.iterdir()) == expected_output_entries(
            canary.LEDGER_NAME
        )
        assert_exact_persistent_markers(output_dir)

    def test_public_live_cleanup_makes_no_identity_check_then_unlink_attempt(
        self,
        output_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
    ) -> None:
        name = canary.reservation_name_for(canary.LEDGER_NAME)
        attacker = b"attacker replacement sentinel\n"
        staging = tmp_path / "attacker-staging"
        moved_owned = tmp_path / "moved-owned-marker"
        staging.write_bytes(attacker)
        real_stat = os.stat
        real_unlink = os.unlink
        cleanup_stat_seen = False
        unlink_attempts: list[str] = []

        def swapping_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
            nonlocal cleanup_stat_seen
            result = real_stat(path, *args, **kwargs)
            if path == name and kwargs.get("dir_fd") is not None:
                cleanup_stat_seen = True
                (output_dir / name).rename(moved_owned)
                staging.replace(output_dir / name)
            return result

        def recording_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
            if path == name and kwargs.get("dir_fd") is not None:
                unlink_attempts.append(path)
            real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", swapping_stat)
        monkeypatch.setattr(os, "unlink", recording_unlink)
        monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, swapping_stat})
        monkeypatch.setattr(
            os,
            "supports_follow_symlinks",
            {*os.supports_follow_symlinks, swapping_stat},
        )
        authorise(monkeypatch)
        wire = Wire()
        monkeypatch.setattr(canary, "open_live_transport", mock_live_transport(wire))
        code = canary.main(live_arguments(output_dir))
        summary = json.loads(capsys.readouterr().out)

        assert len(wire.bodies) == canary.EXPECTED_PROVIDER_CALLS
        assert unlink_attempts == []
        assert (staging.exists() and staging.read_bytes() == attacker) or (
            (output_dir / name).exists() and (output_dir / name).read_bytes() == attacker
        )
        if cleanup_stat_seen:
            assert code != 0
            assert summary["ok"] is False
            assert summary["failure"]["code"] == "output_directory_identity_changed"
        else:
            assert code == 0, summary
            assert summary["ok"] is True
            assert not moved_owned.exists()


class TestDescriptorRelativeSharedApis:
    @staticmethod
    def _error(failure: _WriteOnceFailure) -> Exception:
        return RuntimeError(failure.reason)

    @pytest.mark.parametrize(
        "name", ["", ".", "..", "nested/file", "nested\\file", "/absolute", "nul\x00x"]
    )
    def test_malformed_basenames_mutate_nothing(
        self, output_dir: Path, open_fd_count: Any, name: str
    ) -> None:
        dir_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        before = open_fd_count()
        try:
            with pytest.raises(RuntimeError, match="invalid"):
                _write_once_bytes_at(
                    b"payload",
                    name,
                    dir_fd=dir_fd,
                    display_path=name or "empty",
                    error_adapter=self._error,
                )
            with pytest.raises(ArtifactError):
                read_artifact(name, dir_fd=dir_fd)
            with pytest.raises(LedgerPathError):
                read_execution_ledger(name, dir_fd=dir_fd)
            assert open_fd_count() == before
            assert list(output_dir.iterdir()) == []
        finally:
            os.close(dir_fd)

    def test_renamed_parent_write_stays_anchored_and_fsyncs_file_then_directory(
        self,
        output_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_fd_count: Any,
    ) -> None:
        moved = tmp_path / "anchored"
        dir_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directory_identity = os.fstat(dir_fd)
        fsynced: list[str] = []
        real_fsync = os.fsync

        def recording_fsync(fd: int) -> None:
            info = os.fstat(fd)
            fsynced.append(
                "directory"
                if (info.st_dev, info.st_ino)
                == (directory_identity.st_dev, directory_identity.st_ino)
                else "file"
            )
            real_fsync(fd)

        output_dir.rename(moved)
        output_dir.mkdir(mode=0o700)
        (output_dir / "attacker.txt").write_bytes(b"unchanged")
        before = open_fd_count()
        monkeypatch.setattr(os, "fsync", recording_fsync)
        try:
            result = _write_once_bytes_at(
                b"exact anchored bytes",
                "record.bin",
                dir_fd=dir_fd,
                display_path="record.bin",
                error_adapter=self._error,
            )
            assert result == Path("record.bin")
            assert open_fd_count() == before
        finally:
            os.close(dir_fd)
        assert fsynced == ["file", "directory"]
        assert (moved / "record.bin").read_bytes() == b"exact anchored bytes"
        assert stat.S_IMODE((moved / "record.bin").stat().st_mode) == 0o600
        assert (output_dir / "attacker.txt").read_bytes() == b"unchanged"

    @pytest.mark.parametrize("occupied", ["file", "symlink"])
    def test_occupied_names_are_never_replaced(
        self, output_dir: Path, tmp_path: Path, occupied: str
    ) -> None:
        target = output_dir / "record.bin"
        if occupied == "file":
            target.write_bytes(b"older")
        else:
            target.symlink_to(tmp_path / "elsewhere")
        dir_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with pytest.raises(RuntimeError, match=r"exists|final_symlink"):
                _write_once_bytes_at(
                    b"new",
                    "record.bin",
                    dir_fd=dir_fd,
                    display_path="record.bin",
                    error_adapter=self._error,
                )
        finally:
            os.close(dir_fd)
        if occupied == "file":
            assert target.read_bytes() == b"older"
        else:
            assert target.is_symlink()
            assert not (tmp_path / "elsewhere").exists()

    @pytest.mark.parametrize("fail_on", ["file", "directory"])
    def test_fsync_failures_leak_no_descriptors(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_fd_count: Any,
        fail_on: str,
    ) -> None:
        dir_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directory_identity = os.fstat(dir_fd)
        before = open_fd_count()
        real_fsync = os.fsync

        def failing_fsync(fd: int) -> None:
            info = os.fstat(fd)
            kind = (
                "directory"
                if (info.st_dev, info.st_ino)
                == (directory_identity.st_dev, directory_identity.st_ino)
                else "file"
            )
            if kind == fail_on:
                raise OSError("hostile details")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", failing_fsync)
        try:
            with pytest.raises(RuntimeError, match="write"):
                _write_once_bytes_at(
                    b"payload",
                    "record.bin",
                    dir_fd=dir_fd,
                    display_path="record.bin",
                    error_adapter=self._error,
                )
            assert open_fd_count() == before
        finally:
            os.close(dir_fd)


# -- no exception prose, and no credential, reaches any output ----------------


class TestNoProviderProseOrCredentialReachesOutput:
    """A failure says which class of thing failed, and nothing a provider wrote.

    An SDK exception is prose written by somebody else: it may quote a request
    header, a response body or the credential it was rejected for. Rendering it
    puts all three into whatever collects this tool's stdout. So the failure
    report carries a fixed code from this build's own vocabulary and the fixed
    sentence that code means — never ``str(exc)``, never the exception's type.
    """

    @pytest.mark.parametrize("target", INJECTION_POINTS)
    def test_an_exception_carrying_a_credential_is_never_rendered(
        self,
        output_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: Any,
        target: str,
    ) -> None:
        authorise(monkeypatch)
        if target != "open_live_transport":
            monkeypatch.setattr(
                canary, "open_live_transport", mock_live_transport(Wire())
            )

        def exploding(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(SECRET_PROSE)

        monkeypatch.setattr(canary, target, exploding)
        assert canary.main(live_arguments(output_dir)) != 0
        printed = capsys.readouterr()
        blob = printed.out + printed.err
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in blob
        for path in output_dir.iterdir():
            text = path.read_text(encoding="utf-8", errors="replace")
            for fragment in FORBIDDEN_FRAGMENTS:
                assert fragment not in text
        payload = json.loads(printed.out)
        assert payload["ok"] is False
        assert payload["failure"]["code"] in canary.FAILURE_DETAILS
        assert (
            payload["failure"]["detail"]
            == (canary.FAILURE_DETAILS[payload["failure"]["code"]])
        )
        assert canary.CREDENTIAL_VARIABLE not in os.environ
        assert canary.LIVE_AUTHORIZATION_VARIABLE not in os.environ

    def test_an_offline_client_construction_failure_is_not_rendered_either(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        def exploding(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(SECRET_PROSE)

        monkeypatch.setattr(canary.anthropic, "Anthropic", exploding)
        assert canary.main(offline_arguments(output_dir)) != 0
        printed = capsys.readouterr()
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in printed.out + printed.err
        assert sorted(path.name for path in output_dir.iterdir()) == sorted(
            PERSISTENT_MARKER_NAMES
        )
        assert_exact_persistent_markers(output_dir)

    def test_the_failure_report_renders_a_code_and_a_fixed_sentence_only(self) -> None:
        rendered = json.dumps(
            canary._failure(canary.MODE_LIVE, RuntimeError(SECRET_PROSE))
        )
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in rendered
        failure = json.loads(rendered)["failure"]
        assert failure["class"] == canary.FAILURE_CLASS_INTERNAL
        assert failure["code"] in canary.FAILURE_DETAILS
        assert failure["detail"] == canary.FAILURE_DETAILS[failure["code"]]

    def test_a_refusal_reports_its_own_code_and_nothing_it_was_told(
        self, tmp_path: Path
    ) -> None:
        secret_named = tmp_path / f"dir-{LIVE_KEY}"
        secret_named.mkdir(mode=0o700)
        (secret_named / canary.ARTIFACT_NAME).write_text("older", encoding="utf-8")
        rendered = json.dumps(
            canary._failure(
                canary.MODE_OFFLINE,
                canary.CanaryRefusal("output_target_exists", str(secret_named)),
            )
        )
        assert LIVE_KEY not in rendered
        assert json.loads(rendered)["failure"]["class"] == canary.FAILURE_CLASS_REFUSAL

    def test_every_refusal_and_failure_code_has_a_fixed_sentence(self) -> None:
        for code, sentence in canary.FAILURE_DETAILS.items():
            assert code == code.lower()
            assert sentence and sentence == sentence.strip()
            assert not canary.scan_for_secrets(sentence)


class TestTheSecretScanner:
    """Defence in depth: nothing shaped like a credential leaves this process."""

    @pytest.mark.parametrize(
        "text",
        [
            f"the key is {LIVE_KEY}",
            "Authorization: Bearer ***",
            "api_key=" + "abcdef" + "0123456789",
            "x-api-key: abcdef...89",
            SECRET_PROSE,
        ],
    )
    def test_it_trips_on_credential_shapes(self, text: str) -> None:
        assert canary.scan_for_secrets(text)

    def test_it_passes_this_builds_own_passing_summary(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        assert canary.main(offline_arguments(output_dir)) == 0
        printed = capsys.readouterr()
        assert canary.scan_for_secrets(printed.out) == []
        for path in output_dir.iterdir():
            assert canary.scan_for_secrets(path.read_text(encoding="utf-8")) == []

    def test_output_that_trips_the_scanner_is_withheld(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.setattr(canary, "CANARY_CLAIM", f"leaked {LIVE_KEY}")
        assert canary.main(offline_arguments(output_dir)) != 0
        printed = capsys.readouterr()
        assert LIVE_KEY not in printed.out + printed.err
