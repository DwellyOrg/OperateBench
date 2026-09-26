"""The one-cell canary: what it refuses, and what it does when it does not.

``tools/run_lifecycle_v1_canary.py`` is not a provider CLI. It runs exactly one
pinned cell — openai / Responses / ``gpt-5.6-luna`` / V1 / one episode — and it
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

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from boundarybench.pricing import OPENAI_PRICING_POLICY
from operatebench.agents.model import MODEL_PROTOCOL_VERSION
from operatebench.agents.openai_responses import LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION
from operatebench.artifact import ARTIFACT_VERSION, read_artifact
from operatebench.execution_bundle import audit_execution_bundle_files
from operatebench.execution_ledger import (
    EXECUTION_LEDGER_VERSION,
    TERMINAL_SCORED,
    read_execution_ledger,
)
from operatebench.providers.openai_responses import (
    PARALLEL_TOOL_CALLS,
    STORE,
    TOOL_CHOICE,
)
from operatebench.version import OPERATEBENCH_VERSION
from tests.historical_canary_controls import historical_canary_gate_build  # noqa: F401
from tests.openai_transport import FAKE_API_KEY
from tests.provider_evidence_runs import (
    EXPECTED_PROVIDER_CALLS,
    Wire,
    reference_handler,
)
from tools import run_lifecycle_v1_canary as canary

pytestmark = pytest.mark.usefixtures("historical_canary_gate_build")

LIVE_KEY = "sk-not-a-real-credential-000000000000"

#: What an SDK, a client or a provider is entitled to put in an exception, and
#: what this build is therefore never allowed to render. Every fragment here is
#: invented in this file: no credential exists in this repository.
SECRET_PROSE = (
    f"Incorrect API key provided: {LIVE_KEY}. "
    f"Authorization: Bearer {LIVE_KEY} was rejected by the provider "
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
    ("LARGEST_REQUEST_TOKEN_BOUND", 23_410, 1_375_300),
    ("MAX_OUTPUT_TOKENS", 4_097, 1_375_300),
    ("TOKEN_HARD_CAP", 1_375_251, None),
    ("MAX_PROVIDER_CALLS", 51, 1_402_755),
    ("WALL_CLOCK_DEADLINE_SECONDS", 2_701.0, None),
]

# Every source term that defines the authorised cell. The replacement values
# and derivative updates are literal so this oracle cannot move with production.
EXACT_CELL_MUTATIONS = [
    (
        "CANARY_PROVIDER",
        "another-provider",
        {"LIVE_AUTHORIZATION_VALUE": "another-provider:gpt-5.6-luna:V1:1:max-calls-50"},
    ),
    ("CANARY_API", "chat_completions", {}),
    (
        "CANARY_MODEL",
        "gpt-5.6-luna-moved",
        {
            "GPT_5_6_LUNA_MODEL": "gpt-5.6-luna-moved",
            "LIVE_AUTHORIZATION_VALUE": "openai:gpt-5.6-luna-moved:V1:1:max-calls-50",
        },
    ),
    (
        "CANARY_PROFILE_ID",
        "gpt56luna_moved",
        {
            "GPT_5_6_LUNA_PROFILE": replace(
                canary.GPT_5_6_LUNA_PROFILE, profile_id="gpt56luna_moved"
            )
        },
    ),
    (
        "CANARY_SCENARIO_ID",
        "V2",
        {"LIVE_AUTHORIZATION_VALUE": "openai:gpt-5.6-luna:V2:1:max-calls-50"},
    ),
    ("CANARY_AGENT_ID", "another-agent", {}),
    (
        "CANARY_EPISODES",
        2,
        {"LIVE_AUTHORIZATION_VALUE": "openai:gpt-5.6-luna:V1:2:max-calls-50"},
    ),
    ("EXPECTED_PROVIDER_CALLS", 47, {}),
    (
        "LIVE_AUTHORIZATION_VALUE",
        "openai:gpt-5.6-luna:V1:1:max-calls-51",
        {},
    ),
    ("MAX_PROVIDER_CALLS", 51, {"TOKEN_HARD_CAP": 1_402_755}),
    (
        "MAX_ATTEMPTS_PER_CALL",
        2,
        {
            "LIFECYCLE_RETRY_POLICY": replace(
                canary.LIFECYCLE_RETRY_POLICY, max_attempts=2
            )
        },
    ),
    ("SDK_MAX_RETRIES", 1, {}),
    ("LARGEST_REQUEST_TOKEN_BOUND", 23_410, {"TOKEN_HARD_CAP": 1_375_300}),
    ("MAX_OUTPUT_TOKENS", 4_097, {"TOKEN_HARD_CAP": 1_375_300}),
    ("TOKEN_HARD_CAP", 1_375_251, {}),
    ("TURN_DEADLINE_SECONDS", 31.0, {}),
    ("WALL_CLOCK_DEADLINE_SECONDS", 2_701.0, {}),
    ("STORE", True, {}),
    ("PARALLEL_TOOL_CALLS", True, {}),
    ("TOOL_CHOICE", "auto", {}),
    ("OPERATEBENCH_VERSION", "0.7.1", {}),
    ("CANARY_OPERATION_VERSION", "0.6.1", {}),
    ("MODEL_PROTOCOL_VERSION", "operatebench.model.v5", {}),
    (
        "LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION",
        "lifecycle_openai_responses_model_request_v8",
        {},
    ),
    ("ARTIFACT_VERSION", 9, {}),
    ("EXECUTION_LEDGER_VERSION", 4, {}),
]

EXACT_PRICING_CONSTANT_MUTATIONS = [
    ("FIXED_CANARY_POLICY_ID", "operator_pinned_v2"),
    ("RATE_SOURCE_OPERATOR", "another_operator_source"),
    ("FIXED_CANARY_RATE_SOURCE", "another_operator_source"),
    ("FIXED_CANARY_INPUT_USD_PER_MTOK", Decimal("0.21")),
    ("FIXED_CANARY_OUTPUT_USD_PER_MTOK", Decimal("1.21")),
    ("FIXED_CANARY_COST_CAP_USD", Decimal("0.51")),
]


def test_the_operator_script_is_directly_invocable() -> None:
    """The documented file-path entrypoint must import before any gate or client."""
    result = subprocess.run(
        [
            sys.executable,
            str(canary.REPO_ROOT / "tools/run_lifecycle_v1_canary.py"),
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
    "0.52",
    "--input-usd-per-mtok",
    "0.20",
    "--output-usd-per-mtok",
    "1.20",
]

PRICING_ARGUMENT_MUTATIONS = [
    ("--input-usd-per-mtok", "0.19"),
    ("--input-usd-per-mtok", "0.21"),
    ("--output-usd-per-mtok", "1.19"),
    ("--output-usd-per-mtok", "1.21"),
    ("--cost-cap-usd", "0.49"),
    ("--cost-cap-usd", "0.51"),
]

EXPECTED_PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"


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

    monkeypatch.setattr(canary.openai, "OpenAI", trap("sdk_client"))
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


def mock_live_transport(wire: Wire, **kwargs: Any) -> Any:
    """Answer in process, through the real client-construction path.

    Only the socket is replaced. ``open_live_client`` still runs for real, so
    the wire capture the run installs underneath it is the one the requests go
    through and the ledger's wire digests are digests of bodies the SDK really
    serialised.
    """

    def factory() -> httpx.BaseTransport:
        return httpx.MockTransport(reference_handler(wire, **kwargs))

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
                "tools/run_lifecycle_v1_canary.py",
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
                "tools.run_lifecycle_v1_canary",
                "--offline-preflight",
                "--output-dir",
                "DIR",
                *RATE_ARGUMENTS,
            ],
        ]

    def test_it_is_pinned_to_one_provider_api_model_scenario_and_episode(self) -> None:
        assert canary.CANARY_PROVIDER == "openai"
        assert canary.CANARY_API == "responses"
        assert canary.CANARY_MODEL == "gpt-5.6-luna"
        assert canary.CANARY_PROFILE_ID.startswith("gpt56luna")
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
                    "operator_pinned_v1",
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
        assert canary.LARGEST_REQUEST_TOKEN_BOUND == 27_241
        assert canary.MAX_OUTPUT_TOKENS == 4_096
        assert canary.MAX_ATTEMPTS_PER_CALL == 1
        assert canary.SDK_MAX_RETRIES == 0
        assert canary.TOKEN_HARD_CAP == 1_566_850
        assert canary.TURN_DEADLINE_SECONDS == 30.0
        assert canary.WALL_CLOCK_DEADLINE_SECONDS == 2_700.0

    def test_the_runtime_interface_versions_are_pinned(self) -> None:
        spec = canary.load_spec(canary.FIXTURE)

        assert OPERATEBENCH_VERSION == "0.13.0"
        assert canary.OPERATEBENCH_VERSION == "0.12.0"
        assert spec.operation_version == "0.6.0"
        assert MODEL_PROTOCOL_VERSION == "operatebench.model.v4"
        assert (
            LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION
            == "lifecycle_openai_responses_model_request_v9"
        )
        assert ARTIFACT_VERSION == 8
        assert EXECUTION_LEDGER_VERSION == 3

    def test_the_runtime_request_profile_and_settings_are_pinned(self) -> None:
        profile = canary.GPT_5_6_LUNA_PROFILE

        assert canary.CANARY_PROFILE_ID == (
            "gpt56luna_reasoning_none_sampling_omitted_v2"
        )
        assert profile.reasoning == {"effort": "none"}
        assert profile.temperature is None
        assert profile.omitted_fields() == ("temperature", "top_p")
        assert STORE is False
        assert PARALLEL_TOOL_CALLS is False
        assert TOOL_CHOICE == "required"

    def test_the_fixed_pricing_policy_source_is_pinned(self) -> None:
        assert OPENAI_PRICING_POLICY.source == EXPECTED_PRICING_SOURCE

    def test_the_live_authorization_names_the_fifty_call_envelope(self) -> None:
        assert canary.LIVE_AUTHORIZATION_VALUE == "openai:gpt-5.6-luna:V1:1:max-calls-50"

    def test_pricing_help_describes_numeric_not_lexical_equality(self) -> None:
        help_text = " ".join(canary.build_parser().format_help().lower().split())

        for amount in ("0.52", "0.20", "1.20"):
            assert f"must equal usd {amount} numerically" in help_text
        assert help_text.count("equivalent exact-decimal spellings are accepted") == 3
        assert "required exact decimal string" not in help_text
        assert "accepts only 0.52 usd" not in help_text

    def test_the_pessimistic_fifty_call_cost_fits_fixed_cap_at_stated_rates(
        self,
    ) -> None:
        policy = canary.LifecyclePricingPolicy(
            policy_id=canary.FIXED_CANARY_POLICY_ID,
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
        )
        worst_case = policy.worst_case_usd(
            calls=canary.MAX_PROVIDER_CALLS,
            input_token_bound=canary.LARGEST_REQUEST_TOKEN_BOUND,
            max_output_tokens=canary.MAX_OUTPUT_TOKENS,
        )

        assert worst_case == Decimal("0.51817")
        assert worst_case <= Decimal("0.52")

    def test_the_fixed_canary_pricing_envelope_is_named_as_build_owned_controls(
        self,
    ) -> None:
        assert Decimal("0.20") == canary.FIXED_CANARY_INPUT_USD_PER_MTOK
        assert Decimal("1.20") == canary.FIXED_CANARY_OUTPUT_USD_PER_MTOK
        assert Decimal("0.52") == canary.FIXED_CANARY_COST_CAP_USD
        assert canary.FIXED_CANARY_RATE_SOURCE == canary.RATE_SOURCE_OPERATOR
        assert (
            canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED
            == "fixed_canary_pricing_envelope_required"
        )
        assert (
            canary.CODE_FIXED_CANARY_PRICING_ENVELOPE_REQUIRED in canary.FAILURE_DETAILS
        )
        assert canary.FIXED_CANARY_POLICY_ID == "operator_pinned_v1"

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
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
            rate_source="operator_supplied_pinned_rates",
        )
        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.check_fixed_canary_pricing_envelope(policy, cap=Decimal("0.52"))
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
        monkeypatch.setattr(canary, name, mutated)
        for derivative, value in derivatives.items():
            monkeypatch.setattr(canary, derivative, value)
        environment = CredentialUnreadEnvironment(
            {
                canary.LIVE_AUTHORIZATION_VARIABLE: canary.LIVE_AUTHORIZATION_VALUE,
                canary.CREDENTIAL_VARIABLE: LIVE_KEY,
            }
        )
        monkeypatch.setattr(canary.os, "environ", environment)
        policy = canary.LifecyclePricingPolicy(
            policy_id="operator_pinned_v1",
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
            rate_source="operator_supplied_pinned_rates",
        )
        with pytest.raises(canary.CanaryRefusal):
            canary.run_live_cell(directory=output_dir, policy=policy, cap=Decimal("0.52"))
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
            policy_id="operator_pinned_v1",
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
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
            replace(canary.GPT_5_6_LUNA_PROFILE, reasoning={"effort": "low"}),
            replace(canary.GPT_5_6_LUNA_PROFILE, temperature=0.0),
            SimpleNamespace(
                profile_id="gpt56luna_reasoning_none_sampling_omitted_v2",
                reasoning={"effort": "none"},
                temperature=None,
                omitted_fields=lambda: ("temperature",),
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
        monkeypatch.setattr(canary, "GPT_5_6_LUNA_PROFILE", profile)
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
                    policy_id="operator_pinned_v1",
                    input_usd_per_mtok=Decimal("0.20"),
                    output_usd_per_mtok=Decimal("1.20"),
                ),
                cap=Decimal("0.52"),
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
            policy_id="operator_pinned_v1",
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
        )
        directory = output_dir
        if invalid == "controls":
            monkeypatch.setattr(canary, "MAX_PROVIDER_CALLS", 51)
            monkeypatch.setattr(canary, "TOKEN_HARD_CAP", 1_402_755)
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
                cap=Decimal("0.52"),
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
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
        )
        object.__setattr__(policy, "rate_source", "another_operator_source")

        with pytest.raises(canary.CanaryRefusal) as refusal:
            canary.run_offline_preflight(
                directory=output_dir,
                policy=policy,
                cap=Decimal("0.52"),
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
                "0.520",
                "--input-usd-per-mtok",
                "0.2",
                "--output-usd-per-mtok",
                "1.2",
            ]
        )

        assert code == 0
        assert no_client == []
        summary = json.loads(capsys.readouterr().out)
        assert summary["ok"] is True
        assert summary["mode"] == canary.MODE_OFFLINE
        assert summary["live_provider_call"] is False
        assert summary["budget"]["cost_cap_usd"] == "0.52"
        assert Decimal(summary["budget"]["worst_case_usd"]) == Decimal("0.51817")

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
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
        )
        calculated_worst_case = policy.worst_case_usd(
            calls=canary.MAX_PROVIDER_CALLS,
            input_token_bound=canary.LARGEST_REQUEST_TOKEN_BOUND,
            max_output_tokens=canary.MAX_OUTPUT_TOKENS,
        )
        assert Decimal(summary["budget"]["worst_case_usd"]) == calculated_worst_case
        assert summary["budget"]["cost_cap_usd"] == "0.52"
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
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
        )
        cap = Decimal("0.52")
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
            input_usd_per_mtok=Decimal("0.20"),
            output_usd_per_mtok=Decimal("1.20"),
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
            canary.run_live_cell(directory=output_dir, policy=policy, cap=Decimal("0.52"))

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
                    "openai:gpt-5.6-luna:V1:1:max-calls-50"
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
                    input_usd_per_mtok=Decimal("0.20"),
                    output_usd_per_mtok=Decimal("1.20"),
                ),
                cap=Decimal("1.00"),
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
                    return "openai:gpt-5.6-luna:V1:1"
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
                    "openai:gpt-5.6-luna:V1:1:max-calls-50"
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
            "openai:gpt-5.6-luna:V1:1",
            "openai:gpt-5.6-luna:V1:1:max-calls-90",
            "openai:gpt-5.6-luna:V2:1",
            "openai:gpt-5.6-luna:V1:2",
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
                    "0.01",
                    "--input-usd-per-mtok",
                    "1.25",
                    "--output-usd-per-mtok",
                    "10.00",
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
        assert len(wire.bodies) == EXPECTED_PROVIDER_CALLS
        summary = json.loads(capsys.readouterr().out)
        assert summary["mode"] == "live"
        assert summary["n"] == 1
        assert summary["episodes"] == 1
        assert summary["provider_calls"] == EXPECTED_PROVIDER_CALLS
        assert summary["bundle_audit"]["ok"] is True
        assert summary["replay"]["provider_calls"] == 0

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
        assert sorted(p.name for p in output_dir.iterdir()) == sorted(
            [canary.LEDGER_NAME, "execution_ledger.partial.ndjson"]
        )
        audit = read_execution_ledger(output_dir / canary.LEDGER_NAME)
        assert audit.scored is False

    def test_a_faulted_cell_keeps_the_ledger_writes_no_artefact_and_exits_nonzero(
        self, output_dir: Path, monkeypatch: pytest.MonkeyPatch
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
                wire, faults={4: httpx.Response(500, json={"error": {"m": "mock"}})}
            ),
        )
        assert (
            canary.main(["--live", "--output-dir", str(output_dir), *RATE_ARGUMENTS]) != 0
        )
        written = sorted(path.name for path in output_dir.iterdir())
        assert written == sorted([canary.LEDGER_NAME, "execution_ledger.partial.ndjson"])
        audit = read_execution_ledger(output_dir / canary.LEDGER_NAME)
        assert audit.status == "excluded"
        assert audit.scored is False


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

    The reservation is a sentinel under a name no reader could mistake for
    evidence. It is not the artefact created early: a crash mid-run must leave
    either real evidence or nothing, never a plausible-looking file.
    """

    def test_the_reserved_names_are_exactly_what_a_passing_run_writes(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        assert canary.main(offline_arguments(output_dir)) == 0
        capsys.readouterr()
        assert sorted(p.name for p in output_dir.iterdir()) == sorted(canary.OUTPUT_NAMES)

    def test_the_bundle_audit_and_the_summary_are_reported_and_never_filed(
        self, output_dir: Path, no_client: list[str], capsys: Any
    ) -> None:
        """Which is why there is no third or fourth name to reserve.

        The audit over the pair and the run summary are computed and printed.
        Neither becomes a file, so the reserved set is the whole of what a run
        can put in an operator's directory — and this test is what stops a later
        edit filing one of them without reserving it first.
        """
        assert canary.main(offline_arguments(output_dir)) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["bundle_audit"]["ok"] is True
        assert summary["ledger_audit"]["status"] == "scored"
        assert sorted(p.name for p in output_dir.iterdir()) == sorted(canary.OUTPUT_NAMES)

    def test_a_reservation_cannot_be_mistaken_for_evidence(self) -> None:
        for name in canary.OUTPUT_NAMES:
            reservation = canary.reservation_name_for(name)
            assert reservation not in canary.OUTPUT_NAMES
            assert reservation.startswith(".")
            assert not reservation.endswith((".json", ".ndjson"))
            assert "not-evidence" in reservation

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

    def test_a_collision_on_one_name_releases_the_reservations_it_did_take(
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

    def test_a_faulted_run_leaves_the_ledger_and_no_reservation(
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
        assert sorted(p.name for p in output_dir.iterdir()) == sorted(
            [canary.LEDGER_NAME, "execution_ledger.partial.ndjson"]
        )


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

        monkeypatch.setattr(canary.openai, "OpenAI", exploding)
        assert canary.main(offline_arguments(output_dir)) != 0
        printed = capsys.readouterr()
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in printed.out + printed.err
        assert list(output_dir.iterdir()) == []

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
            "Authorization: Bearer abcdef0123456789",
            "api_key=" + "abcdef" + "0123456789",
            "x-api-key: " + "abcdef" + "0123456789",
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
