"""Four things the Lifecycle provider bridge got wrong, each pinned by a test.

Each section below is one reproduced blocker, and each of them is a way the
bridge could be *silently* wrong rather than loudly broken — which is why they
are tests rather than review notes.

**A request can be tampered with coherently.** ``check_model_request`` recomputed
the prompt digest and stopped there, so an attacker who edited the observation
and *recomputed the prompt digest to match* passed every check and reached the
wire. ``observation_digest_sha256`` — the field the whole replay contract hangs
off — was carried and never verified, and a field the model-visible allowlist
does not name could be inserted into the observation and sent. All three are
refused here before a payload exists, and every one of them is asserted against
a ``MockTransport`` call count of zero.

**``ACT``'s payload was closed.** Core accepts an arbitrary JSON-object payload
on an ``ACT``; the projection sent a closed object with no properties, so the
tool surface admitted only ``{}`` and no model could state an action's
arguments. The payload is an open object here, and the values Core accepts are
values the schema accepts.

**A wait with nothing to wake it was schema-legal.** Core refuses a ``WAIT``
that declares neither an event to wake on nor a fallback deadline; the schema
declared both fields as nullable and so admitted the unbounded wait Core exists
to refuse. The liveness rule is stated in the schema itself, as ``anyOf``
branches derived from the outcome contract.

**The tools claimed to be strict, and could not be.** Both of the fixes above
leave OpenAI's documented strict subset, and there is no third form that keeps
Core's semantics inside it: strict requires *every* object closed with
``additionalProperties: false``, which an arbitrary ``ACT`` payload is not, and
it prohibits a union at the root of a tool's schema, which is where a standalone
``WAIT``'s liveness rule has to be stated. The tools are sent ``strict=false``
here, and :class:`TestStrictModeCannotStateTheCoreContract` proves the choice
rather than asserting it: the generated schemas are held to the documented
subset and *fail*, in exactly the ways Core forces, and each strict-compatible
narrowing is shown to disagree with Core about a concrete value.

**The deadline had two origins.** The transport read the clock to build the
deadline and the executor read it again on entry, so everything between them —
the request checks, the payload build, the endpoint and body checks, the token
bound — was spent outside the budget it was supposed to be spent inside. A turn
configured for ten seconds that took four getting to the executor dispatched
with a ten-second SDK timeout. It dispatches with six here.

No socket, no credential, no provider: every client is a real ``openai.OpenAI``
over ``httpx.MockTransport``, and every clock is a scripted list of readings.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from operatebench.agents.model import ModelAgent, parse_tool_call
from operatebench.agents.openai_responses import (
    ModelRequestRefused,
    OpenAIResponsesTransport,
    outcome_tools,
)
from operatebench.agents.outcome_contract import (
    AGENT_TOOL_NAMES,
    WAIT_FIELDS,
    optional_field_names,
    required_field_names,
)
from operatebench.agents.transport import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    content_digest,
)
from operatebench.core.outcomes import (
    Act,
    Wait,
    WaitContractError,
    outcome_contract_problem,
)
from operatebench.core.protocol import MODEL_VISIBLE_FIELDS, AgentObservation
from operatebench.providers.executor import ProviderRetryPolicy, TurnExecutor
from operatebench.providers.faults import (
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_TIMEOUT,
    TURN_END_BACKOFF_UNAFFORDABLE,
    TURN_END_DEADLINE_EXCEEDED,
    AdapterProviderError,
    Fault,
)
from operatebench.providers.openai_responses import GPT_5_6_LUNA_MODEL
from operatebench.providers.telemetry import TurnDeadline
from operatebench.providers.usage import TokenUsage
from tests.jsonschema import schema_problem
from tests.openai_transport import (
    RecordingTransport,
    function_call_item,
    responses_body,
    scripted_client,
)

MODEL = GPT_5_6_LUNA_MODEL
MODEL_AGENT_ID = "model_reference"

#: The wall-clock budget the deadline section is configured with.
CONFIGURED_SECONDS = 10.0

#: How long this build's own pre-executor work is simulated as taking.
PRE_EXECUTOR_SECONDS = 4.0

#: Deterministic float comparison. Every number under test here is arithmetic
#: over scripted clock readings, so the only slack needed is representation.
TOLERANCE = 1e-9


# --------------------------------------------------------------- the scaffolding


class _NeverCalled:
    """A transport that fails the test if a request reaches it."""

    def send(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("this request must not reach a transport")


def observation(**overrides: Any) -> AgentObservation:
    base: dict[str, Any] = {
        "now": "2025-01-01T09:00:00Z",
        "operation_id": "op",
        "operation_instance_id": "opinst_0123456789abcdef0123456789abcdef",
        "invocation_index": 1,
        "turn_index": 0,
        "policy": {},
        "actors": {},
        "message_fixture_ids": [],
        "last_rejection": None,
    }
    base.update(overrides)
    return AgentObservation(**base)


def request_for(observed: AgentObservation | None = None) -> ModelRequest:
    """One real ``ModelAgent`` request, built without a transport in sight."""
    agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)
    agent.begin_episode({"operation_id": "op", "agent_id": MODEL_AGENT_ID})
    return agent.build_request(observed or observation())


class ScriptedClock:
    """A monotonic clock that reads from a script, holding its last value.

    Not a stepping clock: the point of the deadline section is *which* reading
    an interval is measured from, so each reading is stated rather than derived
    from how many times the code under test happened to consult it.
    """

    def __init__(self, readings: Sequence[float]) -> None:
        self._readings = list(readings)
        self.calls = 0

    def __call__(self) -> float:
        index = min(self.calls, len(self._readings) - 1)
        self.calls += 1
        return self._readings[index]


def no_sleep(_seconds: float) -> None:
    raise AssertionError("this build makes one attempt, so nothing may back off")


def transport_over(
    client: Any,
    *,
    clock: Callable[[], float],
    deadline_seconds: float = CONFIGURED_SECONDS,
) -> OpenAIResponsesTransport:
    return OpenAIResponsesTransport(
        model=MODEL,
        client=client,
        deadline_seconds=deadline_seconds,
        sleep=no_sleep,
        clock=clock,
    )


def answered_client() -> tuple[RecordingTransport, Any]:
    """A client that answers one turn with a well-formed ``COMPLETE``."""
    return scripted_client(
        responses_body(
            [function_call_item("complete", json.dumps({"reason": "done"}))],
            model=MODEL,
        )
    )


# ------------------------------------------- OpenAI's documented strict subset


#: Every JSON Schema keyword OpenAI documents as supported under strict
#: function calling: the structural set, plus the per-type constraint keywords.
#: A schema stating anything outside this is refused by the service rather than
#: ignored by it, so a build that states one has declared a guarantee it will
#: never get to rely on.
STRICT_SUBSET_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "description",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
        "enum",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
    }
)


#: A keyword the subset does not name at all.
STRICT_UNSUPPORTED_KEYWORD = "unsupported_keyword"
#: ``anyOf`` at the root of a tool's schema. OpenAI documents the root of a
#: strict schema as an object that is not itself a union, and a root that
#: narrows itself with ``anyOf`` is stating a union there whatever else it says.
STRICT_ROOT_UNION = "root_states_anyof"
#: An object that is not closed with ``additionalProperties: false``. Strict
#: mode requires *every* object closed — there is no form that means "this
#: object's keys belong to the domain".
STRICT_OPEN_OBJECT = "object_not_closed"
#: A declared property left out of ``required``. Strict mode requires every one
#: of them listed, whatever the contract calls optional.
STRICT_OPTIONAL_NOT_REQUIRED = "property_not_required"


def strict_subset_problems(
    schema: Mapping[str, Any], path: str = "parameters", *, root: bool = True
) -> list[tuple[str, str]]:
    """Every way one schema node leaves OpenAI's documented strict subset.

    Four rules, applied without exemption, each reported as
    ``(path, reason)``:

    #. only keywords the subset names may appear;
    #. the root of a tool's schema may not state ``anyOf``;
    #. **every** object is closed with ``additionalProperties: false``;
    #. every declared property appears in ``required``.

    There is deliberately no carve-out here. Exempting the free-form ``ACT``
    payload from rule 3 would invent a permission the documented subset does not
    grant — a schema is either expressible under strict mode or it is not. Held
    to the rules as they are, this build's schemas are not, and
    :class:`TestStrictModeCannotStateTheCoreContract` reads that incompatibility
    as the adjudication it is.
    """
    problems: list[tuple[str, str]] = []
    for keyword in sorted(set(schema) - STRICT_SUBSET_KEYWORDS):
        problems.append((f"{path}<{keyword}>", STRICT_UNSUPPORTED_KEYWORD))
    if root and "anyOf" in schema:
        problems.append((path, STRICT_ROOT_UNION))
    declared = schema.get("type")
    names = [declared] if isinstance(declared, str) else list(declared or ())
    properties = schema.get("properties")
    if "object" in names:
        if schema.get("additionalProperties") is not False:
            problems.append((path, STRICT_OPEN_OBJECT))
        if sorted(schema.get("required", ())) != sorted(properties or {}):
            problems.append((path, STRICT_OPTIONAL_NOT_REQUIRED))
    for name, child in (properties or {}).items():
        problems.extend(strict_subset_problems(child, f"{path}.{name}", root=False))
    items = schema.get("items")
    if isinstance(items, Mapping):
        problems.extend(strict_subset_problems(items, f"{path}[]", root=False))
    for index, branch in enumerate(schema.get("anyOf", ())):
        problems.extend(strict_subset_problems(branch, f"{path}|{index}", root=False))
    return problems


def open_object_paths(schema: Mapping[str, Any], path: str = "parameters") -> list[str]:
    """Every node that is an object and declares no properties."""
    found: list[str] = []
    declared = schema.get("type")
    names = [declared] if isinstance(declared, str) else list(declared or ())
    properties = schema.get("properties")
    if "object" in names and properties is None:
        found.append(path)
    for name, child in (properties or {}).items():
        found.extend(open_object_paths(child, f"{path}.{name}"))
    items = schema.get("items")
    if isinstance(items, Mapping):
        found.extend(open_object_paths(items, f"{path}[]"))
    for index, branch in enumerate(schema.get("anyOf", ())):
        found.extend(open_object_paths(branch, f"{path}|{index}"))
    return found


def parameters_of(name: str) -> dict[str, Any]:
    return next(tool["parameters"] for tool in outcome_tools() if tool["name"] == name)


# ================================================ 1. the observation is bound


class TestTheObservationIsBoundToItsDigest:
    """A coherent edit of the observation is refused before a payload exists.

    Every test here recomputes whatever digest the previous check would have
    caught, so what is under test is the *new* binding rather than the old one
    firing again by luck.
    """

    def _refused(self, request: ModelRequest) -> ModelRequestRefused:
        wire, client = answered_client()
        transport = transport_over(client, clock=ScriptedClock([0.0]))

        with pytest.raises(ModelRequestRefused) as caught:
            transport.send(request)

        assert wire.calls == 0, "a refused request must not reach the wire"
        return caught.value

    def test_an_edited_observation_with_a_recomputed_prompt_digest_is_refused(
        self,
    ) -> None:
        """The coherent tamper: the prompt digest agrees and the request lies."""
        from operatebench.agents.openai_responses import ObservationDigestMismatch

        request = request_for()
        prompt = dict(request.prompt)
        prompt["observation"] = {
            **prompt["observation"],
            "phase": "INVENTED_PHASE",
        }
        tampered = replace(
            request,
            prompt=prompt,
            prompt_digest_sha256=content_digest(prompt),
        )

        assert isinstance(self._refused(tampered), ObservationDigestMismatch)

    def test_an_edited_observation_digest_alone_is_refused(self) -> None:
        """The other half: the prompt is untouched and the digest is not its."""
        from operatebench.agents.openai_responses import ObservationDigestMismatch

        request = request_for()
        tampered = replace(request, observation_digest_sha256="0" * 64)

        assert isinstance(self._refused(tampered), ObservationDigestMismatch)

    @pytest.mark.parametrize(
        "hidden",
        [
            "scenario",
            "scenario_id",
            "semantic_scenario_id",
            "expected",
            "oracle",
            "control",
        ],
    )
    def test_a_field_outside_the_model_visible_allowlist_is_refused(
        self, hidden: str
    ) -> None:
        """The answer's own name, inserted and made coherent, still cannot travel.

        Both digests are recomputed over the edited observation, so nothing but
        the allowlist is left to catch it — which is the point. A field the
        model-visible projection does not name is a disclosure, and the digests
        would happily certify one.
        """
        from operatebench.agents.openai_responses import RequestObservationStructureError

        request = request_for()
        observed = {**request.prompt["observation"], hidden: "V1"}
        prompt = {**request.prompt, "observation": observed}
        tampered = replace(
            request,
            prompt=prompt,
            prompt_digest_sha256=content_digest(prompt),
            observation_digest_sha256=content_digest(observed),
        )

        assert isinstance(self._refused(tampered), RequestObservationStructureError)

    def test_an_observation_missing_an_allowlisted_field_is_refused(self) -> None:
        """The allowlist is exact in both directions, not a containment check."""
        from operatebench.agents.openai_responses import RequestObservationStructureError

        request = request_for()
        observed = {
            name: value
            for name, value in request.prompt["observation"].items()
            if name != "policy"
        }
        prompt = {**request.prompt, "observation": observed}
        tampered = replace(
            request,
            prompt=prompt,
            prompt_digest_sha256=content_digest(prompt),
            observation_digest_sha256=content_digest(observed),
        )

        assert isinstance(self._refused(tampered), RequestObservationStructureError)

    def test_the_allowlist_the_transport_checks_is_core_s_own_constant(self) -> None:
        """No second, hand-written copy of the model-visible field list.

        The transport imports :data:`MODEL_VISIBLE_FIELDS`; a list re-typed here
        would drift from the projection the moment either moved, and the drift
        would show up as a refusal nobody could explain rather than as a
        disclosure anybody could see.
        """
        from operatebench.agents import openai_responses as bridge

        assert bridge.OBSERVATION_FIELDS == MODEL_VISIBLE_FIELDS
        assert tuple(sorted(request_for().prompt["observation"])) == tuple(
            sorted(MODEL_VISIBLE_FIELDS)
        )

    def test_an_untampered_request_still_reaches_the_wire(self) -> None:
        """The binding refuses tampering and nothing else."""
        wire, client = answered_client()
        transport = transport_over(client, clock=ScriptedClock([0.0]))

        transport.send(request_for())

        assert wire.calls == 1


# ================================================= 2. ACT's payload is open


class TestTheActPayloadIsAnOpenObject:
    """Core accepts an arbitrary mapping payload, so the schema states one."""

    def test_the_payload_schema_is_a_nullable_open_object(self) -> None:
        payload = parameters_of("act")["properties"]["payload"]

        assert payload["type"] == ["object", "null"]
        assert "properties" not in payload
        assert "additionalProperties" not in payload

    def test_the_act_root_is_closed_and_requires_only_core_s_required_fields(
        self,
    ) -> None:
        """Closed at the root, and ``required`` is the contract's, not strict's.

        Under the strict mapping every property was in ``required`` — which is
        strict mode's rule and not Core's — and the contract's own idea of
        optional survived only as a nullable union plus an elision on the way
        back. Non-strict, the schema can simply say what Core says: the required
        fields are required and the rest may be left out.
        """
        act = parameters_of("act")

        assert act["additionalProperties"] is False
        assert act["required"] == list(required_field_names("act"))
        assert set(act["properties"]) == set(act["required"]) | set(
            optional_field_names("act")
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"quote_id": "q1"},
            {"quote_id": "q1", "amount_pence": 12500, "vat": True},
            {"lines": [{"sku": "a", "qty": 2}, {"sku": "b", "qty": 1}]},
            {"nested": {"deeper": {"deepest": ["x", 1, None, {"k": "v"}]}}},
            None,
        ],
    )
    def test_every_payload_core_accepts_is_a_payload_the_schema_accepts(
        self, payload: Any
    ) -> None:
        """The two contracts agree on the same values, in the same direction.

        Both ways of not stating an optional field are checked, because both are
        now on the wire: the nulls the strict mapping forced, which are still
        accepted and still elided, and the plain omission a non-strict schema
        lets a model make.
        """
        with_nulls = {
            "action_type": "submit_quote",
            "payload": payload,
            "evidence_refs": None,
            "rationale": None,
        }
        omitted = {"action_type": "submit_quote"}
        if payload is not None:
            omitted["payload"] = payload

        assert schema_problem(parameters_of("act"), with_nulls) is None
        assert schema_problem(parameters_of("act"), omitted) is None
        assert (
            outcome_contract_problem(
                Act(action_type="submit_quote", payload=payload or {})
            )
            is None
        )

    def test_a_payload_that_is_not_an_object_is_still_refused_by_the_schema(
        self,
    ) -> None:
        for wrong in (["a"], "a", 1, True):
            arguments = {
                "action_type": "submit_quote",
                "payload": wrong,
                "evidence_refs": None,
                "rationale": None,
            }
            assert schema_problem(parameters_of("act"), arguments) is not None

    def test_a_nested_payload_survives_the_bridge_and_reaches_the_outcome(self) -> None:
        """End to end, over the real SDK: the arguments arrive as the payload."""
        nested = {"quote_id": "q1", "lines": [{"sku": "a", "qty": 2}], "vat": True}
        wire, client = scripted_client(
            responses_body(
                [
                    function_call_item(
                        "act",
                        json.dumps(
                            {
                                "action_type": "submit_quote",
                                "payload": nested,
                                "evidence_refs": None,
                                "rationale": None,
                            }
                        ),
                    )
                ],
                model=MODEL,
            )
        )
        agent = ModelAgent(
            transport_over(client, clock=ScriptedClock([0.0])),
            model=MODEL,
            agent_id=MODEL_AGENT_ID,
        )
        agent.begin_episode({})

        decision = agent.decide(observation())

        assert wire.calls == 1
        assert isinstance(decision, Act)
        assert decision.payload == nested
        assert outcome_contract_problem(decision) is None

    def test_the_open_payload_is_the_only_open_object_in_the_whole_surface(
        self,
    ) -> None:
        """One deliberate exemption, named, and no second one by accident."""
        found = {
            f"{tool['name']}.{path}"
            for tool in outcome_tools()
            for path in open_object_paths(tool["parameters"])
        }

        # Two open objects now, and both are open for the same reason: the
        # *domain* owns what an action's arguments are and what a read takes, so
        # any property list here would be this projection inventing a contract
        # Core does not have.
        assert found == {
            "act.parameters.payload",
            "retrieve.parameters.requests[].arguments",
        }


# ============================================== 3. a wait that can never end


#: The ways a ``WAIT``'s two liveness fields can be stated, and whether Core
#: accepts each. Read by both the schema tests and the parser tests, so the two
#: are answering the same question about the same values.
#:
#: Every case is stated twice, once with the unused fields sent as ``null`` and
#: once with them left out. Both are legal answers to a non-strict schema — the
#: nulls because the schema still declares the optionals nullable and this
#: build still elides them, the omissions because ``required`` now names only
#: Core's own required fields — so both have to agree with Core, and a liveness
#: rule that held for one shape and not the other would be a rule about the
#: encoding rather than about the wait.
WAIT_CASES: tuple[tuple[str, dict[str, Any], bool], ...] = (
    ("neither, as nulls", {"wake_on": None, "fallback_after_minutes": None}, False),
    ("neither, omitted", {}, False),
    ("an empty wake list", {"wake_on": [], "fallback_after_minutes": None}, False),
    ("an empty wake list, the rest omitted", {"wake_on": []}, False),
    (
        "a stated wake list",
        {"wake_on": ["quote_received"], "fallback_after_minutes": None},
        True,
    ),
    ("a stated wake list, the fallback omitted", {"wake_on": ["quote_received"]}, True),
    ("a fallback deadline", {"wake_on": None, "fallback_after_minutes": 90}, True),
    ("a fallback deadline, the wake list omitted", {"fallback_after_minutes": 90}, True),
    ("both", {"wake_on": ["quote_received"], "fallback_after_minutes": 90}, True),
)


def wait_arguments(case: Mapping[str, Any]) -> dict[str, Any]:
    return {"reason": "awaiting the contractor", **case}


class TestTheWaitLivenessIsInTheSchema:
    """A wait with nothing to wake it is refused by the schema, not only Core."""

    @pytest.mark.parametrize(("label", "case", "alive"), WAIT_CASES)
    def test_the_standalone_wait_schema_agrees_with_core(
        self, label: str, case: dict[str, Any], alive: bool
    ) -> None:
        problem = schema_problem(parameters_of("wait"), wait_arguments(case))

        assert (problem is None) is alive, f"the schema disagrees with Core about {label}"

    @pytest.mark.parametrize(("label", "case", "alive"), WAIT_CASES)
    def test_the_wait_nested_in_an_ask_agrees_with_core(
        self, label: str, case: dict[str, Any], alive: bool
    ) -> None:
        arguments = {
            "recipient_actor_id": "tenant",
            "message_fixture_id": "fixture_1",
            "correlation_id": None,
            "wait": wait_arguments(case),
        }
        without_correlation = {
            name: value for name, value in arguments.items() if name != "correlation_id"
        }

        problem = schema_problem(parameters_of("ask"), arguments)
        omitted = schema_problem(parameters_of("ask"), without_correlation)

        assert (problem is None) is alive, f"the nested wait disagrees about {label}"
        assert (omitted is None) is alive, f"the nested wait disagrees about {label}"

    @pytest.mark.parametrize(("label", "case", "alive"), WAIT_CASES)
    def test_core_itself_answers_the_same_way(
        self, label: str, case: dict[str, Any], alive: bool
    ) -> None:
        """The oracle the schema is being held to, stated rather than assumed."""
        try:
            wait = Wait(
                reason="awaiting the contractor",
                wake_on=tuple(case.get("wake_on") or ()),
                fallback_after_minutes=case.get("fallback_after_minutes"),
            )
        except Exception:
            assert not alive, f"Core refused {label} and this table calls it alive"
            return
        assert alive, f"Core accepted {label} and this table calls it dead"
        assert outcome_contract_problem(wait) is None

    def test_a_non_positive_fallback_is_refused_by_the_schema(self) -> None:
        for minutes in (0, -1):
            problem = schema_problem(
                parameters_of("wait"),
                wait_arguments({"wake_on": None, "fallback_after_minutes": minutes}),
            )
            assert problem is not None

    def test_an_empty_wake_event_type_is_refused_by_the_schema(self) -> None:
        problem = schema_problem(
            parameters_of("wait"),
            wait_arguments({"wake_on": [""], "fallback_after_minutes": None}),
        )

        assert problem is not None

    def test_the_parser_still_refuses_a_dead_wait_on_its_own(self) -> None:
        """The schema is a second wall, never a replacement for the first.

        A provider that ignored the ``anyOf`` — or a build talking to one that
        does not enforce it — reaches the same refusal it always did, because
        the outcome parser never learned to trust the schema.
        """
        from operatebench.agents.model import ModelToolArgumentsMalformed

        wire, client = scripted_client(
            responses_body(
                [
                    function_call_item(
                        "wait",
                        json.dumps(
                            {
                                "reason": "waiting",
                                "wake_on": None,
                                "fallback_after_minutes": None,
                            }
                        ),
                    )
                ],
                model=MODEL,
            )
        )
        agent = ModelAgent(
            transport_over(client, clock=ScriptedClock([0.0])),
            model=MODEL,
            agent_id=MODEL_AGENT_ID,
        )
        agent.begin_episode({})

        decision = agent.decide(observation())

        assert wire.calls == 1
        assert isinstance(decision, ModelToolArgumentsMalformed)

    def test_a_required_field_left_null_is_still_malformed_after_the_elision(
        self,
    ) -> None:
        """Only semantic optionals are elided; a required null is classified."""
        from operatebench.agents.model import ModelToolArgumentsMalformed
        from operatebench.agents.openai_responses import elide_null_optionals

        arguments = {
            "action_type": None,
            "payload": None,
            "evidence_refs": None,
            "rationale": None,
        }

        elided = elide_null_optionals("act", arguments)

        assert elided == {"action_type": None}
        assert schema_problem(parameters_of("act"), arguments) is not None
        _wire, client = scripted_client(
            responses_body(
                [function_call_item("act", json.dumps(arguments))], model=MODEL
            )
        )
        agent = ModelAgent(
            transport_over(client, clock=ScriptedClock([0.0])),
            model=MODEL,
            agent_id=MODEL_AGENT_ID,
        )
        agent.begin_episode({})

        assert isinstance(agent.decide(observation()), ModelToolArgumentsMalformed)


class TestStrictModeCannotStateTheCoreContract:
    """Why these tools are sent non-strict, proved rather than asserted.

    A claim that these schemas stay inside OpenAI's documented strict subset
    would require exempting the free-form payload that leaves the subset. That
    exemption would be an invention, not a property of the schema.

    So the question is asked the other way round here. The generated schemas are
    held to the documented rules **without exemption** and they fail; the ways
    they fail are exactly two Core semantics; and each of those semantics is
    shown, on a concrete value, to be lost by the narrowing that would put the
    schema back inside the subset. That is the kill criterion this cycle was
    given: drop strict rather than weaken Core.

    Everything here is offline. No request is made to confirm what a provider
    server would do with either form, because confirming it that way needs a
    credential and a live call this phase does not have and does not want — and
    because under ``strict=false`` the answer is "nothing is enforced", which is
    why the parser and not the schema is what refuses.
    """

    def test_every_tool_on_the_wire_states_strict_false(self) -> None:
        """The wire body, not the projection: what the provider is actually told."""
        wire, client = answered_client()
        transport = transport_over(client, clock=ScriptedClock([0.0]))

        transport.send(request_for())

        tools = wire.bodies[0]["tools"]
        assert len(tools) == len(AGENT_TOOL_NAMES) == 6
        for tool in tools:
            assert "strict" in tool, "omission is not an audit trail; false is"
            assert tool["strict"] is False, tool["name"]

    def test_the_projection_states_strict_false_too(self) -> None:
        for tool in outcome_tools():
            assert tool["strict"] is False, tool["name"]

    def test_the_schemas_do_not_fit_the_documented_strict_subset(self) -> None:
        """Held to the rules as documented, without the carve-out, they fail."""
        problems = [
            (f"{tool['name']}.{path}", reason)
            for tool in outcome_tools()
            for path, reason in strict_subset_problems(tool["parameters"], "parameters")
        ]

        assert problems, "if these fit strict mode, strict mode is what to send"

    def test_the_only_ways_they_leave_it_are_the_two_core_semantics(self) -> None:
        """Named, and no third one that a narrower schema could have avoided.

        ``STRICT_OPTIONAL_NOT_REQUIRED`` is here as well, and it is the one that
        strict mode *could* have expressed — the v2 mapping did, with nullable
        unions and an elision. It is listed because it is now a consequence of
        the same decision rather than a fifth problem: once the tools are
        non-strict there is no reason to state a ``required`` that is not Core's.
        No keyword outside the documented set is stated by any of them, which is
        the one strict-subset rule this build has no reason to break.
        """
        reasons = {
            reason
            for tool in outcome_tools()
            for _path, reason in strict_subset_problems(tool["parameters"])
        }

        assert reasons == {
            STRICT_ROOT_UNION,
            STRICT_OPEN_OBJECT,
            STRICT_OPTIONAL_NOT_REQUIRED,
        }
        assert STRICT_UNSUPPORTED_KEYWORD not in reasons

    def test_the_open_object_is_the_act_payload_and_nothing_else(self) -> None:
        found = {
            f"{tool['name']}.{path}"
            for tool in outcome_tools()
            for path, reason in strict_subset_problems(tool["parameters"])
            if reason == STRICT_OPEN_OBJECT
        }

        # Two open objects now, and both are open for the same reason: the
        # *domain* owns what an action's arguments are and what a read takes, so
        # any property list here would be this projection inventing a contract
        # Core does not have.
        assert found == {
            "act.parameters.payload",
            "retrieve.parameters.requests[].arguments",
        }

    def test_the_root_union_is_the_standalone_wait_and_nothing_else(self) -> None:
        found = {
            tool["name"]
            for tool in outcome_tools()
            for _path, reason in strict_subset_problems(tool["parameters"])
            if reason == STRICT_ROOT_UNION
        }

        assert found == {"wait"}

    def test_closing_the_payload_for_strict_refuses_a_payload_core_accepts(self) -> None:
        """The first kill criterion, on a value.

        The strict-compatible ``ACT`` payload is a closed object, and the only
        closed object this projection could honestly declare is one with no
        properties — Core's payload contract names no keys, the domain does. So
        the narrowing that satisfies strict mode refuses every payload that
        states anything, which is every payload the reference agent proposes.
        """
        stated = {"quote_id": "q1", "amount_pence": 12500}
        arguments = {"action_type": "submit_quote", "payload": stated}
        narrowed = {
            **parameters_of("act"),
            "properties": {
                **parameters_of("act")["properties"],
                "payload": {
                    "type": ["object", "null"],
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }

        assert schema_problem(parameters_of("act"), arguments) is None
        assert schema_problem(narrowed, arguments) is not None
        assert (
            outcome_contract_problem(Act(action_type="submit_quote", payload=stated))
            is None
        )
        assert strict_subset_problems(narrowed["properties"]["payload"], root=False) == []

    def test_requiring_optionals_for_strict_refuses_a_core_valid_omission(self) -> None:
        """Strict-required optionals reject an omission the parser accepts."""
        arguments = {"action_type": "submit_quote"}
        current = parameters_of("act")
        narrowed = {**current, "required": list(current["properties"])}

        assert schema_problem(current, arguments) is None
        assert schema_problem(narrowed, arguments) is not None
        assert parse_tool_call(ToolCall("act", arguments)) == Act(
            action_type="submit_quote"
        )
        assert STRICT_OPTIONAL_NOT_REQUIRED not in {
            reason for _path, reason in strict_subset_problems(narrowed)
        }

    def test_dropping_the_root_union_for_strict_admits_a_wait_core_refuses(self) -> None:
        """The second kill criterion, on a value.

        A standalone ``WAIT``'s liveness rule is a disjunction over its two
        optional fields, and the only place it can be stated for the whole
        outcome is the root of the tool's schema. Strict mode prohibits a union
        there, so the strict-compatible ``WAIT`` is the root without its
        branches — which admits exactly the wait that can never end.
        """
        dead = wait_arguments({"wake_on": None, "fallback_after_minutes": None})
        widened = {
            name: value
            for name, value in parameters_of("wait").items()
            if name != "anyOf"
        }

        assert schema_problem(parameters_of("wait"), dead) is not None
        assert schema_problem(widened, dead) is None
        with pytest.raises(WaitContractError):
            Wait(reason="awaiting the contractor")
        assert STRICT_ROOT_UNION not in {
            reason for _path, reason in strict_subset_problems(widened)
        }

    def test_the_wait_branches_are_closed_objects_over_the_same_fields(self) -> None:
        """Each branch is a whole ``WAIT``, minus the optionals it does not need.

        Closed and stating every property, as before — a branch that named only
        the field it is about would admit an ``ASK``'s ``wait`` with a stray key
        — but ``required`` is Core's now plus the one field that branch exists
        to require stated. That is what makes an omitted optional legal and a
        wait with neither still refused.
        """
        branches = parameters_of("wait")["anyOf"]
        root = parameters_of("wait")
        core_required = list(required_field_names("wait"))

        assert len(branches) == 2
        for branch in branches:
            assert branch["type"] == "object"
            assert branch["additionalProperties"] is False
            assert sorted(branch["properties"]) == sorted(root["properties"])
            alive = sorted(set(branch["required"]) - set(core_required))
            assert len(alive) == 1
            assert alive[0] in optional_field_names("wait")

    def test_the_nested_wait_is_the_same_branch_set_as_the_standalone_one(self) -> None:
        nested = parameters_of("ask")["properties"]["wait"]

        assert nested["anyOf"] == parameters_of("wait")["anyOf"]

    def test_the_branches_are_derived_from_the_outcome_contract(self) -> None:
        """One branch per optional ``WAIT`` field, and no hand-written table."""
        optional = [field.name for field in WAIT_FIELDS if not field.required]
        branches = parameters_of("wait")["anyOf"]

        stated = [sorted(set(branch["required"]) & set(optional)) for branch in branches]
        assert stated == [[name] for name in optional]

    def test_the_body_that_goes_on_the_wire_carries_the_exact_schemas(self) -> None:
        """The SDK serialises them: what is compared is the request, not the local
        projection alone.

        Non-strict means the provider is free to ignore all of this, which is
        exactly why what left has to be what this build believes it sent: the
        schemas are guidance to the model, and the only thing they can be
        audited against later is the recorded body.
        """
        wire, client = answered_client()
        transport = transport_over(client, clock=ScriptedClock([0.0]))

        transport.send(request_for())

        sent = {tool["name"]: tool for tool in wire.bodies[0]["tools"]}
        for tool in outcome_tools():
            assert sent[tool["name"]] == tool
        assert sent["wait"]["parameters"]["anyOf"] == parameters_of("wait")["anyOf"]
        assert (
            sent["ask"]["parameters"]["properties"]["wait"]["anyOf"]
            == parameters_of("wait")["anyOf"]
        )
        assert sent["act"]["parameters"]["properties"]["payload"] == {
            "type": ["object", "null"],
            "description": parameters_of("act")["properties"]["payload"]["description"],
        }


class TestTheSchemaIsGuidanceAndTheParserIsAuthoritative:
    """Nothing here relies on a provider having enforced anything.

    The local validator says what this build's JSON Schema means. It does not
    say what a provider does with it, and under ``strict=false`` the honest
    assumption is that a provider does nothing with it at all. So for every
    shape the schema refuses, the thing that actually refuses on the wire is
    asked the same question — and answers the same way, with no schema in the
    path.
    """

    @pytest.mark.parametrize(
        ("label", "arguments"),
        [
            ("a wait with neither liveness field", {"reason": "waiting"}),
            (
                "a wait with an empty wake list",
                {"reason": "waiting", "wake_on": [], "fallback_after_minutes": None},
            ),
            (
                "a wait with a non-positive fallback",
                {"reason": "waiting", "fallback_after_minutes": 0},
            ),
        ],
    )
    def test_what_the_schema_refuses_the_parser_refuses_without_it(
        self, label: str, arguments: dict[str, Any]
    ) -> None:
        from operatebench.agents.model import ModelToolArgumentsMalformed

        assert schema_problem(parameters_of("wait"), arguments) is not None, label

        wire, client = scripted_client(
            responses_body(
                [function_call_item("wait", json.dumps(arguments))], model=MODEL
            )
        )
        agent = ModelAgent(
            transport_over(client, clock=ScriptedClock([0.0])),
            model=MODEL,
            agent_id=MODEL_AGENT_ID,
        )
        agent.begin_episode({})

        decision = agent.decide(observation())

        assert wire.calls == 1
        assert isinstance(decision, ModelToolArgumentsMalformed), label


# =================================================== 4. one deadline, one origin


class TestTheDeadlineHasOneOrigin:
    """Time spent before the executor is time spent out of the turn's budget."""

    def test_the_sdk_timeout_deducts_the_work_done_before_the_executor(self) -> None:
        """Ten seconds configured, four spent getting there, six dispatched."""
        wire, client = answered_client()
        clock = ScriptedClock([0.0, PRE_EXECUTOR_SECONDS])
        transport = transport_over(client, clock=clock)

        transport.send(request_for())

        assert wire.calls == 1
        timeout = _sdk_timeout(wire.timeouts[0])
        assert abs(timeout - (CONFIGURED_SECONDS - PRE_EXECUTOR_SECONDS)) < TOLERANCE

    def test_a_budget_spent_before_the_executor_dispatches_nothing(self) -> None:
        """No wire call, and the turn ends under its own name."""
        wire, client = answered_client()
        clock = ScriptedClock([0.0, CONFIGURED_SECONDS])
        transport = transport_over(client, clock=clock)

        with pytest.raises(AdapterProviderError) as caught:
            transport.send(request_for())

        assert wire.calls == 0
        assert caught.value.fault == PROVIDER_FAULT_TIMEOUT
        telemetry = transport.last_telemetry()
        assert telemetry.attempts == ()
        assert telemetry.terminal_reason == TURN_END_DEADLINE_EXCEEDED

    def test_the_origin_is_captured_before_the_request_is_even_checked(self) -> None:
        """A refused request still costs the clock reading, and nothing else.

        The origin has to be taken before ``check_model_request``, or the checks
        themselves are outside the budget they exist to protect. The observable
        consequence is that a request refused before dispatch has already read
        the clock exactly once.
        """
        wire, client = answered_client()
        clock = ScriptedClock([0.0])
        transport = transport_over(client, clock=clock)

        with pytest.raises(ModelRequestRefused):
            transport.send(
                replace(request_for(), protocol_version="operatebench.model.v99")
            )

        assert wire.calls == 0
        assert clock.calls == 1

    def test_the_deadline_the_transport_builds_states_its_origin(self) -> None:
        deadline = TurnDeadline(
            remaining_seconds=CONFIGURED_SECONDS,
            cancelled=lambda: False,
            started_at=2.5,
        )

        assert deadline.started_at == 2.5
        assert deadline.remaining_seconds == CONFIGURED_SECONDS


def _sdk_timeout(recorded: Any) -> float:
    """The one timeout the SDK put on the request, from httpx's extensions."""
    assert isinstance(recorded, Mapping), recorded
    stated = {value for value in recorded.values() if value is not None}
    assert len(stated) == 1, recorded
    return float(next(iter(stated)))


class TestTheExecutorReadsTheStatedOrigin:
    """The neutral executor's half of it, driven directly and deterministically."""

    def _executor(
        self, clock: ScriptedClock, *, retry: ProviderRetryPolicy | None = None
    ) -> TurnExecutor[str]:
        return TurnExecutor(
            retry=retry or ProviderRetryPolicy(max_attempts=1),
            sleep=lambda _seconds: None,
            clock=clock,
        )

    def _run(
        self,
        executor: TurnExecutor[str],
        deadline: TurnDeadline,
        budgets: list[float],
    ) -> str:
        def dispatch(budget: float) -> str:
            budgets.append(budget)
            return "answered"

        return executor.run(
            payload={},
            deadline=deadline,
            token_bound=1,
            dispatch=dispatch,
            check_before_dispatch=lambda: None,
            classify=lambda _exception: None,
            check_identity=lambda _response: None,
            usage_of=lambda _response: TokenUsage(input_tokens=1, output_tokens=1),
        )

    def test_a_deadline_without_an_origin_behaves_exactly_as_before(self) -> None:
        """Every existing caller states only a remaining budget. Nothing moves.

        The executor's entry is the origin when none is stated, so a deadline
        built the old way gets the whole remaining budget on its first attempt —
        whatever the clock happened to read before it.
        """
        clock = ScriptedClock([5.0])
        budgets: list[float] = []

        self._run(
            self._executor(clock),
            TurnDeadline(remaining_seconds=CONFIGURED_SECONDS, cancelled=lambda: False),
            budgets,
        )

        assert abs(budgets[0] - CONFIGURED_SECONDS) < TOLERANCE

    def test_a_stated_origin_is_deducted_once_and_not_twice(self) -> None:
        clock = ScriptedClock([PRE_EXECUTOR_SECONDS])
        budgets: list[float] = []

        self._run(
            self._executor(clock),
            TurnDeadline(
                remaining_seconds=CONFIGURED_SECONDS,
                cancelled=lambda: False,
                started_at=0.0,
            ),
            budgets,
        )

        assert abs(budgets[0] - (CONFIGURED_SECONDS - PRE_EXECUTOR_SECONDS)) < TOLERANCE

    def test_a_backoff_is_afforded_out_of_the_same_origin(self) -> None:
        """The retry budget shrinks with the turn, not with the executor's entry.

        Six seconds are left when the executor is entered, and the policy's next
        wait is seven. Measured from the stated origin the retry is abandoned;
        measured from the executor's own entry it would look affordable and the
        turn would sleep past its budget.
        """
        clock = ScriptedClock([PRE_EXECUTOR_SECONDS])
        executor: TurnExecutor[str] = TurnExecutor(
            retry=ProviderRetryPolicy(
                max_attempts=3, initial_backoff_seconds=7.0, backoff_multiplier=1.0
            ),
            sleep=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("an unaffordable backoff must not be slept")
            ),
            clock=clock,
        )

        def dispatch(_budget: float) -> str:
            raise RuntimeError("the provider dropped the connection")

        with pytest.raises(AdapterProviderError):
            executor.run(
                payload={},
                deadline=TurnDeadline(
                    remaining_seconds=CONFIGURED_SECONDS,
                    cancelled=lambda: False,
                    started_at=0.0,
                ),
                token_bound=1,
                dispatch=dispatch,
                check_before_dispatch=lambda: None,
                classify=lambda _exception: Fault(
                    PROVIDER_FAULT_NETWORK_ERROR, True, None
                ),
                check_identity=lambda _response: None,
                usage_of=lambda _response: TokenUsage(),
            )

        assert executor.last_telemetry().terminal_reason == TURN_END_BACKOFF_UNAFFORDABLE

    def test_the_turn_latency_is_still_measured_from_the_executor_s_entry(self) -> None:
        """Telemetry semantics are untouched: this is a budget change, not a clock one."""
        clock = ScriptedClock([PRE_EXECUTOR_SECONDS])
        budgets: list[float] = []

        executor = self._executor(clock)
        self._run(
            executor,
            TurnDeadline(
                remaining_seconds=CONFIGURED_SECONDS,
                cancelled=lambda: False,
                started_at=0.0,
            ),
            budgets,
        )

        assert executor.last_telemetry().turn_latency_seconds == 0.0
