"""The standardized scaffold: exactly what an adapter is shown.

One scaffold across every provider is what makes two model runs comparable, so
the prompt and the action schemas are a versioned, content-addressed artefact
rather than strings assembled at call time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from boundarybench.jsonsafe import (
    JsonSafetyError,
    NestingDepthError,
    canonical_json_bytes,
    ensure_json_safe,
    ensure_raw_json_depth,
)

#: The scaffold file format this build reads.
SCAFFOLD_SCHEMA_VERSION = 1

#: The scaffold this build ships. It lives inside the package rather than beside
#: the example cards so an installed wheel runs the same artefact the repository
#: pins, and so a run can never be made against a scaffold that was not released.
STANDARD_SCAFFOLD: Path = (
    Path(__file__).resolve().parent / "scaffolds" / "standard_scaffold_v1.json"
)

#: The argument types an action schema may declare.
PARAMETER_TYPES: frozenset[str] = frozenset({"string", "array_of_string"})

#: The scaffold action that is a *template* rather than an offer.
#:
#: Every other action the scaffold declares is the protocol itself and means the
#: same thing in every case. This one stands for "whatever workflow actions this
#: case has", and a case names its own — ``dispatch_contractor``,
#: ``confirm_contractor_access`` — in its ``allowed_actions``. So it is never
#: offered as itself: :func:`project_action_surface` mints one tool per workflow
#: action the case declares, from this template.
WORKFLOW_ACTION_TEMPLATE = "perform_workflow_action"

#: The argument a fact-acquisition action names its subject with.
#:
#: The scaffold is what decides which actions take one: an action that declares a
#: parameter under this name is asking the agent for a fact *key*, and is
#: therefore the one kind of action whose argument the case itself can enumerate.
#: Nothing else has to be told which actions those are, which is what keeps the
#: elicitation vocabulary from being restated here.
FACT_PARAMETER = "fact"

#: How this build decides what a fact-acquisition action may be *asked for*,
#: recorded in adapter settings so it is inside ``configuration_id``.
#:
#: Named because it changed. Runs before this contract declared ``fact`` as an
#: unconstrained string while the environment scored the answer against the
#: Cube's own set of retrievable keys, which the agent was never shown — so an
#: agent could ask for a key the policy text made plausible and no Cube can
#: deliver, and be graded against a namespace it had never received. A run
#: recorded under this value offered, on every fact-acquisition
#: action, the exact keys that action can retrieve — as a deterministic enum
#: derived from the Cube's ``facts.<key>.elicitation_action`` — and nothing else.
FACT_AFFORDANCE_CONTRACT = "cube_fact_key_enum_v1"

#: How this build decides what a turn may do, recorded in adapter settings so it
#: is inside ``configuration_id``.
#:
#: Named because it changed. Runs before this contract sent the scaffold's whole
#: action list to the model whatever the case allowed, so a Cube that excluded
#: ``call_tool`` still put a ``call_tool`` tool in front of the model and then
#: refused it from card metadata the model had never seen. A run recorded under
#: this value offered exactly the Cube's
#: ``allowed_actions`` and nothing else.
ACTION_SURFACE_CONTRACT = "cube_allowed_actions_exact_v1"

_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "scaffold_id",
    "scaffold_version",
    "content_digest",
    "system_prompt",
    "actions",
)
_ACTION_KEYS: tuple[str, ...] = ("name", "description", "terminal", "parameters")
_PARAMETER_KEYS: tuple[str, ...] = ("name", "type", "required", "description")

_SCAFFOLD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
#: A prerelease is mandatory: there is no stable scaffold to claim a version for.
_SCAFFOLD_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+-[0-9A-Za-z]+(\.[0-9A-Za-z]+)*$")


class ScaffoldError(ValueError):
    """Base class for every scaffold failure."""


class ScaffoldFormatError(ScaffoldError):
    """The scaffold artefact is unreadable, malformed or internally inconsistent."""


class ScaffoldLeakageError(ScaffoldError):
    """The scaffold exposes grading ground truth the agent must not be shown."""


class ScaffoldDigestError(ScaffoldFormatError):
    """The declared ``content_digest`` is not what the scaffold hashes to.

    The declared and computed digests are carried as attributes so a caller can
    act on the values without parsing the message.
    """

    def __init__(self, message: str, *, declared: str, computed: str) -> None:
        super().__init__(message)
        self.declared = declared
        self.computed = computed


@dataclass(frozen=True)
class ActionParameter:
    """One argument an action accepts.

    ``enum`` is the closed set of values the argument may take, or ``None`` for
    an unconstrained one. It is deliberately *not* a field the scaffold file can
    author: the only enumerable argument in this protocol is a fact key, and
    which keys exist is a property of the Cube being run rather than of the
    scaffold, which is one artefact shared by every case. So a scaffold always
    loads with ``enum=None``, and a constrained parameter exists only in a
    projection — see :func:`project_action_surface`.

    That is also why :meth:`as_dict` omits the key when it is ``None`` rather
    than writing an explicit null. The projection is the same rendering the
    scaffold's own ``content_digest`` is taken over, so a field the file cannot
    state must not appear in it; and an absent constraint is exactly what JSON
    Schema means by an absent ``enum``.
    """

    name: str
    type: str
    required: bool
    description: str
    enum: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        rendered: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
        }
        if self.enum is not None:
            rendered["enum"] = list(self.enum)
        return rendered


@dataclass(frozen=True)
class ActionSchema:
    """One action the scaffold exposes, including the terminal call."""

    name: str
    description: str
    terminal: bool
    parameters: tuple[ActionParameter, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "terminal": self.terminal,
            "parameters": [parameter.as_dict() for parameter in self.parameters],
        }


@dataclass(frozen=True)
class Scaffold:
    """A validated scaffold artefact and its verified content digest."""

    schema_version: int
    scaffold_id: str
    scaffold_version: str
    content_digest: str
    system_prompt: str
    actions: tuple[ActionSchema, ...]
    path: Path

    @property
    def terminal_action(self) -> ActionSchema:
        return next(action for action in self.actions if action.terminal)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scaffold_id": self.scaffold_id,
            "scaffold_version": self.scaffold_version,
            "content_digest": self.content_digest,
            "system_prompt": self.system_prompt,
            "actions": [action.as_dict() for action in self.actions],
        }


class ScaffoldProjectionError(ScaffoldError):
    """The scaffold cannot express the action surface a case declares."""


def _fact_parameter(action: ActionSchema) -> ActionParameter | None:
    """The fact-key argument this action declares, if it declares one."""
    for parameter in action.parameters:
        if parameter.name == FACT_PARAMETER:
            return parameter
    return None


def _checked_fact_keys(keys: Any, *, action: str) -> tuple[str, ...]:
    """The enum one fact-acquisition action is offered, or a named refusal.

    Every rejection here is a caller whose action surface and whose affordance
    disagree, and each of them would otherwise reach a model as an interface it
    cannot use:

    * **no keys at all** — an acquisition channel the case offers and no fact
      arrives through. An unconstrained string would restore the very artifact
      this contract closes, and an empty enum is a schema no value satisfies, so
      neither is put on the wire. A suite containing such a card does not
      validate (:mod:`boundarybench.suite`); this is the projection's own answer
      for anything that reaches it anyway.
    * **a duplicate, blank or non-string key** — an enum that is not a set of
      keys. Offered as-is it would either name the same affordance twice or
      offer one no environment can look up.
    """
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
        raise ScaffoldProjectionError(
            f"the fact keys offered for {action!r} must be a sequence of key "
            f"names, got {type(keys).__name__}"
        )
    listed = list(keys)
    if not listed:
        raise ScaffoldProjectionError(
            f"this case offers the fact-acquisition action {action!r} and declares "
            "no fact key it can retrieve, so there is nothing to offer as its "
            f"{FACT_PARAMETER!r} argument. An unconstrained argument would let a "
            "model ask for a key the case then refuses from metadata it was never "
            "shown, and an empty enum is a schema no value can satisfy; remove the "
            "action from allowed_actions, or declare a fact that arrives through it"
        )
    for key in listed:
        if not isinstance(key, str) or not key.strip():
            raise ScaffoldProjectionError(
                f"the fact keys offered for {action!r} must each be a non-empty "
                f"string, got {key!r}"
            )
    if len(set(listed)) != len(listed):
        duplicates = sorted({key for key in listed if listed.count(key) > 1})
        raise ScaffoldProjectionError(
            f"the fact keys offered for {action!r} name {duplicates} more than "
            "once; the affordance is the set of keys this action can retrieve, and "
            "a key is either retrievable through it or not"
        )
    return tuple(sorted(listed))


def project_action_surface(
    scaffold: Scaffold,
    allowed_actions: Sequence[str],
    *,
    fact_affordances: Mapping[str, Sequence[str]],
) -> tuple[ActionSchema, ...]:
    """The exact set of actions a case offering ``allowed_actions`` exposes.

    One name in, one schema out, in the order the card states them — so the
    tool list, ``TurnRequest.available_actions`` and the names dispatch will
    accept are not three lists that happen to agree but one list, projected
    once.

    Two kinds of name arrive here. A *protocol* action —
    ``read_records``, ``ask_user``, ``call_tool``, ``complete_case`` — means the
    same thing in every case and the scaffold declares it directly, so it is
    handed back as the scaffold wrote it. Anything else is a workflow action
    this case named for itself, and is minted from
    :data:`WORKFLOW_ACTION_TEMPLATE` under its own name.

    A minted tool carries no parameters. The template declares one — the name of
    the action to perform — and under this projection the tool's *name* is that
    argument, so keeping it would ask the model to say the same thing twice and
    leave it a way to say two different things at once.

    ``fact_affordances`` maps each fact-acquisition action the case offers to the
    exact fact keys it can retrieve, and every one of them is required: an
    offered action that declares a :data:`FACT_PARAMETER` argument and is not in
    the mapping is refused rather than left open. The keys become a sorted enum
    on that argument, so the values a model may name and the values the
    environment will deliver are one list rather than two that happen to agree.
    Everything else stays exactly as the scaffold wrote it — a disposition, a
    reason code and an evidence handle are the model's own answer, and
    enumerating any of them would hand over the grading contract.

    What this function must never do is offer something the case will refuse.
    The whole point of it is that the model's options and the environment's are
    the same options: a run that offers a fifth tool and then rejects it from
    metadata the model was not shown is measuring whether a model guesses, which
    is the defect :data:`ACTION_SURFACE_CONTRACT` names. The same defect exists
    one level down, in an action's *arguments* rather than its name, and that is
    what :data:`FACT_AFFORDANCE_CONTRACT` names.
    """
    declared = {action.name: action for action in scaffold.actions}
    template = declared.get(WORKFLOW_ACTION_TEMPLATE)
    for action_name in sorted(fact_affordances):
        if action_name not in allowed_actions:
            raise ScaffoldProjectionError(
                f"scaffold {scaffold.scaffold_id} was asked to offer fact keys for "
                f"{action_name!r}, which this case does not allow. The keys an "
                "action can retrieve are part of that action's own schema, so an "
                "affordance for an action that is not offered describes an "
                "interface nothing exposes"
            )
    surface: list[ActionSchema] = []
    for name in allowed_actions:
        action = declared.get(name)
        if action is not None and name != WORKFLOW_ACTION_TEMPLATE:
            parameter = _fact_parameter(action)
            if parameter is not None:
                if name not in fact_affordances:
                    raise ScaffoldProjectionError(
                        f"this case offers the fact-acquisition action {name!r} and "
                        "states no fact keys for it. An acquisition argument is "
                        "either enumerated by the case or it is an open string the "
                        "environment then grades against a namespace the model "
                        "never saw"
                    )
                keys = _checked_fact_keys(fact_affordances[name], action=name)
                action = replace(
                    action,
                    parameters=tuple(
                        replace(existing, enum=keys)
                        if existing.name == FACT_PARAMETER
                        else existing
                        for existing in action.parameters
                    ),
                )
            surface.append(action)
            continue
        if template is None:
            raise ScaffoldProjectionError(
                f"scaffold {scaffold.scaffold_id} does not declare "
                f"{WORKFLOW_ACTION_TEMPLATE!r}, so it cannot offer the workflow "
                f"action {name!r} this case allows"
            )
        surface.append(
            ActionSchema(
                name=name,
                description=template.description,
                terminal=template.terminal,
                parameters=(),
            )
        )
    return tuple(surface)


#: Names that only ever appear on the grading side of the boundary. A scaffold
#: is the agent-facing artefact, so none of them may appear in it at all.
_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "expected_disposition",
    "expected disposition",
    "expected_primary_reason",
    "expected primary reason",
    "expected_mutations",
    "required_evidence",
    "required evidence",
    "matched_rule",
    "answer_key",
    "answer key",
    "ground_truth",
    "ground truth",
    "disposition_table",
    "state_axis",
    "policy_axis",
    "variant_id",
    "variant_digest",
    "content_digest",
)
#: Policy and state arm identifiers, and the cell ids built from them. Which arm
#: a variant is running is the intervention under measurement.
_ARM_PATTERN = re.compile(r"\b[SP][01]_[SP][01]\b|\b[SP][01]\b")
#: Any pinned digest, which would let an agent recognise a variant it has seen.
_DIGEST_PATTERN = re.compile(r"\b[0-9a-f]{64}\b")


def _leakage_failure(text: str) -> str | None:
    lowered = text.lower()
    for token in _FORBIDDEN_SUBSTRINGS:
        if token in lowered:
            return f"exposes {token!r}, which belongs to the grading contract"
    arm = _ARM_PATTERN.search(text)
    if arm is not None:
        return (
            f"names {arm.group(0)!r}, which is a policy/state arm identifier and "
            "would reveal the intervention under measurement"
        )
    if _DIGEST_PATTERN.search(text) is not None:
        return "contains a value that looks like a pinned digest"
    return None


def _reject_leakage(value: Any, context: str) -> None:
    """Walk the whole document, refusing any grading-side name it carries.

    The scaffold is the only thing an adapter is shown, so this is the boundary
    between "what the agent may observe" and "what the evaluator knows". A leak
    here silently invalidates every run made with the artefact, so it is a
    load-time failure rather than a review convention.
    """
    if isinstance(value, str):
        failure = _leakage_failure(value)
        if failure is not None:
            raise ScaffoldLeakageError(f"{context} {failure}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            failure = _leakage_failure(key_text)
            if failure is not None:
                raise ScaffoldLeakageError(f"{context}: key {key_text!r} {failure}")
            _reject_leakage(item, f"{context}.{key_text}")
        return
    if not isinstance(value, (str, bytes)) and isinstance(value, Sequence):
        for index, item in enumerate(value):
            _reject_leakage(item, f"{context}[{index}]")


def scaffold_digest(payload: Mapping[str, Any]) -> str:
    """Canonical SHA-256 over the scaffold document, minus the pin itself.

    Explicit UTF-8 JSON with sorted keys and compact separators, so reflowing
    the file or reordering its keys is free while any change to the prompt or an
    action schema moves the digest. A payload JSON cannot carry back unchanged —
    a non-finite float, a cycle, or a system prompt holding a lone UTF-16
    surrogate — is a named scaffold failure taken before any bytes exist, rather
    than a ``UnicodeEncodeError`` raised from inside the hash.
    ``content_digest`` is the one excluded field: a payload that hashed its own
    pin would have no fixed point, so the pin could never be stated.
    """
    body = {key: value for key, value in payload.items() if key != "content_digest"}
    try:
        return hashlib.sha256(canonical_json_bytes(body, "scaffold")).hexdigest()
    except JsonSafetyError as exc:
        raise ScaffoldFormatError(f"this scaffold cannot be digested: {exc}") from exc


def scaffold_content_digest(scaffold: Scaffold) -> str:
    """What a loaded scaffold hashes to *now*, from the fields it carries now.

    Public because ``content_digest`` travels inside the object it describes,
    which makes it a self-assertion: ``dataclasses.replace`` rebuilds a frozen
    :class:`Scaffold` with a different ``system_prompt`` and the original pin,
    and every comparison against a recorded digest then agrees. A caller that is
    about to execute under a scaffold has to derive the digest again from what
    it is actually holding, and this is the one place that derivation lives.
    """
    return scaffold_digest(scaffold.as_dict())


def _verify_digest(raw: Mapping[str, Any], declared: str, context: str) -> None:
    computed = scaffold_digest(raw)
    if declared != computed:
        raise ScaffoldDigestError(
            f"{context}: declared content_digest {declared} does not match the "
            f"computed digest {computed}. The scaffold's identity has drifted; a "
            "prompt or tool-schema revision requires an explicit, reviewed re-pin.",
            declared=declared,
            computed=computed,
        )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Stock ``json`` keeps the last duplicate; a pinned artefact cannot.

    A scaffold whose ``system_prompt`` is stated twice would be digested over
    whichever copy the parser happened to keep, which is exactly the silent
    ambiguity the pin exists to remove.
    """
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise ScaffoldFormatError(
                f"duplicate JSON key {key!r}; a scaffold must state each key exactly once"
            )
        seen.add(key)
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise ScaffoldFormatError(
        f"{name} is not a finite JSON value; a scaffold carries only strings, "
        "integers, booleans and lists"
    )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ScaffoldFormatError(
            f"{path} is not valid UTF-8: a scaffold must be a UTF-8 text file "
            f"({exc.reason} at byte {exc.start})"
        ) from exc
    except OSError as exc:
        raise ScaffoldFormatError(f"cannot read scaffold {path}: {exc}") from exc
    try:
        ensure_raw_json_depth(text, str(path))
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except NestingDepthError as exc:
        raise ScaffoldFormatError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise ScaffoldFormatError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScaffoldFormatError(
            f"{path}: a scaffold must be a JSON object, got {type(payload).__name__}"
        )
    try:
        ensure_json_safe(payload, f"{path}: scaffold")
    except JsonSafetyError as exc:
        raise ScaffoldFormatError(str(exc)) from exc
    return payload


def _require_exact_keys(
    raw: Mapping[str, Any], allowed: tuple[str, ...], context: str
) -> None:
    missing = [key for key in allowed if key not in raw]
    if missing:
        raise ScaffoldFormatError(f"{context}: missing required field(s) {missing}")
    unknown = sorted(set(map(str, raw)) - set(allowed))
    if unknown:
        raise ScaffoldFormatError(
            f"{context}: unknown field(s) {unknown}; allowed: {sorted(allowed)}"
        )


def _text(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ScaffoldFormatError(f"{context}: {key!r} must be a non-empty string")
    return value


def _flag(raw: Mapping[str, Any], key: str, context: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise ScaffoldFormatError(f"{context}: {key!r} must be true or false")
    return value


def _list(raw: Mapping[str, Any], key: str, context: str) -> Sequence[Any]:
    value = raw[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScaffoldFormatError(f"{context}: {key!r} must be a list")
    return value


def _parse_parameter(entry: Any, context: str) -> ActionParameter:
    if not isinstance(entry, Mapping):
        raise ScaffoldFormatError(f"{context} must be a JSON object")
    _require_exact_keys(entry, _PARAMETER_KEYS, context)
    parameter_type = _text(entry, "type", context)
    if parameter_type not in PARAMETER_TYPES:
        raise ScaffoldFormatError(
            f"{context}: unknown parameter type {parameter_type!r}; allowed: "
            f"{sorted(PARAMETER_TYPES)}"
        )
    return ActionParameter(
        name=_text(entry, "name", context),
        type=parameter_type,
        required=_flag(entry, "required", context),
        description=_text(entry, "description", context),
    )


def _parse_action(entry: Any, context: str) -> ActionSchema:
    if not isinstance(entry, Mapping):
        raise ScaffoldFormatError(f"{context} must be a JSON object")
    _require_exact_keys(entry, _ACTION_KEYS, context)
    parameters = _list(entry, "parameters", context)
    seen: set[str] = set()
    parsed: list[ActionParameter] = []
    for index, parameter in enumerate(parameters):
        parameter_context = f"{context}: parameters[{index}]"
        record = _parse_parameter(parameter, parameter_context)
        if record.name in seen:
            raise ScaffoldFormatError(
                f"{parameter_context}: duplicate parameter name {record.name!r}"
            )
        seen.add(record.name)
        parsed.append(record)
    return ActionSchema(
        name=_text(entry, "name", context),
        description=_text(entry, "description", context),
        terminal=_flag(entry, "terminal", context),
        parameters=tuple(parsed),
    )


def _parse_actions(raw: Mapping[str, Any], context: str) -> tuple[ActionSchema, ...]:
    section = _list(raw, "actions", context)
    if not section:
        raise ScaffoldFormatError(f"{context}: 'actions' must not be empty")
    actions: list[ActionSchema] = []
    seen: set[str] = set()
    for index, entry in enumerate(section):
        action = _parse_action(entry, f"{context}: actions[{index}]")
        if action.name in seen:
            raise ScaffoldFormatError(f"{context}: duplicate action name {action.name!r}")
        seen.add(action.name)
        actions.append(action)
    terminals = [action.name for action in actions if action.terminal]
    if len(terminals) != 1:
        raise ScaffoldFormatError(
            f"{context}: a scaffold must expose exactly one terminal action, got "
            f"{terminals}"
        )
    return tuple(actions)


def load_scaffold(path: str | Path) -> Scaffold:
    """Read and validate a scaffold under the public JSON depth limit."""
    return _load_scaffold(Path(path))


def _load_scaffold(path: Path) -> Scaffold:
    raw = _read_json_object(path)
    context = str(path)

    # Leakage is checked before the shape is, and on the raw document rather than
    # the parsed one. A field named for the answer key would otherwise be
    # reported as an unknown field — the same rejection, but the wrong
    # diagnosis: an unknown field is a typo, a leak invalidates every run made
    # with the artefact. The pin is the one 64-hex value the document may carry.
    _reject_leakage(
        {key: value for key, value in raw.items() if key != "content_digest"},
        context,
    )

    _require_exact_keys(raw, _REQUIRED_KEYS, context)

    version = raw["schema_version"]
    # bool is an int subclass, so the type check has to be exact.
    if type(version) is not int or version != SCAFFOLD_SCHEMA_VERSION:
        raise ScaffoldFormatError(
            f"{context}: unsupported schema_version {version!r}; this build reads "
            f"{SCAFFOLD_SCHEMA_VERSION}"
        )

    scaffold_id = _text(raw, "scaffold_id", context)
    if not _SCAFFOLD_ID_PATTERN.fullmatch(scaffold_id):
        raise ScaffoldFormatError(
            f"{context}: scaffold_id {scaffold_id!r} must use only lowercase "
            "letters, digits, underscores and hyphens"
        )

    scaffold_version = _text(raw, "scaffold_version", context)
    if not _SCAFFOLD_VERSION_PATTERN.fullmatch(scaffold_version):
        raise ScaffoldFormatError(
            f"{context}: scaffold_version {scaffold_version!r} must be an explicit "
            "prerelease such as '0.1.0-dev.1'; there is no stable scaffold version "
            "to claim"
        )

    declared = raw["content_digest"]
    if not isinstance(declared, str) or not _SHA256_PATTERN.fullmatch(declared):
        raise ScaffoldFormatError(
            f"{context}: 'content_digest' must be a lowercase 64-character SHA-256 "
            f"hex digest, got {declared!r}"
        )

    actions = _parse_actions(raw, context)
    _verify_digest(raw, declared, context)
    return Scaffold(
        schema_version=version,
        scaffold_id=scaffold_id,
        scaffold_version=scaffold_version,
        content_digest=declared,
        system_prompt=_text(raw, "system_prompt", context),
        actions=actions,
        path=path,
    )


__all__ = [
    "ACTION_SURFACE_CONTRACT",
    "FACT_AFFORDANCE_CONTRACT",
    "FACT_PARAMETER",
    "PARAMETER_TYPES",
    "SCAFFOLD_SCHEMA_VERSION",
    "STANDARD_SCAFFOLD",
    "WORKFLOW_ACTION_TEMPLATE",
    "ActionParameter",
    "ActionSchema",
    "Scaffold",
    "ScaffoldDigestError",
    "ScaffoldError",
    "ScaffoldFormatError",
    "ScaffoldLeakageError",
    "ScaffoldProjectionError",
    "load_scaffold",
    "project_action_surface",
    "scaffold_content_digest",
    "scaffold_digest",
]
