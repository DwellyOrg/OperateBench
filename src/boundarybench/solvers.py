"""Reference and targeted negative solvers (causal CI, spec §17.1).

These are not agents. They are deterministic programs whose only purpose is to
prove that the evaluator's predicates have bite: the reference solver must pass
every variant, and each negative solver must fail the specific predicate it was
built to trip, while still passing the variants its defect does not touch.

If a negative solver stops failing, either the cube or the evaluator has lost a
discrimination it was supposed to have.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from boundarybench.compiler import Cube, Variant, attempted_rules
from boundarybench.environment import Environment
from boundarybench.evaluator import VariantEvaluation, evaluate
from boundarybench.schema import PolicyArm, Rule
from boundarybench.trajectory import Observation, Trajectory

#: The policy arm a policy-ignorant agent keeps applying out of habit.
HABITUAL_POLICY = "P0"
#: The state arm a state-ignorant agent assumes without checking.
ASSUMED_STATE = "S0"

Solver = Callable[[Variant, Cube], Trajectory]
Selector = Callable[[Cube], tuple[tuple[str, str], ...]]


class SolverError(RuntimeError):
    """Raised when a solver cannot reach a terminal decision."""


# -- shared machinery -------------------------------------------------------


def _presented_rules(env: Environment) -> tuple[Rule, ...]:
    """Read the active policy out of the environment, not out of the answer key."""
    for observation in env.observed():
        if observation.source == "active_policy":
            return tuple(
                Rule.from_dict(raw, "presented_policy")
                for raw in observation.fact["rules"]
            )
    raise SolverError("the active policy was not observed")


def _policy_handle(env: Environment) -> str:
    for observation in env.observed():
        if observation.source == "active_policy":
            return observation.observation_id
    raise SolverError("the active policy was not observed")


def _apply(
    env: Environment,
    rules: tuple[Rule, ...],
    *,
    assumed_facts: Mapping[str, Any] | None = None,
) -> tuple[Rule, tuple[str, ...]]:
    """Walk the rules in order, obtaining exactly the facts each one needs.

    Returns the first matching rule and the observation handles it rests on.
    With ``assumed_facts`` the caller supplies fact values instead of eliciting
    them, which is how the state-ignorant solver skips the state axis.

    Each question is asked once across the whole walk, tracked here rather than
    left to the environment. Successive rules routinely reference the same fact
    — a threshold and its complement, most of all — and the environment records
    a repeated request as a step of its own, because a dispatched action is a
    recorded action. These baselines model an agent that asks what it needs and
    stops asking, so the second rule reuses the first one's answer instead of
    filing a step that means nothing about the strategy under test.
    """
    cited: list[str] = [_policy_handle(env)]
    asked: set[str] = set()
    facts: Mapping[str, Any]
    for rule in rules:
        if assumed_facts is None:
            for fact_key in rule.referenced_facts:
                if fact_key in asked:
                    continue
                asked.add(fact_key)
                observation = env.obtain(fact_key)
                if observation.observation_id not in cited:
                    cited.append(observation.observation_id)
            facts = env.known_facts()
        else:
            facts = assumed_facts
        if rule.matches(facts):
            return rule, tuple(cited)
    raise SolverError("no presented rule covers the observed facts")


def _finish(
    env: Environment,
    *,
    disposition: str,
    primary_reason_code: str,
    evidence_refs: tuple[str, ...],
) -> Trajectory:
    if disposition == "ACT":
        for action in env.variant.irreversible_actions:
            env.act(action)
    env.complete_case(
        disposition=disposition,
        primary_reason_code=primary_reason_code,
        evidence_refs=evidence_refs,
    )
    return env.trajectory()


# -- solvers ----------------------------------------------------------------


def solve_reference(variant: Variant, cube: Cube) -> Trajectory:
    """Apply the presented policy to the observed state, ignoring pressure."""
    env = Environment(variant)
    env.read_records()
    rule, refs = _apply(env, _presented_rules(env))
    return _finish(
        env,
        disposition=rule.decision,
        primary_reason_code=rule.primary_reason,
        evidence_refs=refs,
    )


def solve_policy_ignorant(variant: Variant, cube: Cube) -> Trajectory:
    """Keep applying the habitual policy arm whatever the active policy says."""
    env = Environment(variant)
    env.read_records()
    rule, refs = _apply(env, cube.card.policy_axis[HABITUAL_POLICY].rules)
    return _finish(
        env,
        disposition=rule.decision,
        primary_reason_code=rule.primary_reason,
        evidence_refs=refs,
    )


def solve_state_ignorant(variant: Variant, cube: Cube) -> Trajectory:
    """Apply the right policy to an assumed, never-verified operational state."""
    env = Environment(variant)
    env.read_records()
    rule, refs = _apply(
        env,
        _presented_rules(env),
        assumed_facts=cube.card.facts_for(ASSUMED_STATE),
    )
    return _finish(
        env,
        disposition=rule.decision,
        primary_reason_code=rule.primary_reason,
        evidence_refs=refs,
    )


def solve_pressure_triggered(variant: Variant, cube: Cube) -> Trajectory:
    """Reason correctly, then let irrelevant social pressure flip the decision."""
    env = Environment(variant)
    env.read_records()
    rule, refs = _apply(env, _presented_rules(env))
    flipped = {"alarmist": "STOP", "reassurance": "ACT"}
    disposition = flipped.get(variant.pressure_direction or "", rule.decision)
    return _finish(
        env,
        disposition=disposition,
        primary_reason_code=rule.primary_reason,
        evidence_refs=refs,
    )


def solve_premature_action(variant: Variant, cube: Cube) -> Trajectory:
    """Commit the irreversible action first, then work out the right answer."""
    env = Environment(variant)
    env.read_records()
    for action in variant.irreversible_actions:
        env.act(action)
    rule, refs = _apply(env, _presented_rules(env))
    env.complete_case(
        disposition=rule.decision,
        primary_reason_code=rule.primary_reason,
        evidence_refs=refs,
    )
    return env.trajectory()


def solve_wrong_evidence(variant: Variant, cube: Cube) -> Trajectory:
    """Reach the right decision, then ground it in the wrong observations."""
    env = Environment(variant)
    env.read_records()
    rule, _ = _apply(env, _presented_rules(env))
    decoys = tuple(
        observation.observation_id
        for observation in env.observed()
        if not observation.evidence_relevant
    )
    return _finish(
        env,
        disposition=rule.decision,
        primary_reason_code=rule.primary_reason,
        evidence_refs=decoys,
    )


# -- causal expectations ----------------------------------------------------

ALL_PREDICATES = (
    "disposition",
    "primary_reason",
    "evidence",
    "mutations",
    "ordering",
    "invariants",
)


def _every_predicate(variants: tuple[Variant, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (variant.variant_id, predicate)
        for variant in variants
        for predicate in ALL_PREDICATES
    )


def _required_observations(variant: Variant) -> tuple[Observation, ...]:
    """The observations a variant's required evidence actually names.

    An unresolved handle means the compiled variant is inconsistent with itself,
    which is a structural error rather than something to paper over.
    """
    resolved: list[Observation] = []
    for ref in variant.required_evidence_ids:
        observation = variant.observation(ref)
        if observation is None:
            raise SolverError(
                f"{variant.variant_id}: required evidence {ref!r} does not name "
                "an observation of this variant"
            )
        resolved.append(observation)
    return tuple(resolved)


def _required_fact_keys(cube: Cube, variant: Variant) -> frozenset[str]:
    """Deciding-fact keys among a variant's required evidence."""
    declared = set(cube.card.facts)
    return frozenset(
        observation.key
        for observation in _required_observations(variant)
        if observation.key in declared
    )


def _has_on_request_required_evidence(variant: Variant) -> bool:
    return any(
        observation.availability == "on_request"
        for observation in _required_observations(variant)
    )


def _resolution(
    arm: PolicyArm, facts: Mapping[str, Any]
) -> tuple[str, str, frozenset[str]]:
    """What an arm decides on these facts, and what that rests on."""
    walked = attempted_rules(arm, facts)
    matched = walked[-1]
    keys = {key for rule in walked for key in rule.referenced_facts}
    return matched.decision, matched.primary_reason, frozenset(keys)


def _first_divergence(
    expected: tuple[str, str, frozenset[str]],
    actual: tuple[str, str, frozenset[str]],
) -> str | None:
    """The highest-order predicate a wrong resolution must break, if any."""
    if expected[0] != actual[0]:
        return "disposition"
    if expected[1] != actual[1]:
        return "primary_reason"
    if expected[2] != actual[2]:
        return "evidence"
    return None


def _reference_preserved(cube: Cube) -> tuple[tuple[str, str], ...]:
    return _every_predicate(cube.variants)


def _policy_ignorant_targets(cube: Cube) -> tuple[tuple[str, str], ...]:
    """Whatever applying the habitual arm must get wrong, variant by variant."""
    habitual = cube.card.policy_axis[HABITUAL_POLICY]
    targets: list[tuple[str, str]] = []
    for variant in cube.variants:
        if variant.policy == HABITUAL_POLICY:
            continue
        facts = cube.card.facts_for(variant.state)
        divergence = _first_divergence(
            (
                variant.expected_disposition,
                variant.expected_primary_reason,
                _required_fact_keys(cube, variant),
            ),
            _resolution(habitual, facts),
        )
        if divergence is not None:
            targets.append((variant.variant_id, divergence))
    return tuple(targets)


def _policy_ignorant_preserved(cube: Cube) -> tuple[tuple[str, str], ...]:
    """Every variant already under the habitual arm, plus any it cannot distort."""
    habitual = cube.card.policy_axis[HABITUAL_POLICY]
    preserved: list[Variant] = []
    for variant in cube.variants:
        if variant.policy == HABITUAL_POLICY:
            preserved.append(variant)
            continue
        divergence = _first_divergence(
            (
                variant.expected_disposition,
                variant.expected_primary_reason,
                _required_fact_keys(cube, variant),
            ),
            _resolution(habitual, cube.card.facts_for(variant.state)),
        )
        if divergence is None:
            preserved.append(variant)
    return _every_predicate(tuple(preserved))


def _state_ignorant_targets(cube: Cube) -> tuple[tuple[str, str], ...]:
    """Never checking the state costs the evidence, and sometimes the answer."""
    assumed = cube.card.facts_for(ASSUMED_STATE)
    targets: list[tuple[str, str]] = []
    for variant in cube.variants:
        if not _required_fact_keys(cube, variant):
            continue
        # It cites no deciding fact at all, so the exact-match evidence
        # predicate must fail wherever a deciding fact is load-bearing.
        targets.append((variant.variant_id, "evidence"))
        decision, _reason, _keys = _resolution(
            cube.card.policy_axis[variant.policy], assumed
        )
        if decision != variant.expected_disposition:
            targets.append((variant.variant_id, "disposition"))
    return tuple(targets)


def _state_ignorant_preserved(cube: Cube) -> tuple[tuple[str, str], ...]:
    return _every_predicate(
        tuple(v for v in cube.variants if not _required_fact_keys(cube, v))
    )


def _pressure_targets(cube: Cube) -> tuple[tuple[str, str], ...]:
    return tuple(
        (v.variant_id, "disposition") for v in cube.variants if v.pressure_direction
    )


def _pressure_preserved(cube: Cube) -> tuple[tuple[str, str], ...]:
    return _every_predicate(tuple(v for v in cube.variants if not v.pressure_direction))


def _premature_targets(cube: Cube) -> tuple[tuple[str, str], ...]:
    """Acting first is only detectable where it actually changes the episode."""
    targets: list[tuple[str, str]] = []
    for variant in cube.variants:
        if variant.expected_disposition == "ACT":
            # With every required handle already in the record and read before
            # the action, acting first is behaviourally identical to reference.
            if _has_on_request_required_evidence(variant):
                targets.append((variant.variant_id, "invariants"))
        else:
            targets.append((variant.variant_id, "mutations"))
            targets.append((variant.variant_id, "invariants"))
    return tuple(targets)


def _premature_preserved(cube: Cube) -> tuple[tuple[str, str], ...]:
    preserved: list[tuple[str, str]] = []
    for variant in cube.variants:
        if variant.expected_disposition == "ACT":
            if _has_on_request_required_evidence(variant):
                preserved.extend(
                    (variant.variant_id, predicate)
                    for predicate in ALL_PREDICATES
                    if predicate != "invariants"
                )
            else:
                preserved.extend(_every_predicate((variant,)))
        else:
            preserved.extend(
                (variant.variant_id, predicate)
                for predicate in ALL_PREDICATES
                if predicate not in {"mutations", "invariants"}
            )
    return tuple(preserved)


def _wrong_evidence_targets(cube: Cube) -> tuple[tuple[str, str], ...]:
    return tuple((v.variant_id, "evidence") for v in cube.variants)


def _wrong_evidence_preserved(cube: Cube) -> tuple[tuple[str, str], ...]:
    return tuple(
        (v.variant_id, predicate)
        for v in cube.variants
        for predicate in ("disposition", "primary_reason")
    )


def _none(cube: Cube) -> tuple[tuple[str, str], ...]:
    return ()


@dataclass(frozen=True)
class SolverSpec:
    name: str
    kind: str
    description: str
    solve: Solver
    targets: Selector
    preserved: Selector


SOLVERS: tuple[SolverSpec, ...] = (
    SolverSpec(
        name="reference",
        kind="reference",
        description="Applies the presented policy to the observed state.",
        solve=solve_reference,
        targets=_none,
        preserved=_reference_preserved,
    ),
    SolverSpec(
        name="policy_ignorant",
        kind="negative",
        description=(
            f"Always applies policy {HABITUAL_POLICY}; must fail the live policy edge."
        ),
        solve=solve_policy_ignorant,
        targets=_policy_ignorant_targets,
        preserved=_policy_ignorant_preserved,
    ),
    SolverSpec(
        name="state_ignorant",
        kind="negative",
        description=(
            f"Assumes state {ASSUMED_STATE} without checking; must fail the "
            "evidence wherever a deciding fact is load-bearing, and the "
            "disposition wherever that assumption crosses the boundary."
        ),
        solve=solve_state_ignorant,
        targets=_state_ignorant_targets,
        preserved=_state_ignorant_preserved,
    ),
    SolverSpec(
        name="pressure_triggered",
        kind="negative",
        description="Flips on irrelevant pressure; must fail both probes.",
        solve=solve_pressure_triggered,
        targets=_pressure_targets,
        preserved=_pressure_preserved,
    ),
    SolverSpec(
        name="premature_action",
        kind="negative",
        description=(
            "Reaches the right decision after acting irreversibly; must fail the "
            "invariants on ACT variants whose evidence still had to be elicited, "
            "and the mutations and invariants on every STOP variant. Where all "
            "required evidence was already read, acting first is behaviourally "
            "identical to reference and is preserved."
        ),
        solve=solve_premature_action,
        targets=_premature_targets,
        preserved=_premature_preserved,
    ),
    SolverSpec(
        name="wrong_evidence",
        kind="negative",
        description=(
            "Right outcome, wrong grounding; must fail the evidence predicate "
            "everywhere while keeping outcome and reason correct."
        ),
        solve=solve_wrong_evidence,
        targets=_wrong_evidence_targets,
        preserved=_wrong_evidence_preserved,
    ),
)


def solver(name: str) -> SolverSpec:
    for spec in SOLVERS:
        if spec.name == name:
            return spec
    raise KeyError(name)


# -- report -----------------------------------------------------------------


@dataclass(frozen=True)
class SolverOutcome:
    name: str
    kind: str
    description: str
    evaluations: tuple[VariantEvaluation, ...]
    passed_variants: int
    total_variants: int
    satisfied: bool
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "passed_variants": self.passed_variants,
            "total_variants": self.total_variants,
            "satisfied": self.satisfied,
            "failures": list(self.failures),
            "evaluations": [e.as_dict() for e in self.evaluations],
        }


@dataclass(frozen=True)
class SolverReport:
    cube_id: str
    outcomes: tuple[SolverOutcome, ...]

    @property
    def ok(self) -> bool:
        return all(outcome.satisfied for outcome in self.outcomes)

    def outcome(self, name: str) -> SolverOutcome:
        for item in self.outcomes:
            if item.name == name:
                return item
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cube_id": self.cube_id,
            "ok": self.ok,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def check_solvers(cube: Cube) -> SolverReport:
    """Run every solver over every variant and verify the causal expectations."""
    outcomes: list[SolverOutcome] = []
    for spec in SOLVERS:
        evaluations = tuple(
            evaluate(variant, spec.solve(variant, cube)) for variant in cube.variants
        )
        by_variant = {e.variant_id: e for e in evaluations}
        failures: list[str] = []

        for variant_id, predicate in spec.targets(cube):
            if by_variant[variant_id].predicate(predicate).passed:
                failures.append(
                    f"{spec.name} was expected to fail {predicate!r} on "
                    f"{variant_id} but passed it"
                )
        for variant_id, predicate in spec.preserved(cube):
            result = by_variant[variant_id].predicate(predicate)
            if not result.passed:
                failures.append(
                    f"{spec.name} was expected to pass {predicate!r} on "
                    f"{variant_id} but failed: {result.detail}"
                )

        outcomes.append(
            SolverOutcome(
                name=spec.name,
                kind=spec.kind,
                description=spec.description,
                evaluations=evaluations,
                passed_variants=sum(1 for e in evaluations if e.passed),
                total_variants=len(evaluations),
                satisfied=not failures,
                failures=tuple(failures),
            )
        )
    return SolverReport(cube_id=cube.cube_id, outcomes=tuple(outcomes))
