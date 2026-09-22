# Measurement RFC v2

**Status:** Phase A normative measurement contract; no model evidence
**RFC version:** 2
**Date:** 21 August 2026
**Treatment protocol:** [STATEFUL_TIME_REMOVED_PROTOCOL.md](STATEFUL_TIME_REMOVED_PROTOCOL.md)

Normative terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** apply to later
implementation and campaigns. Statements labelled “baseline” describe current
code only and are not evidence that this RFC has shipped.

## 1. Construct and unit of claim

The v0.x construct is the incremental difficulty of one frozen **bundled
lifecycle protocol**, measured against a matched integrated operation from which
the exact versioned lifecycle/time treatment is removed. The first research unit
is one synthetic UK property-maintenance semantic scenario. Repeated decoding
trials estimate reliability for that fixed scenario; they do not create additional
semantic scenarios.

The primary comparison is:

- `full_lifecycle`: the complete deterministic event/time lifecycle; and
- `stateful_time_removed`: the same substantive operation under the closed
  treatment in the No-Time protocol RFC.

`static`, `boundary_only`, and `ordered_snapshot_control` remain diagnostic or
prerequisite controls. They do not substitute for the confirmatory matched pair.

## 2. Common primary outcome

### 2.1 Mathematical contract

Both primary arms use exactly one tri-state observable primary endpoint `status`:

```text
business_success = legitimate_business_terminal
                   AND no_critical_invariant_violation
                   AND common_decision_obligations_discharged
                   AND common_authority_evidence_finality_predicates_satisfied

status = ERROR    if common evidence or observability is integrity-invalid
         SUCCESS  if evidence is well-formed and business_success is true
         FAILURE  if evidence is well-formed and business_success is false
```

`SUCCESS` and `FAILURE` are observed business verdicts. `ERROR` is an
integrity/observability verdict, not a business verdict. Every assigned trial
identity receives exactly one status. `ERROR` is retained as non-success mass in
the primary estimand and reported separately; it is never coerced to `FAILURE` or
excluded from the assigned denominator.

The contract-only `EpisodeOutcomeV3` integrity tag is a runtime disposition, not this
primary endpoint. It requires a future CommonOutcome `ERROR`; an audit-budget terminal
summary `SUCCESS` states only that the integrity protocol found no error and never
implies business `SUCCESS`. Assigned integrity errors remain non-success mass, are
reported separately from `FAILURE`, permit no retry/replacement/exclusion, and make
affected contrasts `UNINTERPRETABLE`. See
[AUDIT_INTEGRITY_OUTCOME_EVIDENCE_PROTOCOL.md](AUDIT_INTEGRITY_OUTCOME_EVIDENCE_PROTOCOL.md).

No WAIT-, wake-, timer-, cadence-, or deadline-specific metric may alter only one
arm's primary denominator. Those are secondary mechanism metrics.

### 2.2 `CommonOutcomeV1`

Before either arm can produce research evidence, Phase C MUST implement an
independent executable oracle with this conceptual interface:

```text
CommonOutcomeV1(
  ArmNeutralEvidenceProjectionV1
) -> {
  status: SUCCESS | FAILURE | ERROR,
  predicate_verdicts: ordered map<predicate_id, PASS | FAIL | ERROR>,
  finding_codes: ordered unique list<closed code>
}
```

This object is the sole Common Outcome wire representation everywhere: the field
name is exactly `status`, and its JSON value is exactly one of the three uppercase
strings `"SUCCESS"`, `"FAILURE"`, or `"ERROR"`. Numeric/binary `Y`, `0`, `1`,
booleans, lowercase aliases, and a simultaneous `Y` field are forbidden. Analysis
MAY derive the indicator `I(status == "SUCCESS")` in memory for the probability
formula below, but that derived variable is not serialized, hashed, signed, or
accepted by Artifact 9 or a correction envelope. `predicate_verdicts` and
`finding_codes` remain required even when `status` is `"ERROR"`.

`ArmNeutralEvidenceProjectionV1` contains only causally ordered common business
events, proposals, accepted effects, authority/evidence bindings, obligation
transitions, terminal facts, and integrity status. It contains no arm name, raw
reference timestamp, timer identity, WAIT record, wake marker, compiler identity,
or mechanism-specific expectation.

Each conjunct of `business_success` MUST map to a separately named predicate.
Authority and evidence MUST be reconstructed from records available before an
accepted effect; final-state hindsight is forbidden. Any malformed, ambiguous,
missing, unmatched, or integrity-invalid common evidence yields `ERROR`, not an
inferred `SUCCESS` or `FAILURE`. A well-formed observation with one or more false
business conjuncts is `FAILURE`, even when other conjuncts pass.

The oracle MUST NOT import compiler transformation helpers, domain evaluator
selectors, expected-action builders, or compiler-derived constants. It MAY use
stable public domain schemas and independently authored fixtures. Full and No-Time
positive references MUST produce identical `status`; each business conjunct MUST
have a targeted negative fixture that changes only that conjunct or creates one
named integrity error.

If a common primary endpoint cannot be authored and independently reviewed, the
arms are non-commensurable and the comparison is prohibited.

## 3. Estimands

For fixed semantic scenario `s`, model/profile `m`, standardized scaffold `c`, and
provider sampling process `p`:

```text
δ(s,m,c,p) =
    Pr_assigned(status=SUCCESS | stateful_time_removed, s,m,c,p)
  - Pr_assigned(status=SUCCESS | full_lifecycle, s,m,c,p)

Pr_assigned(status=SUCCESS | arm,s,m,c,p)
  = assigned identities with status SUCCESS / all assigned identities
```

This is a descriptive, scenario-specific contrast. Its sign convention is fixed:
a positive value means a higher success probability after removal of the bundled
lifecycle treatment. Each probability is over all assigned trial identities;
`FAILURE` and `ERROR` are both non-success mass, while their rates remain
separately reported.

A future finite-set estimand is:

```text
δ_set(m,c,p,S) = |S|^-1 Σ[s in S] δ(s,m,c,p)
```

It is permitted only for a preregistered frozen official scenario set `S`, with
target population, selection procedure, and finite-set scope frozen before model
calls. Trials are nested decoding attempts; scenario is the semantic sampling
unit.

## 4. Treatment and matching

The No-Time RFC defines the complete transformation and a closed, versioned raw
allowlist. Matching has two independent layers:

1. every raw Full/No-Time difference MUST equal one allowlisted treatment path and
   operation; and
2. arm-neutral substantive projections MUST be byte-equal after canonicalization.

An independent verifier, not the compiler, compares substantive policy,
authoritative facts, actors/authority, action and evidence schemas, retrieval
catalogue/results, causal business-event order, common decision and outcome
predicates, model-visible observations, canonical provider requests,
transcript/memory policy, invocation/retrieval/action opportunities, and cumulative
budgets. It also reports raw opportunity count/order differences separately.
Compiler output and verifier MUST NOT share a derived timestamp, correspondence
ID, or projection constant as their oracle.

Ambiguous or unmatched correspondence fails closed and remains in the denominator.
Reference and independently authored alternate solvers MUST pass both primary arms;
targeted negative agents MUST fail for the intended common predicates.

## 5. Subject continuity

Operation-state continuity and subject/model-session continuity are different
variables. For the primary contrast, both arms MUST use the same:

- fresh-versus-accumulated conversation policy;
- benchmark-owned memory and model-visible history;
- context truncation;
- tool and action schemas;
- request profile and canonical provider projection;
- retry and missing-output policy; and
- call, output-token, cumulative-token, retrieval, action, and wall-time budgets.

The v0.x standardized track freezes
`stateless_across_operation_invocations`. Any approved identity change MUST move
both arms simultaneously. Transcript/model-session continuity is therefore an
identical controlled condition, not a component of the v0.x treatment bundle.
Context length MAY be a measured sensitivity only when realized model-visible
exposure differs; identical policy alone is not evidence of differing exposure,
and any sensitivity is descriptive rather than a mechanism attribution.

## 6. Control requirements

A Boundary control is a prerequisite only when it is reached from real environment
state at a frozen predicate and has the identical model-visible observation,
action/evidence ontology, independently authored admissible set, common point
identity, and common outcome relation. Historical Static/Boundary output remains
diagnostic.

`ordered_snapshot_control` may diagnose order/presentation burden. It cannot
establish lifecycle or temporal capability. No control result may repair a failed
Full/No-Time matching proof.

## 7. ERROR, denominator, and exclusion

Every preregistered trial identity remains accounted for. Reports MUST give, by
arm:

```text
assigned
  authorized
    dispatched
      provider_completed
        runtime_completed
          artifact_valid
            runtime_replay_valid
              scored
excluded_by_reason
```

These are state counts, not progressively selectable denominators. The report MUST
reconcile every `assigned` identity exactly once and assign it one primary endpoint
status. An identity without a well-formed observed business verdict is `ERROR` for
the primary estimand even when the process taxonomy also records an authorized
infrastructure exclusion reason.

- Semantic model behavior is not an infrastructure exclusion.
- Hidden retries and trial replacement are forbidden.
- Provider/infrastructure exclusion is allowed only under a preregistered,
  independently verifiable taxonomy.
- Evaluator or simulator defects make the affected contrast `UNINTERPRETABLE`.
- Arm-differential missingness requires a sensitivity analysis.
- A critical violation blocks release, but is reported as `x/N` with an
  appropriate one-sided upper bound; zero observed is not zero underlying risk.

Under
[`operatebench.domain-generated-audit-budget.v1`](DOMAIN_GENERATED_AUDIT_BUDGET_PROTOCOL.md),
declaration invalidity after assignment and runtime budget, contract, or counter
integrity halts receive primary `ERROR`. They are separate from business `FAILURE`,
remain non-success mass in the assigned denominator, permit no hidden retry or
replacement, and make the affected contrast `UNINTERPRETABLE`. This is a frozen
future accounting requirement, not evidence that the runtime or outcome successor
exists.

## 8. Sampling and analysis

The hierarchy is:

```text
semantic scenario
  -> surface/variant
    -> randomized blocked arm assignment
      -> fresh trial/run
        -> nested decision points
```

Arms MUST run contemporaneously and randomized/counterbalanced within each
model × scenario × surface block. Pair identity, calendar block, service tier,
model/profile, request profile, scaffold, all identity-manifest digests, and run
plan MUST be recorded before dispatch.

One scenario supports one case study. Two support two separate case studies. At
least three may support an unranked cross-scenario methodology report only after
simulation demonstrates adequate decision stability. Ranking or generalization
requires a separate preregistered scenario-count and precision/power design.

There is no fixed `2/3 versus 1/3` confirmatory rule. Before Numerical Alpha a
single confirmatory contrast, primary endpoint, minimum practically important
difference, paired estimator, uncertainty criterion, decision-loss/Type-I target,
operating characteristics, multiplicity family, and missingness rule MUST be
frozen. Very small `n` is reported as raw paired matrices and descriptive
intervals. Hierarchical models are permitted only after estimability is shown.
Decision-point analysis is separate and nests points within run and scenario.
`pass@1`, `pass^k`, or `Stable@k` requires exact formulas and dependence
assumptions or is omitted.

## 9. Mechanism limitations

Full versus No-Time identifies only the versioned bundle. Failure labels are
descriptive, not causal. Wake/re-entry, WAIT ownership, deadline slack,
stale-state refresh, and reopen/finality require separately preregistered narrow
ablations before any mechanism attribution. Transcript/model-session continuity
remains identical and stateless across arms and is outside the treatment bundle.
Context-length sensitivity may be measured only when realized exposure differs;
it does not identify a lifecycle mechanism.

No result from this design is a model ranking, broad operations claim, public
model claim, or production-safety inference.

## 10. Baseline versus required evidence

**Baseline only:** Engine 0.9 combines runtime and evaluator identity; Artifact 8
stores the current evaluation; the current matched-arm compiler has four arms and
no `stateful_time_removed`; current evaluator dimensions are not
`CommonOutcomeV1`.

**Phase B/C evidence required:** strict split identities, Artifact 9 replay/regrade,
independent CommonOutcomeV1 fixtures, independent arm verifier RED tests, the full
No-Time transition matrix, reference/alternate/negative solver parity, zero-provider
replay, and independent construct review with no P0. Passing those gates makes the
contract executable; it does not itself validate the construct or authorize a
provider campaign.
