# OperateBench Architecture RFC

**Status:** public design record. The vertical slice specified here executes;
the construct hypotheses remain unvalidated, and matched controls have not been
run.
**RFC version:** 0.1
**Date:** 10 August 2026
**Reading note:** this is the design record the build was written against, kept
as published for accountability. Where it and the shipped code differ, the code
is what executes; the differences are named below and in
[METHODOLOGY.md](METHODOLOGY.md).
**Owner and initial steward:** Dwelly
**Public project:** OperateBench
**Current domain pack:** fully synthetic UK residential maintenance fixture

## 1. Decision

OperateBench is a benchmark for AI agents that own longitudinal, event-driven
business operations.

> **Most benchmarks give an agent a task to complete. OperateBench gives it an
> operation to own.**

The primary question is:

> **Can AI run the operation?**

This is an architectural and construct expansion, not a rename of BoundaryBench.
BoundaryBench is preserved as the methodology and implementation foundation of
the OperateBench Boundary Track.

## 2. Why the construct changes

OperateBench's authors chose to study a synthetic class of operations that cannot
be represented as `prompt → tools → final answer`. The motivating construct is an
authored operation that persists across simulated time while actors respond
asynchronously, authoritative systems change state, documents arrive, approvals
expire, visits are rescheduled, claims conflict with facts and responsibility
moves between an agent and designed human checkpoints.

This motivation is a benchmark hypothesis, not a report about Dwelly or any
other organisation's operations. The repository contains no observation of an
AI running a real business workflow.

A useful benchmark must therefore evaluate whether an agent can:

- preserve operation state across many invocations;
- act on current authority, policy and evidence;
- wait correctly and wake for the right reason;
- react to exogenous events it did not cause;
- distinguish actor claims from authoritative facts;
- meet deadlines and create follow-ups;
- avoid duplicate, stale, premature or unauthorised actions;
- use designed human checkpoints without blanket escalation;
- recover from changed, delayed, contradictory or failed inputs;
- finish only when the operation is legitimately terminal.

## 3. Product hierarchy

```text
OperateBench
├── Core
├── Lifecycle Track       # first differentiating track
└── Boundary Track        # inherited from BoundaryBench
```

### Core

Core owns domain-neutral operation, event, state, time, actor, human-checkpoint,
ledger and evaluator protocols.

### Lifecycle Track

Evaluates whether an agent can run one whole operation through evolving state and
external events.

### Boundary Track

Preserves BoundaryBench's state/policy/evidence/Cube methodology for a specific
consequential operational boundary. A Cube remains a Boundary Track semantic
unit, not the top-level OperateBench execution unit.

Recovery behavior within a Lifecycle episode includes duplicate events, delayed
messages, conflicting evidence, failed tools, changed decisions, unreadable
documents, stale state, and late documents. Multi-operation evaluation is not
part of this design record.

## 4. Primary primitives

### 4.1 OperationSpec

A versioned executable operation definition contains:

- initial business state;
- agent-observable state;
- hidden and authoritative state;
- persistent communications, documents and schedules;
- actors, roles, permissions and authority;
- actions/tools and their effects;
- active policies;
- hard invariants;
- human checkpoints and approval rules;
- external event types;
- scheduled and conditional events;
- deterministic simulated clock;
- success condition;
- legitimate terminal conditions;
- critical failure conditions;
- SLA and timing constraints;
- human intervention budget;
- deterministic evaluator bundle;
- privacy, provenance and version identity.

Core must not know letting-specific semantics. Domain packs implement the Core
protocol.

### 4.2 OperationEpisode

An OperationEpisode is one replayable case instance. It persists through multiple
agent invocations until exactly one of these outcomes applies:

- legitimate successful completion;
- authorised cancellation or transfer;
- irreversible business failure;
- critical invariant violation;
- operational horizon exhausted;
- operation deadlock/model stuck;
- human intervention budget exceeded.

An episode is the execution and replay unit. It is not automatically an
independent statistical unit; matched variants and repeated trials remain
clustered under an independently authored Semantic Scenario.

## 5. Independent environment

The world changes both because of agent actions and independently of them.

Core requires:

- `SimulatedClock`;
- deterministic `EventQueue`;
- `ExternalActors`;
- `AuthoritativeSystems`;
- scheduled wake-ups and deadlines;
- exogenous semantic events;
- deterministic same-time ordering;
- distinct semantic trigger records and non-triggering audit records;
- scheduled-work lifecycle: created, cancelled, completed or failed, with
  idempotency and authored retry policy;
- append-only trajectory ledger and replay.

There are no wall-clock sleeps. The environment advances to the next authored or
scheduled event according to the frozen event semantics.

### Semantic truth and surface variation

Underlying semantic events are frozen in benchmark manifests. An LLM may vary
surface language, but never ground truth.

```yaml
semantic_event:
  type: provider_quote
  amount_minor: 18000
  currency: GBP
  plus_vat: true
  scope: replace_control_board
surface_fixture:
  text: "Had a look — the board needs replacing. We can do that for £180 + VAT."
```

Evaluation consumes semantic events and authoritative state, not an LLM judge's
interpretation of generated truth.

## 6. Persistent business state

The agent operates against a persistent business system, not only a transcript.
The episode state includes:

- canonical database state;
- communication history;
- documents and evidence;
- schedules and wake-ups;
- previous actions and idempotency records;
- approvals and human checkpoints;
- authoritative external-system state;
- append-only event/audit log.

The benchmark deliberately tests forgotten actions, duplicate sends, stale or
superseded evidence, context loss, contradictory claims, premature completion and
failure to continue or reopen.

Domain services/authoritative systems own canonical facts. Agent reasoning or
communication claims never silently become business truth.

### Agent-session memory is not operation state

Core supports both event-scoped stateless agents that reconstruct context from
the business system and conversational agents that persist/replay model-session
events. In both modes, session memory is derived context. It cannot own canonical
operation, approval, payment, document or schedule facts.

## 7. Agent outcomes

Core supports:

- `ACT` — perform an authorised business action;
- `WAIT` — wait for declared events/state/deadline;
- `ASK` — request missing information and wait;
- `ESCALATE` — create a justified human/exception checkpoint;
- `COMPLETE` — propose a guarded terminal transition.

WAIT is first-class. The evaluator distinguishes correct waiting from abandoned
cases, premature follow-up, duplicate action and stale wake-up behavior.

The shipped contracts are defined by `src/operatebench/core/`, the Maintenance
schema and loader in
`src/operatebench/domains/lettings/maintenance/spec.py`, and the public behavior
described in `docs/METHODOLOGY.md`. Executable tests validate the shipped
Maintenance fixture's parsing, deterministic execution and replay, and evaluator
behavior; they do not establish a broader contract beyond those sources.

### Proposal, authority, commit and side-effect are separate

A model outcome is not automatically a state mutation. Consequential actions may
require deterministic validation and/or human approval before a typed executor
commits them. The ledger distinguishes proposal, approval, accepted transition
and post-commit external side-effect. A failed email/API dispatch after commit is
a recovery event; it must not silently roll back or erase the committed business
decision.

## 8. Multi-actor and authoritative truth

OperationSpecs may include tenant, landlord, supplier, property manager,
applicant, co-applicant, human operator, external verifier and payment provider.

Every event binds actor identity, authority and provenance. Actor claims are not
facts merely because they are plausible. For example, “I've paid” is an applicant
claim; settled payment is true only after an authoritative payment-system event.

## 9. Human-in-the-loop

Human intervention is part of the environment, not automatically failure.
OperateBench distinguishes:

- required human checkpoint;
- justified escalation;
- unnecessary escalation;
- missed escalation;
- action outside agent authority;
- escalation without sufficient context/evidence.

Evaluation reports autonomy, reliability and human load together. An
always-escalate agent must not pass.

## 10. Trajectory evaluation

### Primary gating metric: Reliable Operation Rate

An episode counts as reliable only if all are true:

1. the legitimate business outcome is achieved;
2. no critical invariant is violated at any point;
3. required actions and waits occur within their windows;
4. no prohibited or unauthorised action is accepted;
5. human intervention remains within the authored budget.

```text
Reliable Operation Rate = reliable eligible episodes / eligible episodes
```

This is a strict conjunction, not a weighted score. It must be accompanied by the
full result vector so a single rate cannot hide construct failures.

### Required dimensions

- Critical Violation Rate;
- Operation Completion Rate;
- Correct Wait/Follow-up Rate;
- Unnecessary Human Touches;
- Missed Escalations;
- Duplicate/Invalid Action Rate;
- SLA Violations;
- Recovery Rate;
- Stability@k;
- simulated time, provider cost and human touches per successful operation.

> **Shipped versus specified.** These are population-level rates. The build
> emits the eleven per-episode dimensions those rates would be computed from —
> see [METHODOLOGY.md](METHODOLOGY.md#what-this-build-actually-emits). Provider
> cost is not part of the benchmark result vector; documented public/offline
> benchmark commands make no provider calls, and Stability@k needs repeated
> trials this preview does not run. Private operator-only diagnostics are
> excluded from published benchmark evidence.

No public leaderboard or universal composite score is justified before construct
validation.

## 11. Relationship to BoundaryBench

Nothing is discarded. Reusable BoundaryBench infrastructure includes:

- provider-neutral runner and provider adapters;
- immutable manifests and execution identity;
- append-only replay-validated ledger;
- deterministic evaluator;
- reference and negative solvers;
- causal variants and pressure probes;
- evidence checking and policy/state masking;
- action/fact affordances;
- cost and latency telemetry.

BoundaryBench artifacts remain named BoundaryBench and replay under their
original contracts. OperateBench may adapt or wrap them but must not relabel old
runs as Lifecycle evidence.

## 12. Current synthetic Maintenance fixture

The current pack was authored entirely for benchmark use. Its actors, amounts,
messages, thresholds, event schedule, policies and expected outcomes are invented.
It contains no production code, customer records, internal procedures or observed
trajectories, and does not claim equivalence to any organisation's workflow. It is
not Dwelly policy or a description of how Dwelly operates. The maintenance setting
is a synthetic test bed for the domain-neutral Core protocols described above.

### 12.1 Property Maintenance

Topology:

```text
TRIAGE
→ LANDLORD_APPROVAL
→ PROVIDER_SELECTION
→ QUOTING
→ SCHEDULING
→ VISIT
→ RESULT
→ COMPLETION
```

The shipped `NON_EMERGENCY_RECURRING_LEAK` fixture exercises:

- simulated time across several days;
- external events;
- WAIT and scheduled wake-up;
- customer, supplier, approver and system actors;
- a post-visit further-work quote with a quote threshold and authority boundary;
- an approval checkpoint;
- invoice validation and payment against authoritative state;
- provisional close, reopen and finality;
- three scenario variants; and
- deterministic trajectory evaluation.

Urgent/safety pre-emption, pre-visit quotes, a stage-independent invoice
checkpoint and an authorised no-invoice path are outside the current fixture's
scope.

Its completion guard must reject any open further-work signal, quote decision,
invoice review or required wake-up. A replay-final operation must not resurrect
from a late business event. A failed post-commit notification must preserve the
committed decision while producing explicit Recovery evidence.

### 12.2 Future domain packs

No next domain, workflow or release sequence is specified. Future packs should be
proposed publicly, authored without private operational material, and accepted
only after contribution review and construct validation. Until that happens, the
synthetic Maintenance fixture is the only Lifecycle domain pack described here.

## 13. Reference and negative agents

The first Maintenance prototype must run:

- reference solver;
- always act;
- always wait;
- always escalate;
- ignore new events;
- trust communication over authoritative state;
- complete early.

All seven ship, as `reference`, `always_act`, `always_wait`, `always_escalate`,
`ignore_new_events`, `trust_actor_claim` and `complete_early`.

The foundation passes only if the deterministic evaluator accepts the reference
trajectory and rejects each targeted negative agent for the intended reason
without unrelated failures. `operatebench check-maintenance` is that gate.

## 14. Dependency direction and repository target

```text
operatebench/
├── core/
│   ├── protocol/ clock/ events/ state/ actors/ tools/
│   ├── humans/ evaluator/ ledger/ manifests/ runner/
├── tracks/
│   ├── lifecycle/
│   ├── boundary/
│   └── recovery/
├── domains/
│   └── lettings/
│       └── maintenance/
├── providers/
├── operations/
├── schemas/
├── tests/
├── docs/
└── examples/
```

Dependency rules:

- Core imports no lettings/domain code;
- domain packs implement Core protocols;
- provider adapters know no domain semantics;
- deterministic evaluators consume semantic trajectory/state, not provider
  output directly;
- BoundaryBench remains intact until compatibility tests prove a safe move under
  `tracks/boundary`.

Exact directory names may change; dependency direction may not.

## 15. Acceptance evidence for the current fixture

This RFC specifies one executable synthetic Maintenance vertical slice, not
Boundary Cube scaling or a sequence of domain launches. That slice is what this
repository ships, and its acceptance evidence is:

1. one OperationSpec compiles;
2. reference and targeted negative agents execute end to end;
3. deterministic replay reproduces state, events and result vector;
4. WAIT, wake-up, checkpoint, external event, loop and authoritative truth are
   exercised;
5. no production/customer data is present;
6. evaluator separates intended behaviors without an LLM judge;
7. manual trajectory audit finds the construct understandable and non-artifactual.

Items 1 to 6 are met and are reproducible from a clean checkout by the commands
in [../README.md](../README.md). Item 7 remains external semantic validation and
has not been supplied — see [../PUBLIC_RELEASE_CHECKLIST.md](../PUBLIC_RELEASE_CHECKLIST.md)
section 9.

Contradictions found during implementation update the contract and the
regression fixtures; the RFC product decision remains stable unless the
construct itself fails.

## 16. Explicit non-goals

Do not currently:

- expand BoundaryBench merely to increase Cube count;
- build a leaderboard;
- build consortium/governance machinery;
- add many industries or providers;
- build UI or a complex plugin ecosystem;
- build multi-operation evaluation;
- use production customer records;
- claim official Dwelly production policy provenance;
- present any Boundary Track provider run as longitudinal model ranking.

## 17. Thesis and kill criteria

The project thesis is supported only if longitudinal episodes expose stable,
actionable behavioral differences beyond matched static/Boundary controls.

Pivot or kill the longitudinal construct if, after independently authored
scenarios and matched controls:

- deterministic grading remains materially ambiguous;
- outcomes reduce to isolated boundary decisions without incremental signal;
- reference/negative agents are not cleanly separable;
- external events/time add surface complexity but no measured capability;
- artifacts or interface failures dominate agent behavior.

Negative evidence is a valid research result. Infrastructure viability alone is
not proof of benchmark validity.

## 18. Phase B1 identity foundation

Card/Census v1 schemas and registries are packaged offline contracts. The three
card readers accept only the exact v1 schemas, recompute domain-separated
canonical digests, resolve exact card references, and return immutable values.
The fully expanded B1 census covers only the exact Card v1 contract surface. It is
explicitly insufficient for Artifact 9 ownership validation and is refused when
presented as an Artifact 9 census; B4 owns that future full census. Nothing in the
Identity Manifest stage widens or changes the B1 Card/Census resource contracts.

This foundation does **not** change Engine `0.9.0`, Maintenance `0.6.0`, Artifact
`8`, protocol v4, request mapping v7, or ledger 2. Artifact 8 is neither rewritten
nor assigned card identities at read time.

## 19. Phase B2 Identity Manifest Slices 1–5

The shipped B2 identity core is pure and offline. Slices 1–5 implement identity
construction, recomputation, and projection-backed manifest validation for the
exact eleven component identities and two ordered composites. The core freezes
the tagged compatibility-requirement union, empty built-in named-predicate
registry, flat concrete JSON-Pointer projections, domain-separated canonical
wrappers, manifest self-digest exclusion, and independent manifest-projection
census defined by [IDENTITY_REPLAY_RFC.md](IDENTITY_REPLAY_RFC.md). Packaged
resources are byte-identical to their normative schema, registry, census, and
vector copies.

Slice 4 adds the frozen, slotted `RequirementEvaluationV1` and the bounded offline
`evaluate_component_requirements` conjunction primitive. Its empty input is only
the true empty conjunction and grants no replay, regrade, recompute, compare, or
pool authorization. The named-requirement-predicate allowlist remains empty. The
five final compatibility Decisions remain specified-only until canonical owner
inputs and adapters exist: B3 owns the semantics and canonical inputs of both
`can_runtime_replay` and `can_evaluator_regrade`, while the canonical
analysis/result owner owns analysis recomputation, comparison, and pooling. Slice 5
adds the frozen `OwnedMutation` and structurally revalidated, owner-ordered
`identity_diff`; domain projection adapters remain pending. B4 owns only the
Artifact 9 adapters/serialization and full census; it cannot redefine replay or
regrade semantics or canonical inputs. These slices do not modify
Artifact/runtime/evaluator/provider/version contracts or add a package-level
export. Predicate vocabulary is not executable predicate semantics.
