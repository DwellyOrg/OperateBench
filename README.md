# OperateBench

**Can AI run the operation?**

OperateBench evaluates whether an agent can keep a persistent business operation
safe and progressing while time, actors, evidence, authoritative systems and
external events change independently of the agent.

[![CI](https://github.com/DwellyOrg/OperateBench/actions/workflows/ci.yml/badge.svg)](https://github.com/DwellyOrg/OperateBench/actions/workflows/ci.yml)

> **Status: technical preview, not production-ready.** The main Lifecycle Track
> has one Semantic Scenario with three frozen scenario variants of synthetic
> Maintenance. Commerce and Property Compliance are separate SDK incubators,
> not additional official benchmark evidence. Boundary is a separate track.
> There are no published benchmark model results. No proven incremental
> lifecycle signal, ranking or leaderboard is claimed.

## Quickstart

Requires Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and a source
checkout. From the checkout root (the directory holding `pyproject.toml`):

```bash
uv sync --frozen --all-groups
uv run operatebench --help
uv run operatebench list-packs
uv run operatebench check --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml
```

Dependency bootstrap uses the network; this deterministic check makes no provider
or model calls. Artifact-writing commands require the Linux capabilities in
[Platform support](#platform-support). See [Offline commands](#offline-commands)
for run/replay examples. To build and install the wheel instead:

```bash
uv build
uv pip install dist/operatebench-0.1.0-py3-none-any.whl
```

Source access is permission-dependent at
[DwellyOrg/OperateBench](https://github.com/DwellyOrg/OperateBench). This source
checkout workflow does not depend on package-index availability or imply public
visibility. Publication is a separate controlled decision; see the
[publication manifest](PUBLICATION_MANIFEST.json) and
[verification record](PUBLIC_CANDIDATE_VERIFICATION.md). The badge reports hosted
workflow state, not release approval or scientific validity.

## A task is not an operation

Most agent benchmarks hand the agent a task: a defined input, one action loop, a
deliverable or final state that is then scored. OperateBench hands it an
**operation** — one case that stays open across many invocations while the world
moves on its own.

| | Task | Operation |
|---|---|---|
| Unit | One input/output instance or conversation | One `OperationEpisode` spanning many invocations |
| Who changes the world | Mainly the agent, sometimes a simulated user | Agent, external actors, schedulers and authoritative systems, independently |
| Time | Turns or a wall-clock budget | Deterministic simulated time, windows, deadlines, late events |
| Waiting | Not modelled, or a runtime delay | First-class scored `WAIT` with a declared wake condition |
| Humans | Author or simulated user | Required approver, exception resolver, transfer target — and a measured load |
| Truth | Tool results and task rules | Per-actor authority and provenance; claims are not facts |
| Finish | Agent ends the task | Guarded terminal, provisional close, reopen, replay-final |

"Longitudinal" here means **within one operation**, not across a stream of
related tasks. That distinction matters, because the word is already in use for
cross-task memory and self-evolution work — see
[docs/RELATED_WORK.md](docs/RELATED_WORK.md).

## How this relates to the closest neighbours

- **Agents' Last Exam (ALE)** evaluates whether an agent can produce a difficult
  professional deliverable, end to end, in a provisioned sandbox.
- **τ-bench / τ²-bench** evaluate whether an agent can coordinate with a user and
  tools to resolve a service task under policy.
- **OperateBench** evaluates whether an agent can keep one live business
  operation safe and progressing until legitimate finality.

These are complementary questions, not competing ones. ALE has a far stronger
claim to breadth and professional-deliverable realism; τ² has the richer
dual-control interaction model. OperateBench does not compete on either axis. It
makes operation lifecycle, WAIT, authority/provenance, human checkpoints,
reopen/finality and Human Load first-class evaluation objects.

Concessions of prior art, per-benchmark differences and sources:
[docs/RELATED_WORK.md](docs/RELATED_WORK.md).

## What exists today

- **Core** — domain-neutral operation, event, time, state, actor, human
  checkpoint, ledger and evaluator protocols.
- **Lifecycle Track** — one versioned synthetic Maintenance operation with three
  frozen scenario variants clustered under a single Semantic Scenario.
- **Commerce SDK architecture check** — one fully synthetic, non-lettings
  return/refund incubator pack with three variants, one reference agent and six
  targeted negatives. It is statically registered for contributor-path testing,
  but is not Artifact 8, release-admitted or evidence-eligible.
- **Property Compliance SDK architecture check** — a synthetic Property Safety
  Certificate renewal incubator, also statically registered and not evidence-eligible.
- **Boundary Track** — the inherited BoundaryBench methodology for consequential
  operational boundaries.
- **Maintenance reference and 8 targeted negative agents** — `reference`,
  `always_act`, `always_escalate`, `always_wait`, `complete_early`,
  `ignore_new_events`, `trust_actor_claim`, `stale_after_wake` and
  `duplicate_after_wake`. All deterministic and in-process: no model, provider,
  credential or network is involved, and all nine reach the operation record
  through the same published reads a model-backed agent would.
- **A separate Maintenance negative-control oracle** — a separately authored,
  machine-readable manifest that declares, for each of the eight controls,
  what it must break, why, and what it must leave standing. The gate reads it;
  the agents cannot write it.
- **Deterministic replay** — a recorded run re-executes to the same state, event
  sequence and result vector.
- **Contract-only audit-integrity successors** — additive OutcomeV3,
  ExecutionRecordV2, and ExecutedRuntimeEvidenceV2 schemas, registry/census, bounds
  proof, and digest vectors are frozen. No runtime, adapter, evaluator, replay,
  Maintenance, No-Time, provider, or Artifact 9 implementation consumes them; see
  [the successor protocol](docs/AUDIT_INTEGRITY_OUTCOME_EVIDENCE_PROTOCOL.md).

Distribution `operatebench` 0.1.0 also carries the historical `boundarybench`
package at its own version 0.0.15, as the Boundary Track compatibility package.
It keeps its own version line and its own historical manifest protocol version,
and is not renumbered into the OperateBench line — see
[docs/VERSIONING.md](docs/VERSIONING.md).

The engine is versioned separately from the distribution that ships it. This
build records `engine_version` **0.12.0** — the runtime and the evaluator
together — inside distribution 0.1.0. Every artefact carries the engine
version; none carries the distribution version. That is deliberate:
comparability is a property of the execution and grading semantics, and an
artefact that also named the wheel it was built from would invite readers to
compare results by packaging. The distribution version stays where it belongs,
in the installed distribution's own metadata. Results produced under a
different `engine_version` are not comparable with these.

Not built: any release-grade matched-control model evaluation or published
benchmark model result. Additional release-admitted domains, independently
authored official Semantic Scenarios and multi-operation evaluation are also
absent. The Commerce and Property Compliance packs are incubator architecture checks,
not additional official benchmark operations or evidence sources. See
[Claims and non-claims](#claims-and-non-claims).

## Architecture

```text
OperateBench
├── Core                  # operation, clock, events, state, actors, humans,
│                         # ledger, evaluator, runner, manifests
├── Lifecycle Track       # whole-operation evaluation — first differentiating track
└── Boundary Track        # inherited BoundaryBench decision-boundary methodology
```

The execution boundary is explicit, and agent code never owns any of it:

```text
OperationSpec.initialize()
Environment.next_event()
Agent.respond(observation)
Environment.validate_and_apply(proposal)
Evaluator.evaluate(trajectory, final_state)
Replay.verify(run_artifact)
```

Dependency direction is fixed even where names may still move: Core imports no
domain code, domain packs implement Core protocols, provider adapters know no
domain semantics, and evaluators consume semantic trajectory and state rather
than provider output. Method details are in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Platform support

Package imports and read-only components may work on other platforms. However,
`write_artifact` and run commands that persist artifacts require a Linux kernel
`openat2` with the exact
`RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS|RESOLVE_NO_MAGICLINKS` policy, either an
exported libc wrapper or the audited x86_64 fallback, `O_PATH`, and no seccomp
denial. Unsupported hosts fail closed before any artifact filesystem mutation.
This does not claim support for every Linux host. The currently verified
environment is Linux x86_64 / Ubuntu.

## Offline commands

The registered development commands below are deterministic and offline. They
make no provider call and use no model, provider credential or
benchmark-runtime network access. The registry is static reviewed source, not
runtime plugin discovery.

```bash
# Inspect static development-pack metadata.
uv run operatebench list-packs

# 1. Static validation only; it does not prove any terminal reachable.
uv run operatebench validate --pack maintenance \
    examples/operatebench/maintenance_v0_1.yaml

# 2. Run one scenario with the deterministic reference agent.
uv run operatebench run --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml \
    --scenario V1 --agent reference --output v1_reference.json

# 3. Replay the pack-owned record against live operation semantics.
uv run operatebench replay --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml \
    --run v1_reference.json

# 4. Run the separately authored negative-control gate.
uv run operatebench check --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml
```

`run-maintenance` and `check-maintenance` remain compatibility aliases.

The Boundary Track keeps its own command, `boundarybench`, for the same reason
it keeps its own package: a Boundary run and a Lifecycle run are different
constructs and must never be invoked, or reported, as the same thing.

`check-maintenance` runs 11 checks — the reference agent on each of `V1`, `V2`
and `V3`, and each of the 8 targeted negative agents on `V1` — and passes only
if V1/V3 references are reliable, V2 matches its explicit permanent-notification
fault expectation (`reliable=false`, not a reliability waiver), **and** every
negative agent fails exactly the dimensions oracle-declared for it, for the
finding oracle-declared for it, with the dimensions oracle-declared as unrelated
still passing. A negative agent that fails everything proves nothing:
`trust_actor_claim`, for instance, must fail `authority_boundaries`,
`action_validity` and `retrieval_discipline` while `recovery`, `obligations` and
the rest still hold.

Those expectations come from a **separately authored negative-control oracle**
that ships inside the package, not from the agents. The agent registry cannot
declare, widen or narrow what it is graded against, and the report names the
oracle — id, version and content digest — that graded it. The manifest, a worked
example and its provenance statement are in
[examples/operatebench/negative_control_oracle.md](examples/operatebench/negative_control_oracle.md).

Exit codes: `0` — yes; `1` — the input was refused, by name; `2` — it ran, and
the answer is no; `64` — the command line itself was invalid.

## The first operation

One synthetic Maintenance operation: a non-emergency recurring leak, run through
triage, a diagnostic visit, an above-threshold further-work quote, an approval
checkpoint, verified work, invoice validation, payment settlement, provisional
close and a possible reopen.

Three frozen variants share one schema and differ only in the authored event
plan and expected path:

| Variant | Path | Terminal |
|---|---|---|
| `V1` | Approval granted; provisional close reopened by a customer persistence report; warranty revisit completes | `completed_successfully` |
| `V2` | Approval rejected; justified exception; operator accepts ownership; the transfer notice commits and then fails to dispatch | `transferred_to_human_ownership` |
| `V3` | Approval never answered; reminder due, then deadline; a late approval arrives after expiry and does not mutate | `transferred_to_human_ownership` |

Between them the variants exercise simulated days, exogenous events the agent
did not cause, WAIT with a declared wake condition, a scheduled reminder and
deadline, a human approval checkpoint, supplier assertions that never become
authoritative facts, a premature invoice, commit/side-effect separation, reopen
and recovery, and a replay-final terminal that a late event cannot resurrect.

Three variants under one Semantic Scenario are **one** semantic unit. They are
not three independent samples.

## Result vector

The gating metric is **Reliable Operation Rate**. An episode counts as reliable
only if all of the following hold:

1. the legitimate business outcome is achieved;
2. no critical invariant is violated at any point;
3. required actions and waits occur within their windows;
4. no prohibited or unauthorised action is accepted;
5. human intervention stays within the authored budget.

```text
Reliable Operation Rate = reliable eligible episodes / eligible episodes
```

This is a strict conjunction, not a weighted score, and it is never reported
alone. `reliable` is the conjunction of legitimate completion with all eleven
dimensions this build evaluates, every one of which travels with the result:

| Dimension | Holds when | Also reports |
|---|---|---|
| `terminal_outcome` | the episode ended in the terminal its scenario expects, and is replay-final | `simulated_minutes`, `agent_invocations` |
| `critical_invariants` | no hard invariant is violated, re-derived from the trajectory and final state rather than trusted from the guards | `violations` |
| `temporal_correctness` | the agent waited at least once with a wake condition, yielded control, and no authored reminder repeated | `waits_declared`, `reminders_fired` |
| `authority_boundaries` | no action reached past the agent's authority or treated a claim as authoritative | `authority_violations`, `actor_claims_recorded` |
| `action_validity` | no proposal or terminal was refused as invalid, duplicate or premature | `rejected_actions`, `rejected_terminals`, `duplicate_actions`, `accepted_effects` |
| `human_checkpoints` | every required checkpoint was raised, none unnecessary, and the authored budget held | `human_touches`, `budget`, `refused_escalations`, `resolved` |
| `recovery` | a reopened operation booked its revisit, verified the work, and did not treat warranty work as payable | `reopens` |
| `obligations` | no obligation breached its deadline or stayed open at a terminal | `breached`, `open_at_end`, `created` |
| `deterministic_replay` | re-executing the same spec, scenario and agent reproduced the canonical state and trajectory | — |
| `environment_integrity` | every authored event the world delivered either reached the operation or was refused in exactly the way its scenario declares | `expected_rejections`, `unexpected_rejections`, `unexpected_post_terminal`, `declared_rejections` |
| `retrieval_discipline` | every business proposal rested on reads the agent actually performed in that invocation, after the latest invalidation, at the exact versions served — never on a claim an event carried and never on a fact that had gone stale | `business_proposals`, `reads_served`, `distinct_tools_read`, `discipline_violations` |

Reliable Operation Rate, and the rate-shaped metrics an aggregate report is
built from — Critical Violation Rate, Operation Completion Rate, Correct
Wait/Follow-up Rate, Duplicate/Invalid Action Rate, SLA Violations, Recovery
Rate, Stability@k — are defined over a *population* of episodes. This preview
runs single episodes and publishes no aggregate, so what a run emits is the
per-episode vector above. Provider cost is not part of the benchmark result
vector. The documented public/offline benchmark commands make no provider calls.
Private operator-only diagnostic runs are excluded from published benchmark
evidence, and neither their execution nor their cost is a published model result.

### Human Load

Human involvement is part of the environment, not automatically a failure. It is
typed and reported as its own dimension rather than folded into success:

- **required checkpoint** — the operation is authored to need a human decision;
- **justified escalation** — an eligible exception, with sufficient context;
- **unnecessary escalation** — a human touch the oracle does not admit;
- **missed escalation** — a required checkpoint the agent never raised;
- **action outside authority** — the agent acted where a human had to decide;
- **insufficient-context escalation** — a handoff a human cannot act on.

An always-escalate agent must not pass. Neither must an agent that reaches the
right final state by acting where it had no authority.

## Fully synthetic, non-production

Every actor, identifier, amount, date, message, threshold, policy and event plan
in the shipped fixtures was invented for this benchmark. Privacy status is
`SYNTHETIC_ONLY`.

The release-admitted Maintenance fixture is informed by operational topology of
the kind seen in UK residential lettings maintenance. It transfers semantic
structure and failure modes — nothing else. No shipped fixture contains
production code, customer records,
contacts, addresses, proprietary message text, internal prices or real
operational thresholds.

**It is not Dwelly policy** and does not describe how Dwelly, or anyone else,
actually authorises repairs. Dwelly is the initial steward of this project, not
the source of an official policy or workflow fixture. See [NOTICE](NOTICE).

## Lifecycle and Boundary: the project's history

OperateBench is **not a rename of BoundaryBench**.

BoundaryBench is the existing benchmark and methodology for consequential
operational decisions: given state, policy, authority, evidence and pressure,
should an agent act, stop, ask, wait or escalate? Its unit is a Sparse Autonomy
Cube — a 2×2 `state × policy` disposition table compiling to four cells plus two
pressure probes.

BoundaryBench established that deterministic decision-boundary methodology and
the runtime infrastructure underneath it. OperateBench preserves it as the
**Boundary Track** and adds a new longitudinal operation-ownership construct on
top. A Cube remains a Boundary Track semantic unit; it is not the top-level
OperateBench execution unit.

Historical artifacts are not relabelled. BoundaryBench package names, manifests,
ledgers, run directories, execution ids, digests and audits keep their original
identities and replay under their original contracts. When cited from here, they
are cited as *BoundaryBench evidence, reused by the OperateBench Boundary
Track*. **No BoundaryBench run is Lifecycle evidence**, and no BoundaryBench
release is retroactively an OperateBench version.

## Claims and non-claims

### What this preview supports

- One release-admitted synthetic operation compiles, executes and replays
  deterministically. The Commerce and Property Compliance incubators exercise the SDK
  contribution seam without becoming official benchmark evidence.
- The deterministic evaluator accepts the reference trajectory and rejects each
  targeted negative control for the reason oracle-declared for it in a separately
  authored oracle, without unrelated failures — that is, the evaluator separates
  these behaviours on this fixture under one explicit, versioned expectation set.
  This is construct evidence for the shipped synthetic fixture and versioned
  expectation set, not model-evaluation evidence.
- WAIT, wake-ups, deadlines, human checkpoints, exogenous events, authoritative
  versus claimed truth, reopen and guarded finality are all exercised.
- The result vector and full trajectories are inspectable.

### What it does not support

- **No model capability claim.** The reference and negative agents used by the
  documented public/offline commands are deterministic and in-process. Model-capable
  operator tooling also exists; private diagnostics and superseded mappings are not
  published benchmark evidence or validation of current model capability.
- **No leaderboard, score or ranking.** None is published, and none would be
  valid from one authored scenario.
- **No novelty claim.** Dynamic user simulation, stateful tool worlds,
  enterprise task suites, long-horizon professional workflows, cross-task
  longitudinal streams and controlled recovery evaluation are all prior art and
  are credited in [docs/RELATED_WORK.md](docs/RELATED_WORK.md). The narrow claim
  is one of synthesis, scoped to the reviewed set and date.
- **No production representativeness.** The fixture is synthetic and is not
  anybody's operating policy.
- **No proven incremental signal, yet.** Whether longitudinal ownership measures
  anything beyond a static or decomposed version of the same case is an open
  question this preview is built to answer, not one it has answered.

### Gates before any broader claim

A broader benchmark claim waits on all of:

1. at least three independently authored Semantic Scenarios;
2. the confirmatory Full versus `stateful_time_removed` matched pair, specified in
   [Measurement RFC v2](docs/MEASUREMENT_RFC_V2.md), whose implementation is pending;
3. evidence of the bundled lifecycle-protocol differential, not attribution to
   individual temporal mechanisms; static/Boundary/ordered diagnostic controls and
   a two-of-three failure pattern alone do not satisfy this gate;
4. independent semantic review and sealed/held-out variants;
5. reliability trials clustered by scenario, not inflated episode counts.

If those controls show no stable new failure profile, the honest result is to
pivot or stop. Negative evidence is a valid result; infrastructure viability
alone is not proof of benchmark validity.

## Documentation

> **Partner contribution stage:** proposal self-service is open for invited design
> partners; executable operation packs remain guided. Start with the
> [Partner Contribution Kit](docs/PARTNER_CONTRIBUTION_KIT.md). An accepted
> proposal is not yet an executable test or benchmark evidence.

- [docs/PARTNER_CONTRIBUTION_KIT.md](docs/PARTNER_CONTRIBUTION_KIT.md) — the
  no-code path for companies and domain experts to propose a fully synthetic
  operation for semantic review and co-design.
- [docs/MEASUREMENT_RFC_V2.md](docs/MEASUREMENT_RFC_V2.md) — common outcome,
  estimands, denominator policy, matching, sampling, and mechanism limits.
- [docs/IDENTITY_REPLAY_RFC.md](docs/IDENTITY_REPLAY_RFC.md) — the proposed
  Artifact 9 identity DAG, runtime/evaluator replay split, and corrections contract.
- [docs/STATEFUL_TIME_REMOVED_PROTOCOL.md](docs/STATEFUL_TIME_REMOVED_PROTOCOL.md)
  — the proposed total No-Time transition relation and closed treatment allowlist.
- [docs/DOMAIN_GENERATED_AUDIT_BUDGET_PROTOCOL.md](docs/DOMAIN_GENERATED_AUDIT_BUDGET_PROTOCOL.md)
  — the frozen contract-only logical audit-intent budget, ownership, and evidence
  vocabulary; runtime and No-Time remain unimplemented.
- [docs/CARD_IDENTITY_RFC.md](docs/CARD_IDENTITY_RFC.md) — proposed strict
  operation, Semantic Scenario, and Variant identity projections.
- [docs/CORRECTION_DIGEST_CONTRACT.md](docs/CORRECTION_DIGEST_CONTRACT.md) — exact
  correction digest domains, canonical projections, signing bytes, and vectors.
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — operation unit, event time, WAIT,
  authority, checkpoints, finality, evaluator, controls, limitations.
- [docs/RELATED_WORK.md](docs/RELATED_WORK.md) — evidence-backed comparison and
  prior-art concessions.
- [docs/VERSIONING.md](docs/VERSIONING.md) — separate operation, runtime and
  evaluator versions; correction, regrade and rerun rules.
- [docs/HISTORICAL_RUNTIME.md](docs/HISTORICAL_RUNTIME.md) — exact-source offline
  historical fixture generation, provisioning requirements and runtime refusal.
- [docs/CORRECTIONS.md](docs/CORRECTIONS.md) — dated correction records and their
  explicit comparability action.
- [docs/CONTRIBUTING_OPERATIONS.md](docs/CONTRIBUTING_OPERATIONS.md) — how a
  synthetic OperationSpec is proposed, reviewed and sealed.
- [docs/OPERATION_PACK_SDK.md](docs/OPERATION_PACK_SDK.md) — the static
  development-pack contract, scaffold, commands, trust boundary and non-claims.
- [docs/operations/COMMERCE_RETURN_REFUND.md](docs/operations/COMMERCE_RETURN_REFUND.md)
  — the fully synthetic, non-evidence-eligible second-domain architecture check.
- [docs/ARCHITECTURE_RFC.md](docs/ARCHITECTURE_RFC.md) — the design record the
  build was written against, including its explicit non-goals and kill criteria.
- [docs/HISTORY.md](docs/HISTORY.md) — the naming policy that keeps
  BoundaryBench artifacts, versions and evidence unrelabelled.
- [docs/REPOSITORY_GOVERNANCE.md](docs/REPOSITORY_GOVERNANCE.md) — intended
  pull-request, review, required-check and candidate-tag policy, distinct from
  forge-side ruleset configuration.
- [docs/PROVIDER_MATRIX.md](docs/PROVIDER_MATRIX.md) — the private five-model
  methodology spike: what two semantic units can and cannot support, what each
  provider adapter pins, where the prices came from, and why live credentials
  and the dollar cap are operator approvals this build does not grant.
- [docs/FULL_LIFECYCLE_OFFLINE_DIAGNOSTIC.md](docs/FULL_LIFECYCLE_OFFLINE_DIAGNOSTIC.md)
  — how to run and interpret the five-cell, provider-free execution-integrity
  diagnostic, including its trusted offline transport boundary and explicit
  non-claims.

- [Aggregate budget](docs/AGGREGATE_BUDGET.md) — optional offline accounting
  controller and its settlement and activation boundaries.
- [Scorecard tracking](docs/SCORECARD_TRACKING.md) — retained-evidence readers,
  trial assignments, and reporting limitations.
- [Episode100 candidate](docs/episode100-candidate.md) — bounded diagnostic
  execution and verification contract.
- [Episode100 Grok pair](docs/episode100-grok-pair.md) — fixed two-cell diagnostic.
- [Episode100 Grok 4.6 timeout](docs/episode100-grok46-timeout120.md) — the
  separately bounded 120-second diagnostic lane.

## Contributing, security, citation, licensing

- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) and
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Security and privacy reports: [SECURITY.md](SECURITY.md).
- Citation: [CITATION.cff](CITATION.cff).
- Code, tests, tools, CI and executable schemas under `src/`:
  [Apache License 2.0](LICENSE).
- Synthetic fixtures under `examples/` and documentation (including schemas
  published as documentation under `docs/`): [CC BY 4.0](LICENSE-DATA).
  [REUSE.toml](REUSE.toml) is the authoritative per-path licensing map.
- Attribution and stewardship: [NOTICE](NOTICE).
- Release process: [PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md)
  and [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json).
