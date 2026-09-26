# Versioning, corrections and comparability

Three things can change independently in OperateBench, and each changes what a
past result means. They are versioned separately, and no single project version
number is allowed to stand in for all three.

| Version | Covers | Changing it means |
|---|---|---|
| `operation_spec_version` | One OperationSpec: state, actors, authority, policy, events, scenario variants, guards, oracle | The case itself changed |
| `runtime_version` | Core: clock, event queue, reducers, actor/authority enforcement, ledger, replay, runner, agent protocol | How the case executes changed |
| `evaluator_version` | Predicates, invariant checkers, admissible-outcome sets, result-vector definitions | How the case is graded changed |

Every artifact also carries **content digests** over the operation fixture, the
runtime contract and the evaluator bundle, so a semantic edit cannot hide behind
an unchanged version string. A version that did not move while its digest did is
a defect, and tooling should refuse the artifact rather than accept it.

Package/distribution version is separate again, and is recorded as provenance —
not as the identity of an operation, a runtime contract or an evaluator.

Pricing policies are independently versioned measurement inputs. The current
Anthropic policy, `anthropic-standard-2026-09-07`, supersedes
`anthropic-standard-2026-08-09`. Existing manifests remain byte-preserved and
self-describing under the policy they recorded; this build intentionally refuses
to resume one under a superseded pricing policy rather than silently re-price it.
This correction moves no protocol, artifact, ledger, runtime or evaluator version.

The historical Anthropic Messages Lifecycle request mapping
`lifecycle_anthropic_messages_model_request_v2` projects every tool input to a
root object without a root combinator while retaining nested schemas. The
historical `lifecycle_anthropic_messages_model_request_v1` literal remains an
independent reader-facing identity and is not an alias for the v2 wire bytes.
The historical v3 mapping additionally carries the six-tool instruction correction
described below. The current v4 mapping also includes the compact action-schema
projection described below.
The historical v1-to-v2 request-only change moved no model protocol, operation,
artifact, ledger, runtime, evaluator, or request-profile version; the profiles still
identify the same model-specific sampling and thinking field sets.

Card schema versions are independent contract identities. `CARD_SCHEMA_VERSION = 1`
is a frozen public compatibility alias for the byte-immutable Card v1 family, not a
moving latest-version selector. Module-level family literals identify the additive
Operation/Semantic Scenario/Variant v2 chain, which does not migrate v1 cards and
cannot be mixed into a v1 reference chain. V2 currently changes only the
`NULLABLE_STRING` shape and `EXACTLY_ONE` predicate vocabulary. It is contract-only:
no projectors, runtime evidence, replay, compatibility Decision, or Artifact 9 path
is implemented. Adapter v2 contracts wait for Maintenance source declarations; they
must not be invented before those source leaves exist.

Outcome and executed-evidence generations are also exact, additive wire identities.
The historical OutcomeV2/EvidenceV1 pair remains accepted with no audit-budget claim.
OutcomeV3/EvidenceV1 and OutcomeV2/EvidenceV2 are refused cross-generations;
OutcomeV3/EvidenceV2 is shape-only until runtime authority exists. Card v1 has no
claim, Card v2 without an authorized source declaration refuses before identity, and
Artifact 8 is never migrated or rewritten. The exact matrix is frozen in
[AUDIT_INTEGRITY_OUTCOME_EVIDENCE_PROTOCOL.md](AUDIT_INTEGRITY_OUTCOME_EVIDENCE_PROTOCOL.md).

JSON loading preserves that separation: legacy `load_card_json` is the uncapped,
v1-only compatibility entry point, including arbitrarily whitespace-padded valid
v1 JSON. The explicit `load_card_json_v2` entry point is v2-only and applies an
8 MiB raw UTF-8 limit before parsing. Both reject cross-generation input.

Identity Manifest v1 makes that separation machine-readable without changing the
preview's runtime or Artifact contract. Its fixed component content digests cover
only flat owned projections; requirement metadata is excluded from the component
digest but included through complete member identities in dependent composite
digests. The manifest digest covers the complete manifest except its own digest.
The built-in compatibility vocabulary names separate replay, regrade, analysis,
comparison, and pooling decisions; merely sharing a version or satisfying one
decision never implies another. Slices 1–2 construct these identities, and Slice 3
validates manifests against the independent owned projections and
manifest-projection census. Slice 4 implements only bounded component-requirement
evaluation and freezes predicate names as vocabulary, not executable semantics. The
Slice 5 identity diff now reports detached owner rows for concrete projection and
direct requirement-metadata changes, with actual reverse-composite consequences;
it makes no replay or comparability decision. The preview rules below remain in
force. Final compatibility Decisions, domain projection adapters, and Artifact 9
support remain pending.

### Optional settled aggregate accounting

The repository-only offline diagnostic adds controller profile
`settled-aggregate-bounded-v1`, independently of engine 0.11.0. It preserves
bounded model/call/turn profiles and successful Artifact 8 / ledger 3 records.
Only its optional guard uses a null breach measurement to report durable unknown
forfeiture rather than measured settlement. The legacy ledger remains the initial
reservation projection; the controller journal owns full wire-adjusted exposure.
No historical identity or authority is reused. See [AGGREGATE_BUDGET.md](AGGREGATE_BUDGET.md)
for the accounting distinction and unimplemented live activation boundary.

### Engine 0.13.0: Maintenance wait, notice and guidance repairs

For Maintenance only, unsolicited runtime events interrupting a valid declared
wait are informational only when the diagnostic binds to a unique accepted,
agent-triggering delivery and its immediately preceding event observation, with
matching identity, type, actor, instant and verdict. The delivery must interrupt
the current nonempty WAIT outside its `wake_on`, no later than its fallback;
cleared waits, cancelled events and ambiguous identities do not qualify.
Unproven diagnostics produce `WAIT_INTERRUPT_PROVENANCE_INVALID` and do not
increase `unsolicited_interrupts`. This is a record-integrity failure, not blame
for the runtime's choice to interrupt. A subsequent invocation is not required:
the runtime can reach its invocation limit after recording the delivery.
Invalid waits, missing yields, unresolved waits, SLA
failures and unsafe actions retain their checks. Event delivery is unchanged.

An identical completion notice is required again when a new open duty arose
after its prior accepted effect. The evaluator independently binds the reporting
recipient, cycle, authoritative verification, obligation creation and actual
current delivery/discharge to accepted proposals. Early notices remain allowed;
they cannot discharge later duties. Wasted notices, duplicate business actions
and forged or unrelated obligations remain failures.

Maintenance exposes the enforced scenario checkpoint budget. Malformed decision
feedback distinguishes ACT evidence references from RETRIEVE requests. These
model-visible changes are bound to engine 0.13.0. Provider mapping algorithms,
Artifact 8, ledger shape and operation fixture 0.6.0 are unchanged. Source-owner
projection v5 is additive; v1–v4 remain frozen. Commerce grading is unchanged.

Historical artifacts remain readable with stored grades; replay requires the
original engine identity and refuses older engines before execution. These
changes do not authorize relabelling historical results. Focused synthetic
regressions do not establish full-suite, hosted-CI or live-model acceptance.

### Engine 0.12.0: communication obligations

This historical combined runtime/evaluator identity corrects recipient, correlation,
and actual-delivery semantics for reminder and transfer notices. V2 deliberately
retains its permanent transfer-notice fault: legitimate human transfer survives,
but `reliable=false` with exactly recovery and obligations failing. The diagnostic
check uses the separately authored
`maintenance.reference.V2.permanent-notification-fault.v1` expectation; V1/V3
remain reliable positive controls. Check PASS is not three reliable references.

Artifact 8 reads the unchanged state shape for both 0.11.0 and 0.12.0; execution
still refuses foreign-engine replay. Fixture, ledger, protocol and provider
mapping algorithms do not move. Engine-bound domain guidance and resulting
observation/request digests change. Source-owner projection v4 is additive and
historical projections stay frozen. See the
[correction record](CORRECTIONS.md#ob-corr-2026-09-17-communication-obligations).

### Engine 0.11.0: benchmark validity correction

`OB-CORR-2026-09-12-BENCHMARK-VALIDITY` moves the combined runtime/evaluator
identity to `0.11.0`, distinct from both `0.10.0` and the separate development
identity `0.11.0.dev1`. It corrects six-tool instructions, classification-only
malformed-action feedback, completion recipient enforcement, and independent
completion-notice causal grading. It does not incorporate other development
runtime changes. All affected experiments require fresh runs; no regrade,
relabel, or pooling with earlier runs is supported.

The accepted issue-reporting actor is disclosed by `get_case_record.v2`; the
other retrieval schemas remain v1. Maintenance canonical state and the
Maintenance Card-v2 declaration profile advance from `@1.0.0` to `@2.0.0`.
Their original profile, semantic oracle, and source-owner freeze remain intact.
Artifact 8 keeps its outer record, trajectory vocabulary, and provider binding.
Its domain-state validation dispatches on the recorded engine identity: engine
0.11.0 requires the reporting-actor field, while historical engines retain the
exact old state field set. No stored state is migrated. Historical artifacts
remain readable but cannot replay under the corrected engine. This is not the
pending Card/Identity Manifest Artifact 9 integration.

The six provider request projections advance together: OpenAI Responses v8,
Anthropic Messages v3, xAI Chat Completions v2, Mistral Chat Completions v2,
xAI Responses v3 (4096 output tokens), and xAI Responses v4 (the fixed Grok 4.6
8192-token contract). Historical mapping literals remain independent. Request
profile field sets, model protocol v4, ledger 3, operation fixture 0.6.0 and
distribution 0.1.0 do not move: the correction changes runtime enforcement and
disclosure of the existing completion obligation, not authored fixture inputs.
Settings and their digests name the mapping used by their producing build;
historical settings and ledger evidence keep the mapping they recorded. No new
live evidence is claimed by this correction.

### xAI Responses schema projection correction

The historical xAI Responses schema-spelling mappings are
`lifecycle_xai_responses_model_request_v4` for the 4096-token contract and
`lifecycle_xai_responses_model_request_v5` for the fixed Grok 4.6 8192-token
contract. They supersede the v3/v4 mappings introduced by the benchmark-validity
correction above: all fifteen nonempty-string patterns in xAI's tool schemas
use `^[\s\S]+$` instead of `[\s\S]`. The spellings are equivalent under JSON
Schema search semantics; all other schema leaves and the canonical Core schema
remain unchanged. This xAI-only wire-spelling correction
does not change engine 0.11.0, Artifact 8, execution ledger 3, model protocol v4,
operation 0.6.0, distribution 0.1.0, or model-specific request-profile field sets.
Historical evidence remains bound to its original mapping and is not rewritten
or silently pooled with fresh requests.

### Compact action-schema projection

The current provider mappings are OpenAI Responses v9, Anthropic Messages v4,
Mistral Chat Completions v3, xAI Chat Completions v3, xAI Responses v6 for
4096 output tokens, and xAI Responses v7 for the fixed Grok 4.6 8192-token
contract. They supersede the mappings above for current requests, omitting
empty optional and evidence-reference declarations from model-facing action
schemas while preserving the nonempty contract and unmet-read boundary.
Mapping identities remain bound into transport settings and request instructions;
model-specific request profiles, token ceilings, canonical action declarations,
and historical evidence identities are not relabelled.

The repository-only episode100 operators additionally bind explicit invocation,
call and turn ceilings. The Grok 4.6 timeout120 diagnostic changes that
operator's fixed turn deadline from 30 to 120 seconds and its episode wall-clock
admission bound to 12000 seconds; it does not change the request mapping or the
shared engine. These profiles are distinct diagnostic
execution settings, not a new operation or published benchmark result. See
[the episode100 candidate](episode100-candidate.md),
[the Grok pair](episode100-grok-pair.md), and
[the Grok 4.6 deadline profile](episode100-grok46-timeout120.md).

### Unpublished engine 0.12.0 guidance correction

The historical engine 0.12.0 candidate removes repeated message-type
spelling from the two `send_message` guidance fields. Completion, approval-reminder
and transfer rules still specify their recipients and cycle conditions; an
accepted completion mismatch still does not discharge an obligation. All other
schema, evidence-origin, accepted issue-reporting actor and post-approval record
requirements are unchanged. This is shorter wording, not a weaker information contract.

Observation and request digests bind the exact engine 0.12.0 guidance bytes.
Reproduction requires the matching source identity; a shared engine label alone
does not establish request-byte equality. The provider mapping algorithms and
versions do not change. Every canonical SDK request must remain within its pinned
per-request vector, independently of the unchanged aggregate projection ceiling.
Financial, token, call, retry and deadline controls remain fixed. Historical
artifacts, bundles and source-owner projections before v4 are not regenerated.

### Compact action-schema disclosure

The next request mappings omit repeated empty `optional: {}` and
`evidence_refs: {required: {}}` declarations from model-visible action schemas.
Omission means exactly those empty declarations; every nonempty requirement,
selector, payload field and read guard remains unchanged. Completion-message
field guidance is concise but still discloses the cycle binding, accepted but
non-discharging mismatch, reporting-customer recipient and other-message rules.
This changes prompt bytes, not Core execution semantics or authored inputs.

| Provider/request profile | Predecessor | Current mapping suffix |
| --- | --- | --- |
| OpenAI Responses | v8 | v9 |
| Anthropic Messages (Haiku and Sonnet) | v3 | v4 |
| Mistral Chat Completions | v2 | v3 |
| xAI Chat Completions | v2 | v3 |
| xAI Responses, 4096 output tokens | v4 | v6 |
| xAI Responses, Grok 4.6 / 8192 output tokens | v5 | v7 |

These suffixes belong to each provider's existing full mapping-name prefix.
Predecessor constants stay literal; historical evidence and output-policy enum
identities are not changed. All current consumers use the successor mapping.
Request digests and exact SDK byte/token-bound vectors must be recomputed from
fresh synthetic requests; old request pins do not authorize new wire bodies.
No new paid or live evidence is claimed. Engine 0.11.0, Artifact 8, ledger 3,
model protocol v4 and the output/profile policy field sets remain unchanged.

## Comparability statement

> Results are comparable only between runs that share `operation_spec_version`,
> `runtime_version`, `evaluator_version` and their content digests. Any
> difference in any of the three makes the numbers different experiments, and
> they must not be pooled, averaged or plotted on one axis.

This is stricter than semantic versioning, and deliberately so. A "patch" to a
scenario's event plan can change which behaviours the case rewards.

Published results therefore state all three versions, their digests, the
scenario ids, the trial count, and the agent or model identity. A result without
that provenance is not a result this project will cite.

## What counts as which change

### Operation change

Any edit to authored semantics: actor authority, policy values or thresholds,
event plan, timings, payloads, guards, terminal conditions, admissible outcomes,
human checkpoint budgets, or the set of scenario variants.

- Bump `operation_spec_version`.
- Prior results **are not comparable** and are not silently carried forward.
- Affected runs must be **re-run** to produce comparable numbers.
- Formatting, comment and reflow changes are not semantic and do not bump —
  digests are taken over parsed semantic content, not file bytes.

### Identity-coverage change

A third case, which the two rules above do not between them settle: the authored
semantics do not move, and the digest does — because the projection that the
digest is taken over was widened to cover a field it always should have covered.
Nothing executes differently; what changes is what the identity *determines*.

- Bump `operation_spec_version`. The rule at the top of this document is
  unconditional, and the reflow carve-out above is the opposite case: it excuses
  file bytes moving while the digest stands still, not a digest moving while a
  version stands still.
- Do **not** bump `runtime_version` or `evaluator_version` for this alone. A
  projection consumed only by the digest is neither execution nor grading.
- Prior results are **not comparable** and must be **re-run**, not regraded — and
  for a stronger reason than an ordinary operation change. The old identity was
  not merely different; it was not injective, so it never determined which
  executable case an artefact bearing it ran under. There is nothing to regrade
  *against*.
- Say in the correction note which field was uncovered and what it changed about
  execution, so a reader can tell how far the old identity's ambiguity reached.

The worked instance is
[OB-CORR-2026-09-06-MAINTENANCE-POLICY-PROJECTION](CORRECTIONS.md).

### Runtime change

Any edit to execution semantics: event ordering, wake-up dispatch, reducer
behaviour, authority enforcement, idempotency, commit or side-effect handling,
observation projection, ledger or replay contract.

- Bump `runtime_version`.
- Prior results **are not comparable**; affected runs must be re-run.
- A stored trajectory produced under an older runtime replays under that
  runtime's contract, never under the new one.

### Evaluator-only change

A grading correction where the operation and runtime are untouched — a predicate
that scored a correct trajectory as failing, a missing forbidden-delta
assertion, an admissible outcome that should have been admitted.

- Bump `evaluator_version`.
- Stored trajectories that remain **contract-compatible** — same operation
  version, same runtime version, complete recorded trajectory and final state —
  may be **regraded** without re-execution.
- Regraded results are published as regraded, naming the old and new evaluator
  versions. They are never presented as the original numbers.
- If the correction requires information the stored trajectory does not contain,
  regrading is impossible and the runs must be re-executed. Say so rather than
  approximating.

The current preview has a narrower executable boundary than that future policy:
terminal-suffix grading accepts only an authenticated `EpisodeOutcome` minted by
the current Engine process, or that exact authenticated outcome exposed by a
successful replay. A digest-consistent serialized trajectory or caller-built
outcome is not equivalent. Replay failures expose diagnostics but withhold the
outcome, and successful replay is not publication or rescore authorization. This
process-local capability defends against serialized/caller-built evidence, not
arbitrary code already executing inside the evaluator process; Python private
names are not access control. The combined-version and no-rescore rule below
therefore still governs engine 0.11.0.

## Corrections policy

Benchmarks have errors. The policy is to make them visible rather than tidy.

1. **Publish the defect.** Every accepted correction gets a dated correction
   note: what was wrong, which scenarios and versions are affected, and which
   published numbers it invalidates.
2. **Preserve the pre-fix state.** The pre-correction commit is tagged, and the
   pre-correction fixture stays reproducible. Reproducing an old result must
   remain possible after the fix.
3. **State comparability explicitly.** Each note says whether affected results
   are still comparable, must be regraded, or must be re-run.
4. **Never edit published results in place.** Superseded results are marked
   superseded and kept, with a pointer to the replacement.
5. **Never rewrite history to hide a naming or scoping correction.** The history
   of a benchmark's construct is part of its evidence.

A correction note carries at minimum: correction id and date; affected
`operation_spec_version` / `runtime_version` / `evaluator_version` and digests;
affected scenario and variant ids; the class of change (operation, runtime,
evaluator-only); the resulting action (rerun, regrade, no action); and the list
of superseded published results.

## Rerun, regrade, no action

| Change | Stored trajectories | Published numbers |
|---|---|---|
| Operation semantics | Invalid as evidence for the new version | Rerun |
| Runtime semantics | Replay only under their original runtime | Rerun |
| Evaluator correction, compatible | Reusable | Regrade, labelled as regraded |
| Evaluator correction, incompatible | Insufficient | Rerun |
| Documentation, formatting, comments | Unaffected | No action |
| Package/tooling with no semantic effect | Unaffected | No action, provenance recorded |

## Scenario lifecycle and sealed variants

Scenario variants are either **public** — shipped, readable, used for
development and reproduction — or **sealed**, held out for evaluation and
rotated. A sealed variant's semantics are not published while it is sealed; its
identity and version are, so results can name what they ran against.

Rotation is a version event. Moving a variant from sealed to public, or
retiring one, bumps `operation_spec_version` and invalidates comparability with
results measured against the previous set. See
[CONTRIBUTING_OPERATIONS.md](CONTRIBUTING_OPERATIONS.md).

## Historical artifacts and the Boundary Track

BoundaryBench artifacts keep their original identities: package names, benchmark
and suite ids, scaffold and configuration ids, manifests, ledgers, run
directories, audit filenames and digests. They are not renumbered into an
OperateBench version line, and no OperateBench version is applied retroactively
to them.

- Cite them as *BoundaryBench evidence, reused by the OperateBench Boundary
  Track*.
- Replay each artifact **under its original contract**. Never claim a
  BoundaryBench run executed the OperateBench Core protocol.
- Boundary Track results and Lifecycle Track results are different constructs
  and are never pooled into one number.

## Preview status of this policy

The policy above describes three independent versions. **This build does not yet
carry three.** What a run artefact actually records is:

| Field | Where it lives | Value in this preview |
|---|---|---|
| `operation.operation_version` | the fixture, echoed into every artefact | `0.6.0` |
| `operation.spec_digest_sha256` | derived from the fixture's parsed semantic content | pinned per fixture |
| `engine_version` | `OPERATEBENCH_VERSION` in `src/operatebench/version.py` | `0.13.0` |
| `artifact_version` | the artefact contract itself | `8` |
| Card schema contract | `CARD_SCHEMA_VERSION` in `src/operatebench/version.py` | `1` |
| `final_state_digest_sha256`, `trajectory_digest_sha256` | derived per run | pinned per run |

One more literal exists and is deliberately **not** in that table:
`EXECUTION_LEDGER_VERSION`, currently `3`, in
`src/operatebench/execution_ledger.py`. It versions the provider execution
journal described in
[METHODOLOGY.md](METHODOLOGY.md#provider-execution-evidence-draft-bound-to-the-artefact-and-not-yet-live),
and it moves on any change to what a row of that journal states. It is not a run
identity: no evaluation reads it, and a run's result does not depend on whether
one was written. A contract-7 artefact of a model run does now *state* it, inside
`provider_execution.execution_ledger_version`, so a reader knows which journal
shape the binding is a reference to — but it remains the sidecar's number rather
than the artefact's, it moves independently of the four above, and moving it
moves none of them.

Execution-ledger version 3 extends the closed attempt-fault vocabulary with the
three Anthropic HTTP 400 classifications:
`provider_bad_request_unclassified`,
`provider_invalid_request_or_spend_limit` and `provider_spend_limit`. It also
gives `response_received` one provider-neutral meaning across every Lifecycle
provider lane: a typed HTTP-status exception records `true` because an HTTP
response reached the SDK, even though no accepted model response, usage or model
output was recorded. Transport failures remain `false`. The reader retains
version 2 under its independent historical vocabulary: valid v2 rows still read,
while a v2 row that claims any v3-only fault is refused rather than interpreted
under live meaning.

Distribution `operatebench` is **0.1.0** and is not in that table, because it is
not an identity of a run — it is provenance about the build that produced one.
The numbers move independently, and in this build they already have: the
fixture's authored semantics moved to `operation_version 0.5.0` and its identity
projection was corrected at `0.6.0`, the runtime and evaluator moved to
`engine_version 0.2.0`, then to `0.3.0`, `0.4.0`, `0.5.0`, `0.6.0`, `0.7.0`, `0.8.0`, `0.9.0` and `0.10.0`,
the artefact contract moved to `2`, then to `3`, `4` and `5`, then to `6`, `7` and
`8`, and the wheel is still 0.1.0.
`__version__` (distribution) and
`OPERATEBENCH_VERSION` (engine) are therefore two separate literals in
`src/operatebench/version.py`, pinned separately by the release scanner. A check
that required them to be equal is what previously let a runtime and evaluator
change ship while still calling itself `engine_version 0.1.0`.

`engine_version` covers the runtime **and** the evaluator together. Splitting it
into `runtime_version` and `evaluator_version` is the point of the regrade path
above, and until the split ships, an evaluator-only correction is not separable
from a runtime one: it moves the same number, so affected runs are re-run rather
than regraded. Saying that is cheaper than implying a regrade path that this
build cannot honour.

For engines 0.9.0 and 0.10.0, a Full terminal suffix is graded only from an
authenticated current-process Engine outcome or the outcome released by an exact
successful replay. Stored rows alone remain diagnostic and cannot be officially
rescored under this combined identity.

The distribution version is separate again. `operatebench` 0.1.0 is provenance —
which build produced an artefact — and is not the identity of an operation, a
runtime contract or an evaluator. The distribution also carries the historical
`boundarybench` package at **0.0.15**, with its own benchmark version
(`0.2.0-dev.1`) and its own manifest protocol version (`1`). None of those three
numbers is derived from the distribution version, and a distribution bump moves
none of them.

The preview ships `operation_version 0.6.0` for one synthetic operation, under a
runtime and evaluator that are expected to change (`engine_version 0.13.0`). Until
the construct is validated, treat every
version boundary as breaking, and expect that early results will need re-running
rather than regrading — including across the 0.1.0 → 0.2.0 → 0.3.0 → 0.4.0 → 0.5.0 → 0.6.0 → 0.7.0 → 0.8.0 → 0.9.0 → 0.10.0 engine
boundaries, which this build cannot regrade because it carries no separate
`evaluator_version`.

### Engine 0.10.0: invocation admission

Engine 0.10.0 enforces the declared invocation allowance before both event-wake
and fallback-wake calls. This changes admission and potentially the trajectory,
not the wire shape: Artifact 8, Card 1, model protocol v4, operation 0.6.0 and
distribution 0.1.0 do not move. Historical records remain readable, but replay
requires both Artifact 8 and the exact current engine identity; a different
engine is refused before re-execution. Old trajectories must use their original
runtime. Results across 0.9.0 and 0.10.0 must not be pooled or regraded; rerun
affected experiments. See [the correction note](CORRECTIONS.md#ob-corr-2026-09-09-invocation-admission).

### Retrieval tape recorder correction (2026-09-10)

The tape writer previously sorted and deduplicated non-conflicting `RETRIEVE`
requests. That was not transparent recording: the Engine records the original
`request_count` on refusal, so a repeated unknown request produced a tape that
could not reproduce its own trajectory. The writer now preserves request order
and multiplicity, including unknown names. Runtime serving still canonicalizes
valid batches; refusal counts and exact replay equality are unchanged.

This is a recorder implementation correction, not a new runtime, request, or
reader contract. Engine `0.10.0`, Artifact `8`, model protocol `v4`, provider
request mappings/profiles, and execution-ledger `3` remain unchanged. The ledger
still states the digest of the decision actually stored in the tape, using the
same digest function. The separate `operatebench.decision_tape.v1` schema also
remains unchanged: its request array has no sortedness or uniqueness invariant.
No evidence-generation schema or identity projection is changed. No new version
is introduced merely because faithfully recorded **content** now differs.

Compatibility is not byte equivalence across recorder builds. For unsorted or
repeated requests, new tape bytes, decision digests and dependent ledger/artifact
digests differ from the old writer's output. Record the producing source commit.
Old tapes remain readable as stored; missing multiplicity/order cannot be
recovered from a deduplicated tape. A historical replay failure remains a failure,
not a license to rewrite its count, tape, ledger, authority, or evaluation. Even
an old tape whose execution happens to match is not proof that the old writer
preserved the original decision. Affected experiments require fresh runs after
review, never in-place repair, rescore, or pooling based on this correction.
Synthetic offline regression results do not validate any historical model run.

### The artefact contract, and what moved it to 8

Contract 8 keeps contract 7's exact top-level shape and its
`provider_execution` binding, but moves the evaluation meaning at the version
boundary. It owns the hardened typed per-action evidence-selection contract and
the `REQUIRED_EVIDENCE_REF_NOT_CITED` finding; a contract 7 record remains
readable only under the historical semantics that produced it. This build does
not relabel or reproduce that history as contract 8.

### What moved the contract to 7

Contract 7 adds exactly one top-level field, `provider_execution`, and it is the
binding between an episode and the **execution ledger** that witnessed the
provider session behind it.

It is `null` for every run that reached no provider — a deterministic in-process
agent or a recorded playback — and it is required, non-null and exact for a run
whose decisions came off one. Always present rather than optional, because
"absent" and "null" would otherwise be two ways to write one document and a
reader could refuse neither.

What it carries is a reference, never the evidence: the execution run identity,
the ledger contract version, the digest of the complete journal, the provider,
API and model, the digests of the settings and the rate table the run executed
under, the call and attempt counts, the measured cost and forfeited reservation
as exact decimal strings, the token counts, and one call index per recorded
decision. There is no field in it for provider prose, a credential, a raw
response, a header or a filesystem path, and there is deliberately nowhere one
could be put.

**A model run with no complete scored ledger writes no episode artefact.** Not a
null binding and not an empty one: a refusal, at the point the record would be
built. An excluded or aborted run keeps its journal and has no episode artefact,
exactly as before.

**What reading the binding establishes, and what it does not.** The artefact's
own reader checks *coherence*: the shape, the identity against the execution
record, the call and attempt cardinality against the recorded attempts, and the
exact contiguous mapping of one call index per decision. It cannot check what
the ledger contains, because the ledger is not in the document — and
`replay_artifact` never opens it, carrying the binding forward beside the
execution record and reproducing the episode against a forbidden transport at
zero provider calls. Verifying the sidecar is
`operatebench.execution_bundle.audit_execution_bundle`, a separate surface
reading a separate file. Even a passing bundle audit establishes that two local
documents describe one run. It is not provider attestation, and no digest here
is.

Contract 6 stops being reproducible for the ordinary reason: it carries no
binding, so replaying one would compare a field the record does not have against
a field the reconstruction does. It stays readable.

Contract 7 remains readable under its historical request and evaluation
semantics. Contract 8 is the current reproducible contract. No release-grade
live-provider evidence, raw provider output, or published benchmark model results
are included in this publication candidate. Private diagnostics under superseded
mappings are execution history, not capability evidence.

### What moved the contract to 6

Contract 6 is a bump for a change of **meaning**, not of shape. No top-level
field is added or removed. What changed is what a record describes.

The model-visible observation no longer carries the normalized projection of the
operation record, and no longer carries the bare `trigger` envelope either. An
agent is told the coarse `phase`, the waking `event` as a *claim* — with the
authority class of whoever produced it and whether that authority is
authoritative — the action and terminal contracts including each one's exact
required reads, and the read affordances: the published catalogue, what this
invocation has been served, and what is left of the budget. Every authoritative
fact reaches an agent as the answer to a read it asked for.

Two consequences are in the record. `agent_invoked` now states
`trigger_event_authority`, `trigger_is_authoritative` and
`trigger_claim_values`, so a reader can ask, from the document alone, whether a
proposal cited what somebody *told* the agent rather than what the record holds.
And the result vector carries `retrieval_discipline`, which binds each business
proposal to the reads that established it — same invocation, after the latest
invalidation, exact tools, exact versions, exact authority — and reaches that
verdict from the trajectory rather than from the runtime's own refusals.

Because every `observation_digest_sha256` in a contract-5 tape was taken over a
projection this build cannot rebuild without putting the answer set back in
front of a provider, contract 5 stops being reproducible. It stays readable.

### What moved the contract to 5

Contract 5 made information acquisition an explicit part of the execution
record. It added the `RETRIEVE` decision and three trajectory row types:
`retrieval_served`, `retrieval_refused`, and `derived_context_invalidated`.
A served row carries the exact records shown to the agent and their source,
authority, record identity, version, schema identity, and simulated as-of time.
The evaluator can therefore recompute the record version and independently bind
the read to its invocation, turn, catalogue entry, and causal invalidation.

Contract 5 published the normalized `state` alongside the same facts served by
retrieval, so the construct claim was not closed there: an agent was still
handed the answer set unasked. Contract 6 is where that duplication is removed.

Contract 6 is a **draft** while Lifecycle Model Alpha is under construction. It
does not record provider telemetry, token usage, latency, retries, reservation or
settlement evidence. No live provider call is evidence for this phase.

**Reading and reproducing remain separate.** In this build, contracts 1 to 7 are
read and not reproduced. It reproduces contract 8 only,
re-serving every recorded retrieval from the live deterministic domain,
requiring the recorded provenance and versions to match, and carrying the
provider execution binding forward without opening the sidecar it names.

### What moved the contract to 4

Contract 4 changes what the *agent* was shown and what the record therefore has
to state, in one field each direction.

It **adds** `operation_instance_id`: the opaque, random identity of this run,
minted once and carried in every observation the agent answered. It replaces the
scenario identity at the model boundary. `scenario_id` is the index into the
authored expectation — the semantic scenario, the expected terminal and the
oracle control are one lookup from it — so an agent that reads it can be tuned
per case without running the operation, and everything measured afterwards is
recognition rather than operation. The record still carries `scenario_id`,
`semantic_scenario_id` and `expected_terminal`, unchanged and in the same place:
a grader reads the artefact, and an agent does not. What moved is only what the
agent is shown, and because every recorded decision is bound to the digest of the
observation it answered, the record has to name the instance identity or the
decisions cannot be offered the observations they were produced from.

It **removes** `reachable` from every `wait_declared` row. That flag said whether
anything still queued could deliver the declared wake condition — authored-future
truth, private to the environment, which no agent can see and no operator running
a real operation can ask. A wait is now accepted on the public contract alone (the
wake types are in the operation's published vocabulary; the operation is still
live) and *time* decides what ends it: a delivery, the declared fallback, the
operational horizon, or deadlock. A wait that never resolves is recorded as
`wait_unresolved_at_horizon` when the horizon runs out, which is a fact about how
the episode ended rather than a prediction made when it began.

**Reading and reproducing are separate.** A contract 1, 2 or 3 record still
validates under its own contract, including the `wait_declared` body it was
written with. It is not *replayed*: this build produces neither the observation
projection those decision digests were taken over nor the reachability claim
those rows carry, so a replay would report a divergence the record had no way to
avoid. The refusal is by name, before anything is executed. Refusing to guess is
the whole point of versioning the contract.

### What moved the contract to 3

Contract 3 adds no top-level field. What it changes is the durable body of one
trajectory row, and that is a contract change for the same reason a new field
is: a reader meeting the two shapes could not tell which one it was holding.

Every `effect_accepted` record now carries `bindings` — the string-to-string map
of canonical identities that effect *established*, written at the ledger
position where it committed. `request_invoice_validation` binds the invoice the
operation actually accepted, after resolving the name the agent proposed against
the authoritative invoice record.

What that is for: the evaluator now re-derives which invoice a validation was
about from evidence the ledger holds, rather than from a final state assembled
after every later event has had its say. Before contract 3 the only exact
proposal correlations were work authorisation and payment, so a genuine run
whose `request_invoice_validation` payload had been rewritten to name an
unrelated invoice was graded reliable: the ending still looked correct, and
nothing in the record disagreed with it. A proposal and a binding forged
*together* do not pass either — agreement between two rewritten fields is not
authority, and the invoice, its cycle and its evidence must have been admissible
at that ledger position.

**Row shapes are selected by the document's own contract version.** A contract 1
or 2 record is held to the `effect_accepted` body those contracts were written
under and validates exactly as it did before; a contract 3 record must carry
`bindings`, and one without them is refused. A reader that accepted either body
at any version would enforce neither, and "this build always writes them" would
stop being a claim anything could check.

### What moved the contract to 2

Contract 1 recorded what happened: identity, the events as delivered, the
trajectory, the final state and the evaluation. It did not record *what the
agent decided* separately from what the environment did with it, which is
enough for an in-process agent — re-run it and it decides the same way — and
not enough for a model, which need not agree with itself.

Contract 2 adds exactly three top-level fields, and nothing is removed:

| Field | What it carries |
|---|---|
| `decisions` | the decision tape: every outcome the agent returned, each bound to the digest of the observation it answered, in the order the engine asked for them |
| `decisions_digest_sha256` | a content digest over that tape, recomputed by the reader |
| `agent_execution` | how those decisions were produced: the outcome source, the model and protocol version where there was one, the output ceiling, the transport calls and one attempt record per call |

What the three enable together is **playback**: a run whose decisions came from
a model is reproduced by replaying its own tape **without calling a provider
again**. The environment is still fully re-executed — clock, events, domain
validators, ledger and evaluator all run — and the rebuilt request for each
decision must hash to the attempt recorded beside it, so the recorded provider
identity is part of the run rather than a label on it. The replayed agent holds
a transport that refuses to reach anything, and a replay report names every
field it took from the record instead of re-deriving.

Two limits are part of the contract rather than caveats on it.

**This is consistency, not provider attestation.** The reader binds the
execution record, the decision tape and the agent identity to each other and
refuses a record whose three contradict, so a deterministic control's run
relabelled as a model's fails by name. What no local file can establish is that
a provider was ever asked: anyone who can run this build can compute the same
digests from the same observations and mint a record that is consistent
throughout. Closing that needs evidence a provider signs, which this format does
not hold.

**Contract 1 and 2 records are still read, and replay is where compatibility
ends.** A contract 1 artefact still validates under this build, against the
contract 1 field set exactly: it does not acquire the contract 2 fields by being
read here, and a contract 1 document carrying one is refused rather than
half-read. The same holds one contract on: a contract 2 record is read under
contract 2's field set and contract 2's row bodies, and does not acquire
contract 3's `bindings` by being read by a build that writes them.
Replaying one is a different question, because a replay re-executes the
operation, the engine and the evaluator the record names. Where those semantics
have moved — as they have across the `operation_version` 0.1.0 → 0.2.0 boundary
— the replay is refused as a **named incompatibility** stating both spec
digests. That refusal is the honest answer: the alternatives are rewriting the
frozen record or narrowing the comparison until it agrees, and the second is how
a replay that compares three fields comes to be described as a replay.

## See also

- [METHODOLOGY.md](METHODOLOGY.md) — what the evaluator asserts.
- [CONTRIBUTING_OPERATIONS.md](CONTRIBUTING_OPERATIONS.md) — intake and QC.
- [../PUBLIC_RELEASE_CHECKLIST.md](../PUBLIC_RELEASE_CHECKLIST.md) — release
  gates.


### Current fixed-canary envelope reconciliation

The current request mappings retain the compact disclosure contract above.
Fixed one-cell operators pin the canonical UTF-8 SDK request census plus
1024 bytes of token-bound headroom. These build-owned admission controls do not change
provider rates, the 46-call authored episode, the 50-call ceiling, or any
historical evidence or authorization. Previous numeric controls are refused.

| Operator | Maximum input bound | 50-call token cap | Worst-case USD | Fixed USD cap |
|---|---:|---:|---:|---:|
| Haiku | 25881 | 1498850 | 2.31805 | 2.32 |
| Sonnet | 25884 | 1499000 | 4.6364 | 4.68 |
| Luna, including the original generic canary | 27241 | 1566850 | 0.51817 | 0.52 |
| Mistral Small | 27331 | 1571350 | 0.3278625 | 0.33 |
| Grok 4.5 Responses | 27278 | 1568700 | 3.9566 | 3.96 |
| Grok 4.6 Responses | 27279 | 1773550 | 5.1855 | 5.20 |

### Illustrative offline Grok 4.6 envelope v1

The public one-cell Grok 4.6 operator has no enabled live path. Its illustrative
fleet/profile and authorization-record identities are distinct from live
records; example fleet ceilings are fictional and grant no spending authority.
External consumption also refuses without accessing records or credentials.
Explicit mock injection permits offline durable-record tests, never a paid
client. This is an operator-envelope boundary, not a provider mapping, engine
or evaluator version change. Per-cell technical caps are retained.

The output reservation remains 4096 tokens except for Grok 4.6's unchanged
8192-token lane. Worst-case exposure is fifty times the input-bound and
output-bound cost at the fixed rates, computed with exact Decimal arithmetic.
Sonnet retains its existing 4.68 cap; it is no longer the minimal cent cap for
this request census. Grok 4.5's ten-cell declared worst-case total is 39.566 USD.
Mistral's SDK adds `stream: false` after initial reservation: the 46-request
wire-bound total is 0.25666935 USD, while initial reservations total
0.25656585 USD. These are distinct accounting observations, not interchangeable
ledger values.

All seven current entrypoints reserve `execution_ledger.partial.ndjson` before
client construction alongside the ledger and artifact names. The partial file
is non-scored diagnostic evidence, including on faults; it is never accepted as
an Artifact. Existing permanent one-shot claim markers retain their semantics.
Frozen Alpha execution and historical readers are not assigned this new output
contract. Its test-only historical dependency identities are obtained from the
exact archived runtime, while current causal replay still rejects old tapes.
