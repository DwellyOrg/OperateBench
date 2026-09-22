# Audit-integrity outcome and executed-evidence successor protocol

**Status:** normative wire contract frozen; runtime, adapters, evaluator, replay, Maintenance declarations, No-Time, and Artifact 9 unimplemented

**Contract ID:** `operatebench.audit-integrity-successors.v1`

**Resources:** [registry](schemas/audit-integrity-successor-registry-v1.json), [new-field census](schemas/audit-integrity-successor-census-v1.json), and [independent vectors](schemas/audit-integrity-successor-v1.golden.json)

This contract-only phase freezes additive successor resources. It changes no production
class, validator, loader, export, Engine, runner, evaluator, replay, artifact, provider,
Card, runtime-identity adapter, Maintenance declaration, CommonOutcome, or No-Time path.
Schema conformance provides no runtime or evaluation authority.

## 1. `EpisodeOutcomeV3`

[EpisodeOutcomeV3](schemas/episode-outcome-v3.schema.json) has discriminator
`operatebench.episode_outcome.v3`, version `3`, and retains every V2 field and exact
field schema except `status`. Status is a closed tagged union:

- `{kind: "domain_terminal"}` requires a non-null domain terminal and
  `replay_final=true`.
- `{kind: "engine_halt", code: "operation_deadlock" |
  "operational_horizon_exhausted"}` preserves V2's state-derived
  `terminal_outcome` and `replay_final` semantics.
- `{kind: "integrity_error", contract_id:
  "operatebench.domain-generated-audit-budget.v1", code: <one of the exact three
  DOMAIN_AUDIT codes>}` is an assigned, non-retryable runtime integrity halt.

The closed schema itself enforces the expressible `domain_terminal` terminal/finality
branch. Engine-halt and integrity-error terminal/finality values remain state-derived;
the independent oracle pins their exact values in each golden rather than inventing a
global terminal blacklist. The generic content-coherence oracle does not authenticate a
domain terminal or rollback snapshot: a semantically legal coherent terminal/state/
trajectory replacement remains content-valid. Only the five exact fixture goldens pin
those concrete values, and future process-local Engine authentication plus replay owns
execution truth.

An integrity outcome is the exact pre-transition rollback snapshot. `terminal_outcome`,
`replay_final`, `final_state`, trajectory, events, clocks, and counters preserve the
state that existed before the refused transition. It may therefore be preterminal,
provisional, or post-finality. It creates no synthetic terminal, finality, trajectory
row, or domain event. Only dedicated audit evidence changes outside the outcome.
Transition-horizon admission has precedence over domain-budget admission.

The outcome carries no primary `SUCCESS`/`FAILURE` field, audit evidence, audit digest,
self-digest, or new digest domain. A domain-audit summary `SUCCESS` means only that the
audit protocol found no integrity error; it never implies business `SUCCESS`.

## 2. `ExecutionRecordV2`

[ExecutionRecordV2](schemas/execution-record-v2.schema.json) has discriminator
`operatebench.execution_record.v2`, version `2`, retains every V1 field and shape, and
adds required `provider_dispatch_state` with exact values `not_applicable`,
`not_started`, and `started`.

- Deterministic and recorded executions use `not_applicable`, null model/protocol/token
  fields, zero calls and attempts, and no provider binding. They are never relabelled
  from a selected model plan.
- Model executions retain non-null selected model, protocol, and positive token ceiling.
  `not_started` requires zero calls, attempts, and decisions and a null provider binding.
  It is legal only for assigned-preexecution integrity-error evidence with a null
  finding invocation index. The selected model plan remains visible.
- Model `started` requires calls = attempts = decisions >= 1 and a non-null provider
  binding. Existing ordering, fault, model, ledger, and contiguous call-index rules
  remain strict.

The schema closes source/plan/dispatch/call/attempt combinations. The independent
content-coherence oracle additionally rejects model-to-recorded laundering; mismatched
model identity where execution and binding both carry it; non-contiguous, out-of-range,
descending, fault-incoherent, or excluded attempts; and content-visible provider
activity whose invocation is later than a non-null integrity-finding invocation.
It cannot prove that no same-invocation or omitted post-finding activity occurred.

## 3. `ExecutedRuntimeEvidenceV2` and digests

[ExecutedRuntimeEvidenceV2](schemas/executed-runtime-evidence-v2.schema.json) has
`operatebench.executed_runtime_evidence.v2`, version `2`, retains every required V1
field, references OutcomeV3 and ExecutionRecordV2, and requires one non-null complete
`domain_generated_audit_budget_evidence` v1 object plus its non-null digest.

The outer self-digest domain is
`operatebench.digest.executed_runtime_evidence.v2`. Its input is exactly the complete
wrapper `{domain,schema,content}`, recursively key-sorted canonical UTF-8 JSON with no
trailing newline. `content` is the complete evidence object with only
`runtime_evidence_digest_sha256` omitted. The sibling audit digest is included.
The nested digest independently uses the already frozen
`operatebench.digest.domain-generated-audit-budget-evidence.v1` projection and domain.

V2 use itself claims protocol applicability. Non-applicable historical runs remain V1.
OutcomeV2 + EvidenceV1 is accepted only as historical content with no audit claim;
V3 + V1 refuses `unsupported_outcome_schema_version`; V2 + V2 refuses
`outcome_evidence_generation_mismatch`; V3 + V2 is shape-accepted only and has no
runtime authority.

## 4. Cross-record reconciliation

The dedicated audit evidence is the sole Core witness. It is not duplicated in generic
trajectory and no causal-witness wire is invented.

- A final audit summary `SUCCESS` and no finding require a non-integrity outcome.
- A final summary `ERROR` requires exactly one finding immediately before it and an
  `integrity_error` outcome with exactly the same contract and code. The refused batch
  commits zero units.
- Integrity outcomes require `ERROR`; non-integrity outcomes forbid `ERROR` and a
  finding.
- Outer and embedded `operation_instance_id` are exact. The accepted Operation Core
  digest and future Card aggregate/source declaration bind normatively. Missing
  preidentity facts construct no identity, outcome, evidence, finding, or denominator
  claim.
- The content oracle recomputes the outer, audit, final-state, trajectory, and decision-
  tape digests; validates exact calendar instants, elapsed minutes, event range/order,
  status branches, source-prefix counts, invocation coverage and known finding-prefix
  order; and reconciles every field actually duplicated on this wire (model, invocation/
  turn pairs, execution/attempt/decision counts, and contiguous call indices).
- Attempt request digest and decision observation/action/content digest joins are
  required when a future provider decision/request record supplies their counterpart.
  This wire supplies no such independent counterpart. Likewise token/cost totals,
  provider/API/protocol details beyond their closed schema/model join, execution run ID,
  settings/pricing digests, and execution-ledger identity/digest are single-record
  assertions here. `provider_binding` therefore remains a wire candidate until a fresh
  process-local provider sidecar audit compares the complete ledger and provider rows.
- JSON Schema and content coherence cannot authenticate rollback, finality, provider
  execution, ledger history, pre-transition state, or absence of omitted/same-invocation
  post-finding work. Future process-local Engine authentication, runtime adapters/replay,
  and provider sidecar audit own those obligations.

Assigned evidence selects one independently literal-pinned row of the complete
fourteen-row domain-audit refusal matrix by `refusal_class`. Model `not_started` is
limited to assigned-preexecution rows (including the two frozen assigned-defect forms),
and budget-overrun classes cannot launder their required budget-exceeded code. The
generic content-coherence oracle admits aggregate overrun only when committed intents
plus the requested count exceed the accepted Card aggregate declaration. It rejects
`valid_reservation_above_declared_source` unconditionally: this contract-only slice has
no authenticated canonical Operation Core declaration input or content-digest verifier.
The exact audit-v1 source-overrun golden remains a frozen wire/reference vector, not
successor execution authority.

Source-class semantic admission is fail-closed and deferred to a future runtime
verifier. That verifier MUST receive an Engine-authenticated Operation Core declaration,
verify its content digest against `operation_core_content_digest_sha256`, and prove the
source maximum from `max_accepted_occurrences * max_events_per_transition`. A
caller-supplied source limit or self-consistent evidence rows are insufficient.

All reference JSON is parsed from raw UTF-8 with the raw byte ceiling checked before
parsing, strict UTF-8 decoding, duplicate object-key rejection, and lone-surrogate
rejection. The executable oracle then applies all canonical-byte and iterative structural
bounds described in section 5 before JSON Schema, content coherence, or digest
projection. This is an executable content-contract oracle only; no production loader or
authority is introduced by this phase.

### Authority boundary

| Layer | What it establishes | What it does not establish |
|---|---|---|
| JSON Schema | Closed wire shape, types, enums, and expressible branch constraints | Cross-record coherence or execution truth |
| Generic content-coherence oracle | Invariants recomputable from the record and supplied frozen contracts, including all section digests and visible prefix/order joins | Authentic terminal/state/provider/ledger/pre-transition truth, omitted activity, or caller-independent provenance |
| Exact five-fixture golden validator | The concrete literal fixture bytes and pinned terminal, state, trajectory, provider identity, and outer hash | That an arbitrary run actually executed those contents |
| Future authenticated execution verifier | Process-local Engine provenance, runtime adapter/replay agreement, provider-sidecar ledger audit, and no-post-finding execution provenance | Not implemented or claimed by this contract-only phase |

Artifact 8 is neither migrated nor rewritten. Card v1 has no protocol claim. Card v2
without an authorized source declaration refuses before identity. No complete Artifact
9 census, Maintenance declaration, runtime adapter v2, or support lane is claimed.

## 5. Bounds and independent arithmetic

The successor profile caps depth at 64; non-audit content at 100,000 nodes and 8 MiB
canonical UTF-8; total content at 2,100,000 nodes and 268,435,456 bytes; ordinary
containers at 4,096 members; the exact audit `records` exception at 100,000; and each
string at 1 MiB UTF-8. Existing safe-integer profiles remain, including the frozen
signed-int64 audit declaration field and 99,998 record integer caps.

The executable convention counts containers only for depth: a root object or array is
depth 1, each nested object or array adds one, and scalar leaves add no depth. A
structural node is each JSON value/container, each object key, and each array-element
slot. Thus a maximum nine-field committed-record object consumes nineteen nodes and its
slot in `records` consumes the twentieth. Object and array cardinality is 4,096 except
only the array at the exact JSON Pointer
`/domain_generated_audit_budget_evidence/records`; a same-named array anywhere else has
no exception.

The non-audit projection starts with the complete root object and removes exactly the
root member `domain_generated_audit_budget_evidence`, both its key and its complete value
subtree. The exception is available only when that value is an object with exactly the
seven frozen audit-envelope keys (`schema`, `schema_version`, `contract_id`,
`operation_core_content_digest_sha256`, `operation_instance_id`, `declared_max_events`,
and `records`). An extra, missing, relocated, or nested lookalike key is rejected or
charged to non-audit limits; it cannot launder arbitrary payload. The 8 MiB check is the
canonical UTF-8 size of that exact projection. Total canonical size and total nodes
always include the complete audit member. Every string value and object key is measured
after strict UTF-8 encoding. Safe integer/type bounds remain JSON Schema obligations;
in particular, booleans are not treated as integers by those schemas.

The independent no-fixture bound vector computes the longest schema-valid committed
record as 1,102 canonical bytes. At 100,000 records, 99,999 commas, a 591-byte audit
envelope, and the complete 8 MiB non-audit allowance, the upper bound is 118,689,198
bytes, below 256 MiB. The records and their array slots can consume 2,000,000 structural
nodes. The audit member key, seven-key envelope, and `records` array add sixteen more,
so an independently saturated 100,000-node non-audit projection and 100,000 maximum
records would total 2,100,016. The independent 2,100,000 total-node ceiling therefore
binds first by sixteen nodes; the profile does not promise that every independent maximum
is simultaneously attainable. Boundary tests use reduced immutable profiles to prove
the same accounting without huge fixtures.

## 6. Frozen scope

The registry names at least sixteen historical resources that remain byte-identical,
including OutcomeV2, ExecutionRecordV1, EvidenceV1 and its golden, all frozen audit
budget resources, Card v1/v2, adapter-v1 registry/census, and Artifact 8 behavior. The
successor census is independently pinned and covers only four new pointers; it is not
an Artifact 9 census.

The five golden vectors independently freeze deterministic zero-event audit success, started model
success with a synthetic provider wire candidate, assigned-preexecution model
`not_started` error with null provider, midexecution started-model error, and
post-finality error. Every vector pins nested audit digest, exact outer wrapper text,
UTF-8 hex, byte count, SHA-256, and semantic literals while explicitly claiming no
runtime authority. Their complete literal pins include manifest and operation identity,
the full status object, exact terminal/finality values, execution/provider identity and
cardinalities, audit joins and record kinds, nested digest, canonical byte count, and
the full outer SHA-256, so coherent terminal/state/trajectory/provider mirror-and-digest
repins of those fixtures are rejected by the exact golden validator even when they
remain semantically legal to the generic content-coherence oracle.
