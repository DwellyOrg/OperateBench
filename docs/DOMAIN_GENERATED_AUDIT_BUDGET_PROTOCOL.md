# Domain-generated audit budget protocol v1

**Status:** normative contract frozen; runtime unimplemented
**Protocol ID:** `operatebench.domain-generated-audit-budget.v1`
**Wire resources:**
[declaration/source schema](schemas/domain-generated-audit-budget-v1.schema.json),
[evidence vocabulary](schemas/domain-generated-audit-budget-evidence-v1.schema.json),
[registry](schemas/domain-generated-audit-budget-registry-v1.json),
[ownership/consumer census](schemas/domain-generated-audit-budget-census-v1.json), and
[independent vectors](schemas/domain-generated-audit-budget-v1.golden.json)

Normative terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** govern a future
implementation. This document freezes a contract, not runtime behavior. Nothing in
this amendment activates a counter, reservation API, No-Time runtime, provider path,
evaluator path, or research comparison.

## 1. Normative boundary and status

The protocol bounds audit-event intents authored by operation execution but created
by Core. The counting unit is exactly one **distinct logical operation-authored
audit-event intent successfully committed through the future Core budget API**. It
is charged once at atomic creation. A request, reservation, refused transition,
ledger row, delivery, or later lifecycle action is not by itself a charge.

The declaration and evidence schemas freeze wire shapes only. This prose and the
registry govern ownership, transition semantics, arithmetic, cross-record
invariants, precedence, accounting, and compatibility that JSON Schema cannot
express. The class-complete test oracle is an independent executable specification:
it checks declaration arithmetic and every record-order, batch, source, finding, and
terminal invariant in section 9 without importing a future production verifier. It
is not a production validator or parser; that verifier remains deferred.

The following remain unimplemented hard dependencies:

1. operation-owned source declarations and independently reviewed literal maxima;
2. Core reservation, counter, transaction, integrity, and provenance behavior;
3. an additive EpisodeOutcome successor that can express integrity `ERROR`;
4. an additive ExecutedRuntimeEvidence successor that authenticates this evidence;
5. an additive runtime-identity adapter registry/census;
6. evaluator and Measurement alignment; and
7. separately authorized No-Time implementation and verification.

`EpisodeOutcomeV2`, `ExecutedRuntimeEvidenceV1`, Artifact 8, Operation Card v1 and
Operation Card v2 are not changed or reinterpreted by this protocol.

## 2. Ownership

Ownership is exact:

- **Operation Core** owns the authored scalar maximum, source identifiers, source
  meaning, accepted-occurrence maxima, per-transition fanout maxima, and their
  operation-version authorization. The existing Operation Card v2 scalar
  `content.transition_contract.declared_max_domain_generated_audit_events` carries
  only the authored aggregate maximum.
- **Core runtime** owns the closed counted/excluded taxonomy, checked aggregate
  alignment, operation and per-source counters, reservations, duplicate-intent
  protection, atomic commit/refusal, integrity codes, bounded findings, committed
  evidence vocabulary, and replay/evidence cross-check.
- **Card** projects the Operation Core scalar; it owns no source table, counter,
  reservation, remaining count, finding, or emitted count.
- **Semantic Scenario, Variant, scaffold, provider, evaluator, and No-Time** own no
  counter and cannot authorize, increment, reset, infer, or repair one.
- **No-Time** is only a future consumer of the same operation-owned logical count.
  Its current protocol remains specified-only and unauthorized.

The versioned ownership/consumer census contains exactly twelve tuples. It is
additive and does not edit the frozen Card-v2 field census or
`runtime-identity-adapter-registry-v1`.

## 3. Declaration and checked aggregate

A declaration conforms to `operatebench.domain_generated_audit_budget.v1` and binds:

- the exact protocol ID;
- the accepted Operation Core content digest;
- `declared_max_events`, equal to the Card-v2 scalar; and
- an ordered source table.

Each source row is closed and contains `source_id`, `counted_kind`,
`max_accepted_occurrences`, and `max_events_per_transition`. Source IDs MUST be
unique. The Card-v2 scalar remains signed int64, but protocol v1 declarations are
representable only when each source maximum, fanout, checked source product, and the
aggregate are exact JSON integers within `0..99,998`; booleans are not integers and
fanout MUST be positive. The evidence array capacity is exactly 100,000 records.
The v1 aggregate cap of 99,998 leaves room for at most `D` committed intents, one
integrity finding, and the mandatory terminal summary. Eligibility is established
before authorization and before trial identity assignment. A Card scalar above
99,998 cannot bind protocol v1 and MUST refuse at that pre-identity boundary as
`DOMAIN_AUDIT_CONTRACT_VIOLATION` or use a future protocol version. That refusal has
no episode evidence or denominator claim. The checked formula is:

```text
D = checked_sum(each source.max_accepted_occurrences
                checked_multiply source.max_events_per_transition)
```

Every multiply and add MUST first fit signed int64 and then the v1 capacity, and `D` MUST equal both
`declared_max_events` and the accepted Card scalar. Overflow, type confusion, or
mismatch refuses before authorization and identity assignment. Source order is identity-bearing and MUST equal
the operation-version-authorized order; sorting a caller candidate cannot authorize
it.

No source occurrence maximum, fanout, or `D` may be inferred from an observed
trajectory, fixture event count, Artifact 8, timer count, invocation limit,
transition limit, policy duration, or the same production formula used as a test
oracle. This contract defines no authoritative Maintenance source census and
assigns no Maintenance source maximum or `D`; Maintenance readiness is unresolved.

## 4. Closed counting taxonomy

The v1 counted kinds are exactly:

1. `scheduled_audit_event_intent`: a domain-created scheduled/timer audit intent;
2. `immediate_audit_event_intent`: a domain-created immediate ordinary audit intent.

Both MUST cross the future Core budget API. Generic `context.record(...)` calls are
never implicitly countable. A new counted kind requires a new protocol version.

The registry's `excluded_kinds` tuple is authoritative, exact, ordered, and closed;
its 32 literals correspond one-for-one to the following excluded classes, all of
which cost zero:

- authored absolute and conditional events;
- activation or queueing of an authored conditional;
- authored audit-only, non-triggering, rejected, or post-terminal deliveries;
- generic business-state or evidence records, including `context.record(...)`;
- checkpoint and obligation creation, discharge, cancellation, or breach;
- proposals, refusals, accepted effects, retrievals, agent/invocation/scaffold,
  provider, execution-ledger, side-effect, wait, and wake records;
- timer scheduling/cancellation/firing/delivery/disposition/resolution/normalization
  lifecycle after the originating logical intent was charged;
- `finality_cut` and `UNTRIGGERED_AT_FINALITY`;
- transition-horizon, budget-integrity, and observability findings; and
- terminal budget findings and summaries.

An obligation costs zero. A separately created reminder, deadline, follow-up,
finalization, or ordinary immediate audit intent costs one. No record name,
disposition, or projection can change these classes.

## 5. Identity and digest domains

Each counted intent has a domain-supplied namespaced `intent_id`, unique within
`operation_instance_id`. Core MUST reject reuse after commit, including after
cancellation, firing, delivery, disposition, or resolution. A distinct accepted
replacement uses a new identity and costs one.

Domains are disjoint:

- contract: `operatebench.domain-generated-audit-budget.v1`;
- intent digest: `operatebench.digest.domain-generated-audit-intent.v1`;
- budget evidence digest: `operatebench.digest.domain-generated-audit-budget-evidence.v1`.

A digest input is exactly the canonical JSON wrapper
`{"content":<closed projection>,"domain":<exact literal>}` serialized as UTF-8 by
the repository canonical JSON algorithm: recursively sorted object keys, separators
`,` and `:`, no insignificant whitespace, no ASCII escaping, no NaN, and no trailing
newline. SHA-256 is computed over those exact bytes and rendered in lowercase hex.
There is no NUL-delimited or naked-value alternative.

The complete intent projection key set is exactly `contract_id`,
`operation_core_content_digest_sha256`, `operation_instance_id`, `intent_id`,
`source_id`, `transition_id`, and `batch_ordinal`. The complete evidence projection
key set is exactly `contract_id`, `operation_core_content_digest_sha256`,
`operation_instance_id`, `declared_max_events`, and `records`; records retain every
field defined by their closed record branch. The evidence schema `$defs` freeze both
closed projections and both exact-domain wrappers. Intent projection excludes
`counted_kind`, due time, queue position, timer sequence, disposition, counters, arm
name, ledger index, and every unspecified key. Evidence projection excludes wire
schema tags, any enclosing artifact/outcome fields, credentials, capabilities,
provider/model text, and every unspecified key. Golden verifier vectors hash only
these wrapped projections.
Full/No-Time arm-neutral correspondence remains independently derived from the
accepted common cause and MUST NOT reuse a run-scoped intent digest.

## 6. Reservation lifecycle and atomicity

Core MUST open one transition-local transaction around each reducer, accepted
action, and terminal transition. Before any authoritative business-state mutation,
identity allocation, checkpoint/obligation/cycle creation, queue insertion, timer
sequence increment, domain ledger append, side-effect dispatch, or finality change,
the domain requests one reservation with the exact source ID and complete batch.

Core validates, in order:

1. higher-priority transition-horizon admission;
2. supported protocol and accepted declaration/Card/operation alignment;
3. exact int64 counter state and arithmetic;
4. known source, counted kind, unique intent identities, and exact batch shape;
5. positive requested batch not above per-transition fanout;
6. checked next source occurrence/emission counts; and
7. checked `operation_committed + requested_count <= D`.

Capacity is reserved for the complete batch or not at all. Every successful creation
consumes exactly one matching token unit. Accepted transition completion requires
exact token exhaustion and atomically publishes business state, emitted intents,
queue entries, records, source counters, and operation counter. Refused, malformed,
exceptional, unconsumed, partially consumed, or rolled-back transitions publish none.
A two-unit batch with one unit available therefore represents one refused request,
zero committed intents, and zero partial mutation.

The authoritative operation counter starts at zero once per operation instance and
never resets across agent turns, invocations, wakes, transport retries, proposal
retries, accepted effects, reopen cycles, or provisional-close activations.
Transition reservations are transient and end on commit or rollback.

## 7. Duplicate, retry, reschedule, and finality rules

Malformed/rejected proposals, transport/model retries, reducer refusals, and failed
uncommitted transitions cost zero. Re-execution of a committed transition MUST be
idempotent at the Core boundary or halt as duplicate-intent contract violation; it
cannot silently create a second intent.

Cancellation, firing, delivery, disposition, resolution, and normalization of an
existing identity cost zero and never refund capacity. An explicit atomic move of
the same unconsumed identity costs zero. Cancel-and-replace, re-arm, reschedule, or
reopen with a new identity costs one when the replacement commits.

After replay finality, Core MUST NOT enter the domain reducer or admit a new
operation-authored intent. Authored queued events remain excluded `post_terminal`
evidence. Any post-finality reservation attempt is a contract violation with no
resurrection or mutation.

## 8. Integrity errors and precedence

The stable codes are exactly:

- `DOMAIN_AUDIT_BUDGET_EXCEEDED`: a well-formed request exceeds a source or
  operation maximum;
- `DOMAIN_AUDIT_CONTRACT_VIOLATION`: unsupported contract/source/kind, wrong fanout,
  duplicate/unknown identity, missing/foreign/reused/unconsumed reservation,
  unbudgeted emission, or post-finality request;
- `DOMAIN_AUDIT_COUNTER_INTEGRITY`: malformed counter or checked-int64 overflow.

An already-exhausted transition horizon has higher precedence and prevents domain
entry or reservation. Otherwise counter-shape/integer integrity is established
before capacity comparison; a valid over-capacity request is budget exceeded.
The registry machine table freezes refusal stage and every finding field rule.
Unsupported/wrong contracts, null/string/boolean/non-exact values, negative or
out-of-signed-int64 values, duplicates or unauthorized source order,
source/aggregate mismatch, and values above v1 capacity are eligibility failures.
They map to `DOMAIN_AUDIT_CONTRACT_VIOLATION` and MUST be rejected before
authorization and identity assignment. Checked signed-int64 arithmetic overflow or a
malformed runtime counter maps to `DOMAIN_AUDIT_COUNTER_INTEGRITY`. Only a valid
runtime reservation above a declared aggregate/source maximum maps to
`DOMAIN_AUDIT_BUDGET_EXCEEDED`. Before identity assignment this is a configuration
refusal with the same stable code but no finding, terminal evidence, primary result,
or denominator claim. It is forbidden to manufacture an operation instance merely
to evidence an ineligible declaration. Once an identity is assigned, its accepted
Card declaration is necessarily an exact built-in, nonnegative signed-int64 and is
immutable. A stale or manipulated runtime collaborator, or a collaborator value
that disagrees with that accepted Card value, is an assigned
`DOMAIN_AUDIT_CONTRACT_VIOLATION`: exactly one finding and summary record primary
`ERROR` and assigned-denominator membership, and both bind the accepted Card value,
never hostile raw input.

All three runtime codes are non-retryable integrity `ERROR`, not business
`FAILURE`, operation deadlock, operational horizon success, proposal refusal,
provider fault, or infrastructure exclusion. There is no hidden reducer retry,
replacement trial, fresh invocation, provider call, compensating emission, or
counter restoration.

## 9. Refusal evidence, outcomes, and measurement

On refusal, authoritative counters and domain state remain at their exact
pre-transition values. For every assigned identity Core MUST append exactly one
bounded Core-owned integrity finding followed by exactly one terminal summary. On
success there is no integrity finding and Core MUST append exactly one terminal
summary. A summary is therefore mandatory exactly once on every assigned success or
refusal. The finding vocabulary contains contract ID, stable code, exact
`refusal_class`, nullable source ID, exact requested count, pre-attempt operation
count, nullable pre-attempt source count, declared limit, nullable transition
identity, and nullable invocation index. It excludes domain
payloads, provider/model text, exception prose, credentials, reservation capability,
queue bodies, and unrestricted diagnostics.

Committed-intent records bind contract ID, intent digest/ID, source, counted kind,
transition identity, batch ordinal, and resulting operation count. A terminal
summary binds contract ID, Card-sourced declaration, committed count, exact
`SUCCESS`/`ERROR` result, and nullable integrity code. `SUCCESS` requires no finding
and null code; `ERROR` requires exactly one finding and the same non-null code.
These are protocol vocabulary, not `EpisodeOutcome` or executable
runtime schemas.

A future verifier MUST require committed intents first, then at most one finding,
then exactly one summary last; commits after a finding/summary are forbidden. It
MUST rederive contiguous operation counts `1..N`, unique intent IDs/digests, exact
summary count/declaration, source existence/kind, per-`(transition_id, source_id)`
ordinals `0..batch_size-1`, fanout, distinct source-batch occurrence maxima, and
source committed-intent maxima. Findings MUST match the registry row field-by-field. Its fourteen ordered rows freeze
`stage`, source presence/null semantics, requested-count zero/positive/exact rule,
transition and invocation presence/null semantics, operation/source pre-count
semantics, accepted-Card declaration source, finding requirement, and finding code.
The six `pre_identity` rows require no evidence. `assigned_transition` rows bind the
exact attempted source/reservation, transition, batch size, invocation scope, and
authoritative pre-counts as stated by their row. `assigned_preexecution` rows use the
row's exact null/zero fields. The separately frozen assigned accepted-declaration
mismatch matrix is exactly `source_id=null`, `requested_count=0`,
`transition_id=null`, `invocation_index=null`, `operation_committed_before=0`,
`source_committed_before=null`, and the accepted Card declaration; it permits zero
committed intents, exactly one `DOMAIN_AUDIT_CONTRACT_VIOLATION` finding, and exactly
one `ERROR` summary. No other interpretation is permitted.
Budget-exceeded requests are exact positive integers and must prove aggregate or
source over-capacity; contract/counter classes and precedence MUST match the registry
table. The attempted refused batch commits zero. The result/code/finding matrix above
is exact. JSON Schema cannot prove these cross-record invariants; accepting shape
alone is forbidden.

The current successor contract's test-only generic content-coherence oracle proves an
aggregate overrun from committed intents plus the requested count against the accepted
Card aggregate declaration. It MUST reject
`valid_reservation_above_declared_source` unconditionally because this contract-only
slice has no authenticated canonical Operation Core declaration input or content-digest
verifier. The exact source-overrun audit-v1 golden remains frozen wire/reference, not
successor execution authority. Source-class semantic admission is fail-closed and
deferred until a future runtime verifier receives an Engine-authenticated Operation Core
declaration, verifies its content digest against
`operation_core_content_digest_sha256`, and proves the source maximum from
`max_accepted_occurrences * max_events_per_transition`. Caller-supplied limits or
self-consistent evidence rows are insufficient.

Evidence `declared_max_events` fields are diagnostic nonnegative signed-int64 values,
not protocol-v1 capacity declarations. This distinction makes a system defect
representable if identity was incorrectly assigned to a Card-valid signed-int64
value above 99,998. Such evidence MUST use the exact assigned-over-capacity matrix:
zero commits, exactly one `v1_capacity_exceeded` contract-violation finding, and one
`ERROR` summary, all binding the accepted Card value. Above-cap `SUCCESS`,
`DOMAIN_AUDIT_BUDGET_EXCEEDED`, any committed intent, or any other matrix is invalid.
Boolean, negative, string, null, and out-of-signed-int64 diagnostic declarations are
invalid. They can never be accepted Card declarations and therefore never enter
assigned evidence.

Every assigned trial gets exactly one primary `SUCCESS`, `FAILURE`, or `ERROR`.
Assigned overflow and contract/counter integrity failures are `ERROR`, contribute non-success
mass to the assigned denominator, leave `FAILURE` unchanged, and are reported
separately. Pre-identity eligibility refusal contributes no primary result, evidence,
or denominator mass. Assigned errors never authorize exclusion or replacement. Because they identify a
simulator/declaration/runtime-integrity defect, an affected contrast is
`UNINTERPRETABLE` while raw assigned `ERROR` mass remains reported.

`EpisodeOutcomeV2` cannot express this halt and `ExecutedRuntimeEvidenceV1` cannot
authenticate this vocabulary. Additive successors are hard dependencies. This PR
MUST NOT name, freeze, or implement `EpisodeOutcomeV3` or `ExecutedEvidenceV2`.

## 10. Provenance and replay

Future live evaluator authority MUST bind status, finding, source, request,
counters, committed records, trajectory, events, and final state into a complete
mutation-sensitive Engine-authenticated snapshot. Persisted authority MUST be
released only by successful exact replay. Caller-recomputed digests, copied rows,
self-consistent summaries, and coherent repins are not authority.

Reservation and live evaluation capabilities are process-local and non-wire. Copy,
replace, deepcopy, pickle, filesystem round-trip, fork, spawn, pipe, socket, or IPC
transfer MUST NOT transfer them. Exact replay reconstructs the same pre-error state,
single request, zero refused committed units, code, and summary with zero provider
calls.

## 11. Compatibility and dependency order

Operation Card v1 has no claim under this protocol. Operation Card v2's existing
signed-int64 scalar is unchanged; values above 99,998 cannot pass protocol-v1
eligibility. The evidence vocabulary's wider diagnostic field does not widen the
declaration protocol or authorize such a value.
The scalar's schema bytes remain unchanged. The frozen Card v2
census, runtime-adapter-v1 registry/census, EpisodeOutcomeV2,
ExecutedRuntimeEvidenceV1, their goldens, and Artifact 8 remain byte-identical.
Legacy readers/runtimes remain legacy and cannot advertise enforcement.

Implementation proceeds only in registry order: contract resources; operation
source declaration/literal maxima; Core runtime; additive outcome successor;
additive evidence successor; additive runtime identity registry; evaluator and
measurement alignment; and separately authorized No-Time. No downstream stage may
infer or bypass an unresolved upstream dependency.

The golden resource contains a closed mutation-operation schema and executable
mutations for every malformed/semantic class. Owning tests independently pin the
complete literal mapping from every malformed case name to base, target, layer,
code, and exact operations, and from every semantic case name to its exact operations
array and complete result. It also carries an independently pinned complete
operation/delta table covering existing-identity delivery and
lifecycle zero-cost rules, new-identity creation/reopen/rearm/reschedule charges,
refused/uncommitted retries, committed-identity retry violation, and post-finality
refusal. Wrapped intent/evidence digest vectors, exact-capacity construction, all
reviewer attacks, and coherent repins are executed by the owning tests. It is
conformance evidence for this contract only.
