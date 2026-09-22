# `stateful_time_removed` protocol

**Status:** Phase A normative protocol; Phase C implementation pending
**Protocol identity:** `operatebench.arm.stateful_time_removed.v1`
**Date:** 21 August 2026
**Outcome contract:** [MEASUREMENT_RFC_V2.md](MEASUREMENT_RFC_V2.md)
**Identity contract:** [IDENTITY_REPLAY_RFC.md](IDENTITY_REPLAY_RFC.md)

This document defines a total deterministic transition relation, not an example of
one. It specifies the bundled treatment paired with `full_lifecycle`. Phase C MUST
implement this protocol and an independent verifier; current Engine 0.9 does not.

## 1. Interpretation

`stateful_time_removed` preserves one integrated operation, causal business order,
authority/evidence rules, state mutation, retrieval, actions, obligations,
reopen/finality, and the common terminal relation. It removes simulated elapsed
time, WAIT ownership, timers/reminders/deadlines/cadence, and model-visible
reference timestamps as one named bundle.

Accordingly Full versus No-Time supports only a **bundled lifecycle-protocol
differential**. It does not isolate temporal capability or any individual removed
mechanism.

## 2. Input and transformed plan

The transformer accepts only a valid, identity-verified Full `EpisodePlan`, its
operation/scenario/variant identities, and protocol identity v1. Invalid or
unknown fields fail before producing a plan.

```text
NoTimePlanV1 = {
  protocol: "operatebench.arm.stateful_time_removed.v1",
  origin_plan_digest,
  events: ordered list<NoTimeEventV1>,
  finality_cut_specs: ordered list<NoTimeFinalityCutSpecV1>,
  max_transitions: positive integer,
  max_invocations: same as Full,
  max_turns_per_invocation: same as Full,
  max_retrieval_batches_per_invocation: same as Full
}

NoTimeEventV1 = {
  event_id, event_type, actor_id, authored_sequence,
  payload, triggers_agent,
  source_class, cause | null,
  finality_relations: ordered map<provisional_close_common_cause_key,
    pre_finality | post_finality>
}

NoTimeFinalityCutSpecV1 = {
  provisional_close_common_cause_key,
  complete_common_point_key,
  reachable_pre_finality_closure: ReachablePreFinalityClosureV1
}

ReachablePreFinalityClosureV1 = {
  conditional_entries: ordered unique list<ReachableConditionalV1>,
  authored_descendant_event_keys: ordered unique list<Digest>,
  closure_digest_sha256: Digest
}

ReachableConditionalV1 = {
  conditional_semantic_key: Digest,
  cause_occurrences: ordered unique list<ReachableCauseOccurrenceV1>
}

ReachableCauseOccurrenceV1 = {
  semantic_cause_key: Digest,
  cause_kind: "action" | "event",
  required_disposition: "accepted" | "accepted_or_audit",
  predecessor_semantic_keys: ordered unique list<Digest>
}
```

`Digest` is exactly 64 lowercase hexadecimal characters. All four records are
closed and every listed field is required. “Ordered unique” means duplicate-free
and sorted by `(conditional_semantic_key)` for conditional entries,
`(semantic_cause_key, cause_kind, required_disposition)` for cause occurrences,
and raw digest bytes for predecessor/descendant keys. `closure_digest_sha256` is
`SHA256(CJ({"domain":"operatebench.notime.reachable_pre_finality_closure.v1",
"conditional_entries":conditional_entries,
"authored_descendant_event_keys":authored_descendant_event_keys}))`; it excludes
itself. Empty arrays are valid only when the compiler proves the cut has no such
reachable member. Raw timestamps, scheduler keys, liveness verdicts, and compiler
provenance are not members. Runtime liveness is evaluated against this immutable
set; it never mutates the closure.

The plan contains cut specifications, not activation IDs. An accepted runtime
`COMPLETE` effect instantiates exactly one activation from the matching
specification. Define the arm-neutral identities in this order:

```text
complete_occurrence_key = SHA256(domain_complete_occurrence_v1, {
  common_point_key, proposal_key, accepted_effect_key,
  predecessor_common_transition_key,
  accepted_complete_ordinal_at_common_point
})

provisional_close_activation_id = SHA256(domain_finality_activation_v1, {
  complete_occurrence_key, accepted_effect_key,
  provisional_close_common_cause_key
})
```

`predecessor_common_transition_key` identifies the immediately preceding common
runtime transition and the ordinal counts accepted `COMPLETE` effects at that
common point in this run. Both are derived from the accepted common trajectory,
not an authored occurrence, clock, timestamp, timer identity, queue/sorted
position, compiler surrogate, or arm-specific ledger row. Consequently every
accepted `COMPLETE` effect creates one unique activation; reopening cancels that
exact activation, and a later accepted completion creates a new occurrence key
and activation ID even if its normalized proposal and effect content repeat.
Proposal and accepted-effect identities are computed before the occurrence and
activation identities and MUST exclude both of them; finality records may refer
downstream to the activation ID. This one-way dependency forbids an identity
cycle. Cut specifications and instantiated activation IDs are evaluator/runtime
control data, not event payloads, and are never model-visible.

`starts_at`, absolute `at`, `delay_minutes`, runtime timer due instants, and
`horizon_minutes` are not executable time in this arm. `max_transitions` is a
protocol safety bound derived solely as:

```text
1 + authored_event_count + conditional_event_count
  + declared_max_domain_generated_audit_events
  + max_invocations * (
      max_turns_per_invocation
      + max_retrieval_batches_per_invocation
      + 1
    )
```

The declared audit bound is the Operation Core scalar defined by
[`operatebench.domain-generated-audit-budget.v1`](DOMAIN_GENERATED_AUDIT_BUDGET_PROTOCOL.md):
one unit per distinct logical operation-authored audit-event intent successfully
committed through the future Core budget API. It is identical across arms and is
not inferred from a trajectory. Its term is disjoint from authored events,
conditionals, the leading finality transition, and invocation/scaffold terms.
Overflow of this protocol is a domain-audit integrity `ERROR`; exhaustion of the
separate `max_transitions` control yields `ERROR:TRANSITION_HORIZON`.

The extra invocation term accounts for every permitted retrieval response and
the one final call on which the shared invocation contract can report a call- or
retrieval-budget refusal. A transformer MUST also prove that removing timestamps
does not change the Full arm's common authored-event causal order. After removed
timer rows are excluded,
the Full order and the No-Time order MUST be the same linearization of the
declared causal graph, including every finality cut defined in section 5.3. A plan
that fails either proof is ineligible for this protocol rather than silently
reordered.

Absolute authored events enter the initial queue. Conditional events enter the
conditional registry. Full timers are represented only by normalized
`timer_obligation_removed` audit facts and never enter the executable queue. A
future authorized implementation MUST create one removal fact for each corresponding
Full logical timer intent without recharging it, preserve the same aggregate logical
count, and exclude timer lifecycle rows, `finality_cut`, closure, and post-terminal
delivery from that count. This protocol remains specified-only and unauthorized;
the audit-budget contract does not activate it.

## 3. State and queue

A machine state is:

```text
Σ = (
  business_state,
  queue,
  conditionals,
  active_finality_activation | null,
  used_event_ids,
  delivered_semantic_keys,
  active_invocation | null,
  retrieval_session,
  trajectory,
  common_evidence,
  replay_final: bool,
  halt: RUNNING | DEADLOCK | TRANSITION_HORIZON | TERMINAL | ERROR,
  transition_count
)
```

The queue is a set keyed by `event_id`. Outside an active provisional-close
activation it is read through this total ordering key:

```text
(authored_sequence, event_id)
```

Whether an authored event was absolute or conditional is retained in origin-plan
provenance and `cause`, but is not a queue priority. Giving all absolute events a
higher priority than all conditionals would move a conditional effect behind
unrelated later authored events and violate the common causal-order contract.

During provisional-close activation `a`, queue selection uses:

```text
(finality_epoch_rank(a), authored_sequence, event_id)
```

where `pre_finality` has rank 0 and `post_finality` has rank 2. Rank 1 is reserved
for the explicit `finality_cut(a)` protocol transition and is never assigned to
an authored event or an ordinary domain audit. The transition is selected after
all materialized queued `pre_finality` entries have been resolved, no invocation
is active, and no retained conditional has a still-live cause in the statically
classified reachable-pre-cut closure. It is selected before every
`post_finality` entry. It is therefore a common causal boundary, not a
low-priority audit event.

There is no timestamp in either No-Time key. A conditional keeps its original
authored sequence. Ordinary domain audits use the sequence of the accepted cause
and a namespaced, deterministically derived event ID; they do not receive a rank
that can move them across a finality cut. Duplicate order keys are impossible
because `event_id` is unique; any duplicate identity is a plan/runtime integrity
error.

Events made ready by one transition are inserted as a batch before the next pop,
then globally sorted by the same key. This defines simultaneous conditional
fan-out without scheduler insertion-order dependence.

## 4. Normalized identity and correspondence

No correspondence key may depend on a Full timestamp, timer ID, compiler-produced
surrogate, or position after sorting.

```text
semantic_event_key = SHA256(domain_event_key_v1, {
  event_type, actor_authority_class,
  semantic_payload_without_allowlisted_time_fields,
  cause_key | null,
  authored_occurrence_index_within_equal_semantics
})

common_point_key = SHA256(domain_point_key_v1, {
  semantic_scenario_digest, point_id, causal_predecessor_point_ids
})

proposal_key = SHA256(domain_proposal_key_v1, {
  common_point_key, outcome_kind,
  normalized_action_or_terminal_payload, evidence_refs
})

accepted_effect_key = SHA256(domain_effect_key_v1, {
  proposal_key, normalized_bindings, pre_effect_common_state_digest
})
```

`authored_occurrence_index_within_equal_semantics` is assigned in original authored
sequence order before transformation and is independently recomputable from the
Full card. Equal semantic events with different authored occurrence indices are
not duplicate identities.

The verifier MUST establish a one-to-one correspondence for common events, points,
proposals, accepted effects, accepted-`COMPLETE` occurrence keys and activation
IDs, obligations, evidence/authority bindings, terminal facts, observations,
requests, and opportunities. Missing, extra, or ambiguous correspondence is
`ERROR`, remains in the denominator, and cannot be normalized away.

## 5. Total transition relation

At every step exactly the first matching row applies. Every input shape not named
below maps to `ERROR:UNKNOWN_TRANSITION_INPUT`.

| Priority | Preconditions | Input | Next state and emitted record |
|---:|---|---|---|
| 1 | malformed state, queue, identity, payload, or counter | any | halt `ERROR:INTEGRITY`; no business mutation |
| 2 | `transition_count >= max_transitions` and not replay-final | any | halt `TRANSITION_HORIZON`; emit integrity finding |
| 3 | `replay_final=true` and queue non-empty | next event | consume event; do not call reducer or agent; emit `post_terminal` audit; remain replay-final |
| 4 | `replay_final=true` and queue empty | none | halt `TERMINAL` |
| 5 | active invocation returns `RetrieveBatch` | retrieval response | validate and atomically serve or refuse the batch under the unchanged retrieval catalogue, authority, evidence, per-batch, and per-invocation contracts; record the same normalized refusal/served class as Full; remain in the invocation |
| 6 | active invocation has a proposed `ACT` | valid outcome | validate reads, evidence, authority, payload, and domain guard; on refusal record proposal/refusal and continue invocation; on acceptance commit once, emit effect/common evidence, invalidate reads, activate matching conditionals, then continue invocation |
| 7 | active invocation has `WAIT` | contract-valid WAIT | record `wait_removed`; end invocation; immediately continue to queue selection; wake list/fallback do not filter or delay the queue |
| 8 | active invocation has `ASK` | contract-valid ASK | apply its message action under unchanged authority/evidence rules; if refused continue invocation; if accepted commit and activate action conditionals, record nested wait as `wait_removed`, end invocation, continue queue |
| 9 | active invocation has `ESCALATE` | contract-valid ESCALATE | apply checkpoint action under unchanged guard; if refused continue invocation; if accepted commit, activate action conditionals, record its deadline as removed, end invocation, continue queue |
| 10 | active invocation has `COMPLETE` | valid outcome | apply unchanged common terminal guard; refusal continues invocation; acceptance commits the common effect, derives its runtime occurrence and unique activation ID in the order specified in section 2, enters provisional/final state, instantiates the matching cut specification, and activates action conditionals; end invocation |
| 11 | active invocation exceeds a shared call/business-turn/retrieval bound or returns malformed/unknown input | input | emit the same common refusal/integrity class as Full after treatment normalization; continue or end the invocation exactly where the shared runtime contract says so |
| 12 | no active invocation; finality activation active; a materialized queued `pre_finality` event exists | pop first such event by the active queue key | consume identity; validate and deliver normally; emit its common disposition; accepted/audit delivery may activate conditionals and a triggering event may open one invocation; a reopening effect cancels this exact activation before any later row can apply |
| 13 | no active invocation; finality activation active; no queued `pre_finality` event exists; a retained conditional is classified blocking but no live pre-cut producer for it exists | none | halt `ERROR:FINALITY_REACHABILITY_MISMATCH`; compiler/runtime reachability evidence is inconsistent, so do not guess finality or deadlock |
| 14 | no active invocation; finality activation active; no queued `pre_finality` event and no blocking retained conditional remains | `finality_cut(activation_id)` | atomically close each nonblocking retained conditional governed by this activation as audit-only `UNTRIGGERED_AT_FINALITY`, then apply the unchanged finality predicate; emit common finality evidence and set replay-final, with no event delivery, reducer mutation, or subject invocation; a stale/cancelled/wrong activation is an integrity error |
| 15 | no active invocation; queue non-empty | pop first event under the applicable queue key | consume identity; validate payload/authority; deliver to domain unless replay-final; emit accepted/audit/rejected disposition; activate event conditionals only on accepted or audit delivery; if `triggers_agent` and operation not replay-final, open one invocation; otherwise continue queue |
| 16 | no active invocation; no finality activation; queue empty; unresolved conditionals exist | none | halt `DEADLOCK`; list unresolved normalized trigger keys, without future payload disclosure |
| 17 | no active invocation; no finality activation; queue empty; no unresolved conditionals; not replay-final | none | halt `DEADLOCK` |

A transition increments `transition_count` once even if it emits several adjacent
ledger rows. Retrieval requests within an invocation preserve the Full action and
batch contracts and each agent retrieval response consumes one transition. Reads
are served from the current business state; their `as_of` is transformed as
specified below and cannot reveal Full time.

### 5.1 Conditional semantics

A conditional trigger is satisfied only by an **accepted** action effect or an
accepted/audit event disposition with matching normalized kind/type/correlation.
A refused prerequisite does not satisfy it. The conditional is retained, not
consumed or rescheduled. If the prerequisite identity is one-shot and no other
valid matching cause can occur, the retained conditional contributes to deadlock
when no finality activation is active; active-finality closure follows the
precedence and blocker rule below.

When a cause is accepted, every matching conditional is removed atomically from
the registry and queued as one batch. A conditional is consumed exactly once. A
rejected conditional event is itself consumed; its descendants remain retained
because rejection is not an accepted event cause. There is no implicit retry or
reschedule.

For each cut specification and retained conditional `c`, compilation computes
`ReachPre(c)` as the closed set of statically reachable matching causes and their
authored predecessor chains whose exact Full order keys are before that cut. The
classification includes action and event causes, direct and indirect conditional
descendants, and disposition requirements. It is not the broader set of causes
that merely share a trigger label. A runtime cause in `ReachPre(c)` is **live**
only while that exact occurrence or a remaining predecessor chain can still
produce an accepted/audit matching disposition before the cut under the Full
plan. Runtime disposition evidence monotonically removes a possibility when its
one-shot prerequisite is rejected or consumed without satisfaction, is cancelled,
or becomes impossible. A cause classified after the cut is never live for this
predicate.

During active finality, a retained conditional blocks the cut if and only if at
least one independently classifiable cause in `ReachPre(c)` is live. Therefore a
retained conditional whose one-shot cause was rejected, consumed, or made
impossible, and one whose every remaining cause is `post_finality`, does not
block. Once all materialized queued pre-cut events and the active invocation are
also absent, row 14 closes each such conditional as
`UNTRIGGERED_AT_FINALITY`. That record names the conditional semantic key,
activation ID, and closed reason class, but carries no undisclosed cause payload;
it does not deliver an event, satisfy descendants, mutate business state, or open
an invocation. This matches Full when no conditional event was scheduled before
finalization. Outside an active finality activation there is no such closure:
retained conditionals continue to contribute to row 16 `DEADLOCK`.

If pre-cut reachability, liveness, or a required disposition depends on an
unbounded/dynamic source or model choice and cannot be classified independently
from the Full plan, the compiler rejects the scenario before execution with
`ERROR:FINALITY_CUT_UNCLASSIFIABLE`. It MUST NOT retain the conditional as a
conservative blocker or close it optimistically. At runtime, claiming a live
pre-cut cause after every producer path has disappeared is the reachability
evidence inconsistency handled by row 13.

### 5.2 Duplicate semantics

- Reuse of an `event_id`, including after cancellation/consumption, is
  `ERROR:DUPLICATE_EVENT_ID` before a second mutation.
- Two distinct IDs with equal semantic content are two authored occurrences. Both
  are delivered in authored occurrence order and the unchanged domain idempotency
  guard decides whether the second is accepted or rejected.
- Repeating an identical proposal is recorded and passed to unchanged domain
  duplicate guards; the protocol never deduplicates agent behavior.
- Duplicate common correspondence keys or a many-to-one pairing are integrity
  errors, not exclusions.

### 5.3 Terminal and post-terminal behavior

`COMPLETE` uses unchanged domain guards. Provisional close is a business state, not
replay-final. An authored persistence event may therefore reopen it in causal
order. Full's provisional-close timer is removed, but its position in Full's total
scheduler order is retained as a timestamp-free causal cut.

At compile time, for **every** possible provisional-close activation, the compiler
expands all relevant directly and indirectly authored events and the corresponding
Full finalization transition. It evaluates the exact Full scheduler comparator
`(simulated_instant, authored_sequence, event_id)` and records only the normalized
relation:

```text
event Full order key < finalization Full order key  => pre_finality
finalization transition itself                      => finality_cut
event Full order key > finalization Full order key  => post_finality
```

Raw Full instants, derived due instants, and the finalization timer identity remain
compiler provenance and MUST NOT enter `NoTimePlanV1`, the No-Time runtime, model
observations, or arm-neutral evidence. A same-instant tie is not collapsed: Full's
authored-sequence field and then event-ID field decide the side exactly as they do
in `Event.order_key`. An authored event cannot compare equal to the finalization
transition because Full event identity is unique; a duplicate complete comparator
key is `ERROR:FINALITY_CUT_AMBIGUOUS` before execution.

The partition and `reachable_pre_finality_closure` include pending absolute
events, registered conditional events, every statically reachable matching cause
and predecessor chain, and every statically reachable authored descendant,
whether or not it triggers the subject. On provisional-close activation, all
materialized `pre_finality` events retain normal delivery, mutation, conditional
activation, and invocation opportunities. Thus a
pre-cut persistence event can reopen the operation; reopening cancels that
activation's cut, and a later provisional close receives a distinct activation
ID bound to its applicable independently compiled cut specification. The cut
waits only for materialized queued pre-cut events, an active invocation, and the
blocking retained conditionals defined in section 5.1. When none remains,
`finality_cut` first records nonblocking retained conditionals as
`UNTRIGGERED_AT_FINALITY`, then applies the unchanged domain predicate without
elapsed time or a subject invocation. Every `post_finality` event is then consumed
as audit-only `post_terminal`, with no reducer mutation or invocation; a
conditional whose only cause is such an event has already closed untriggered and
is not activated by that audit.

Compilation MUST reject with `ERROR:FINALITY_CUT_UNCLASSIFIABLE` any plan whose
partition depends on runtime/model choice, an unbounded or dynamic schedule, or a
cause not statically orderable against the Full finalization transition. Ambiguous
correspondence, a cycle, incompatible partitions across reachable activations, or
an authored dependency that crosses the cut in a way Full cannot deliver is
likewise ineligible before execution. The compiler may not repair such a plan by
placing all authored events before finality.

The independent verifier MUST recompute every partition and the complete
reachable-pre-cut cause/predecessor closure directly from the Full plan, Full
runtime disposition evidence, and Full scheduler comparator, without importing
compiler helpers, normalized relations, compiler constants, or the compiler's
liveness result. It independently derives which retained conditionals block and
which close `UNTRIGGERED_AT_FINALITY`, and requires finality-versus-deadlock,
conditional disposition, pre-cut disposition, invocation-opportunity, exact
activation cancellation/recreation, and post-cut audit-only parity. Any mismatch
is `ERROR`, remains in the assigned denominator, and prohibits the comparison. In
the current V1 Full
fixture this proof necessarily classifies `v1_e18` as `post_finality`; No-Time must
therefore record it audit-only and must not invoke the subject. This fixture is a
witness of the generic rule, not an exception to it.

After replay-final, remaining events are consumed only as `post_terminal` audit
records; they never mutate state or invoke the subject. This preserves late-event
evidence and the common finality predicate.

## 6. Deadline-bearing outcomes and observations

The outcome vocabulary and JSON/tool schemas remain present in both arms. To avoid
interface guessing while removing duration semantics:

- input fields named `deadline_after_minutes` and `fallback_after_minutes` remain
  positive integers at the model boundary;
- in No-Time, any valid positive value canonicalizes to the literal `1` before
  proposal correspondence and is recorded as `deadline_removed`/`wait_removed`;
- no timer or deadline event is scheduled from that value;
- policy duration fields exposed to the model canonicalize to `1` under the same
  named transform; and
- the verifier compares Full and No-Time observations/requests after removing only
  these allowlisted transformed values and reports their raw difference.

`WAIT` means “yield the current invocation and let the next causally ordered event
run”; it never chooses a future event. `ASK` preserves the ask/message effect then
yields. `ESCALATE` preserves the checkpoint effect then yields. `ACT` and
`COMPLETE` retain their domain semantics.

## 7. Closed raw transform allowlist v1

No raw difference outside this table is permitted. “Remove” means absent from the
No-Time executable projection but retained in origin-plan provenance where stated.

| Path/category | Operation | Required normalized treatment |
|---|---|---|
| `EpisodePlan.starts_at`, `PlannedEvent.at` | remove | excluded from common projection and correspondence |
| `PlannedEvent.delay_minutes` | replace with causal edge | compare cause key and authored sequence |
| `EpisodePlan.horizon_minutes` | replace with `max_transitions` formula | both are safety bounds, not common outcomes |
| runtime timer due instants, timer queue entries, reminder/deadline/follow-up events | remove | emit closed `*_removed` audit category; never model-visible |
| event `sequence` used with time | retain as `authored_sequence` | same authored order |
| `AgentObservation.now`, retrieval `as_of`, record-version time input | replace with literal `notime:v1` anchor and No-Time record-version domain | verifier strips anchor and compares substantive records |
| policy fields ending in `_minutes` that encode cadence/deadline/finality | canonicalize positive value to `1` | raw differences reported; semantics removed |
| WAIT wake/fallback scheduling | remove scheduling; retain declared vocabulary for interface parity | normalized as invocation yield |
| ASK nested WAIT scheduling | same | ask effect retained; yield normalized |
| ESCALATE deadline scheduling | remove scheduling | checkpoint effect retained |
| provisional-finalization timer | replace its Full comparator position with a compiler-derived `finality_cut` relation; discard raw time | explicit common causal transition with the same reopen/finality predicate |
| ledger rows solely for timer schedule/cancel/fire, wait declaration, wake reason, elapsed minutes | map to closed removal audit codes or omit | absent from arm-neutral outcome; raw counts reported |
| outcome elapsed/simulated duration | set `null` in No-Time mechanism report | never part of common primary endpoint status |

The allowlist does **not** permit changes to business payloads, actors/authority,
policy thresholds other than duration semantics, action/evidence schemas,
retrieval content, accepted-effect guards, obligation/finality predicates,
terminal legitimacy, model/scaffold/history policy, provider settings, or budgets.
Unknown transformed fields fail closed.

## 8. No-future-information rule

The subject sees the same static event vocabulary, never the pending queue,
conditional registry, future payloads, future order, remaining event count,
Full timestamps, correspondence keys, expected terminal, scenario ID, or oracle.
The transform MUST be completed before subject execution, but model observations
are projected only from current state and the event just delivered. Deadlock
diagnostics, cut specifications, reachable-pre-cut closures, activation IDs,
`UNTRIGGERED_AT_FINALITY` reasons, and origin-plan provenance are
evaluator/operator-only.

## 9. Arm-neutral common outcome relation

The independent verifier produces `ArmNeutralEvidenceProjectionV1` exactly as
specified in [Measurement RFC v2](MEASUREMENT_RFC_V2.md#21-mathematical-contract).
It must show one-to-one common business events, proposals, accepted effects,
accepted-`COMPLETE` occurrence/activation identities, authority/evidence bindings,
obligation transitions, terminal facts, and integrity status. Full and No-Time
reference outcomes MUST satisfy identical named `CommonOutcomeV1` predicates.
Mechanism-only rows in the allowlist never become a common predicate.

## 10. Compiler/verifier ownership and independence gate

Phase C assigns the transformer/compiler exclusively to
`src/operatebench/domains/lettings/maintenance/stateful_time_removed.py` and the
independent matcher exclusively to
`src/operatebench/domains/lettings/maintenance/stateful_time_removed_verifier.py`.
The verifier module MUST NOT import the compiler module, any private compiler
helper, compiler fixture generator, generated expected projection, normalized
relation table, correspondence-key builder, finality closure builder, liveness
helper, or compiler-derived constant. The compiler likewise MUST NOT import the
verifier. Stable public card schemas and Core wire dataclasses may be common
inputs; each side must separately implement derived timestamps, correspondence
keys, partitions, closures, liveness, and arm-neutral projections.

`tests/test_stateful_time_removed_import_graph.py` MUST parse the transitive import
graph and fail either direction or any shared third module that exports a derived
algorithm/constant to both. `tests/test_stateful_time_removed_independence.py` MUST
mutate each side independently and prove the hand-authored matrix catches drift;
ordinary equality against compiler output is not an oracle.

Before a research release, `.github/CODEOWNERS` MUST contain separate entries for
the two modules using distinct role aliases (for example `@notime-compiler-owners`
and `@notime-verifier-owners`) and branch protection MUST require an approval from
the opposite independent-review role for changes to either surface. Because the
current private repository has not established those aliases, this RFC does not
invent people or require two unavailable humans at merge time. Until governance
assigns distinct roles and records the independent review, the release gate stays
closed; one current catch-all owner is not evidence of independence.

## 11. Independently hand-authored fixture matrix

Before transformer implementation merges, Phase C MUST add fixtures authored
without importing compiler helpers or generated expectations. Every cell contains
Full input, hand-written No-Time expected queue/trajectory/common projection,
expected disposition/halt, and exact finding code where applicable.

| Fixture family | Required cells |
|---|---|
| queue base | empty queue deadlock; one absolute; same-source sequence tie; event-ID tie breaker |
| simultaneous fan-out | two and three conditionals from one accepted action; from one accepted event; insertion permutations |
| collision | authored absolute plus conditional plus removed timer at the Full instant; Full sequence/event-ID tie orders on both sides of the cut |
| rejected prerequisite × finality | rejected action and rejected event, each crossed with no active finality (retained-to-deadlock) and active finality; active-finality cells cover one-shot rejected/consumed/impossible cause closing `UNTRIGGERED_AT_FINALITY`, an alternate live pre-cut cause remaining blocking then triggering, and no delivery/mutation/invocation on closure |
| post-cut conditional causes | only post-cut cause closes `UNTRIGGERED_AT_FINALITY` at the cut; mixed rejected pre-cut plus post-cut causes closes; mixed live pre-cut plus post-cut cause blocks only for the live pre-cut cause; later post-terminal audit does not activate the closed conditional |
| conditional descendants | accepted parent/accepted child; accepted parent/rejected child; child descendants retained |
| duplicates | duplicate ID before run; duplicate ID after consume/cancel; semantic duplicate distinct IDs accepted; domain-rejected duplicate |
| WAIT | wake-only; fallback-only; both; unknown wake refusal; repeated yields to deadlock |
| ASK/ESCALATE | accepted then yield; refused then continue; deadline canonicalization |
| COMPLETE/finality | premature refusal; accepted runtime COMPLETE effect creates exactly one activation ID; repeated equal-content completion after reopen creates a new ID; exact old activation cancellation; every provisional-close activation partitioned before/equal/after by the Full comparator; same-instant sequence and ID ties; pre-cut reopen preserved; explicit finality cut; post-cut late event audit-only/no invocation; V1 `v1_e18` post-cut witness; ambiguous, cyclic, dynamic/model-choice, and crossing plans refused before run; activation identity contains no authored occurrence, timestamp/timer, sorted position, compiler surrogate, or identity cycle |
| audit events | non-triggering accepted audit; domain audit ordering; audit conditional fan-out |
| horizon | exact last allowed transition; one-over bound; malformed/negative bound |
| correspondence | timestamp-independent match; duplicate key; missing/extra event; ambiguous occurrence; altered business payload |
| no leakage | observation/request excludes queue, future count/order/payload, scenario/expected terminal, and Full time |
| allowlist | one positive per allowed path; one targeted negative per forbidden category |
| common outcome | `SUCCESS` parity; one well-formed `FAILURE` per business conjunct; one integrity `ERROR` per malformed common category |

Expected matrices are reviewed separately from compiler and verifier. Any ambiguous
cell blocks C3. Mutation tests MUST demonstrate that moving any event across its
independently recomputed finality cut, treating finality as an ordinary domain
audit, invoking on a post-cut event, losing a pre-cut reopen, consuming a
conditional on rejection, blocking finality on a rejected/consumed/impossible
one-shot cause, failing to block on an alternate live pre-cut cause, activating a
conditional from its post-cut audit, replacing `UNTRIGGERED_AT_FINALITY` with an
event delivery/mutation/invocation, sharing compiler reachability or liveness
logic with the verifier, reusing an activation ID after reopen, cancelling the
wrong activation, introducing an activation/effect identity cycle, deduplicating
semantic events, mutating after final, leaking a future field, or allowing one
forbidden raw difference is killed.

## 12. Phase boundary

This RFC freezes Phase A semantics. Phase C evidence—not this prose—must demonstrate
totality, deterministic fixtures, independent matching, reference and alternate
solver parity, targeted negatives, zero-provider replay, and no future leakage.
Until then `stateful_time_removed` is specified-only and no research comparison or
provider run using it is authorized.
