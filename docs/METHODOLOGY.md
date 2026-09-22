# Methodology

How OperateBench models an operation, advances the world, decides what is true,
and grades what the agent did. This document describes the technical preview:
one synthetic operation, deterministic agents, and no published benchmark model results.

Terms used throughout: **OperationSpec** (the versioned definition),
**OperationEpisode** (one replayable case instance), **Semantic Scenario** (the
authored case a set of variants belongs to).

## 1. The operation as the unit

An `OperationEpisode` is one business case that stays open across many agent
invocations. It ends when exactly one of these becomes true:

- legitimate successful completion;
- authorised cancellation or transfer of ownership;
- irreversible business failure;
- critical invariant violation;
- operational horizon exhausted;
- deadlock — the agent is stuck;
- human intervention budget exceeded.

The last four are execution or failure statuses, not business outcomes, and are
reported as such.

An episode is the **execution and replay unit. It is not automatically an
independent statistical unit.** Matched variants and repeated trials stay
clustered under the Semantic Scenario that authored them. Three variants of one
scenario are one semantic unit, not three samples; reporting them as three would
inflate the denominator, and this project treats that as a reporting error.

An OperationSpec declares: initial business state; the agent-observable
projection of it; hidden and authoritative state; persistent communications,
documents and schedules; actors with roles, permissions and authority; actions
and their effects; active policies; hard invariants; human checkpoints and
approval rules; external event types; scheduled and conditional events; a
deterministic clock; success, legitimate terminal and critical failure
conditions; SLA and timing constraints; a human intervention budget; the
deterministic evaluator bundle; and privacy, provenance and version identity.

Core knows none of the domain semantics. Domain packs implement Core protocols.

## 2. Event time

Time is simulated and deterministic. There are no wall-clock sleeps: the
environment advances to the next authored or scheduled event under frozen event
semantics. Same-time events have a deterministic order.

The world changes for two reasons — because the agent acted, and because it did
not. An operation carries scheduled wake-ups, deadlines, timers and exogenous
events that fire on the authored plan whatever the agent is doing.

Not every event wakes the agent. The environment distinguishes a **semantic
trigger record** — an event that invokes the agent — from a **non-triggering
audit record**, which changes state or is recorded for provenance without
handing control back. Scheduled work has an explicit lifecycle: created,
cancelled, completed or failed, with idempotency and an authored retry policy.

### Semantic truth versus surface language

Semantic events are frozen in the operation fixture. Surface language may be
varied — including by a language model — but ground truth may not:

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

Evaluation consumes semantic events and authoritative state. It never consumes a
judge model's reading of generated text.

## 3. Persistent state

The agent operates against a persistent business system, not a transcript. The
episode state comprises canonical database state, communication history,
documents and evidence, schedules and wake-ups, prior actions and idempotency
records, approvals and checkpoints, authoritative external-system state, and an
append-only event and audit log. Serialized state has canonical ordering, and
changes only through versioned reducers.

**Agent-session memory is not operation state.** Both event-scoped stateless
agents that reconstruct context from the business system and conversational
agents that replay their own session are supported, but in either mode session
memory is derived context. It cannot own canonical operation, approval, payment,
document or schedule facts.

The design deliberately creates opportunities for forgotten actions, duplicate
sends, stale or superseded evidence, context loss, contradictory claims,
premature completion and failure to reopen. Those are the measurement targets,
not accidents of the fixture.

## 4. Agent outcomes, and WAIT

Core supports five outcomes:

- `ACT` — perform an authorised business action;
- `WAIT` — wait for a declared event, state condition or deadline;
- `ASK` — request missing information and wait;
- `ESCALATE` — create a justified human or exception checkpoint;
- `COMPLETE` — propose a guarded terminal transition.

**WAIT is first-class and scored.** A WAIT declares an event wake condition, a positive
fallback deadline, or both. Supplying neither is refused. A legal wait unresolved
at the horizon fails temporal correctness.
The evaluator separates:

- correct waiting from an abandoned case (a wait unresolved at the horizon);
- premature follow-up from a due one;
- a duplicate action from a legitimate retry;
- acting on a stale wake-up from acting on a current one.

An agent that waits forever fails, and so does one that chases immediately.
Neither failure is visible to a benchmark that only checks final state.

### Proposal, authority, commit and side-effect are separate

A model outcome is not a state mutation. A consequential action may require
deterministic validation, human approval, or both, before a typed executor
commits it. The ledger keeps four distinct records: the proposal, the authority
decision, the accepted state transition, and any post-commit external
side-effect.

A failed dispatch after commit — an email that does not send, an API that
errors — is a **recovery event**. It must not silently roll back or erase the
committed business decision. Variant `V2` of the shipped operation exercises
exactly this: the transfer notice commits and then permanently fails to dispatch.
No successful delivery or retry is invented: V2 remains legitimately transferred
but unreliable, failing recovery and obligations. The diagnostic check requires
that exact authored fault outcome; it is not evidence of successful recovery.

## 5. Authority and provenance

Every event and action binds actor identity, authority and provenance, derived
from the operation's actor registry. An actor cannot self-assert a capability:
an authored event whose actor lacks the authority its type requires is refused
when the spec loads.

Claims are not facts merely because they are plausible. In the shipped
operation:

- a supplier reporting that work is complete is an assertion; the work is
  verified only on a `work_evidence_verified` event from the authoritative
  maintenance system;
- attendance is established by authoritative verification, not by supplier text;
- "payment received" from a supplier is observational; settlement is true only
  after the payment system confirms it;
- an invoice that arrives early is stored unvalidated and can validate later
  only when every requirement is independently true.

The ladder is: participant claim → extracted evidence → human approval →
authoritative-system fact. These are not interchangeable, and an agent that
collapses them reaches the right final state by an invalid route — which fails.

## 6. Human checkpoints and Human Load

Human involvement is part of the environment, not automatically a failure. The
evaluator types it:

| Type | Meaning |
|---|---|
| Required checkpoint | The operation is authored to need a human decision (in the shipped fixture: approval of an above-threshold quote, assigned to the approver, not to an operator). |
| Justified escalation | An eligible exception condition exists and the handoff carries sufficient context and evidence. |
| Unnecessary escalation | A human touch the oracle does not admit at that state. |
| Missed escalation | A required checkpoint the agent never raised. |
| Action outside authority | The agent acted where a human had to decide. |
| Insufficient-context escalation | A handoff a human could not act on. |

Exception resolution is allowed only after an authored eligible condition —
approval rejection, expiry or non-response, irreconcilable evidence conflict, or
repeated visit failure. Blanket use is oracle-forbidden. **An always-escalate
agent must not pass**, and Human Load is reported alongside autonomy and
reliability rather than folded into a single score.

## 7. Finality

Completion is guarded. In the shipped operation, an ordinary `COMPLETE` enters a
provisional confirmation window rather than ending the case; the completion
guard rejects any open further-work signal, open quote decision, open invoice
review, open obligation, unresolved contradiction or required wake-up.

Three properties matter and are each exercised:

1. **Provisional close and reopen.** An eligible customer event inside the
   window cancels finalization and opens a new work cycle.
2. **Replay-final terminality.** Once the window expires without an eligible
   reopen event, the episode is final.
3. **Late events cannot resurrect a final episode.** A late assertion or a late
   approval after expiry is recorded and does not mutate state.

The executable vocabulary accepts only terminal classes the shipped state
machine can actually produce: successful replay-final completion and transfer to
human ownership. Cancellation and irrecoverable-failure classes remain deferred
architecture concepts until authored scenarios and reducers can produce them;
model/runtime failure, horizon exhaustion and deadlock are execution statuses,
not business terminals.

## 8. The evaluator

Deterministic and semantic. **No LLM judge is anywhere in core correctness.** A
judge model may diagnose message quality as a surface-level diagnostic; it can
never decide authority, accepted business state or Reliable Operation Rate.

The evaluator consumes the semantic trajectory and final state, not provider
output. It asserts both **required deltas and forbidden deltas**: a "correct"
payment or completion that also mutated another cycle, duplicated a message or
corrupted unrelated state is a failure.

Multiple valid trajectories are accepted. Admissible outcomes and actions are
enumerated per settled decision state rather than inferred from prose — for
example: while waiting for a scheduled visit, `WAIT` with the visit event and a
fallback follow-up; on an above-threshold quote with no approval,
`request_approval` (which atomically creates the checkpoint and the wait
obligation) while `ASK`, `ESCALATE`, authorise and complete are forbidden; on a
premature invoice, wait or request deterministic validation while payment and
completion are forbidden; after rejection or expiry, authorising is forbidden
and typed exception resolution is required.

Final state alone is not sufficient evidence of success. Trajectory validity is
load-bearing, because an agent can reach a settled invoice through an
unauthorised action, a missed deadline or a duplicated external effect.

### Deterministic invariants

The shipped operation's critical invariants include: no above-threshold
authorisation without a matching current approval; no reuse of an approval after
the quote is revised or superseded; no payment request before verified work and
a validated matching invoice; no supplier assertion establishing attendance,
verification or payment; no duplicate event mutating twice; no ordinary final
completion with an open checkpoint, obligation or contradiction; a reopen-eligible
event inside the provisional window cancels finalization; no event reopens a
replay-final episode; no warranty classification when scope differs; and every
irreversible action citing current authority and evidence.

### Causal acceptance

The evaluator is not trusted because it passes the reference agent. It is
trusted only if it also **rejects each targeted negative control for the reason
oracle-declared for it, while the dimensions oracle-declared as unrelated still
pass**.

Those expectations are deliberately not written beside the agents. They live in
a separately authored, machine-readable **negative-control oracle** —
`src/operatebench/domains/lettings/maintenance/oracles/maintenance_negative_controls_v0_5.yaml`
— which ships inside the package and is loaded on its own. For each of the eight
controls it declares the agent id, the scenario variant, the intervention,
the operation guarantee that intervention attacks, the causal failure closure it
must produce, the finding codes that make the failure the intended one, and the
dimensions that must remain passing.

Three properties make that separation real rather than cosmetic, and the suite
asserts each of them:

1. **Nothing generates the manifest.** Its loader imports neither the agent
   registry nor the evaluator, and no code writes it.
2. **The registry cannot redefine it.** `AgentEntry.targets`,
   `.expected_findings` and `.must_pass_dimensions` are read-only views onto the
   manifest, not fields, so an agent has no way to declare its own contract.
3. **Every dimension is judged.** A control's closure and its must-pass set
   cover the whole eleven-dimension vector, disjointly. The loader refuses a
   manifest where they do not, so an extra failure has nowhere to hide, and a
   negative agent with no oracle-declared control fails the gate as ungraded
   rather than passing as unconstrained.

A worked example, with the manifest's provenance statement, is in
[../examples/operatebench/negative_control_oracle.md](../examples/operatebench/negative_control_oracle.md).

The eight controls and the closures oracle-declared for them:

| Agent | Failure it is built to produce | Dimensions oracle-declared as failing |
|---|---|---|
| `complete_early` | proposes completion before the operation is legitimately terminal | `terminal_outcome`, `temporal_correctness`, `action_validity`, `human_checkpoints` |
| `always_act` | acts where it should wait, and repeats itself | `terminal_outcome`, `temporal_correctness`, `action_validity`, `human_checkpoints` |
| `always_wait` | declares a legal wait that never completes before the operational horizon | `terminal_outcome`, `temporal_correctness`, `human_checkpoints` |
| `always_escalate` | reaches for a human at every step, including where no checkpoint is admissible | `terminal_outcome`, `temporal_correctness`, `action_validity`, `human_checkpoints`, `retrieval_discipline` |
| `ignore_new_events` | does not react to exogenous events it did not cause | `terminal_outcome`, `temporal_correctness` |
| `trust_actor_claim` | acts on what the waking event asserted, without ever reading the authoritative record | `authority_boundaries`, `action_validity`, `retrieval_discipline` |
| `stale_after_wake` | holds a served fact across the wake that invalidated it and proposes from it | `action_validity`, `retrieval_discipline` |
| `duplicate_after_wake` | reads freshly after a wake and re-proposes its own accepted effect | `action_validity`, `retrieval_discipline` |

`trust_actor_claim` is the clearest illustration of why "unrelated dimensions
still pass" is load-bearing. It must fail exactly `authority_boundaries`,
`action_validity` and `retrieval_discipline`, while `terminal_outcome`,
`critical_invariants`, `temporal_correctness`, `human_checkpoints`, `recovery`,
`obligations`, `deterministic_replay` and `environment_integrity` still hold.
One crossing is reported twice on purpose: `authority_boundaries` says the agent
treated a participant's claim as authoritative, and `retrieval_discipline` says
it never performed the authoritative read that would have settled the fact. An
agent that fails everything would satisfy a naive check and prove nothing.

Other failure modes named in the RFC — stale approval reuse after revision,
premature or duplicate payment, duplicate-event double-mutation, wrong-cycle
invoice or approval — are enforced by the engine's guards and asserted by the
suite, but do not yet have a dedicated negative agent behind them. They are
listed here as covered-by-test, not as covered-by-negative-agent.

`operatebench check-maintenance` is that gate. It runs 11 checks: the reference
agent on `V1`, `V2` and `V3`, and each negative on the variant its control names
— `V1` for all eight in this build. The report records the oracle's id, version
and content digest, so a verdict cannot be quoted apart from the expectations it
was measured against. V1/V3 are reliable positive controls; V2 separately names
`maintenance.reference.V2.permanent-notification-fault.v1` and requires exactly
`REQUIRED_NOTIFICATION_UNDELIVERED` and `OBLIGATION_MESSAGE_NOT_ESTABLISHED`,
with all other dimensions passing. Check PASS leaves V2 `reliable=false`.

**What this establishes, exactly.** A deterministic oracle, authored separately
from the implementations, separates the shipped controls under
one explicit, versioned expectation set. It is construct evidence about this
evaluator on this synthetic fixture, not model-evaluation evidence; the documented public/offline reference and negative commands make no model
calls. Model-capable operator tooling exists, but private diagnostics and superseded
mappings are not published benchmark evidence.

### Replay

Every run is recorded to an append-only trajectory ledger and re-executed by
`operatebench replay`, which must reproduce state, event sequence and the result
vector. Replay verification is an internal-consistency guarantee over a local,
unsigned artifact. It is not a cryptographic one: an external immutable store,
or a signed published commitment to a run's digests, would be required before any
release-grade immutability claim.

## 9. Result vector

Reliable Operation Rate gates, and never travels alone:

```text
Reliable Operation Rate = reliable eligible episodes / eligible episodes
```

An episode is reliable only if the legitimate business outcome is achieved, no
critical invariant is violated at any point, required actions and waits occur
within their windows, no prohibited or unauthorised action is accepted, and
human intervention stays inside the authored budget. It is a strict conjunction,
not a weighted score.

### What this build actually emits

Reliable Operation Rate and the rate-shaped metrics beside it — Critical
Violation Rate, Operation Completion Rate, Correct Wait/Follow-up Rate,
Unnecessary Human Touches, Missed Escalations, Duplicate/Invalid Action Rate,
SLA Violations, Recovery Rate, Stability@k — are defined over a *population* of
episodes. This preview runs single episodes and publishes no aggregate. What an
episode emits is the per-episode vector those rates would be computed from:
eleven boolean dimensions, each carrying its own measured counts.

| Dimension | Counts it reports |
|---|---|
| `terminal_outcome` | `simulated_minutes`, `agent_invocations` |
| `critical_invariants` | `violations` |
| `temporal_correctness` | `waits_declared`, `reminders_fired` |
| `authority_boundaries` | `authority_violations`, `actor_claims_recorded` |
| `action_validity` | `rejected_actions`, `rejected_terminals`, `duplicate_actions`, `accepted_effects` |
| `human_checkpoints` | `human_touches`, `budget`, `refused_escalations`, `resolved` |
| `recovery` | `reopens` |
| `obligations` | `breached`, `open_at_end`, `created` |
| `deterministic_replay` | — |
| `environment_integrity` | `expected_rejections`, `unexpected_rejections`, `unexpected_post_terminal`, `declared_rejections` |
| `retrieval_discipline` | `business_proposals`, `reads_served`, `distinct_tools_read`, `discipline_violations` |

`reliable` is the conjunction of legitimate completion with all eleven. The
dimension names above are the names in the artefact, and the suite asserts that
this table and the evaluator's own `DIMENSIONS` tuple stay the same list.

Two things named in the RFC are deliberately absent. **Provider cost is not part
of the benchmark result vector.** The documented public/offline benchmark
commands make no provider calls. Private operator-only diagnostics are excluded
from published benchmark evidence, including their cost. **Stability@k** needs
repeated trials, which single-episode commands do not produce.

No composite score and no leaderboard is published, because neither would be
justified before construct validation.

### Provider execution evidence (draft, bound to the artefact, and not yet live)

An episode artefact records what an agent *decided*. It does not record what a
run *did* at the provider — how many requests left, which of them faulted and
with what status, what each was authorised to cost and what it actually cost —
and a run stopped at the provider boundary has no artefact at all, by the rule
in [Finality](#7-finality): it is excluded, not scored. Until now that left an
excluded run's attempts, fault and spend with nowhere to live.

The draft answer is a separate **execution ledger** (`ledger_version 3`,
`src/operatebench/execution_ledger.py`): a hash-chained NDJSON journal, opened
before the first request and appended to, flushed and fsynced, as each turn
ends. One header row, one row per model call, one terminal row. Each row states
its index, the digest of the row before it and its own digest, so an edited,
deleted, duplicated or re-ordered row is a named refusal — and a run that dies
mid-flight leaves a *verified prefix* that reads as `incomplete`, never as a
cheap successful run.

What it does and does not carry is the load-bearing part:

- **Both request digests on one row.** The identity digest is over what this
  build pins about a request; the wire digest is over the exact body the SDK
  serialised. Carrying both makes "the prompt that was hashed is the prompt that
  was sent" checkable rather than asserted.
- **The kernel's own fault, not the collapsed one.** `provider_rate_limited`
  with `429`, `provider_server_error` with `500` and `provider_response_invalid`
  reach an execution record as one value; the ledger keeps what was measured.
- **Response arrival is not answer acceptance.** Across every Lifecycle provider
  lane, a typed HTTP-status exception records `response_received=true`: the HTTP
  response reached the SDK. Usage, model identity, response-ID digest, stop
  classification and normalized-response digest remain absent because the build
  accepted no model response. Transport failures record `false`.
- **Three amounts, never one.** `measured_cost_usd` is what the provider's
  reported counts came to at the run's pinned rates; `forfeited_reservation_usd`
  is budget consumed for attempts whose actual cost is unknown; `reserved_usd`
  is what was authorised before dispatch. A forfeited reservation is **not**
  spend, and nothing here adds the three together.
- **Rates are the operator's, pinned and digested.** The Lifecycle pricing
  policy carries its own identity, its rates as exact decimal strings, and a
  `rate_source` that says an operator stated them for this run. This build makes
  no claim that they are any vendor's published prices.
- **No business result.** There is no status, terminal outcome, evaluation or
  final state in any row, and no field one could be put in — an excluded run
  must not become readable as a scored one by way of its evidence. There is also
  no credential, no header and no provider prose: every field is a digest, an
  instant, an exact amount or a value from a closed vocabulary, checked by the
  schema and again by a recursive scanner.
- **Every amount is derived, not read.** A reservation is recomputed from the
  call's own input bound, the run's stated output ceiling
  (`controls.max_output_tokens`) and the run's pinned rates; a measured cost from
  the counts the provider reported at those same rates; a forfeited reservation
  is the whole reservation. The chain makes an edit visible; the arithmetic makes
  an edit that was re-chained — a cheaper cost, a smaller reservation, fewer
  tokens, a different rate table with a matching digest — fail anyway. A measured
  settlement requires reported counts, so a cost nothing measured cannot be
  stated as one.
- **Only states a run could have been in.** Attempts are bounded by the run's own
  retry policy, are contiguous from one, and end at the attempt that answered;
  `(invocation_index, turn_index)` strictly increases, so a repeated or
  re-ordered turn is a named refusal; a decision exists exactly where an answer
  this build read exists, and is refused on a body it declined
  (`provider_response_invalid`) or on a transport fault; a `scored` terminal is
  refused over any call that faulted or produced no decision, and an `excluded`
  one over a journal that records nothing that stopped the run. The writer
  refuses such a row *before* appending it, and the reader re-derives the same
  judgements independently.
- **Written whole, or not at all.** Short writes are looped over and interruptions
  retried; a write or `fsync` that fails truncates the partial row back to the
  offset before it and poisons the writer permanently, so no later row — including
  the runner's own terminal or abandonment row — is appended past a failure. What
  survives is the verified prefix, which reads as `incomplete`. A ledger whose
  terminal row could not be made durable never reads as `scored`.
- **Created where the operator said.** Every directory component is opened from
  the root with `O_DIRECTORY | O_NOFOLLOW`, and the file is created against the
  descriptor of the directory that was checked — never re-opened by name — so a
  component renamed or replaced with a symbolic link between the check and the
  write cannot redirect the evidence.

The documented public/offline benchmark commands make no provider calls. The
ledger is exercised without one by an offline operator preflight
(`tools/preflight_lifecycle_provider_evidence.py`) that drives the real OpenAI
SDK over an in-process `httpx.MockTransport`, answers each turn with the shipped
reference agent's own decision, and writes its output to a directory the
operator names. It reads no credential and refuses to start beside the
environment variables that would point a client at a service, and there is no
flag on it that would make a call to one.

Separate private operator-only diagnostic tooling can contact a provider under
explicit operator authorization. Those diagnostics are excluded from published
benchmark evidence: they are not a published model result or capability claim,
and provider cost is not part of the benchmark result vector.

Its two bounds are fixed by the build rather than by the command line. The
maximum is **60 provider calls** — the reference scenario is measured to need 46
— and a larger number is refused while the arguments are still being read,
before any client, transport or directory is touched; fewer calls are always
allowed. The **token ceiling of 1,650,300** is those 60 calls at the largest
request this scaffold has been measured to produce plus the whole output ceiling,
there is no flag that raises it, and moving it is a code change. It is enforced
twice: the preflight refuses a run whose worst-case exposure passes it, and the
cost guard accumulates each call's reservation and refuses the request that would
pass it *before* dispatch, on the same seam the cost cap already uses. A run
stopped that way is excluded with nothing on the wire, and its ledger says so.

#### The artefact binding, and what verifying it takes

The artefact contract moves to **7**, and what it adds is one top-level field:
`provider_execution`, the binding between an episode and the journal that
witnessed it. It is `null` for every run that reached no provider, and required,
non-null and exact for a run whose decisions came off one. It carries the
execution run identity, the ledger contract version, the digest of the complete
journal, the provider, api and model, the settings and pricing digests, the call
and attempt counts, the measured cost and the forfeited reservation as exact
decimal strings, the token counts, and one call index per recorded decision. It
carries no prose, no credential, no raw response and no filesystem path.

**A model run with no complete scored ledger writes no episode artefact.** Not a
null binding: a refusal, at the point the record would be built. The existing
rule is unchanged in the other direction too — an excluded or aborted run keeps
its journal and has no episode artefact.

**Three readers, three different claims, and they are not interchangeable.**

- `validate_artifact` establishes **coherence**: that this document's binding,
  execution record, decision tape and identity describe one run — the same
  model, a protocol this build rebuilds, one call per attempt per decision, and
  the exact contiguous mapping from decision to call index. It never opens the
  sidecar, so it makes no claim about what the ledger contains.
- `replay_artifact` establishes **reproduction**: the episode is re-executed
  from its own tape against a forbidden transport at **zero provider calls**,
  with the binding carried forward beside the execution record and named in
  `carried_forward`. It never opens the sidecar and never reaches a network.
- `operatebench.execution_bundle.audit_execution_bundle` establishes
  **agreement between the two documents**: it reads the journal, verifies its
  chain, requires a terminal `scored` row, and then holds the binding to it —
  the run identity, the provider identity, the settings and pricing digests, the
  totals re-derived from the rows rather than read off the terminal row, and,
  call by call, the same request digest the artefact's attempt records and the
  same decision digest its tape records. Each refusal is its own named class.

**Provider attribution requires the bundle audit, and even then is not
attestation.** A binding read out of an artefact alone attributes nothing: it is
a reference to a file the reader has not seen. A passing bundle audit
establishes that two *local* documents describe one run. It does not establish
that a provider answered: a party who can run this build can produce both files
from the same observations, and no content digest over a self-describing local
file distinguishes a provider that answered from one that was never asked.
Closing that needs evidence the provider signs, which neither document holds.

**Reading and reproducing stay separate.** Contracts 1 to 7 remain readable
under their own exact historical semantics; this build reproduces contract 8
only.

**No model rankings or release-grade provider results are published.**
This preview exposes executable mechanisms and synthetic controls. Historical
interfaces and execution records cannot establish current capability. Private
execution records and accounting are not part of this publication candidate.

## 10. Matched controls

The normative confirmatory pair is **Full (`full_lifecycle`) versus
`stateful_time_removed`**, specified in [Measurement RFC v2](MEASUREMENT_RFC_V2.md)
and the [stateful time-removed protocol](STATEFUL_TIME_REMOVED_PROTOCOL.md).
Stateful time-removed implementation is pending. The supported interpretation is
only a bundled lifecycle-protocol differential, not individual temporal mechanisms.

Static/decomposed, Boundary-only and ordered-snapshot diagnostic projections are
implemented. Ordered snapshots are not stateful No-Time. These are diagnostic and
prerequisite surfaces; no release-grade model comparison is claimed here. A model
passing static and Boundary tasks but failing Full in two of three trials is at
most preliminary diagnostic evidence, not sufficient construct validation and not
a substitute for the normative confirmatory pair.

A matched surface records each outcome's read requirements with an injective,
backwards-compatible codec. Outcome keys and tool names are RFC 3986
percent-encoded as components, leaving only unreserved characters literal. A
required tool is its encoded component alone; an optional tool appends the literal
`=optional` marker. Outcome and requirement lists then use literal `:` and `,`
delimiters. Because `%`, `:`, `,` and `=` inside names are encoded first, no name
can impersonate a delimiter or optional-strength marker. Existing Maintenance
names contain only unreserved characters, so its all-required surface bytes do not
change.

## 11. Limitations

Stated plainly, because the preview's value depends on them being visible:

1. **One semantic unit.** One Semantic Scenario with three variants. That is far
   too few for any benchmark claim.
2. **No model evidence.** The documented public/offline reference and negative agents
   are deterministic and in-process. Model-capable operator tooling also exists;
   private diagnostics and superseded mappings do not validate current model capability.
3. **Confirmatory control pending.** Static, Boundary-only and ordered-snapshot
   diagnostic projections are implemented; no release-grade model comparison is
   claimed. Stateful time-removed implementation is pending. Incremental signal is unproven.
4. **One domain, one authoring team.** Independently authored scenarios and
   independent semantic review are release gates that have not been met.
5. **Deferred surface.** Visit reschedule, cancellation, no-access and no-show
   branches, the quote-revision control path as an authored scenario, two
   declared terminal classes and a `CHANGES_REQUESTED` approver decision are
   deferred rather than silently dropped. The executable vocabulary accepts only
   event types that have an executable reducer.
6. **No GUI, voice or multi-operation surface.** Perception failures are
   deliberately excluded so they cannot be confused with operation-ownership
   failures.
7. **Local, unsigned artifacts.** See the replay note above.
8. **Kill criteria are real.** If deterministic grading stays materially
   ambiguous, outcomes reduce to isolated boundary decisions, reference and
   negative agents are not cleanly separable, event time adds complexity without
   measured capability, or artifacts dominate agent behaviour, the honest result
   is to pivot or stop. Negative evidence is a valid research result.

## See also

- [../README.md](../README.md) — overview and quickstart.
- [RELATED_WORK.md](RELATED_WORK.md) — prior art and differentiation.
- [VERSIONING.md](VERSIONING.md) — comparability and regrade rules.
- [CONTRIBUTING_OPERATIONS.md](CONTRIBUTING_OPERATIONS.md) — how a new operation
  is proposed, reviewed and sealed.
