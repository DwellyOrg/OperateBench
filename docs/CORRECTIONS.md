# Private repository corrections

## OB-CORR-2026-09-17-COMMUNICATION-OBLIGATIONS

- **Affected:** engine 0.11.0 and earlier maintenance reminder/transfer grading.
- **Correction:** engine 0.12.0 binds reminder and transfer recipients and cycles,
  discharges reminders only on delivery, and independently reconstructs delivery.
  Durable human transfer is not undone by notification failure. V2 retains its
  permanent authored dispatch fault: reliable is false, recovery and obligations
  fail with exact notification findings, and the diagnostic check separately
  accepts that explicitly authored fault outcome. V1/V3 must remain reliable.
- **Control isolation:** the early claim stressor remains; an otherwise-valid
  payment control changes only authoritative evidence to an actor assertion. Its
  unchanged amount is 64000 minor units (GBP 640), not GBP 64000.
- **Identity:** Artifact 8 and ledger 3 shapes, fixture 0.6.0 and provider projection
  algorithms are unchanged. Changed domain guidance changes observation/request
  bytes and their digests under engine 0.12.0; it does not change provider mapping
  algorithms. Source-owner projection v4 is additive; v1-v3 and historical
  negative-oracle files remain frozen. Current operator AST changes include engine
  admission and the Grok 4.6 offline-only execution boundary, exact MockTransport
  admission, and separately identified illustrative fleet metadata. Exact
  affected-source identity is retained outside public Git in controlled records
  under this correction identifier; authorship provenance still requires owner
  confirmation.
- **Evidence boundary:** rerun affected experiments; do not regrade or relabel old
  results. No paid/provider calls or current-model validation is claimed. The
  historical read-byte baseline remains 295404; publishing recipient/cycle rules
  yields 368239 bytes in the V1 census (approximately 1.24656 times the baseline).
  The unchanged test-only overhead ceiling is 1.25;
  provider per-cell financial/token/call caps are not changed.


## OB-CORR-2026-09-12-BENCHMARK-VALIDITY

- **Date:** 2026-09-12
- **Class:** runtime and evaluator correction under the combined preview identity.
- **Affected:** Maintenance V1, V2 and V3, operation fixture 0.6.0, engines through
  0.10.0. The unchanged fixture digest is recorded in PUBLICATION_MANIFEST.json.
- **Defect:** instructions incorrectly described five decision tools despite six
  being available; malformed ACT feedback lacked actionable fixed guidance;
  completion messages could target someone other than the accepted reporting
  customer; grading did not independently bind recipient and acceptance
  causality to the completion obligation.
- **Correction:** engine 0.11.0 and successor provider request mappings, canonical
  state @2.0.0, get_case_record v2, and declaration profile @2.0.0. Runtime and
  evaluator bundle digests are not emitted by this preview; no digest is invented.
  Original fixtures, historical state validation and request identities remain.
  Exact affected-source provenance is held in the external controlled record
  under this correction identifier.
- **Current Sonnet scaffold envelope:** the model-visible request has a measured
  maximum native input bound of 25,884 tokens. With unchanged rates, 4,096 output
  tokens and 50 maximum calls, its exact worst case is USD 4.6364. The build
  retains a USD 4.68 cap (minimum sufficient cent cap: USD 4.64) and pins a
  1,499,000-token hard cap. Historical paid output, authorizations and raw
  histories are not rewritten by this recalculation.
- **Action:** rerun affected experiments, never regrade or pool with prior engines.
  Historical artifacts remain readable and exact-engine replay refusal remains.
- **Superseded published results:** none shipped in this publication candidate.
  Private diagnostics are not promoted to corrected benchmark results.
  Only offline synthetic controls were run for this correction.

See [versioning and compatibility](VERSIONING.md#engine-0110-benchmark-validity-correction).

## OB-CORR-2026-09-10-RETRIEVAL-TAPE

- **Date:** 2026-09-10
- **Class:** recorder implementation correction; execution/evaluation and reader
  contracts unchanged. See the detailed [version decision](VERSIONING.md#retrieval-tape-recorder-correction-2026-09-10).
- **Affected source:** the writer that sorted and deduplicated requests in
  `_tape_requests`, identified by correction `OB-CORR-2026-09-10-RETRIEVAL-TAPE`.
  Exact affected-source provenance is retained in the external controlled record
  under this correction identifier. Earlier writers using the same normalization
  are potentially affected, not exhaustively dated.
- **Affected identity:** engine `0.10.0`, Artifact `8`, model protocol `v4`,
  execution-ledger `3`; no separate runtime/evaluator contract digests are emitted
  by this preview. None of these versions moves for this writer-only fix.
- **Operation:** retrieval-bearing scenarios, including synthetic maintenance
  `lettings_maintenance_synthetic_v1@0.6.0`, `V1`–`V3`; reproduced with fresh `V1`
  fixtures. Operation content is untouched, digest
  `9e99b409b153e03aeb240c6402e14ff21c8bf0df3b9a1df2016c8fa64138e44d`.
- **Defect:** sorting/deduplicating retrieval decisions loses request order and
  multiplicity. On unknown-tool refusal, replay then records a different original
  `request_count`. The recorder now serializes every request as supplied. Engine
  serving canonicalization, refusal counts and equality gates are untouched.
- **Action:** fresh runs for affected experiments after review; no regrade,
  historical rewrite, or retroactive success. Lost information is not recoverable
  from the old tape. Source provenance distinguishes writer builds; changed tape
  bytes propagate to decision and dependent evidence digests.
- **Superseded published results:** none identified. No historical evidence is
  rewritten, replayed, rescored, or published by this correction. Offline synthetic
  regression is not capability evidence or wire attribution of any past response.

## OB-CORR-2026-09-09-INVOCATION-ADMISSION

- **Date:** 2026-09-09
- **Class:** runtime correction under the combined preview engine identity
- **Affected identity:** engine `0.9.0`, including builds with unversioned
  invocation admission changes. Exact source provenance must distinguish these
  builds; the version label alone is insufficient.
  These builds emit no separate runtime/evaluator versions or contract digests.
- **Successor identity:** engine `0.10.0`.
- **Operation:** `lettings_maintenance_synthetic_v1@0.6.0`; all scenarios/variants
  using Core invocation admission, including shipped `V1`–`V3`.
- **Operation digest:**
  `9e99b409b153e03aeb240c6402e14ff21c8bf0df3b9a1df2016c8fa64138e44d`
- **Superseded published results:** none identified; historical records
  and frozen fixtures remain unchanged, not relabelled or promoted as new results.

### Defect and correction

PR78 enforces the declared invocation allowance before event and fallback wakes.
That changes which calls are admitted and can change termination and trajectory,
but the initial PR still emitted engine `0.9.0`. Engine `0.10.0` gives that runtime
its own identity. Current release/operator pins follow the successor; old
plans and stored evidence do not. The existing Full terminal suffix
rules continue unchanged. Artifact 8, Card 1, model protocol v4, operation 0.6.0,
ledger 3 and distribution 0.1.0 do not move: this correction adds no wire fields
and changes no artifact row shape or evaluator vocabulary.

### Compatibility and action

Reading and execution are separate. Historical Artifact 8 records and ledger
bundles still validate under their recorded contracts; replay rejects a different
engine identity before executing any trajectory. Reproduce old evidence only
with its original source/runtime, never by relabelling it as 0.10.0. The unversioned
PR78 head and pre-PR78 runtime share a defective identity, so source commit
provenance is needed to distinguish them; a version string alone cannot repair
that ambiguity. No historical evidence is rewritten by this correction.

Results across this boundary are not comparable and must not be pooled. Affected
experiments require fresh reruns, not regrading under the combined engine identity.
No provider call, official rerun or rescore is part of this correction.

## OB-CORR-2026-08-17-BOUNDARY-OWNERSHIP

- **Date:** 2026-08-17
- **Class:** evaluator-only correction under the intended split-version policy;
  combined preview engine identity correction in this build
- **Affected identity:** `engine_version 0.7.0` (the build did not emit separate
  runtime/evaluator versions or their digests)
- **Operation:** `lettings_maintenance_synthetic_v1@0.5.0`
- **Operation digest:**
  `ddf486c641f51901814c612fcf7551b5852dcf7f66454074c93db6b750e4d46c`
- **Scenario / variant:** `maintenance_recurring_leak_synthetic_v1`, `V1`
- **Semantic scenario digest:**
  `f8ce63aa5b2b4b82300e6d12cb373ca86cdfd7897dc6c1599a978027ed6da052`
- **Superseded published results:** none; the affected Alpha material was sealed
  private diagnostic material and was not a published benchmark result.

### Defect

After the subject began a Boundary target timestamp and exhausted its supplied
choices, the internal driver delegated another turn at that same timestamp to the
reference agent. `run_boundary` then searched the timestamp for an exact accepted
proposal without recording who supplied it. An invalid subject action could
therefore be followed by, and receive credit for, the reference agent's exact
action. The sealed Alpha diagnosis included **8/9 false negatives**; this ownership
defect makes its Boundary attribution non-comparable even where the stored
trajectory remains internally readable.

Engine 0.8.0 makes the target timestamp subject-owned once the subject starts.
When its decisions are exhausted, the driver emits a deterministic protocol-valid
non-crediting wait and permits reference driving only after simulated time moves
outside the target timestamp. The operation semantics, Maintenance `0.5.0`,
protocol v4, request mapping v7, artefact contract 8 and ledger contract 2 do not
move.

### Compatibility and action

The original sealed Alpha remains preserved and is not edited or relabelled.
Although this is evaluator-only in the intended identity model, the current build
has one combined `engine_version` and cannot name or publish a standalone regrade.
An official comparable result therefore requires a rerun under `engine_version
0.8.0`; no rerun is part of this correction.

The existing **45-point zero-provider replay is diagnostic only, not a replacement
result**. It does not supersede the original sealed Alpha and must not be presented
as an official comparable rerun.

## OB-CORR-2026-08-17-FULL-TERMINAL-SUFFIX

- **Date:** 2026-08-17
- **Class:** evaluator-only correction under the intended split-version policy;
  combined preview engine identity correction in this build
- **Affected identities:** `engine_version 0.7.0` and `engine_version 0.8.0`
  (neither build emitted separate runtime/evaluator versions or their digests)
- **Operation:** `lettings_maintenance_synthetic_v1@0.5.0`
- **Operation digest:**
  `ddf486c641f51901814c612fcf7551b5852dcf7f66454074c93db6b750e4d46c`
- **Scenario / variant:** `maintenance_recurring_leak_synthetic_v1`, `V1`
- **Semantic scenario digest:**
  `f8ce63aa5b2b4b82300e6d12cb373ca86cdfd7897dc6c1599a978027ed6da052`
- **Superseded published results:** none; the affected Alpha material remains
  private diagnostic material and was not an official benchmark result.

### Defect

The Full-arm mapper could infer `UNREACHED` from a missing boundary merely because
an earlier point was `NOT_ESTABLISHED`, without evidence that execution had
terminated before the missing boundary. This conflated a malformed or incomplete
row sequence with a genuine terminal suffix. It also allowed a missing current
boundary to hide a later boundary marker.

Engine 0.9.0 permits a missing boundary and every boundary after it to map
`UNREACHED` only when the missing positions form a true suffix and the evaluator
holds either an authenticated current-process Engine `EpisodeOutcome` or the same
authenticated outcome released by an exact successful replay, recording deadlock
or horizon termination. Raw or serialized rows, caller-constructed outcomes,
contradictory markers, mapping/execution errors, authentication or digest
mismatches, and a missing boundary followed by a later semantic boundary fail
closed as `ERROR` at the affected point and every remaining denominator point.
Only a trusted terminal true suffix remains `UNREACHED` through the remainder.
The process-local authentication is an integrity capability
against serialized or caller-built evidence, not a sandbox against arbitrary code
already executing in the evaluator process; Python private names are not access
control. No diagnostic outcome vectors are included in this preview.

The operation semantics, Maintenance `0.5.0`, protocol v4, request mapping v7,
artefact contract 8 and ledger contract 2 do not move. The semantic scenario,
model, profile, cost envelope and pricing policy are unchanged.

### Compatibility and action

Historical 0.7.0 and 0.8.0 records remain preserved and are not edited,
relabeled, or officially rescored. No private provider evidence is published and
no official rescore or rerun is part of this correction. A replay failure exposes
diagnostics but withholds the authenticated outcome, and even a successful replay
does not by itself authorize an official rescore. Because the current
artefact has only the combined `engine_version`, an official comparable result
requires a fresh rerun under `engine_version 0.9.0`.

## OB-CORR-2026-09-06-MAINTENANCE-POLICY-PROJECTION

- **Date:** 2026-09-06
- **Class:** operation identity correction; the authored semantics do not move
  and the engine does not move
- **Affected identity:** `operation_version 0.5.0` under `engine_version 0.9.0`
  (every build that shipped the eleven-key policy projection)
- **Operation:** `lettings_maintenance_synthetic_v1@0.5.0`
- **Operation digest:**
  `ddf486c641f51901814c612fcf7551b5852dcf7f66454074c93db6b750e4d46c`
- **Scenario / variant:** `maintenance_recurring_leak_synthetic_v1`, `V1`
- **Semantic scenario digest:**
  `f8ce63aa5b2b4b82300e6d12cb373ca86cdfd7897dc6c1599a978027ed6da052`
- **Replacement identity:** `lettings_maintenance_synthetic_v1@0.6.0`, operation
  digest `9e99b409b153e03aeb240c6402e14ff21c8bf0df3b9a1df2016c8fa64138e44d`,
  semantic scenario digest
  `593b82e0cd8ce9cb0b25bafefc5c2f61e9cf405b6f66b7e8ec8c5b15a40e8b89`
- **Superseded published results:** none; no benchmark model result was
  published under the affected identity. Private diagnostic execution history
  is excluded from `PUBLICATION_MANIFEST.json`; this correction provides no
  benchmark capability evidence or publication authority.

### Defect

`OperationSpec.semantic_payload()` enumerated the policy by hand and named
eleven of `PolicySpec`'s twelve fields. The omitted one was
`max_retrieval_batches_per_invocation`, which bounds how many retrieval batches
one invocation may be served: the loader accepted it, `_parse_policy` required
it, `MaintenanceOperation.episode_plan` passed it to the engine, and
`semantic_arms_v1` read it as the arm's retrieval budget — but the content
digest never saw it.

The identity was therefore not injective. Two specs differing only in that
value both hashed to
`ddf486c641f51901814c612fcf7551b5852dcf7f66454074c93db6b750e4d46c` while
executing differently: with the shipped value of `6` and with `7`, the same
scenario produced different trajectories under one published digest. Any
artefact bearing that digest states an operation identity that does not
determine the executable policy it ran under, so `0.5.0` could not support the
claim that two runs carrying it are the same case.

The projection was the sole outlier. `card_v2_declarations` already declared all
twelve fields as `operation.policy_contract.*`, the loader's accepted-key set
already listed all twelve, and the commerce pack already projected its policy by
construction. `PUBLICATION_MANIFEST.json`'s `digest_scope` note — "taken over the
parsed semantic content" — was, for this field, false; it is true again now.

The projection is now taken over one list, `POLICY_FIELDS`, which is also the
loader's accepted-key set and is proved equal to `PolicySpec` at import time. A
field added to the policy and left out of that list fails the import rather than
leaving the digest blind to it.

### Compatibility and action

The operation's authored semantics did not change: the policy value is `6`
before and after, no event, guard, terminal, actor or variant moved, and no
episode executes differently. What changed is what the identity *covers*.

`docs/VERSIONING.md` places `policy` inside `operation_spec_version`'s scope and
states, without exception, that "a version that did not move while its digest did
is a defect, and tooling should refuse the artifact rather than accept it". The
reflow carve-out at the end of that section excuses the opposite case — file
bytes moving while the digest stands still — and gives no cover here. The
operation therefore moves to `0.6.0`.

Engine `0.9.0`, protocol v4, request mapping v7, artefact contract 8 and ledger
contract 2 do **not** move. `semantic_payload()` is consumed only by
`compute_digest()` and `verify_identity()`; no runtime behaviour and no
evaluator predicate changed, and `engine_version` covers exactly those two
things. This correction is the mirror image of the two above it, which moved the
engine while freezing the operation.

Results recorded under `ddf486c6…d46c` are **invalid as evidence for the new
identity and must be re-run, not regraded**. A stored artefact bearing the old
digest is refused by `replay_artifact` against the corrected spec, which compares
the recorded and live operation digests before it compares anything else. The
sealed Engine 0.7 Alpha remains `UNINTERPRETABLE` and is not revived, rescored
or relabelled by this correction.

The B3.1 evidence freeze under `tests/fixtures/b3` was **regenerated**, not
edited. `tools/regenerate_b3_evidence.py` is new and is the command the freeze's
own docstring called an explicit maintenance operation without ever naming; it
rebuilds all four files from the corrected fixture with pinned identities and a
pinned clock, and `--check` proves the committed bundle is exactly what this
build produces. The regenerated bundle is byte-reproducible, which the previous
one was not: its call latencies and per-call timestamps were real measurements,
so it could not have been rebuilt to the same bytes by anyone. The bundle stays
an **execution-ledger 2** document — the suite asserts that version and the
filename carries it — although the build's writer has since moved to ledger 3
(`OB` change preserving Anthropic HTTP fault semantics). The generator pins the
contract it writes under to 2 for exactly that reason: a fault-free reference
run uses nothing ledger 3 added, the build still owns the complete ledger-2
vocabulary, and regenerating a historical oracle as a current document would be
the relabelling this policy forbids. The pre-correction
bundle remains recoverable from the pre-correction commit, which is where the old
digest stays reproducible — `spec_digest_sha256` is derived at load time and is
authored nowhere, so no checkout of the corrected code can recompute
`ddf486c6…d46c` from any input.

`tools/matched_arm_alpha.py` is re-pinned to the corrected identity, including
its plan digest and its own normalized source digest. A consequence worth stating
plainly: Alpha material held outside this repository under plan digest
`1c19af35af7834ab667f680f6f9811b39fd62b16d16c13793ee2f44086a6cc32` is refused by
that validator after the re-pin, consistent with the re-run rule above.

`PUBLIC_CANDIDATE_VERIFICATION.md` is **not** edited. It is a dated 16 August
2026 record whose engine row the suite pins to the historical `0.7.0`; its
operation and digest rows are that record's own contemporaneous values, and this
note supersedes them rather than rewriting them in place.

The negative-control oracle keeps `authored_against_operation_version: 0.5.0`.
It was authored against that operation and its expectations are unchanged by a
projection correction; nothing binds that field to the live spec, and the
matched-control manifests are deliberately stale in the same way.
