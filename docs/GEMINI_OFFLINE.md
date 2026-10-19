# Gemini native offline campaign

This candidate adds two independent series over the existing frozen 44-case
roster: `gemini-4-argon` and `gemini-3.8-flash`. The Core tool contracts,
synthetic cases, strict evaluators and historical campaign modes are unchanged.
Mock scores describe plumbing and evaluator execution, never model performance.

The versioned mapping is `three-flow-gemini-developer-native-low-v4`; the roster
is `gemini-standard-same44-low-v1`. Native Developer API REST uses the existing
`BudgetWire` HTTP boundary for exact request bytes, durable dispatch evidence,
response capture and one-use accounting. This avoids adding an SDK dependency
or a second serialization layer. The mapping uses native `generateContent`,
not an OpenAI-compatible endpoint. The user's example URL
`https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent`
is topology evidence only; it is never selected or rewritten into a model ID.

The sole Gemini model authority is `tools/three_flow_gemini.py`. Its immutable
profiles bind the exact stable model IDs, LOW thinking, a nominal physical output maximum
of 65,536, and a conservative complete-request-plus-output operational envelope
of 1,048,576. Flash's envelope is a local conservative intersection, not a claim
that Google documents a unified bound on its separate input/output limits. The complete serialized history and tool declarations count toward
that local bound. The versioned policy
`gemini-complete-input-context-intersection-v1` selects each request's output as
`min(65536, 1048576 - complete_input_bound)` and refuses dispatch when no positive
room remains. Input is conservatively counted with the nominal numeric ceiling
before its digits shrink. Request identity, wire bytes, reservation, response
decoding, settlement, turn evidence and replay all bind the actual ceiling. `parametersJsonSchema` carries all six frozen Core schemas
unchanged. Duplicate JSON keys are refused at every response depth; native
argument objects and JSON-looking string values are preserved. The existing
Core optional-null normalization and strict outcome parser remain authoritative.
Native model parts, including opaque thought signatures, are retained across
turns. The initial prompt is text; each subsequent native function response carries
the next exact scaffold prompt and actual public Core observation once. No
observations, model parts or signatures are dropped to make history fit.

Standard estimates verified October 14, 2026:

| Exact model | Input USD/M | Output USD/M including thinking | Valid through |
|---|---:|---:|---|
| gemini-4-argon | 2 | 10 | November 30, 2026 |
| gemini-3.8-flash | 0.75 | 3.75 | December 31, 2026 |

Thinking tokens are added to candidate output tokens. Reported implicit cached
input remains visible and is conservatively estimated at the full input rate.
A nonzero `toolUsePromptTokenCount` is retained separately in raw usage; for the
accepted four-addend total variant (`prompt + tool-use prompt + thoughts +
candidates`), prompt and tool-use prompt are settled together at the full input
rate, and checked together against the admitted input bound. An over-bound
aggregate excludes the execution with unknown liability rather than silently
omitting tool-use tokens. Absent tool-use usage is zero; a mismatched reported
total is rejected. The generateContent UsageMetadata prose lists prompt,
thoughts and candidates for `totalTokenCount` despite also defining the
separate tool-use field; the four-addend semantics are accepted here, not
claimed universal for real generateContent responses. A separately authorized
live canary must establish actual server behavior and whether tool-use pricing
needs a distinct category. Additional fees remain unknown, not zero; these
estimates are not invoice guarantees. No Batch, Flex, Priority, explicit cache,
Search or Maps configuration is emitted.

A fresh Gemini journal uses `three-flow-gemini-uncapped-v1` and a null monetary
cap. It preserves reservations, measured costs and unknown exposure and cannot
reopen a historical capped journal or consume Meta policy evidence. The offline
acceptance fixture's synthetic limit-policy evidence grants no launch authority.
Failures stop that transport without automatic retries. HTTP error evidence
uses the existing body-free projection; reflected response text is not retained.

Run offline acceptance in an environment with the repository dependencies:

```bash
PYTHONPATH=src python -m pytest tests/test_three_flow_gemini.py
```

For the complete campaign alone:

```bash
PYTHONPATH=src python -m pytest tests/test_three_flow_gemini.py -k all_88
```

The suite prohibits socket connections, injects an HTTP mock and checks all 88
unique assigned slots, the two distinct series IDs, 44 unique cases per model,
exact scored/excluded denominators, mock provenance and the null-cap scorecard.
Replay reconstructs native history and checks request identities, byte digests,
LOW/output settings, exact response model, usage, estimates and Core decisions.

The offline constructor and mock tests need no credentials. The live factory
can admit paid Gemini only through the explicit source-bound, owner-approved
one-shot admission claim and the Gemini null-cap limit policy. Calling
`GeminiCampaignTransport.live(admitted=True)` alone grants no launch authority.
No new spend or account quotas are authorized or inferred here, and no model
scores or results are published.

For live v4 settings, `source-bound-admission-reference-v1` retains the admission
digest, the exact admitted profile digest and the limit policy in the trial
binding and record. Replay checks the exact settings shape, profile digest and
policy structure, and equality against the independently retained trial binding.
Offline settings equality and native request bytes remain unchanged. These
hashes are recorded links, not cryptographic attestations of approval. Without
the original admission sidecar, replay cannot reconstruct that admission or
verify that its digest represents a valid owner-approved claim. A coherently
rewritten binding and record require an independently pinned closure digest or
source/custody evidence to detect. Source versioning, same-SHA quality gates,
account/key availability and any separately approved launch remain parent-owned.
DCO runs after the parent commit.

Primary documentation:

- [Argon model](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon)
- [Flash model](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Standard pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Native function declarations](https://ai.google.dev/api/generate-content#FunctionDeclaration)
- [Function calling and signatures](https://ai.google.dev/gemini-api/docs/function-calling)

## Frozen context regression

The two frozen `SC_reopen` regressions previously failed before request 40:
985,399 complete input bound plus nominal 65,536 exceeded 1,048,576 by 2,359.
The parent-selected physical context intersection reserves and submits 63,177
for that historical v2 request, preserving every history part. The v3 visible
action-schema projection (also used by v4) gives 985,445 input and 63,131
output for the frozen request; the sum remains exactly 1,048,576. Both original
regression cases remain, with exact wire and replay assertions. Boundary tests exercise
unchanged output, positive room, no-room refusal, reduced-cap usage and tampered
replay through the native REST mock and budget guard. Missing or over-bound
usage retains unknown liability; the Gemini USD cap stays null.

Set `PRIVATE_SCRATCH` to the owner-approved private scratch directory, then run
without credentials or network:

```bash
evidence=$(mktemp -d "${PRIVATE_SCRATCH:?}/gemini44-reproduction.XXXXXX")
TMPDIR="$evidence" PYTHONPATH=src python -m pytest tests/test_three_flow_gemini.py \
  --basetemp="$evidence/pytest"
```

The two parametrized `all_88` tests create separate fresh campaign roots and
journals for the exact models, each with 44 assignments. Each writes
`offline-counts.json` with assigned/scored/excluded/aborted counts. These are
reference mock plumbing outcomes and never Gemini model-performance scores.
The previous interrupted 75 closures remain diagnostic evidence and are never
resumed or relabeled as acceptance.

## Scored model behavior (v4)

Mapping `three-flow-gemini-developer-native-low-v4` preserves strict native
envelope, exact model identity, usage categories, part types and function-call
signatures. Well-formed STOP responses with zero or multiple calls, and
MAX_TOKENS responses, now retain their actual call cardinality and stop reason.
Validated input includes prompt plus tool-use prompt tokens; output includes
candidate plus thinking tokens. Cache tokens remain a prompt subset. Usage
settles before the shared ModelAgent classifies model behavior. No call or
Core decision is invented from prose or truncated output. Such responses add
no native function-response history. If the engine's ordinary turn policy
asks again, that is a genuine new request using the current observation; it
does not replay or continue the malformed native output. The one-RPC end-to-end
regressions use an explicitly synthetic one-turn plan, while a separate test
retains the real fixture's ordinary turn limit.

The roster and live profile consumers import the adapter mapping constant.
Replay requires the exact current profile and mapping; v2/v3 evidence must be
replayed with its original source tree, never relabeled or rescored under v4.
The frozen v3 regression verifies historical parser behavior and rejection of
its old mapping by the current validator. It does not prove refusal of a
coherently rewritten record without an independently pinned source or digest.
The frozen v2 source regression remains pinned to its original Git object.
The prior v3 fixture correction changes only stale expected context numbers
and mutation accounting, and is retained in the tests.
