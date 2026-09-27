# Three-flow evaluation controller

The repository tooling runs the three Maintenance scenarios, sixteen Commerce
profiles and twenty-five Compliance profiles through their real Engine domains
and original evaluators. This is a separate, nonofficial evidence contract. It
does not change Artifact 8, development records, the public pack CLI or business
grading. The selected Maintenance fixture models verified notification recovery;
its reference agent reaches reliable closure for V1, V2 and V3. The historical
permanent-fault fixture retains its separate operation identity and grading.

`python -m tools.three_flow_campaign` prints a non-authorizing preflight. Every
model remains unapproved. It cannot create owner authority or dispatch a paid
campaign. Fresh approval must independently bind source, account, settings,
endpoint limits and all applicable prices. DeepSeek V4.1 remains unpriced.

`python -m tools.three_flow_campaign offline --output /absolute/fresh/directory`
executes all 44 configurations through the actual OpenAI SDK and an in-process
reference-driven HTTP mock. `--providers openai anthropic mistral river` assigns
all 616 roster cells: HTTP providers exercise their SDKs; River uses its actual
optional SDK, pinned tokenizer assets, native renderer/parser and protobuf mock
transport in a qualified Python environment with explicit `--river-assets`.
Missing River dependencies or assets fail closed. Neither mode reports mock
decisions as LLM performance. The command forbids network connections and uses literal mock
credentials. It never loads account credentials or creates approval documents.

The controller assigns unique trial slots and uses bounded independent HTTP and
River worker lanes (see the consuming-entrypoint resource policy below).
One durable USD 1000 journal covers every worker. Synthetic offline prices test
accounting mechanics and are not current-price admission. The controller retains
pending requests as unknown exposure on restart and never releases unknown
liability by timeout. `--resume` reads the same journal, reuses completed closures,
and marks interrupted allocated slots aborted without resubmitting them. The
Python API supports separately identified corrections of excluded or aborted
originals; scored model mistakes cannot be rerun through that path.

Each trial captures provider settings, SDK version, source/spec/profile binding,
actual wire bodies, request identities, decisions, events and original evaluation.
Records and incremental partial captures have owner-only permissions. Headers and
credentials are not captured. Scored, excluded and aborted closures stay distinct.
The JSON/CSV/Markdown reports retain causal accepted event receipts, dimensions,
completion, usage, latency and separate estimated/unknown costs. Missing outcomes
remain null; there is no weighted ability score.

Replay needs the independently retained Trial binding and closure digest. It
reexecutes the original domain and evaluator, consumes the entire decision tape
and compares full state/events/evaluation with zero provider calls. Digests are
integrity checks against separately retained anchors, not signatures or proof of
provider authorship. An owner who replaces both evidence and every trusted anchor
is outside this trusted-controller contract.

HTTP transports use explicit injected wire capabilities, one SDK attempt per
turn, no artificial episode wall deadline or call cap, and candidate endpoint
output profiles. Fable uses automatic tool choice and adaptive thinking; hidden
thinking remains in captures and output usage. Mistral subtracts a conservative
input bound from its candidate context allowance. These candidate settings and
limits still require independent admission. River offline integration uses the
optional SDK and explicit pinned assets.
Synthetic tests verify transport mechanics, not provider capability or a
production memory envelope. No real-provider results are claimed.

## Consuming entrypoint

`python tools/three_flow_live.py preflight` is the separate, inert live-path
preflight. It checks clean source before importing repository runtime modules,
then prints the exact roster, ordered 616 slots, settings registry, interpreter,
dependency versions, lock digest and runtime digest. It never inspects a
credential path, consumes authority, creates output, or reserves money. `--help`
needs no SDK imports. Preflight is not approval and cannot produce approval.

The external owner may later invoke:

```sh
python tools/three_flow_live.py run \
  --expected-source REVIEWED_FULL_COMMIT \
  --admission /absolute/private/series/admission.json
```

No admission document or default credential location is supplied. The closed
schema is defined by `tools/three_flow_admission.py:validate` and
`validate_profile`. An admission must bind the exact runtime/preflight registry,
canonical external campaign and custody directories, one series ID, a unique
attempt ID, the ordered selected initial slots, global cap `1000`, resource
policy, provider endpoints, asset catalog and actual asset hashes. Each selected
model needs positive finite decimal input/output rates per million tokens,
provider input/context/output bounds and an explicit network timeout. The default
`conservative-all-tier-envelope-v1` policy requires provider-origin rate/bounds
evidence digests and every named billing/access guarantee. The
settings registry binds the source implementation; arbitrary request overrides
are not accepted. All-tier conservative rates must cover cache writes/reads,
long-context/tier/geography modifiers and all hidden generated reasoning. Missing
coverage is not zero. Public price proposals cannot admit a campaign, including
the presently unpriced DeepSeek V4.1 model.

The supported billing policy reserves the entire admitted provider input maximum
and request output limit, then settles reported total input and generated output
at conservative envelope rates. Anthropic cache creation/read tokens join ordinary
input in this successor projection; raw provider usage remains captured. This
can overestimate cost and is not invoice settlement. Unknown usage keeps the
full reservation. The admitted guarantees must exclude additional unbounded fees.
The global cap may stop the campaign before completion; it is not a spend target.

### Explicit conditional pricing alternative

`owner-assumed-token-envelope-v1` is an opt-in external admission policy, not an
approval generator or a fallback for missing prices. Its closed `owner_pricing`
record requires explicit `owner_accepted=true`, USD per 1,000,000 tokens, an
unknown-category rate of `10`, and an inventory for `input`, `cache_read`,
`cache_write`, `output` and `reasoning` (separately billed reasoning tokens).
Each category lists all applicable `verified_usd_per_million` rates and explicitly
marks `assume_unknown`. Known rates, including rates above the assumption, remain
in the inventory; a known free category is represented by `"0"`, not by absence.
Rates must be effective total per-token rates, including any additive premiums.
The input/output envelopes must equal the maximum applicable verified or assumed
rate across their categories. Aggregate usage is charged once at that envelope:
cache or reasoning subsets are not added again to already-inclusive counts.

The record binds `source_sha`, the exact model's `registry_entry_sha256`,
`owner_evidence_sha256` and `verified_rate_evidence_sha256`. Its `profile_sha256`
is the canonical admission digest of the profile excluding `owner_pricing` and
`rate_evidence_sha256`; the profile's `rate_evidence_sha256` in turn binds the
entire owner record. Digests use `three_flow_admission.digest`. Evidence review
must establish that the inventory preserves all applicable known rates; hashes
provide integrity, not signatures or proof that the supplied material is accurate.

Only the two tariff-coverage guarantees may be conditional. A coverage guarantee
must stay **false** when its envelope uses unknown-category assumptions; the
record instead states accepted owner pricing coverage. Unknown prices do not
resolve context/output bounds, account access, complete billable usage, hidden
reasoning within output usage and limits, or absence of other fees. Those retain
their original requirements unless the separate explicit diagnostic policy below
is admitted; price consent alone does not waive them.

Provider settings retain the explicit assumption record and set
`model_contract_verified=false`; scorecards label the cost as an estimate under
owner assumptions, **not a guaranteed actual invoice**. The existing journal's
pricing-policy digest binds the whole admitted profile and its envelope rates.
Neither this policy nor a new rate record resets the campaign, journal, cap,
retry or correction scope. Historical and default verified-only policies are
unchanged. No owner record, account evidence or executable admission is supplied
by this repository.

### Optional reported-usage diagnostic accounting

An owner-priced profile may additionally contain the closed `accounting_policy`
record: `mode="reported-usage-diagnostic-v1"`, `owner_accepted=true`, and
`owner_evidence_sha256` (SHA-256 of explicit owner risk acceptance). This is
**estimated-usage-not-invoice**: the shared USD 1000 ceiling covers known reported
usage estimates plus pending and unresolved request reservations at the admitted
rates, not an actual-invoice guarantee. The owner accepts possible incomplete
usage, hidden billing, additional fees and invoice divergence. Unknown fees are
**unknown-not-included**, never a zero-fee assertion.

Only this additional consent permits honest boolean false values for
`account_access_verified`, `all_billable_input_in_usage`,
`hidden_reasoning_in_output_usage_and_limit` and `no_other_fees`. Preserve true
values supported by evidence, including documented OpenAI/Anthropic reasoning
coverage; do not force all guarantees false or true. First-live unknown access
may fail safely as an excluded provider request, not proof of access. Price
coverage and published-context policy remain separate. Without this accounting
policy the previous strict requirements are unchanged.

Rebind `owner_pricing.profile_sha256` over the complete profile except
`owner_pricing` and `rate_evidence_sha256`, then `rate_evidence_sha256` over the
owner record. The existing owner source/registry bindings must match the final
source SHA and registry; runtime and complete admission digests must also be
regenerated. Any changed consent, guarantee, timeout or bound invalidates the
old binding. This repository supplies no owner consent or consuming authority.

Reported usage settlement, unknown-reservation retention on missing/malformed
usage, one shared journal, no automatic resubmission and no reset are unchanged.
Settings and all scorecard formats retain the policy/provenance and estimate
label for scored and excluded requests; `model_contract_verified` stays false.
Genuine provider model results remain live-eligible by origin, while injected
SDK mocks remain mock-only; unverified billing guarantees do not decide origin.

The existing documented native per-RPC operational timeout is 120 seconds
(`THREE_FLOW_RIVER.md`); an operator can explicitly select 120 seconds for the
external profiles and rebind them. This is not an episode/call/output cap or a
new hardcoded policy default. Historical profiles still require a positive reviewed timeout.
The explicit successor below instead requires null.

### Explicit optional limits for a new series

`three-flow-optional-limits-v1` is a reusable opt-in configuration of the same
consuming entrypoint, not a new launcher, default, approval or authorization to
repeat indefinitely. Historical admission records without this policy still
require exactly `cap_usd="1000"` and a positive timeout. Their journal genesis,
reducer, settlement and replay interpretation are unchanged.

For an explicitly authorized **new** series, set top-level `cap_usd=null` and add
a closed `limit_policy` object with these exact fields:

- `mode="three-flow-optional-limits-v1"` and `owner_accepted=true`;
- `owner_evidence_sha256`: hash of the explicit owner decision;
- `source_sha`: the reviewed source commit (must match admission);
- `historical_liability_manifest_sha256`: SHA-256 of the retained external
  historical-liability manifest. This is mandatory, not an assertion of zero.

Each selected profile additionally sets
`execution_policy="three-flow-optional-limits-v1"` and
`network_timeout_seconds=null`, and retains its explicit published-context
`bounds_policy`. Rebind owner pricing/profile/rate evidence and runtime exactly
as described above. Pricing assumptions and reported-usage uncertainty are
separate consents; optional execution does not invent any true guarantee.

This removes the monetary ceiling and network-generation deadlines. No artificial
call, cumulative token or total-duration ceiling is introduced: the actual model
agent already uses `max_transport_calls=None`. Nullable execution controls and
owner/history provenance are bound into provider/trial identity, records,
assignment/consumption receipts, journals and reports. Replay checks the retained
independently bound provider settings and per-request outputs without dispatch.

Every request still has a **finite integer** output bound: operational context
minus the complete input bound, intersected with any *known* dedicated output
maximum and the wire integer range. Unknown dedicated maxima remain null; a
context-derived output bound is not a fabricated provider maximum. Native input
is tokenizer-counted; HTTP uses the existing conservative serialized-payload
projection including SDK-added defaults. No monetary division is performed for
null remaining allowance. Known, pending and unknown estimates remain nonzero
when applicable, and unknown usage does not refund liability or trigger retries.

OpenAI/Anthropic receive explicit null SDK and HTTP client timeouts. Mistral's
SDK replaces `timeout_ms=None` with 300000ms; the explicit successor therefore
sets **all four actual HTTP transport timeout extensions to None** before network
access. Historical capped requests retain their previous SDK interpretation.
Native submission and every poll receive `timeout=None`; there is no overall
poll deadline or automatic resubmission. Pending polls concern the same durable
acknowledged request, not retrying a generation. External operator termination
remains available; abrupt termination can leave incomplete evidence and pending
liability, which synthetic reopening tests conservatively convert to unknown.
No connection-specific finite deadline is retained in this successor.

Core, authored invocation turn/retrieval budgets, business horizons, deadlines,
schemas and graders are unchanged. These are benchmark semantics, not artificial
provider-execution caps. Worker concurrency and serialized tokenizer construction
remain resource-safety controls, not a bound on generation duration or total work.
Credential custody, source/runtime gates, exact scope, one-shot durable claims,
wire admission and zero automatic retry remain required.

A fresh journal is **not** settlement, refund or erasure of the previous series.
The operator must retain and review the referenced historical manifest (including
all known/pending/unknown liabilities), retire any conflicting outstanding launch
authorities, and keep old evidence immutable. The hash binds that external review;
it is not proof of financial completeness or fleet-wide exclusion. A capped
journal cannot reopen under the optional profile. No real historical recovery,
credential access, provider dispatch or invoice reconciliation is implied by
these offline tests. Any live execution requires independent source, account,
pricing and execution-policy admission.

The owner-only external admission directory has a single permanent
`SERIES-CONSUMED` claim. The credential file must be canonical, private, single
link and bounded, within an owner-only mode-0700 directory. Its descriptor is pinned without
reading contents. Exclusive claim and nonauthorizing receipt writes must finish
and both file and directory fsyncs complete before the strict environment parser
reads that descriptor. It accepts literal assignments only; no shell evaluation.
Only selected provider keys enter in-memory SDK constructors. Errors printed by
the CLI are fixed classifications, never arbitrary provider exceptions or keys.
Failed or malformed credential reads do not restore consumption.

HTTP uses actual SDKs with retry-disabled, proxy-disabled HTTP transports and
fixed HTTPS endpoints. River uses the actual generated SDK stub over an explicit
TLS channel to `api.river.ai:443`, with retries and ambient proxy routing disabled.
Acknowledgements are persisted before polling; polls never resubmit generation.
Every raw HTTP/gRPC attempt, including polls, is journaled before dispatch.
Request captures omit headers. Synthetic capability injection is a Python test
seam, with unapproved `mode=synthetic` admission and explicit mock-only transports;
there is no CLI switch that silently substitutes mocks for paid execution.

## Continuation and resource policy

`operation=resume` or `operation=correction` requires a distinct attempt ID,
original series-claim digest, exact existing journal digest and the same canonical
campaign root and ID. The controller locks and checks that journal before
recovery or credential release; it never creates a replacement pool. Resume
requires the explicit still-unassigned `pending` subset and does not resubmit
started trials. Retained closures are reconciled without provider calls. A source
correction cannot relabel historical evidence as executed under the new source.
Correction requires separately linked excluded/aborted originals and confirmed
technical-fix evidence; scored model behavior is not automatically retried.
Each continuation permanently consumes its own attempt claim and preserves prior
claims, trial IDs, closures, unknown liabilities and financial history.

River has at most two active trials and serialized native tokenizer construction.
Each trial owns its tokenizer; there is no shared mutable tokenizer or full-roster
cache. A separate HTTP lane has at most four workers, so a blocked River request
cannot occupy every HTTP worker. This is resource scheduling, not a model-call,
output or business-horizon restriction. The same lanes serve offline and live
execution. Resource qualification must cover the selected combined workload;
small focused mocks cannot establish its resource envelope.

The common executor writes per-model/per-process JSON, CSV and Markdown scorecards
as well as campaign reports, with truthful provider/mock origins and eligibility.
Terminal reports bind the journal sequence, tip and complete journal digest.
Original historical launchers and outstanding historical authorities are unchanged
and are outside this series. The operator's no-intervening-spend obligation is
an operational prerequisite, not mechanically enforced fleet-wide exclusion.
Source review, offline qualification and real account, pricing and limit
admission must precede any paid launch.
