# Three-flow evaluation controller

The repository tooling runs the three Maintenance scenarios, sixteen Commerce
scenarios and twenty-five Compliance scenarios through their real Engine domains
and original evaluators. This is a separate, nonofficial evidence contract. It
does not change Artifact 8, development records, the public pack CLI or business
grading. The selected Maintenance fixture models verified notification recovery;
its reference agent reaches reliable closure for V1, V2 and V3. Execution and
replay share `tools.three_flow_runtime.SPECS`, selecting
`maintenance_delivery_recovery_v0_7.yaml` with operation identity
`lettings_maintenance_delivery_recovery_v2`. The historical permanent-fault YAML,
artifacts, operation identity and grading remain unchanged, as do Commerce and
Compliance spec selections, business guards and scoring.

## Authority boundary

Preflight is not approval: every model remains unapproved. The repository supplies
no approval writer, recovery command, owner/account evidence or executable admission.
Fresh approval must independently bind reviewed source/runtime, account access,
settings, endpoint bounds, all applicable prices and execution policy. DeepSeek
V4.1 remains unpriced. Hashes are integrity bindings, not signatures, provider
authorship or proof of evidence accuracy; external owner custody and approval are
required. Replacing evidence and every trusted anchor is outside this contract.
Offline controls prove mechanics, not LLM performance, production resource bounds,
real historical recovery, invoice reconciliation or authority for provider access,
paid launches/corrections or re-opening a real journal. No real-provider results
are claimed.

## Offline execution

`python -m tools.three_flow_campaign` prints a non-authorizing preflight.

`python -m tools.three_flow_campaign offline --output /absolute/fresh/directory`
executes all 44 configurations through the actual OpenAI SDK and an in-process
reference-driven HTTP mock. `--providers openai anthropic mistral river` assigns
all 616 default `legacy-v1` cells (14 models × 44 configurations). The separately
selectable [`latest-http-v1` roster](THREE_FLOW_LATEST_HTTP.md) has 16 models and
704 cells. HTTP providers exercise their SDKs; River uses its actual
optional SDK, pinned tokenizer assets, native renderer/parser and protobuf mock
transport in a qualified Python environment with explicit `--river-assets`.
Missing River dependencies or assets fail closed. The command forbids network
connections, uses literal mock credentials and never loads account credentials.

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

## Replay and error evidence

Replay needs the independently retained Trial binding and closure digest. It
reexecutes the original domain and evaluator, consumes the entire decision tape
and compares full state/events/evaluation with zero provider calls.

The record container remains `operatebench.three-flow-candidate.v1`: this is not a
claim that every source revision has the same evaluator or fault semantics.
Admission pins the runtime source digest and interpreter/dependency identity;
records pin the source digest, fixture bytes, semantic spec digest, operation ID
and provider settings. Replay checks these bindings before executing decisions.
Old records remain readable as recorded. Exact historical replay requires the
original pinned source/runtime/spec, not this successor's evaluator. Do not rewrite
an old refusal's label, replay it as a successful successor, or count it as new
safe-reason evidence. No new registry mode is needed to select the successor:
the source-bound spec selection is authoritative; reasoning modes are orthogonal.

New error projections are independently named: `sanitized_http_error_v2`,
`sanitized_grpc_error_v2`, and `sanitized_river_failure_v1`. Partial evidence retains
its NON-SCORED v1 container and source bindings; projections and exclusion codes
are observations from that source, never instructions to reclassify old records.

## HTTP transport and pacing

HTTP transports use explicit injected wire capabilities, one SDK attempt per
turn, no artificial episode wall deadline or call cap, and candidate endpoint
output profiles. Fable uses automatic tool choice and adaptive thinking; hidden
thinking remains in captures and output usage. Mistral subtracts a conservative
input bound from its candidate context allowance. These candidate settings and
limits still require independent admission. River offline integration uses the
optional SDK and explicit pinned assets.

The Mistral-only `paced-safe-errors-v1` transport policy serializes calls across
lanes in one process, spaces them, and honors validated bounded delta-seconds
Retry-After cooldown for subsequent work. It never resubmits a failed query,
releases uncertain liability, or proves a pre-generation rejection. Manual
technical correction remains explicit and separately authorized by campaign
controls. Generic unknown causes remain unknown.

Trusted Python injection examples (neither supplies credentials nor admission):

```python
# Existing callers retain the legacy transport and reasoning defaults.
HTTPCampaignTransport(..., provider="mistral", transport_policy="legacy-v1")
# Separate explicit opt-ins; neither enables automatic retries.
HTTPCampaignTransport(
    ...,
    provider="mistral",
    mode_profile="off-or-minimum-v1",
    transport_policy="paced-safe-errors-v1",
)
```

The reasoning-mode registry and provider settings still distinguish legacy from
`off-or-minimum-v1`. Transport settings and request mapping additionally record
pacing opt-in. These Python examples are not new live-CLI admission fields.

## Consuming entrypoint

`python tools/three_flow_live.py preflight` is the separate, inert live-path
preflight. It checks clean source before importing repository runtime modules,
then prints the default `legacy-v1` roster, ordered 616 slots, settings registry, interpreter,
dependency versions, lock digest and runtime digest. It never inspects a
credential path, consumes authority, creates output, or reserves money. `--help`
needs no SDK imports.

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
must establish that the inventory preserves all applicable known rates.

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
unchanged.

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

Rebind profile/rate evidence as in [owner pricing](#explicit-conditional-pricing-alternative).
Owner source/registry bindings must match the final source and registry; regenerate
runtime and complete admission digests too. Any changed consent, guarantee,
timeout or bound invalidates the old binding.

Reported usage settlement, unknown-reservation retention on missing/malformed
usage, one shared journal, no automatic resubmission and no reset are unchanged.
Settings and all scorecard formats retain the policy/provenance and estimate
label for scored and excluded requests; `model_contract_verified` stays false.
Genuine provider model results remain live-eligible by origin, while injected
SDK mocks remain mock-only; unverified billing guarantees do not decide origin.

### Explicit optional limits for a new series

`three-flow-optional-limits-v1` is a reusable opt-in configuration of the same
consuming entrypoint, not a new launcher, default, approval or authorization to
repeat indefinitely. Historical admission records without this policy still
require exactly `cap_usd="1000"` and a positive timeout. Their journal genesis,
reducer, settlement and replay interpretation are unchanged. The documented
[native per-RPC timeout](THREE_FLOW_RIVER.md) is 120 seconds; external profiles may
explicitly select it and rebind, not as a new hardcoded default or episode/call/
output cap. The optional policy below instead requires null.

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
journal cannot reopen under the optional profile.

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

### Source-changing corrections

The optional `correction_binding` admission extension uses schema
`three-flow-correction-binding-v1`. It authorizes execution under the current
reviewed source/runtime/registry, **not a replacement financial policy**.
Absent this extension, existing admission/receipt/genesis serialization is unchanged.
Model-mode selection remains independent and retains its existing default.
The optional `mode_profile: off-or-minimum-v1` and
`transport_policy: paced-safe-errors-v1` fields bind the current execution
registry exactly. The latter selects process-wide Mistral pacing only; other
providers retain their existing transport policy. Both settings are forwarded
to the real SDK transport factory. Omitting either retains its legacy default,
not permission to drop a setting from an already-bound registry.

The embedded original admission retains its original mode and transport fields
and structurally validated historical registry; it is not compared with today's
settings. Its consumed closed receipt authenticates the complete original
admission. New-series pricing binds the new series source and selected registry;
correction pricing instead remains identical to the original admission, even
when the current execution mode or pacing changes.

The closed binding contains explicit `owner_accepted`, `owner_evidence_sha256`,
`original_source_sha`, `execution_source_sha`, `original_admission`,
`financial_policy_sha256`, `genesis_sha256`, `manifest_sha256`, and
`original_trial_artifacts_sha256`. The latter maps the exact selected original
trial IDs to `assigned`, `binding`, and `closure` file-byte SHA256 values; these
must exist unchanged before recovery or credential access.
The financial digest uses admission `digest` on exactly `cap_usd` and
`limit_policy` (null for the historical capped controller). Genesis hashes the
complete first canonical journal line, including newline; manifest hashes the
unchanged `assignment.json` bytes. The original admission must hash to the
original consumed receipt's `admission_sha256`. Retain that external document;
never reconstruct it by inventing financial consent.

Existing admission fields additionally bind campaign/custody roots, namespace,
original consumed claim, current journal digest, current runtime/registry,
distinct attempt, ordered subset, and confirmed technical-fix evidence. The
consuming path verifies the original receipt before opening the existing budget;
the journal and genesis hashes are checked under its exclusive lock before
recovery. Every prior settled/unknown reservation survives; unresolved exposure
is conservatively converted to unknown by existing recovery, never refunded.

Selected profiles must remain exactly equal to the original admitted profiles.
Owner pricing stays verified against its **original** source and registry entry;
the outer admission checks the **new** execution source/runtime/registry.
This extension does not authorize price upgrades or changed financial assumptions.
Such changes require separately designed explicit consent, not editing this
binding or the immutable genesis. Historical liability references stay unchanged.

Only correction operations can carry the extension. A changed-source correction
without it refuses, including the capped legacy controller. Existing same-source
operations keep their existing behavior. Technical eligibility, subset scope,
new correction IDs, cap exhaustion, and one-shot durable claims still apply.
Consumption precedes credential reads; rejection never creates a fresh wallet.
Tests use disposable private synthetic custody and actual SDK mock transports.

### Worker resources and reports

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
Offline qualification and the authority boundary above apply before any paid launch.
