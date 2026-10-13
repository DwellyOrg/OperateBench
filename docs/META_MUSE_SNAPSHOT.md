# Meta Muse Standard snapshot comparison

The candidate supports exactly `muse-wonder-1.0` and `muse-spark-1.3` in
Standard mode through two distinct integrations. Both use independent observation
snapshots for an equal-scaffold comparison. Neither claims Meta-native reasoning
continuity or full-capability agentic performance. The original evaluator and
grades are unchanged. Current implementation verification is offline;
**no scored runs, model results or ranking exist for the current candidate**.

## Standalone Lifecycle adapter

`operatebench.agents.meta_responses.MetaResponsesTransport` implements
`send(ModelRequest) -> ModelResponse` using the separate
`lifecycle_meta_responses_snapshot_auto_v3` mapping. It uses Core's existing
non-strict function schemas, common executor, finite turn deadline and scaffold
output ceiling of 4096 tokens. That ceiling is policy, not a published Meta
output maximum. Reasoning remains provider-default in this adapter.

The factory pins `https://api.meta.ai/v1`, accepts only `MODEL_API_KEY`, disables
SDK retries and rejects endpoint overrides and unsupported models. Local
credential validation cannot establish account access. Requests carry fixed Core
instructions and one user snapshot, with `store:false`; reasoning items are
validated and discarded as non-actions. Output usage already includes reasoning
and is never charged twice.

The Meta-only response reader accepts an optional root `store` only when it is
literal `false`, and removes it solely from a validation projection. When
`input_tokens_details` supplies `cached_tokens` without `cache_write_tokens`,
the projection inserts zero to satisfy the pinned SDK schema. This zero is a
synthetic schema placeholder, not billed provider telemetry. A supplied
cache-write count must be an exact integer zero. Unknown fields and malformed
counts fail closed. The unchanged OpenAI validator checks the projection, and
strict SDK validation supplies typed tool actions. Original response bytes and
raw input/output usage remain unchanged in captures and accounting. The v3
mappings identify this reader change; they do not reclassify earlier records.

Lifecycle execution ledgers and deterministic Core replay work offline. Its
frozen generic artifact reader still rejects Meta-bound bundles. This standalone
adapter is not the campaign route.

## Added 44 × 2 candidate campaign

The `meta-standard-same44-v1` roster adds 88 slots: the same 44 public synthetic
cases for each Standard model, across maintenance, commerce and compliance.
The shared campaign registry and Meta HTTP bridge use explicit
`reasoning.effort:minimal` and the registry/runtime mapping
`three-flow-meta-snapshot-minimal-auto-v3`, validated Responses projection and the unchanged
evaluator. The campaign has its own durable `three_flow_runtime` records and
replay; it does not depend on admitting Meta to the Lifecycle artifact reader.

Campaign requests dynamically bound output by remaining published context and,
for the original capped profile, available money. There is no generation
wall deadline or read/write timeout; connect and pool timeouts remain finite.
The unknown native output maximum is retained as unknown. Snapshot turns do not
replay encrypted reasoning, previous response IDs or native tool-result state.

The original capped admission profile uses two disjoint persistent budgets
with a **USD 1000 hard total**:
**MAX USD 10** for at most a few real Meta API compatibility probes (not scored),
and **USD 990 shared across both models** for all 88 scored campaign cells,
campaign retries and unknown exposure. The total includes provider diagnostics
and retries; it is authorization, not a spending target. Separate durable
diagnostic and campaign ledgers track totals; neither budget may be reset,
reused or transferred to the other.

Campaign reservations and accounting persist across restarts. Unknown reserves
are irreversible and continue consuming the campaign budget. Estimated spend
under the admitted pricing and reported-usage assumptions is not an invoice
guarantee. It does not establish provider tariff coverage, account access or
billing completeness.

## Explicit uncapped Meta successor

A distinct `three-flow-meta-uncapped-v1` policy permits `cap_usd: null` only
for the same Meta Standard roster. Each selected Meta profile must carry the
matching `execution_policy`. The existing generic optional policy does not
admit Meta; other providers and their existing USD 1000 capped behavior remain
unchanged. The original USD 990 Meta campaign profile remains available as a
separate legacy policy without optional execution controls.

This successor requires external owner evidence, the exact clean source SHA,
and a historical-liability manifest digest in `limit_policy`; code supplies no
approval or real evidence record. A distinct series and null-cap ledger genesis
cannot reuse an original capped journal. The fixed 44 cases per model, 88 initial
slots and one trial per case remain unchanged. No automatic paid resubmission
or historical grade changes are authorized.

There is no owner monetary ceiling, artificial model-call or cumulative-token
limit, or local generation-wall cap. Requests still respect published physical
context, finite connect/pool transport safety and required account quota pacing.
No arbitrary large money sentinel replaces the null cap. Exact Standard pricing,
account-access and usage guarantees remain admission requirements. Known usage
estimates, pending reservations and unknown exposure remain durably accounted;
unknown exposure is retained across restart. Uncapped means no monetary ceiling,
not zero cost or a known invoice. Offline validation is not proof of live success
across the 44 scenarios.

## Tool selection and comparability

Meta forces `tool_choice:auto` as a provider constraint in both integrations;
other tool-required providers retain `required`. The shared Core parser and
outcome validation are unchanged. Plain text without a valid Core function call
is excluded and not scored as a tool outcome, while any incurred usage remains
accounted. Auto selection can produce fewer valid tool outcomes, so comparisons
with tool-required providers must disclose this difference. This snapshot does
not establish equivalent tool-selection behavior or live compatibility.

## Pricing and scope

Pinned Standard rates in USD per million tokens:

| Model | Uncached input | Cached input | Output |
| --- | ---: | ---: | ---: |
| `muse-wonder-1.0` | 4.00 | 0.20 | 20.00 |
| `muse-spark-1.3` | 1.25 | 0.15 | 4.25 |

Lifecycle's separately identified pricing policy conservatively charges all
input at the uncached rate; recorded cost is not an exact discounted invoice.
Campaign profiles bind the Standard tariff and published context diagnostic.
Neither integration adds Contributor, Wonder Full, older Spark models,
streaming, background execution, multimodal input or built-in web tools.

## Request compatibility review

The public Responses and tool-calling docs explicitly support `store:false`,
`parallel_tool_calls:false`, string `instructions` and string `input`; the
reasoning docs support `reasoning.effort:minimal` for both Standard models. No
other explicit unsupported field was found in the current campaign payload.
These documented contracts are not successful live acceptance evidence.
Non-streaming generations remain subject to Meta's server-side time limit and
can return HTTP 504 even though the campaign has no local generation deadline.
The snapshot deliberately does not add streaming or reasoning continuity.

## Official sources

First-party references: [models](https://dev.meta.ai/docs/models),
[Responses](https://dev.meta.ai/docs/protocols/responses),
[authentication](https://dev.meta.ai/docs/authentication),
[reasoning](https://dev.meta.ai/docs/reasoning),
[pricing](https://dev.meta.ai/docs/pricing-rate-limits),
[overview](https://dev.meta.ai/docs/overview) and
[tool calling](https://dev.meta.ai/docs/tool-calling).

## Verification and remaining acceptance

Offline tests exercise SDK request bytes, exact model selection, minimal campaign
reasoning, dynamic caps, usage including reasoning, refusals, partial/fault
accounting, pre-dispatch rejection, durable records and deterministic replay.
They also preserve the distinction between the campaign runtime and Lifecycle's
Meta-rejecting artifact reader. Mocked transport evidence is not live endpoint
acceptance or model performance evidence.

The final clean source must pass the full 88-slot mocked consume before any paid attempt;
this check proves source-bound offline behavior, not Meta live compatibility or scored quality.
Live compatibility and model performance remain unproven.

The capped predecessor retains USD 990 for the campaign plus USD 10 for diagnostics.
An optional `three-flow-meta-uncapped-v1` successor requires explicit source-bound
owner admission for a distinct series. Its single global ledger has a null monetary
cap and retains measured costs and unknown liabilities with a historical-liability
manifest binding. Preflight describes this policy without admitting it.
Meta execution keeps finite 10-second connect and pool timeouts, unbounded local
read/write time, physical context bounds, and per-model rolling quota pacing at
3000 RPM and 4 million TPM. Reservations cover complete serialized input plus
requested output; there is no automatic resubmission. HTTP 429 stops fleet dispatch
and leaves unsent trial IDs pending. Resume audits previously started trials without
resubmitting them; unsent slots remain eligible under explicit admission.
No ranking is established by offline mocked results.

Meta pacing uses all four successful-response `x-ratelimit-*` token/request
limit and remaining headers documented in [Meta rate limits](https://dev.meta.ai/docs/pricing-rate-limits).
Exact nonnegative ASCII integers must fit the published 4,000,000 TPM / 3000 RPM
ranges and remaining values must not exceed their positive reported limits.
Under each model's network lock, a fresh observation authorizes the next request
when remaining tokens cover its complete serialized input bound plus requested
output and at least one request remains, even if local maximum reservations exceed
TPM. Each success replaces that observation; missing or invalid headers restore
conservative rolling reservations. Insufficient quota blocks until the observation
expires after 60 seconds, then conservative pacing resumes. Mock transports do
not wait. This does not establish a native output maximum or add owner ceilings.
Team traffic outside this process can consume quota after observation; HTTP429
still stops dispatch without retries. Provider headers are not persisted as raw
response evidence; the policy and signal are bound in settings and registry.
