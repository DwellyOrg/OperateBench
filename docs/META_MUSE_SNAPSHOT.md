# Meta Muse Standard snapshot adapter

The provider-isolated Lifecycle transport supports exactly `muse-wonder-1.0`
and `muse-spark-1.3` in Standard mode. It is an **equal-scaffold stateless
comparison**, not Meta-native or full-capability agentic performance. No live
provider admission or model score is claimed. No paid calls were made.

`operatebench.agents.meta_responses.MetaResponsesTransport` implements the
existing `send(ModelRequest) -> ModelResponse` boundary. Its separate
`lifecycle_meta_responses_snapshot_v1` mapping uses Core's existing non-strict
function schemas, including arbitrary ACT payloads and retrieval. The Meta
exchange uses the common executor for cost reservations, deadlines, one attempt,
usage and telemetry. The OpenAI SDK serializes requests and supplies raw and typed
Responses readings; the existing checked Responses projection reads outcomes.
No shared runner, historical mapping, fixture or existing provider is changed.

## Construction and identity

Use `operatebench.providers.meta_responses.build_client(model=..., environ=...)`
and pass that client to `MetaResponsesTransport` with an explicit turn deadline.
The factory accepts credentials only from `MODEL_API_KEY`, pins
`https://api.meta.ai/v1`, and disables SDK retries. It never falls back to
`OPENAI_API_KEY`. It rejects redirect/account environment overrides, unsupported
model IDs (including Contributor and older Spark IDs), and credentials that are
not 1–4096 visible ASCII characters without whitespace or control characters.
Opaque bearer values need no particular prefix; header-valid punctuation such
as `:` is permitted. Backend authorization and model access are checked via a
separate account `GET https://api.meta.ai/v1/models`, not a credential regex.
The factory does not perform that request; local validation cannot prove
validity, revocation status or model access offline.

Transport settings bind Meta identity, endpoint, mapping/profile, Standard rates,
SDK version, provider-default reasoning and snapshot comparison scope. Each
request sends one canonical user snapshot plus fixed Core instructions and its
request identity digest. Function tools are explicitly non-strict; `store:false`
is explicit. There is no built-in web tool, encrypted-reasoning include, previous
response ID, function-call-output replay or server-side conversation state.
Reasoning output items are validated and discarded as non-actions. Refusals
produce the existing no-tool-call outcome and still retain billable usage.

The **4096 output-token ceiling is scaffold policy**, not Meta's advertised
maximum. This change establishes no provider output maximum. Output usage
already includes reasoning tokens; those tokens are validated but never added
a second time. Documented cached-input and reasoning detail fields are accepted;
unsupported server fields fail closed, including OpenAI-specific extensions.

## Pricing and limitations

Dated Standard rates in USD per million tokens (verified 2026-10-13):

| Model | Uncached input | Cached input | Output |
| --- | ---: | ---: | ---: |
| `muse-wonder-1.0` | 4.00 | 0.20 | 20.00 |
| `muse-spark-1.3` | 1.25 | 0.15 | 4.25 |

`pricing_profile(model)` returns a separately identified, operator-pinned
Lifecycle pricing policy based on these published rates. The existing guard
and ledger settle only total input and output counts. They conservatively charge
all input at the uncached rate; **recorded cost is an upper bound when cache hits
occur, not an exact discounted invoice**. Cached rates are disclosed in settings.
No Contributor rates, web-search charges or paid budget are authorized here.

Known incompatibilities and excluded capabilities:

- No reasoning continuity, native tool-result association or conversation reset
  protocol. Each turn independently reads the fixed observation snapshot.
- The frozen provider-execution-binding-v1 schema and artifact reader allow only
  existing providers. Meta execution ledgers validate, and full Core episodes
  deterministically replay, but Meta-bound artifact bundles fail the current
  artifact reader. The adapter is a smaller offline PR deliverable, **not an
  admitted production campaign route**. A separately authorized artifact binding
  successor is the next integration step; do not rewrite the frozen schema or
  label Meta as OpenAI to bypass it.
- No campaign registry, CLI canary, Boundary Track route or public model admission
  is added. Parent admission requires a secure key and explicit paid budget.
- No streaming, background execution, multimodal inputs, built-in tools, custom
  endpoint, Contributor, older models, Wonder Full or SDK-external response
  extensions. Non-streaming service time limits remain provider faults.
- Provider-default reasoning and sampling remain the default for this mapping;
  no unsupported `reasoning.effort:none` is sent. The 4096 scaffold ceiling can
  truncate reasoning and answers and does not establish provider capacity.
- Cache discount accounting and details retention need a separately versioned
  usage/ledger contract if exact invoice reconciliation is required.

## Official sources

Verified the supplied official [models](https://dev.meta.ai/docs/models),
[Responses](https://dev.meta.ai/docs/protocols/responses),
[authentication](https://dev.meta.ai/docs/authentication),
[reasoning](https://dev.meta.ai/docs/reasoning), and
[pricing](https://dev.meta.ai/docs/pricing-rate-limits) documentation.
Additional supplied references:
[overview](https://dev.meta.ai/docs/overview) and
[tool calling](https://dev.meta.ai/docs/tool-calling).
Meta recommends encrypted reasoning replay for agentic continuity; this mapping
intentionally implements independent turns under the shared snapshot scaffold.

## Offline acceptance

TDD began with the new Meta transport import failing. Tests use real OpenAI SDK
clients over `httpx.MockTransport`; no credential files, secrets or model network
requests are read or made. Both exact model IDs have byte-level request assertions,
repeated snapshot request/response identity, request-ID digests, non-strict tools,
usage including reasoning, refusal handling, sanitized errors and zero-dispatch
preflight checks. Full maintenance V1 episodes pass through the fixed Core runner
and produce verified execution ledgers with exact billable token totals and
re-derived conservative cost; Core's deterministic replay check passes. Tests
explicitly prove the frozen artifact reader rejects Meta-bound bundles.

Final verification: **30 Meta tests passed on Python 3.11.15 and 3.14.4** with
OpenAI SDK 2.53.0. The broader affected-provider/compatibility suite produced
**558 passes and one failure on each interpreter**: the existing
`test_reasoning_combined_consuming_cli_and_zero_call_replay` refuses staged
changes through its clean-source gate. That gate was neither changed nor bypassed.
Historical projection/fault digests and frozen schema contracts passed.
Repository-wide Ruff lint/format, mypy (`src tools`, 200 source files), all
14 public-release checks and REUSE (871/871 files) passed. No commit or push;
only the four reviewed adapter/test/document paths are staged. The superseded
untracked gap diagnostics were removed.
