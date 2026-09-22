# The provider matrix: a private methodology spike

This document describes a **private, staged, cost-capped methodology spike**
across five exactly-pinned models from four providers. It is not a benchmark
run, and nothing produced under it may be published, cited or reported as one.

## What it measures, and what it cannot

The plan is 180 episodes: **2 semantic units × 6 variants × 3 trials × 5
models**.

Two is the number that governs how the results may be read. A semantic unit is
one Construct Card — one authored decision situation — compiled into six
variants. The five models are therefore all answering **two questions**, six
ways each. Two questions cannot separate five models: any ordering that fell out
of 180 episodes would be an ordering of two cases, and the confidence interval
around it would be wide enough to contain almost any conclusion.

What the repeated trials *do* support is a within-model quantity. Running the
same model on the same variant three times, under one fixed scaffold, at pinned
sampling settings, estimates **decoding repeatability**: how often the model
gives the same answer to the same question. That is a property of the model's
decoding under this configuration, not of its capability, and it is the only
thing three trials are enough for.

So, explicitly:

- **This is not a leaderboard.** There is no ranking, no score, no "best model".
- **This is not a public claim** about any model, provider or product.
- **Cross-model comparison is out of scope at this n.** The matrix exists to
  find out whether the *methodology* — the adapters, the identity binding, the
  cost controls, the staging — holds up across four vendors' APIs. That is a
  question about this repository, answered by running it.
- **A model that fails here has not been shown to be worse.** It may have been
  shown that this build's request shape for it was wrong, which is exactly the
  kind of finding a methodology spike is for.

## The five models, exactly

Each cell runs **36 episodes** (6 variants × 3 trials × 2 semantic units), for
180 in total. Every identifier below is the exact string sent on the wire and
recorded in the run's adapter identity; the response's own stated model is
checked against it before the answer's action or its token counts are accepted.

| Provider | Exact model identifier | Episodes | Credential variable |
|---|---|---|---|
| Anthropic | `claude-haiku-4-5-20251001` | 36 | `ANTHROPIC_API_KEY` |
| Anthropic | `claude-sonnet-5` | 36 | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-5.6-luna` | 36 | `OPENAI_API_KEY` |
| xAI | `grok-4.5` | 36 | `XAI_API_KEY` |
| Mistral | `mistral-small-2603` | 36 | `MISTRAL_API_KEY` |

The separate fixed one-cell Sonnet operator is
`tools/run_lifecycle_v1_anthropic_sonnet_canary.py`. It is **offline-verified only**:
the real pinned Anthropic SDK was exercised over `httpx.MockTransport`, including
all 46 native requests, Artifact 8, bundle audit, and zero-call replay. Its official
rates were rechecked on 2026-09-07 at the
[model overview](https://platform.claude.com/docs/en/models/overview) and
[pricing page](https://platform.claude.com/docs/en/about-claude/pricing): USD 2
input / USD 10 output per million tokens. The fixed envelope has a 25,884-token
maximum native input bound, 4,096 output-token reservation, 50-call hard ceiling,
1,499,000-token hard cap, exact USD 4.6364 worst case, and USD 4.68 retained fixed
cap. No live Sonnet cell has been executed, and this offline verification makes
no model result or score claim.

The separate repeat-ready fixed OpenAI Luna operator is
`tools/run_lifecycle_v1_openai_luna_canary.py`. It is **offline-verified only**
through the real pinned OpenAI SDK over `httpx.MockTransport`: all 46 native
Responses requests completed, followed by Artifact 8, execution-ledger-v3,
bundle audit, and zero-provider-call replay. No live Luna cell has been executed.
The operator-pinned USD 0.20 input / USD 1.20 output per million-token envelope
was recorded on 2026-09-08 from the published
[model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md)
and [pricing page](https://developers.openai.com/api/docs/pricing); those rates
are operator controls, not a claim about current prices. The fixed envelope has
a 27,241-token maximum native input bound, 1,441,502 expected-call ceiling,
1,566,850-token hard cap, exact USD 0.518170 worst case, and USD 0.52 minimal
cent cap. This offline verification makes no model result or score claim.

The separate fixed xAI Responses operator is
`tools/run_lifecycle_v1_xai_grok_4_5_responses_canary.py`. Its 46-call offline
preflight, Artifact 8, execution-ledger-v3, bundle audit, and zero-call replay
pass through the pinned OpenAI SDK over `httpx.MockTransport`; no live cell has
been executed. Its operator-pinned USD 2 input / USD 6 output per million-token
envelope has a 27,278-token maximum native input bound, 1,443,204 expected-call
ceiling, 1,568,700-token hard cap, exact USD 3.9566 worst case, and USD 3.96
minimal cent cap. The moving `grok-4.5` identifier is not public reproducibility
evidence, and the cap treats `max_output_tokens` as covering reasoning and
function-call output because xAI does not separately document argument-token
accounting. This is a private execution assumption, not a model result or score.
The response-contract-only correction preserved request mapping
`lifecycle_xai_responses_model_request_v1` because its request bytes, bounds, and
controls did not change at that boundary. The subsequent benchmark-validity
correction and nonempty-schema projection are reflected in the current request
mapping v6; the Grok 4.6 8192-token lane uses v7. Historical diagnostics retain their original
identities; see [the version decision](VERSIONING.md#engine-0110-benchmark-validity-correction). It separately pins the provider-observed response
extension contract `xai_responses_server_extensions_v2` and its digest. The v2
contract keeps every v1 field unchanged and admits `context_details` only as a
closed union of the two observed complete exact-count pairs:
`{image_tokens, text_tokens}` or `{input_tokens, output_tokens}`. Mixing,
partial pairs, unknown children, booleans, nulls, and out-of-bound counts remain
refused; server-tool and source counts remain exact zero. Trial 01 remains under
its original v1 response contract and settings digest, while the private Trial
02 structural diagnostic remains an excluded diagnostic rather than an
Artifact or benchmark retry. Neither is relabelled. This response-only change
makes no public model claim and leaves execution authorization and cost controls
unchanged.

The separate Grok 4.6 one-cell tool,
`tools/run_lifecycle_v1_xai_grok_4_6_responses_canary.py`, is **public offline-only**.
Its default preflight uses the real OpenAI SDK over `httpx.MockTransport`.
All of its live entrypoints and `tools/consume_grok46_external_authorization.py`
refuse before credentials, authority consumption, output reservation or clients.
No environment setting enables paid execution. Illustrative offline envelope v1
uses a distinct profile, policy and closed record identity; its invented fleet
metadata is neither historical allocation, spending permission nor a ledger.
Mock-only one-shot tests require an explicitly injected mock transport. Technical
per-cell rates, token/call/deadline limits and conservative refusal remain pinned.
This envelope version is independent of engine and provider-mapping versions.

The separate fixed one-cell Mistral operator is
`tools/run_lifecycle_v1_mistral_small_2603_canary.py`. It is implemented and
**offline-verified only** through the native pinned Mistral SDK over
`httpx.MockTransport`: all 46 native requests completed, followed by Artifact 8,
execution-ledger-v3, bundle audit, and zero-provider-call replay. No live Mistral
cell has been executed. The USD 0.15 input / USD 0.60 output per million-token
standard rates were verified on 2026-09-07 at the
[dated model page](https://docs.mistral.ai/models/mistral-small-4-0-26-03),
[API guide](https://docs.mistral.ai/api),
[OpenAPI document](https://docs.mistral.ai/openapi.yaml), and
[pricing page](https://docs.mistral.ai/inference/pricing). The fixed operator
uses dated `mistral-small-2603`, not the moving `mistral-small-latest` alias. The
model page describes a 256k text context without an exact integer claim here;
4,096 reserved output tokens is the benchmark cap, not an official model-output
ceiling. `reasoning_effort` remains omitted because the provider's absent-field
default is unresolved, and `service_tier` is omitted rather than opting into
Global Standard. The fixed envelope has a 27,331-token maximum native input
bound, 1,445,642 expected-call ceiling, 1,571,350-token hard cap, exact USD
0.32786250 worst case, and USD 0.33 minimal cent cap. This offline verification
makes no model result or score claim.

**No aliases, and no defaults.** `--model` is required everywhere; nothing in
this build picks a model, and no moving alias (`-latest` or otherwise) is
substituted, matched by prefix or case-folded — not in the adapters, and not in
the price tables, which match the `(provider, model)` pair whole.

**One identifier cannot claim as much as the other four, and the plan says so.**
xAI's published model page exposes `grok-4.5` and no dated snapshot identifier,
so it is the most exact string obtainable rather than a pinned point in time:
what it resolves to can change without the name changing, and the per-turn check
that the response names the model this run requested would still pass. That
limitation travels in the plan artefact as each cell's `identifier_limitation`
(null for the other four), so an operator authorising the xAI cell authorises it
along with everything else. It is a limitation of the evidence, not a caveat
about the model.

## What is pinned, and why comparability is fragile

Every result-affecting request setting is frozen per provider *and per model*,
and hashed into the run's `configuration_id`: the API surface, the request
mapping, the exact model identifier, the tool-choice behaviour, the output
ceiling, which sampling and reasoning fields are sent and which are deliberately
omitted, the SDK and its version, and the retry policy.

The output ceiling is 1024 tokens for every model in the matrix, and that is a
comparability decision rather than a per-provider preference: two models cut off
at different lengths are not answering the same question.

**Two models' request contracts are read from the vendor; two are not.**

Anthropic documents how its models treat both a sampling parameter and an absent
thinking field, so Sonnet 5 is sent an explicit `thinking: {"type": "disabled"}`
block and a plan against it can state as a fact that adaptive thinking is off.

Anthropic Messages tool inputs are a provider-specific projection of the shared
Lifecycle contract. Every wire `input_schema` has an object root and no root
`oneOf`, `anyOf`, or `allOf`; nested unions remain intact. A root union is widened
provider-side only to the deterministic object superset that covers every branch,
while conflicting property definitions, non-object branches, unresolved or cyclic
references, and excessive schema graphs are refused locally. Tool choice is still
non-strict provider-side. The authoritative Core schemas are unchanged, and the
local Core parser remains the strict authority for every response.

`gpt-5.6-luna` is in the same position, from its own published model page at
`developers.openai.com/api/docs/models/gpt-5.6-luna.md`. That page states that
the model supports the Responses API and function calling, and that
`reasoning.effort` accepts `none`, `low`, `medium` (the default), `high`, `xhigh`
and `max`. So the profile sends `reasoning: {"effort": "none"}` **explicitly**
rather than leaving the field out: an absent field selects `medium`, and a
standardized no-reasoning track that relied on that default would be running
under a reasoning setting its manifest never recorded. The same page does not
establish that the model accepts a sampling parameter, so `temperature` is
omitted from the body entirely rather than pinned to a value nobody published a
contract for. Its profile carries `model_contract_verified: true`, which is a
claim about exactly the two fields the profile decides — the reasoning object it
sends and the sampling parameter it omits — and a preflight against it reports
`optional_extended_thinking: false`.

For `grok-4.5` and `mistral-small-2603`, this repository verified each model's
price and its API identifier against the provider's published pages, but did not
find a published statement of how the model treats a sampling parameter or an
absent reasoning field. Those two profiles carry
`model_contract_verified: false`, and a preflight against them reports
`optional_extended_thinking: null` with a note saying why.

If either of those two models rejects the pinned sampling parameter, the run
fails closed — an HTTP 4xx recorded as a request rejection — rather than
silently running under a shape nobody checked.

## The xAI adapter is not the xAI SDK

The xAI integration talks to **xAI's documented OpenAI-compatible Chat
Completions surface**, through the OpenAI SDK, pinned to `https://api.x.ai/v1`.
It is named `xai_openai_compat_chat_completions` everywhere it is recorded — in
the adapter identity, in the settings (`sdk: "openai"`, `api_compatibility:
"openai_compatible"`) and here — so a run executed through it can never be read
as a run executed through the vendor's own client.

Why `xai-sdk` was not used:

- It is a gRPC/protobuf client whose `Client` exposes no channel, interceptor or
  transport injection point. Every other adapter in this build proves what it
  puts on the wire by running the *real* SDK over an in-process HTTP transport
  and asserting the serialised body; against `xai-sdk` the equivalent test would
  have to monkeypatch the generated stub, which proves only that the adapter
  agrees with the stub.
- It would add `grpcio` and `protobuf` to a published distribution's dependency
  closure for one of five models.

**What this costs, stated plainly:** a behaviour that differs between xAI's
compatibility layer and its first-party API — a field the layer drops, a usage
counter it rounds, a tool-call shape it normalises — would be invisible from
here. An operator who needs the first-party client needs a different adapter,
and it would be a different configuration identity.

One consequence of that is visible rather than hypothetical, and it is why this
lane carries a response-extension contract of its own: the layer states fields
the OpenAI SDK does not model, on the message and inside the usage block. What
this build does about them is below. What it cannot do is tell you whether the
first-party API states the same ones.

A second limitation belongs beside it, because it is about the model rather than
the transport. The published `grok-4.5` model page states that the model reasons
and that it calls functions, and publishes **no parameter that disables
reasoning**. This build therefore sends no reasoning field to this lane and
invents none; `model_contract_verified` is `false` in its settings, and an
operator authorising a live xAI configuration authorises that claim with the
rest.

## The response is checked as the wire sent it

Each adapter reads two things out of a provider response: one action, and the
two token counts the turn is priced and capped on. Both used to be read off the
SDK's *typed* model, and a typed model is not a transcript of the wire. Three
things happen between the bytes and the object, and none of them is visible from
the far side:

- **coercion** — `"prompt_tokens": true` reaches the object as `1`, and `"12"`
  reaches it as `12`. Both then look exactly like a measurement;
- **dropping** — the Mistral SDK discards a field its schema does not declare
  before any object exists, so an undeclared field could not be refused by
  inspecting one. That is true of every response model in that SDK this build
  reads *except* `UsageInfo`, which is configured to keep undeclared fields; not
  declaring a field and dropping it are two different things, and which one a
  given object does decides what a contract over it can check;
- **leniency** — a `function` that arrives as a list survives the union
  resolution and fails several frames later as an `AttributeError`, after the
  attempt has already been published as a successful response with usage.

So the exact JSON is validated first, and only then is the SDK's object trusted.
Field presence, field types, nested shape and undeclared extras are checked at
every level this build reads — the response, the choices or output items, the
message, the tool call, the function object and the usage block — with the
allowed field set for each taken from the pinned SDK's own model for it rather
than hand-listed. Anything outside that set fails closed.

How the bytes are obtained is part of what a run records, under
`response_capture` in adapter settings:

| Lane | Mode | Note |
|---|---|---|
| OpenAI, xAI | `openai_sdk_with_raw_response_v1` | the SDK's own `with_raw_response` surface; it marks the request with a header |
| Mistral | `mistral_sdk_captured_http_response_v1` | the configured HTTP client is wrapped in an audited object implementing the SDK's `HttpClient` protocol |

The Mistral wrapper only observes: the SDK builds the request and the inner
client sends it. It is cleared immediately before each call and takes exactly one
response after it, so what a turn is measured from is that turn's own body. No
raw body or response text is ever written anywhere.

**Usage is required.** A response must state a usage object and both token count
fields, as exact non-negative integers on the wire. A missing block, an empty
one, a null, a boolean, a string or a fractional number is
`provider_response_invalid`: the measured usage stays null, and the conservative
reservation is kept in full as exposure rather than settling at a cost nobody
measured. A body the SDK read but could not parse — Mistral's
`ResponseValidationError` on an HTTP 200, for instance — is recorded as an
attempt that *received a response*, because it did.

All three adapter versions moved to `0.2.0` for this, and the settings that name
the capture mode are hashed into `configuration_id`, so a run made before it can
never be read as a run made under it. All three have since moved further, for
what they *accept* rather than for what they send: the OpenAI adapter is at
`0.5.0` (and what it sends to `gpt-5.6-luna` changed too), the xAI adapter is at
`0.3.0`, and the Mistral adapter is at `0.3.0` — all three are below.

### Three lanes accept named fields their SDK does not declare

The OpenAI service may return, on an otherwise ordinary HTTP 200, a small block
of **top-level fields the pinned SDK's `Response` model does not declare** —
covering billing attribution, two sampling-penalty parameters, a retention flag
and a server-side tool-usage summary. Under the rule above, a body carrying one
is refused whole, which is the correct default and the wrong answer for a field
set that is stable, named, and read by nothing here.

So the OpenAI lane names **one exact response-extension contract** —
`openai_response_server_extensions_v2` — and accepts precisely it. What the
contract guarantees:

- **Closed, recursively.** A fixed set of top-level names, fixed children under
  each object, and an unknown key at *any* depth is still
  `provider_response_invalid`. Nothing else about the response contract is
  relaxed: an undeclared field anywhere outside this block fails exactly as it
  did before.
- **Optional at the top, complete underneath.** Each of the top-level names may
  be absent, independently of the others: this build reads none of them, so a
  service that sends no such block at all has not broken anything a run depends
  on. An object that *is* present is a different matter — it must state every
  child the contract fixes for it, at every depth. A partly stated object, an
  empty one included, is `provider_response_invalid`, because one exact shape is
  what this contract describes and every subset of that shape is not.
- **Typed on the wire, not after coercion.** Counts are exact non-negative
  integers and never booleans; the two penalties are finite JSON numbers inside
  the published `[-2, 2]` range, and never booleans, strings, nulls, `NaN` or
  infinities; the flag is a JSON boolean; the attribution string is non-empty
  and length-bounded.
- **Checked on both readings.** It is validated on the exact JSON the provider
  sent *and* on the typed extras the SDK parks in `model_extra`, and the two
  readings must state the same block. Neither is edited to agree with the other,
  and `model_extra` is never cleared to hide what arrived; the SDK's typed parse
  stays exactly as it produced it.
- **Read for nothing.** No value in the block is used for the action, for the
  model's stated identity, or for the token counts a turn is priced and capped
  on. It is validated so that a body carrying it can be accepted at all, and then
  it is dropped. No name or value from it is written to any durable row — a
  refusal states which *kind* of shape was wrong and nothing else.

**Provenance, stated honestly.** This contract is **provider-observed**: it
transcribes the field names and type shape this repository saw the service
return. It is *not* an official vendor declaration and it is not read off the
SDK — the pinned SDK declares none of these fields, which is precisely why the
block needed a contract of its own. A provider-observed contract can move
without notice, so it is named **and hashed** in adapter settings
(`response_server_extensions`, `response_server_extensions_digest`): the name
alone would let two builds accept different bodies while their manifests agreed.
The digest covers the whole contract — its name, the transcribed names and
types, and the two rules above that decide what a present object has to state —
rather than the transcription alone, because two builds can share every field
name and still read one body differently. Both are inside `configuration_id`,
and the OpenAI adapter version is `0.5.0`: what a run will read as an answer is
part of the integration, so it moves with it. What the lane *sends* is not
affected — the request body is byte-identical to `0.3.0`. No other lane accepts
these fields, and everywhere else in this build an undeclared response field is
still refused.

#### The xAI compatibility layer states its own, on nested objects

The xAI lane speaks a **compatibility surface rather than the vendor's own API**
(see above), and the fields it returns are where that shows. On an ordinary
HTTP 200 from `grok-4.5`, the endpoint may state fields the pinned OpenAI SDK
does not declare — on the assistant **message**, on the **usage block**, and as a
pair inside `prompt_tokens_details`, which is an object the SDK *does* declare.
None of them arrives at the root, so this is a different contract from the OpenAI
one over different names at different places, and it is recorded under its own
settings keys.

So the xAI lane names **one exact response-extension contract** —
`xai_compat_response_extensions_v1` — and accepts precisely it:

| Object | Field | Kind |
|---|---|---|
| `choices[].message` | `reasoning_content` | bounded string |
| `usage` | `cost_in_usd_ticks` | non-negative bounded integer |
| `usage` | `num_sources_used` | non-negative bounded integer |
| `usage.prompt_tokens_details` | `image_tokens`, `text_tokens` | non-negative bounded integers, **as a pair** |

What the contract guarantees:

- **Closed, and per object.** A name this contract fixes for one object buys
  nothing on another: `reasoning_content` on the usage block, `cost_in_usd_ticks`
  at the root, or any name outside the table is still
  `provider_response_invalid`, exactly as it was before. Nothing else about the
  response contract is relaxed, at any depth.
- **Optional where absence was observed; whole where a group was.** Each message
  and usage member may be absent on its own — this build reads none of them, so a
  service that stops sending one has not broken anything a run depends on. The
  `prompt_tokens_details` pair is the exception: both may be absent, and if
  either arrives both are required, because half of it was never observed and
  accepting half would widen an exact contract into a guess.
- **Typed and bounded on the wire, not after coercion.** The counts are exact
  non-negative integers within the same exact-integer bound the token counts are
  read under, and never booleans, strings, fractional numbers or nulls. The
  reasoning string is an exact JSON string of valid Unicode scalars — a lone
  surrogate is refused — bounded at `MAX_OUTPUT_TOKENS × 16` characters, a limit
  tied to this run's own pinned output ceiling at a per-token character count far
  above what any tokenizer emits.
- **Checked on both readings, over the union of what each states.** Each field is
  validated on the exact JSON the provider sent *and* on the typed extras this
  SDK parks in `model_extra` at each of the three objects, and the two readings
  must state the same thing. The objects those sites live on are compared first:
  a message, a usage block or a `prompt_tokens_details` one reading states and
  the other does not is `provider_response_invalid` before any value is read,
  because the reading that has it could have stated extensions under it that the
  other reading had nowhere to state. Neither reading is preferred, neither is
  edited to agree with the other, and `model_extra` is never cleared.
- **Read for nothing, and priced into nothing.** No value reaches the action, the
  model's stated identity, or the counts a turn is priced and capped on. In
  particular `cost_in_usd_ticks` is **not** a cost: a measured cost in this build
  is the run's pinned price policy times the two token counts, and that is
  unchanged. The reasoning text is never persisted and never quoted in a failure
  detail — a refusal states which *kind* of shape was wrong and nothing else.

**Provenance, stated honestly.** This contract is **provider-observed**: it
transcribes the names and type shape this repository saw the compatibility
endpoint return, established by two bounded schema-only diagnostics whose raw
bodies were not kept. It is *not* an official vendor declaration and it is not
read off the SDK, which declares none of these fields. So it is named **and
hashed** in adapter settings (`response_extensions`,
`response_extensions_digest`), the digest covers the rules and the bounds as well
as the transcribed names, and both are inside `configuration_id`. The xAI adapter
version is `0.3.0`.

**What did not move.** The request is byte-identical to `0.2.0`'s: temperature
pinned to zero, no reasoning field, `model_contract_verified` still `false`, and
pricing untouched. The published `grok-4.5` model page states that the model
reasons and that it calls functions, and publishes **no control that disables
reasoning** — so this build sends none and invents none. Describing what comes
back is the whole of the change.

### Mistral: one object on the usage block

`mistral-small-2603` returns, on an otherwise ordinary HTTP 200, a
**`prompt_tokens_details` object on the usage block** that the pinned SDK's
`UsageInfo` does not declare, carrying a single `cached_tokens` count. Under the
rule above the whole body was refused, so a real HTTP 200 could not be read at
all: the action went unparsed and the turn's reservation was held as exposure
over a field nothing here reads.

So the Mistral lane names **one exact response-extension contract** —
`mistral_response_extensions_v1` — and accepts precisely it:

| Object | Field | Kind |
|---|---|---|
| `usage` | `prompt_tokens_details` | object, closed to exactly the row below |
| `usage.prompt_tokens_details` | `cached_tokens` | non-negative bounded integer, **required** |

What the contract guarantees:

- **Closed, at both levels.** One name on the usage block and one name inside the
  object it carries. The same object arriving at the root or on a message is
  still `provider_response_invalid`, an unknown key inside the object is refused
  at that depth, and nothing else about the response contract is relaxed.
- **Optional as a whole, complete when present.** The object may be absent — this
  build reads nothing out of it — but an object that arrives states its one
  member. A vacuous `{}`, or one stating some other name instead, was never
  observed and is refused rather than read as an empty statement.
- **Typed and bounded on the wire, not after coercion.** `cached_tokens` is an
  exact non-negative integer within the same exact-integer bound the token counts
  are read under, and never a boolean, a string, a fractional number, a null or a
  negative.
- **Checked on both readings.** `UsageInfo` is the one model in this SDK
  configured to keep undeclared fields, so the object is stated twice — in the
  exact JSON the provider sent, and in `model_extra` on the typed reading of the
  same bytes — and the two must state the same thing. Neither reading is
  preferred, neither is edited to agree with the other, and `model_extra` is
  never cleared. The typed allowance is as narrow as the wire's: it is granted on
  the usage block's SDK model alone, and every other object at every other depth
  still accepts exactly zero undeclared fields.
- **Read for nothing, and priced into nothing.** `cached_tokens` says the
  provider served part of the prompt from a cache and charged less for it. **This
  build does not model that discount.** The pinned Mistral price charges every
  provider-reported `prompt_token` at the uncached input rate, which is
  conservative in the only direction that matters: a run never under-reports what
  it spent, and a measured cost stays re-derivable from the run's own pinned
  policy times the two counts rather than from a number the provider supplied.
  Moving `cached_tokens` across its whole accepted domain changes neither the
  accepted action, nor the standard usage, nor the cost.

**Provenance, stated honestly.** This contract is **provider-observed**: it
transcribes the name and type shape this repository saw the service return,
established by two bounded schema-only diagnostics whose raw bodies were not
kept. It is *not* an official vendor declaration and it is not read off the SDK,
which does not declare the field. So it is named **and hashed** in adapter
settings (`response_extensions`, `response_extensions_digest`), the digest covers
the rules and the bound as well as the transcribed name, and both are inside
`configuration_id`. The Mistral adapter version is `0.3.0`.

**What did not move.** The request is byte-identical to `0.2.0`'s: temperature
pinned to zero, the same pinned output ceiling and tool choice,
`model_contract_verified` still `false`, and pricing untouched. No parameter that
would turn the object off is invented here. Describing what comes back is the
whole of the change.

## Pricing provenance

Each provider carries its own reviewed price policy — its own version, its own
published source, its own date this repository read it — and its own digest.
Prices are exact `Decimal` values; a cost is never computed in binary floating
point.

| Provider | Source | Recorded |
|---|---|---|
| Anthropic | `platform.claude.com/docs/en/about-claude/pricing` | 2026-09-07 |
| OpenAI | `developers.openai.com/api/docs/pricing` | 2026-08-11 |
| xAI | `docs.x.ai/developers/pricing` | 2026-08-11 |
| Mistral | `docs.mistral.ai/inference/pricing` | 2026-08-11 |

The xAI policy carries a `snapshot_limitation` the others do not. Its source
page exposes no dated price snapshot, so those rates and that model's tier are
what the page showed on the recorded date rather than a price the provider
published as of a stated day. A later reader cannot re-fetch the version they
were read from. That limitation is inside the policy payload, and therefore
inside the digest a run records.

A model is priced by the **pair** `(provider, model)`. `grok-4.5` under
`openai` is refused, not silently priced at xAI's rate. An unknown provider or
model fails closed: a cost-capped run against an unpriced model does not start.

### Two of the five models are priced in tiers

`grok-4.5` and `gpt-5.6-luna` publish a **long-context tier**: past a stated
input-token threshold the *whole request* — prompt and completion alike — is
priced at a higher pair of rates. The other three models publish one rate at any
prompt length and are priced at it, with no tier field in their policy payloads
at all.

| Model | Base (in / out per 1M) | Threshold | Long (in / out per 1M) |
|---|---|---|---|
| `grok-4.5` | $2 / $6 | **from** 200,000 input tokens | $4 / $12 |
| `gpt-5.6-luna` | $0.20 / $1.20 | **above** 272,000 input tokens | $0.40 / $1.80 |

The word in bold is the part that cannot be guessed. xAI's page states that
every token in a request uses the long rates once the threshold is *reached*, so
a prompt of exactly 200,000 tokens is already long-context. OpenAI's states
`≤272K input tokens` against `>272K input tokens`, so a prompt of exactly
272,000 is still the base tier and the long one begins one token later. Each
policy therefore records its threshold, its inclusivity and both long rates
explicitly, under `long_context_tier`, inside the payload the run hashes — so a
later reader can recompute any cost the run reported without knowing either
vendor's wording.

The tier is selected by the request's **input-token count**, and the same
selection is made everywhere a cost is computed: the reservation the cost guard
authorises before a request (from a conservative upper bound on that count), the
settlement it banks from the provider's reported counts, and the re-derivation a
replay checks a stored row against. A reservation is still a bound because a
long tier is never cheaper than its base tier — a policy that stated one would
be refused.

Two shortcuts are deliberately not taken. Charging the long rate for every
request would be safe against the cap and false about every model; imposing a
maximum prompt length would turn a pricing question into a capability limit this
build has no grounds to state.

Because the tier moved what these two tables say, both moved their policy
version and their digest: `openai-tiered-2026-08-11` and `xai-tiered-2026-08-11`
supersede the `-standard-` versions. The Anthropic and Mistral policies are
untouched, byte for byte, including their digests.

## Stages

The order is fixed, and it is described by **two** numbers per stage, which are
not interchangeable.

*The cumulative episode target per model* is where a model's run stands once the
stage is complete. The targets are a strict prefix chain, so each executing
stage *resumes* the last rather than re-running episodes already paid for.

*The new episodes per model* is what one invocation of that stage actually
executes: this stage's target minus the one the stage before it already made
durable. This — and only this — is the number that belongs beside
`--stop-after`, which bounds new episodes in a single invocation. Handing
`--stop-after` a cumulative target runs the prefix a second time: 1 for the
smoke stage and then 12 for "a trial block of twelve" executes thirteen episodes
against a twelve-episode block.

| Stage | Cumulative target / model | New episodes / model | Notes |
|---|---|---|---|
| `offline_preflight` | 0 | 0 | Contacts nothing. Computes each cell's `configuration_id`. |
| `smoke` | 1 | 1 | The smallest thing that proves the wiring is real. |
| `audit_gate` | 1 | 0 | **Manual.** A person reads the smoke evidence. No episodes. |
| `trial_block` | 12 | 11 | Completes the first trial block. |
| `remainder` | 36 | 24 | The rest, to 36 per model / 180 total. |

The new-episode column sums to 36 — the whole per-model plan — while the targets
do not sum to anything meaningful. A resumed run still computes its own pending
suffix from its ledger; `--stop-after` bounds the invocation and never restarts
or shortens the plan.

Two ceilings sit outside the stage table and do not move with it. Each model's
own run manifest is capped at exactly **36** episodes (`--max-episodes 36` on
that cell's `run-suite`), whatever stage is being executed — it is the run's own
immutable authority, and a run capped at anything else is a different
`configuration_id`. And `provider-matrix --max-episodes` is one *global* exact
authorisation over the whole matrix: exactly **180** is accepted and is the
default, because this is a fixed methodology — a smaller ceiling is one the plan
cannot run under, and a larger one authorises episodes nothing would execute.
`--trials` is fixed at **3** here for the same reason; `run-suite --trials`
remains general.

Plan it offline:

```
boundarybench provider-matrix <suite.yaml> \
    --output-root DIR --max-cost-usd AMOUNT --stage smoke --json
```

That command contacts nothing, creates no run directory, and prints each cell's
`configuration_id`, episode cap, dollar allocation and output directory, along
with both stage numbers under the explicit names
`stage_cumulative_episode_target_per_model` and
`stage_new_episodes_per_model`. Its `--execute` flag is refused: this build
ships no pre-approved authorisation.

Each cell's `slug` is one directory name under `--output-root`, and it is
checked as one: non-empty, no separator in either spelling, never `.` or `..`,
never absolute or drive-qualified, and no control characters. Two cells may
share neither a `(provider, model)` pair nor a slug — the pair is what the
allocation, the price, the configuration identity and the approval are all held
under, so a duplicate would collapse two cells into one approval.

## Money

One global hard cap is split into exact per-model allocations. The parts must
equal the whole **exactly** — a short split silently under-spends the approval,
an over-split spends more than was approved, and neither is a cap. There is no
borrowing between models.

The split is proportional to each model's worst-case unit cost, because a flat
split would starve the two expensive models while leaving the cheap ones with
headroom they cannot use. Shares are floored to the cent and the whole remainder
goes deterministically to one model, so the sum is exact by construction.

Each model runs in its **own** run directory, with its own `configuration_id`
and its own execution identity. Two cells sharing a directory is refused: one
directory is one run's immutable evidence, and two runs sharing one produce a
ledger whose rows cannot be attributed to the model that produced them.

## Credentials and authorisation are operator approvals

**This build authorises nothing.** Both of the things that make a run cost money
are supplied from outside the repository, at runtime, by an operator.

*Credentials.* Each provider reads exactly one environment variable —
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY` — and
nothing falls back to an SDK's own resolution order, an on-disk profile or a
second variable. They come from a single external, root-only environment or
policy file supplied at runtime. The matrix plan records only the **name** of
each variable and a **boolean** for whether it is set; it never reads, stores or
renders a value, so the artefact can be committed and attached to an approval.

That single-variable rule is load-bearing for the xAI adapter in particular: it
uses the OpenAI *SDK*, which would read `OPENAI_API_KEY` from the environment if
it were not given a key explicitly. A run must not authenticate to xAI with a
key meant for OpenAI, or send one to xAI's endpoint.

*Authorisation.* `AUDITED_CONFIGURATIONS` is empty in this build and empty by
construction. A live run requires the operator to export
`BOUNDARYBENCH_AUTHORISED_CONFIGURATIONS` with identities their own audit
approved, and `BOUNDARYBENCH_LIVE_CONFIGURATION` naming which one this
invocation is. A `MatrixAuthorization` binds an exact provider, model,
configuration identity, episode cap, cost allocation and output directory, and
defaults to `live_enabled: false` — writing an approval down is not the same act
as granting it.

Endpoints are pinned and redirection is refused rather than ignored: an operator
who set `OPENAI_BASE_URL` meant it to take effect, and a run that quietly
disregarded it would record a provenance nobody agreed to while sending the
credential somewhere the manifest does not name.

## Every completed model action is in the trajectory

A turn is one attempted adapter call, and a schema-valid action the environment
accepted spends that turn and records exactly one trajectory step. So a row's
`turns_used` and its step count are two readings of the same episode: for a run
that completed, and for one that stopped on its turn or message limit, they are
equal, and the ledger refuses a row where they are not.

That holds for a question the model asks twice. An acquisition reveals at most
once — the second `ask_user` or `call_tool` for a key the episode already holds
returns the same observation handle, delivers no new observation, and changes no
state — and it is still recorded, as a step naming the action and the key it
requested, revealing nothing and mutating nothing. The alternative, dropping it,
would spend a turn that appears nowhere in the evidence.

Grading and replay read the same rule from both sides. A zero-reveal acquisition
step is accepted only where the handle it would have delivered was already
revealed; a repeat that claims to reveal that handle a second time, and a first
request that claims to reveal nothing, are both refused. Neither the offered
query keys nor the channel each one arrives on is relaxed by any of this: a step
is still held to the one channel the Cube's registry assigns its key.

## What never becomes durable

No provider message, response body, header, request id, exception string, tool
name, argument or model identifier from a response reaches a ledger row. A
durable failure detail is a fixed sentence plus closed-set values this build
chose — a fault name, an HTTP status, an attempt count. This is not a debugging
convenience being withheld: a ledger row is evidence an audit reads, and text
that arrived from outside the build does not belong in one.

In every Lifecycle provider lane — Anthropic, OpenAI, xAI and Mistral — a typed
SDK HTTP-status exception is recorded as an attempt with
`response_received=true`. That means an HTTP response reached the SDK; it does
not mean this build accepted a model response. Response model, model-response ID,
usage, stop classification and normalized-response digest therefore remain
absent, while transport failures record `response_received=false`.

Anthropic documents HTTP 400 as covering both invalid requests and an
organization or workspace spend limit
(<https://platform.claude.com/docs/en/api/errors>). The Anthropic lane therefore
uses a provider-owned closed taxonomy: an exact locally allowlisted
spend-limit sentence is `provider_spend_limit`; a structured
`invalid_request_error` that does not match that sentence is
`provider_invalid_request_or_spend_limit`; and an unrecognized or malformed 400
body is `provider_bad_request_unclassified`. Arbitrary provider prose is never a
fallback. Request IDs remain deliberately omitted rather than being encoded in
the field reserved for an accepted model-response ID.

BoundaryBench ledger schema 7 remains historical and frozen. Its independent
eight-fault vocabulary and retryable subset do not import the mutable Lifecycle
taxonomy; the shared arrival meaning does not add Lifecycle-v3-only fault names
to BoundaryBench evidence.
