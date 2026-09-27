# Three-flow successor and safe error semantics

The controller uses `maintenance_delivery_recovery_v0_7.yaml` through the single
`tools.three_flow_runtime.SPECS` selection shared by campaign execution and replay.
Its operation identity is `lettings_maintenance_delivery_recovery_v2`; the original
maintenance YAML and historical artifacts are unchanged. Commerce and compliance
spec selections, business guards and scoring are unchanged.

## Version boundary, not historical migration

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

## What the installed River contract can prove

The reviewed `river-client==0.12.0` generated `RequestFailedResponse` has string
`error_category`, string `message`, and `google.protobuf.Struct details`. It has
**no typed failure-reason enum or numeric reason field**. A `details.code` or
`details.reason` entry is not a documented enum merely because it is a number or
string. These arbitrary values are not retained or interpreted. Presence of the
Struct is retained; finer cause remains `unknown`.

Actual installed gRPC `StatusCode` members are closed structured codes. Their name
and numeric value are retained (for example INVALID_ARGUMENT = 3,
RESOURCE_EXHAUSTED = 8), and classifications distinguish request rejection, rate
limiting, authentication, timeout, network and generic server failures. Unknown
status objects yield UNKNOWN with no numeric code. INVALID_ARGUMENT alone does
not identify which argument failed, establish context overflow, prove that no
generation occurred, or authorize a retry. No raw error body, text, headers,
trailers, reasoning, credentials or hashes of those values are diagnostic evidence.

Diagnostics expose these safe status names/numbers and the presence of
failed-response structured details. They cannot establish a finer numeric
River reason through this SDK contract. Supporting one requires a
provider-documented field/enum contract and a reviewed closed projection first.
This document and offline tests grant no provider access or launch authority.

## Pacing is not retrying

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
