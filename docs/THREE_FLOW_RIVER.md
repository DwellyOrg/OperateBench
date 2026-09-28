# River three-flow offline transport

The optional River route requires Python 3.12 or newer and published
`river-client==0.12.0`. The ordinary package and HTTP suite still support
Python 3.11 without River installed. No SDK or tokenizer assets are vendored.

`python -m tools.three_flow_campaign offline --providers river --output <fresh-directory> --river-assets <asset-cache>`
executes the real Maintenance, Commerce and Compliance engines through the
installed SDK's generated `RiverServiceStub` and protobuf serializers. The
channel mock decodes the native rendered request, gives only its public
observation to the domain reference agent, and returns native model-family
syntax. These are **reference-driven SDK mock controls, not LLM results**.
The command never creates a network channel or loads credentials.

The eleven exact River identities are fixed in the campaign roster. Tokenizer
revisions, public asset URLs and SHA256 hashes are in
`tools/three_flow_river_assets.json`. Its `model_slug` field is the public catalog
identifier, not a credential. Put those files under
`<asset-cache>/<tokenizer_model-or-model>/<filename>`. Only pinned data is read:
Rust tokenizer JSON, sandboxed Jinja templates and Kimi tiktoken ranks; no
`trust_remote_code` or downloaded Python is executed. Strict native parsers
retain typed JSON arguments. Native-family reserved-string domains are
announced in the prompt; ambiguous interfaces are excluded, while malformed
model syntax remains an invalid model decision. No domain is cast to Maintenance.

## Public asset provenance policy

The release checker permits immutable Hugging Face revision and URL values only
in the exact `tools/three_flow_river_assets.json` path. This is a closed JSON
schema: duplicate keys, extra fields, missing or duplicate roster rows, unknown
asset names and mismatched model/model_slug/tokenizer repository bindings receive no
exception. Each asset requires a lowercase 64-hex SHA256, a positive integer byte
length, string HTTP status `200` and integer exit status zero. These metadata
checks validate structure, not downloaded bytes or hosted model weights.

Only the validated revision and URL value spans are exempt from the commit-token
rule. URLs must exactly use HTTPS, `huggingface.co`, the row's tokenizer repository
(or model repository), `resolve`, the row's lowercase revision and the declared
filename, without queries, fragments or traversal. The independent publication
roster and file-name sets live in the checker, not in the candidate input.
The same rules apply in the working tree and archives; an sdist must first have
one safe root before its relative manifest path can qualify. Copies at other
paths, comments and unrelated identifiers remain rejected. Existing pinned
GitHub Action rules and the rejection of same-repository commit URLs are unchanged.
This exception changes no runtime asset bytes, settings or replay source binding.

Each actual `InferenceGenerate` crossing reserves and fsyncs its bound in the
same `CampaignBudget` used by HTTP, then journals dispatch before entering the
injected channel. Submission is never retried. Each `RetrieveFuture` poll is
captured as another network RPC against the acknowledged request; it is not
another generation or token charge. A per-RPC 120-second network timeout is
fresh for every RPC; there is no episode or hosted-generation wall limit.
Failures retain unknown liability, including failures after acknowledgement.
Restart never resubmits an interrupted slot. Corrections retain the campaign
journal and create separate trial and closure identities.

The adapter uses the installed generated stub rather than the high-level SDK
Client, whose session/retry machinery is unnecessary here. A trusted caller
must supply the channel with `grpc.enable_retries=0`; no default channel exists.
Request and response protobuf bytes, requested model, rendering digest, full
billable token counts, network RPC counts and current settings are captured.
The protobuf has no returned-model identity or stop-reason field: evidence
reports those limitations explicitly. Local token-cap classification is not
represented as a provider-reported stop reason. Replay uses only recorded
semantic decisions and requires the independently assigned Trial and retained
closure digest, making zero RPCs.

All real admission flags remain false. The CLI's output setting 32768 and
synthetic rates are offline fixture settings, **not verified endpoint bounds
or current prices** (including unsuffixed Kimi). DeepSeek V4.1's missing price
blocks real admission, not offline qualification. The campaign's
[authority boundary](THREE_FLOW_CAMPAIGN.md#authority-boundary) and
[consuming entrypoint](THREE_FLOW_CAMPAIGN.md#consuming-entrypoint) govern live use,
including account access, context/output limits and cache/reasoning/tier billing.

## Failure evidence

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
Error projections are `sanitized_grpc_error_v2` and `sanitized_river_failure_v1`;
[replay and historical evidence boundaries](THREE_FLOW_CAMPAIGN.md#replay-and-error-evidence) apply.

Run the separate River matrix with the optional SDK environment and explicit
`THREE_FLOW_RIVER_ASSETS=<asset-cache>`:

```sh
pytest tests/test_three_flow_river.py
```

Python 3.11 skips that optional matrix. Normal HTTP tests remain a separate gate.
