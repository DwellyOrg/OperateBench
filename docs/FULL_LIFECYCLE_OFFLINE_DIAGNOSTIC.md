# Full-lifecycle offline diagnostic

`tools/full_lifecycle_diagnostic.py` is the private Slice 2B execution-integrity
diagnostic. It contains five `implemented_offline` cells: two Anthropic models,
one OpenAI model, one xAI model, and `mistral-small-2603` through the native
Mistral Chat Completions SDK lane.

The fixed one-cell Sonnet operator is
`tools/run_lifecycle_v1_anthropic_sonnet_canary.py`. It is **offline-verified only**
through the real Anthropic SDK over `httpx.MockTransport`. Its USD 2 input / USD
10 output per million token rates were verified on 2026-09-07 against Anthropic's
[model overview](https://platform.claude.com/docs/en/models/overview) and
[pricing page](https://platform.claude.com/docs/en/about-claude/pricing). The
fixed envelope is 46 expected and 50 maximum calls, a 25,884-token maximum native
input bound, 4,096 reserved output tokens per call, a 1,379,080 expected-call
ceiling, a 1,499,000-token hard cap, exact USD 4.6364 worst case, and USD 4.68
retained fixed cap. No live Sonnet cell has been executed.
This status makes no model result or score claim.

The repeat-ready fixed OpenAI Luna operator is
`tools/run_lifecycle_v1_openai_luna_canary.py`. It is **offline-verified only**
through the real pinned OpenAI SDK over `httpx.MockTransport`; no live Luna cell
has been executed. All 46 native Responses requests completed, followed by
Artifact 8, execution-ledger-v3, bundle audit, and zero-provider-call replay.
Its operator-pinned USD 0.20 input / USD 1.20 output per million-token envelope
has a 27,241-token maximum native input bound, 1,441,502 expected-call ceiling,
1,566,850-token hard cap, exact USD 0.518170 worst case, and USD 0.52 minimal cent
cap. The rates are operator controls recorded on 2026-09-08, not a current-price
claim. This status makes no model result or score claim.

The fixed one-cell Mistral operator is
`tools/run_lifecycle_v1_mistral_small_2603_canary.py`. It is implemented and
**offline-verified only** through the native pinned Mistral SDK over
`httpx.MockTransport`; no live Mistral cell has been executed. All 46 native
requests completed, followed by Artifact 8, execution-ledger-v3, bundle audit,
and zero-provider-call replay. Mistral's USD 0.15 input / USD 0.60 output per
million-token standard rates were verified on 2026-09-07 against the
[dated model page](https://docs.mistral.ai/models/mistral-small-4-0-26-03),
[API guide](https://docs.mistral.ai/api),
[OpenAPI document](https://docs.mistral.ai/openapi.yaml), and
[pricing page](https://docs.mistral.ai/inference/pricing). The operator uses the
dated `mistral-small-2603` identifier, not the moving `mistral-small-latest`
alias. The model page describes a 256k text context without publishing an exact
integer here; the operator's 4,096 output-token reservation is this benchmark's
cap, not a claim about the model's official output ceiling. The request omits
`reasoning_effort` because the absent-field default remains unresolved, and
omits `service_tier` rather than opting into Global Standard. Its fixed envelope
has a 27,331-token maximum native input bound, 1,445,642 expected-call ceiling,
1,571,350-token hard cap, exact USD 0.32786250 worst case, and USD 0.33 minimal
cent cap. This status makes no model result or score claim.

The operator's `provider_calls_dispatched` counts SDK request attempts at the final
provider boundary, immediately before `messages.with_raw_response.create`; an
attempt still counts when its transport fails, while earlier refusals do not.
`network_access` is a separate transport-boundary fact.
MockTransport does not count as network access, including when it is injected
into the live path for an offline test.

Run it without provider credentials or network access:

```console
uv run python tools/full_lifecycle_diagnostic.py --offline-verify \
  --output-dir /var/tmp/operatebench-full-lifecycle-diagnostic
```

Each cell runs one wholly synthetic Maintenance V1 episode through an in-process
`httpx.MockTransport` and its real pinned SDK. The custom `MockTransport` is a
trusted offline test boundary: client-state admission closes the reachable
endpoint, retry, SDK-hook, and HTTPX request-state channels, but does not claim
to inspect transport internals. Any future supported live builder must own and
pin its transport rather than accepting caller-supplied plumbing. A successful
report means each cell made 46 mocked calls, wrote Artifact 8 and
execution-ledger-v3 evidence, passed bundle audit, and replayed with zero further
transport calls.

This is execution-integrity evidence only. It is not a live-provider result,
model capability claim, matched-arm comparison, ranking, benchmark publication,
or production-support statement. `--live` always refuses before environment
reads, client construction, output creation, or network access. In particular,
the Mistral offline cell does **not** claim live Mistral support or vendor
verification of the pinned request profile.
