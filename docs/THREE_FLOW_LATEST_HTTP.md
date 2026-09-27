# Explicit latest HTTP roster

Historical readers and `ROSTER`, `MODELS`, `registry()`, `assignments(id)` remain
legacy-v1 (14 models / 616 initial slots). Nothing migrates existing series,
financial authority, trial identities or evidence.

New external admissions may explicitly select `roster_profile: latest-http-v1`
with `mode_profile: off-or-minimum-v1`. Use `registry(roster_profile=...,
mode_profile=...)`, `assignments(id, roster_profile)`, or the non-consuming
`proposal(source_sha, roster_profile)` to describe that source-bound selection.
Existing admission, source, budget, custody and approval checks still apply.

This roster includes ten River models, Claude Fable
5.1 and Mistral Medium 3.5, and selects these confirmed OpenAI/Claude models:

| API model | Thinking | Context | Max output | Standard input / output USD per million |
|---|---|---:|---:|---:|
| gpt-6-luna | OFF: reasoning.effort=none | 1,050,000 | 128,000 | 0.10 / 0.50 |
| gpt-6-sol | OFF: reasoning.effort=none | 1,050,000 | 128,000 | 2 / 10 |
| gpt-6-astra | MINIMUM: reasoning.effort=low | 1,050,000 | 128,000 | 10 / 50 |
| claude-opus-5-5 | MINIMUM: adaptive thinking, effort=low | 1,000,000 | 128,000 | 4 / 20 |

Opus 5.5 adaptive thinking cannot be disabled. Existing Fable keeps adaptive
thinking / low effort. The actual resolved model reaches the official SDK and
its response-model validator; SDK retries remain disabled. OpenAI non-strict
function schemas and reasoning metadata validation are unchanged.

Terra and Qwen/Qwen3.5-122B-A10B-FP8 are not selected by `latest-http-v1`.
The legacy registry and model implementations remain available unchanged.
The River-only owner output cap mechanism is unchanged (not applied to HTTP).

## Pricing safeguards

Standard prices above are not conservative envelopes. OpenAI cache read is
0.1x input, cache write 1.25x; input/cache doubles above 272K prompt tokens,
output multiplies by 1.5; fast mode multiplies applicable rates by two.
Sol/Luna document a regional 1.1x premium. Opus 5.5 cache read is 0.20,
5-minute write 5, 1-hour write 8; fast mode doubles prices and US inference
multiplies by 1.1. Fable cache write is up to 20, with US inference x1.1.

The latest roster's minimum input/output envelope checks conservatively retain
these published tier maxima: Luna 0.55/1.65, Sol 11/33, Astra 50/150,
Opus 17.6/44, Fable 22/55. They do not enable fast or regional requests.
Higher known/owner-reviewed rates are never lowered. Owner-assumed unknown
rates, usage-risk consent and profile/source/registry digest checks remain
separate. Public documentation is not account access or billing verification.

Sources:
- https://developers.openai.com/api/docs/models/gpt-6-luna
- https://developers.openai.com/api/docs/models/gpt-6-sol
- https://developers.openai.com/api/docs/models/gpt-6-astra
- https://platform.claude.com/docs/en/models/overview
- https://platform.claude.com/docs/en/build-with-claude/effort
- https://platform.claude.com/docs/en/about-claude/pricing

`tests/test_three_flow_latest_http.py` uses actual SDK MockTransport business
trials and zero-provider replay for every selected HTTP model, wrong-response
model rejection, source/roster/price mutation rejection, and the real capability
factory plus grouped campaign reports. These are synthetic mechanics tests,
not paid availability or model-performance claims.
