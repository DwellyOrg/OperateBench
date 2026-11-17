# OpenAI 6.2 opt-in SAME44 profile

`openai62-standard-same44-low-v1` selects exactly `gpt-6.2-astra` and
`gpt-6.2-sol`, one observation per existing configuration per model (3
maintenance, 16 commerce, 25 compliance). Historical/default rosters are unchanged.
Use the existing `tools.three_flow_live` operator with `off-or-minimum-v1`;
transport policy is the omitted/default `legacy-v1`. This is a capability,
not authorization to run or a claim of account availability.

Requests use synchronous `/v1/responses`, `reasoning.effort=low`,
`service_tier=default` (Standard), no automatic retries, and explicit non-strict
function tools for the existing ACT payload and WAIT alternatives. Output is
bounded by 128,000 and remaining context; input is bounded by 922,000 within
1,050,000 total context. Existing Core liveness and fixture limits still apply.
The optional-money policy adds no model-call, token, or generation-time cap.
Connection and pool acquisition retain the existing 10-second liveness bounds.
The shared named, versioned response contract admits its closed inert root
metadata and cache-write TTL partition. Model identity, actions and unknown fields
continue to fail closed. For these two models, the raw and SDK response must both
report `service_tier=default` before usage is settled.

The closed published profile uses `openai62-standard-token-envelope-v1` with
reported usage accounting, unknown additional fees, and no invoice guarantee.
Category maxima (input, cache read, cache write, output), in USD per million,
are `(16, 0.8, 40, 60)` for Astra and `(4, 0.2, 10, 15)` for Sol. The controller
uses input envelopes 40/10 and output envelopes 60/15, covering long-context
12-hour cache writes. These are conservative Standard envelopes, not ordinary
measured rates or invoices. Original reported cache and reasoning categories
remain in raw evidence; reasoning is an output subset.

Ordinary Standard short-context rates (input/cache read/30m write/12h write/output)
are Astra `(8, 0.4, 10, 20, 40)` and Sol `(2, 0.1, 2.5, 5, 10)`.
Above 272,000 input tokens, full-request input/cache rates double and output
rates multiply by 1.5. No background, Fast, Ultrafast, or Batch discount is used.

Official sources:

- [Astra model card](https://developers.openai.com/api/docs/models/gpt-6.2-astra.md)
- [Sol model card](https://developers.openai.com/api/docs/models/gpt-6.2-sol.md)
- [Pricing](https://developers.openai.com/api/docs/pricing.md)

The existing operator requires external source-bound admission, exact selected
assignment IDs, profiles, runtime/registry identity, optional-limit consent,
private custody, and one-shot consumption. Its HTTP thread pool supports both
model lanes concurrently. Prepare and review authority externally; preflight
creates no authority. Freeze a clean reviewed source before using the real
operator's source identity gate.
