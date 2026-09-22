# Six-model episode100 candidate (NOT paid authorization)

Controller `synthetic-demo-episode100-v1`, agent `synthetic-demo-model-episode100-v1`.
The existing corrected engine and ledger versions are unchanged; no core source
or historical operator constants are rewritten. The opt-in spec has its own
recomputed content digest, not the historical canary's digest.

## Count contract

- At most **100 model response decisions over the entire episode per model**.
  RETRIEVE and malformed structured responses count. Normal completion can occur
  earlier; this does not force 100 calls after a terminal business outcome.
- The engine's canonical `turn_index` counts business turns per invocation;
  retrieval has a separate budget. It is NOT the requested global counter.
- The episode-lifetime recorder enforces 100 calls, and the agent independently
  enforces 100 sends. With retries zero and provider faults terminal, this bounds
  decisions and actual API requests without free retrieval loops. Outcome JSON
  reports `decision_turns`, `api_requests`, `retries` separately. Ledger call
  coordinates preserve invocation/turn identity.
- The profile raises business turns per invocation to 101 **only so the global
  recorder can refuse before a scored invocation-limit failure**. This is not a
  100-turn-per-invocation contract. Authored retrieval-service refusals, business
  deadlines and episode horizon remain in effect.
- Ledger max calls is 100, never the historical 50. The required finite legacy
  token cap is derived as ceil(100 * NEW_CAP * 1e6 / min(input_rate,output_rate));
  it is redundant given admission and the call bound, not a reused native-input
  estimate. Per-request output cap and existing per-turn time deadline remain;
  wall deadline is explicitly 100 times that deadline. These are bounded runs,
  not an unlimited-time or unlimited-output promise.
- At exhaustion: complete `excluded/provider_budget` ledger and NON-SCORED
  partial evidence; controller cause `episode_decision_limit`. No fabricated
  scored failure/artifact. A 503 on API request 100 after 99 responses remains
  `provider_transport`, not a decision-limit attribution.

## Real execution path

`python -m tools.run_episode100` selects MockTransport for `--offline` and
`--mock-authority`; `--live` selects HTTPTransport(retries=0) only after external
one-shot consumption and credential parsing. All modes then use the same six
SDK clients, dispatch reservation wrapper, SharedGuard, recorder and runner.
Cells: haiku45, sonnet5, luna56, grok45, grok46, mistralsmall; exact models and
request settings remain in their existing operators. Scheduling is independent
six-worker execution; the shared accounting lock does not span network calls.
Known + pending + unknown wire-adjusted exposure is bounded by the explicit cap.
Unknown charges are retained, not silently refunded.

## Activation prerequisites — currently UNSATISFIED

1. Owner approves a **new dollar ceiling**. A request for 100 turns is NOT dollar
   authorization. No former financial ceiling, reservation or consumption is reused.
2. Parent independently verifies the final clean source SHA and tests, and the
   changed count/spec semantics and retained time/output bounds are accepted.
3. Owner retires outstanding historical/sibling authorities. This is an explicit
   operational prerequisite: unchanged historical launchers do not share the new
   consumption marker. This candidate does NOT mechanically revoke them.
4. An external owner-only directory (0700) receives a fresh approval JSON (0600,
   regular, single-link). No real seal, nonce or approval is shipped here.
   Exact fields: profile, source_sha, output (new absolute path), cap_usd (exact
   decimal text), cells (ordered list above), max_episode_decisions=100, retries=0,
   approved=true, historical_authorities_retired=true, mode="paid",
   credential_file (canonical absolute 0600 JSON file mapping all six cell IDs to
   their keys). Credential custody is checked without reading; an O_EXCL
   directory-wide EPISODE100-CONSUMED receipt and directory fsync precede reading.
   Failure does not reset it. This is trusted owner-file authority, not a
   cryptographic signature service or invoice reconciliation.

Non-authorizing template; **never execute before all prerequisites**:

```sh
env -i PATH=/usr/bin:/bin PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 "${EPISODE100_PYTHON:?set the absolute path to the verified Python interpreter}" -m tools.run_episode100 --live --authority /ABS/NEW-OWNER-AUTHORITY/approval.json --output /ABS/NEW-OUTPUT --cap-usd NEW_EXPLICITLY_APPROVED_USD
```

Run from the exact approved checkout. Offline rehearsal uses `--offline` instead
of `--live`, no authority argument, a new output directory and an explicit
simulation cap. Dummy consuming tests create only placeholder credential JSON;
real credential values must not be inspected during preparation.
