# Meta Muse Standard snapshot comparison

The candidate supports exactly `muse-wonder-1.0` and `muse-spark-1.3` in
Standard mode through two distinct integrations. Both use independent observation
snapshots for an equal-scaffold comparison. Neither claims Meta-native reasoning
continuity or full-capability agentic performance. The original evaluator and
grades are unchanged. All current implementation evidence is offline;
**no paid inference, model results or ranking exists yet**.

## Standalone Lifecycle adapter

`operatebench.agents.meta_responses.MetaResponsesTransport` implements
`send(ModelRequest) -> ModelResponse` using the separate
`lifecycle_meta_responses_snapshot_v1` mapping. It uses Core's existing
non-strict function schemas, common executor, finite turn deadline and scaffold
output ceiling of 4096 tokens. That ceiling is policy, not a published Meta
output maximum. Reasoning remains provider-default in this adapter.

The factory pins `https://api.meta.ai/v1`, accepts only `MODEL_API_KEY`, disables
SDK retries and rejects endpoint overrides and unsupported models. Local
credential validation cannot establish account access. Requests carry fixed Core
instructions and one user snapshot, with `store:false`; reasoning items are
validated and discarded as non-actions. Output usage already includes reasoning
and is never charged twice.

Lifecycle execution ledgers and deterministic Core replay work offline. Its
frozen generic artifact reader still rejects Meta-bound bundles. This standalone
adapter is not the campaign route.

## Added 44 × 2 candidate campaign

The `meta-standard-same44-v1` roster adds 88 slots: the same 44 public synthetic
cases for each Standard model, across maintenance, commerce and compliance.
The shared campaign registry and Meta HTTP bridge use explicit
`reasoning.effort:minimal`, validated Responses projection and the unchanged
evaluator. The campaign has its own durable `three_flow_runtime` records and
replay; it does not depend on admitting Meta to the Lifecycle artifact reader.

Campaign requests dynamically bound output by remaining published context and
available money, rather than the Lifecycle 4096 ceiling. There is no generation
wall deadline or read/write timeout; connect and pool timeouts remain finite.
The unknown native output maximum is retained as unknown. Snapshot turns do not
replay encrypted reasoning, previous response IDs or native tool-result state.

Two disjoint persistent budgets sum to the user's **USD 1000 hard total**:
**MAX USD 10** for at most a few real Meta API compatibility probes (not scored),
and **USD 990 shared across both models** for all 88 scored campaign cells,
campaign retries and unknown exposure. The total includes provider diagnostics
and retries; it is authorization, not a spending target. The parent operator
maintains separate durable diagnostic and campaign ledgers and verifies totals;
neither budget may be reset, reused or transferred to the other.

Campaign reservations and accounting persist across restarts. Unknown reserves
are irreversible and continue consuming the campaign budget. Estimated spend
under the admitted pricing and reported-usage assumptions is not an invoice
guarantee. It does not establish provider tariff coverage, account access or
billing completeness.

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

The parent must commit the reviewed candidate and run the source-bound gates,
including the full 88-slot mocked consume, against that clean source. That
clean-source qualification and live endpoint acceptance remain pending. Final
verification must be reported against the resulting source binding; this snapshot
does not claim those gates have passed.
