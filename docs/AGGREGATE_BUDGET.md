# Optional settled aggregate budget (offline activation only)

`tools.aggregate_budget` is repository-supported optional controller code, not
part of the installed wheel CLI. `tools.diagnose_aggregate_budget` is its explicit
six-interface offline diagnostic. It uses the bounded runtime and fixed
interface profiles. The diagnostic accepts no live authorization and does not
change the fixed operators.

From a source checkout with the project dependencies installed:

```sh
unshare -n env -i PATH=/usr/bin:/bin HOME=/nonexistent \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD" \
  "$VIRTUAL_ENV/bin/python" -m tools.diagnose_aggregate_budget \
  --output /tmp/operatebench-new-offline-budget
```

The output directory must not exist. The CLI creates an owner-only directory and
permanent dummy-only consumption marker before constructing SDK clients. All six
clients use dummy keys and HTTPX MockTransport; internet sockets are additionally
refused. There is no live flag, credential lookup, resume, migration, refund, or
financial-release command. Use a fresh output name for each diagnostic. The
controller API assumes a trusted in-process single controller and private local
filesystem, not adversarial Python callers or a network filesystem.

## Accounting contract

The exact Fraction invariant is `known + pending + unknown <= cap`. Reserve and
wire-supplement admission, settlement, unknown forfeiture, and proven prewire
cancellation are serialized only for journal transitions. The append, file fsync,
and directory fsync complete before reduced exposure becomes visible. I/O failure
freezes admission. Recovery is audit-only, even for a complete journal; no old
journal grants renewed spend authority. Each cell has one active admission. Each
HTTP dispatch consumes it exactly once, admits any nonnegative serialized-wire
supplement before calling the inner transport, and releases the controller lock
before the independent HTTP request. SDK retries are disabled.

Validated integer usage exceeding the admitted input/output bounds is a specific
`SettlementBoundsError`, not an arbitrary `ValueError`. The adapter durably closes
it as unknown, then raises the existing `CostReservationBreachedError` with null
measurement. The executor records a forfeited attempt; the model boundary maps
this dedicated breach to the existing `provider_budget` exclusion and preserves
the breach as its exception cause. The runner completes the excluded ledger and
closes partial evidence against that ledger's terminal and digest. It does not
invent a transport fault or a pre-dispatch refusal for the answered request.
Internal ValueErrors are not translated by the accounting guard.
Missing or malformed usage never releases the bound. Accepted validated zero
usage can settle to zero; complete truthful billable counters, conservative
request bounds and applicable fixed rates are assumptions, not invoice assurance.

### Two explicitly different evidence projections

`successor-events.jsonl` is the authoritative aggregate accounting journal. Its
profile is `settled-aggregate-bounded-v1`. Reservations retain `initial_bound` and
`initial_input_bound`; dispatch records the complete wire-adjusted bound and wire
payload digest. Unknown retains that entire bound. Per-cell outcomes include
`budget_accounting` with the records and explicit reconciliation:

`legacy_forfeited_usd + wire_supplement_forfeited_usd = full_unknown_usd`.

Ledger 3 remains an **initial-reservation projection only**, not the aggregate
account. Its forfeited amounts intentionally match its own recorded initial
bounds. Do not present its totals alone as full aggregate exposure. Aggregate
summary totals and the journal, not that legacy projection, own settled-budget
accounting. The diagnostic retains normal complete bundle audit/replay on success,
and a strict-reader-verified complete excluded ledger with closing partial
sidecars on an over-bound SDK response. Other failures retain their existing
runner evidence guarantees. Summary files are
convenience projections, not additional crash-resumption authority.

## Version decision and activation boundary

No engine, evaluator, request mapping, model output ceiling, call/turn/token or
wall-clock profile is relaxed or bumped. Successful execution uses engine 0.12.0
and existing Artifact 8 / ledger 3 records. Existing measured-breach guards keep
their accounting behavior; dedicated reservation breaches now reach the existing
budget exclusion rather than an incorrect transport exclusion. No durable
taxonomy or ledger validation changes are needed: `provider_budget` already
represents this build's own bounds, and the answered/forfeited attempt stays
truthful. The additional nullable breach measurement is used only by the
new optional controller. This is an independent accounting convention, explicitly
identified by `settled-aggregate-bounded-v1` in journal genesis and diagnostic
summary/outcomes, not authority to reuse an old run identity for a new experiment.

Live operator integration and fresh exact-source launch authority are **not
implemented** here. Any future activation requires a separately reviewed binding
of this controller profile, current exact source, fixed interfaces, pricing and
fresh authorization. No live evidence, full-series readiness, parent verification,
or invoice-cap guarantee is claimed. The diagnostic writes only to its new
output directory; it does not rewrite retained journals or artifacts.
