# Offline scorecard tracking (v1)

`tools/track_scorecard.py` tracks **assigned trials**, not a new benchmark, grader,
ranking, dashboard or weighted score. It consumes an explicit private manifest,
existing artifacts/ledgers, and existing original evaluation reports. It never
runs an agent, calls a provider, settles a budget, edits frozen evidence or replays
an episode. No historical model results or operator paths are shipped here.

## Run

Use an existing Python environment with the dependencies required by each original
reader checkout. No install/fetch or automatic historical-source fallback occurs.
From this repository, on Linux:

```sh
# PYTHON is an absolute path to that environment's Python.
# MANIFEST is an absolute path to a private manifest; OUTPUT must not exist.
env -i PATH=/usr/bin:/bin HOME=/tmp \
  unshare -n "$PYTHON" -B tools/track_scorecard.py "$MANIFEST" "$OUTPUT"
```

`unshare -n` requires appropriate OS permission. If unavailable, arrange equivalent
network isolation; Python socket guards alone are **not** a security sandbox.
The tool invokes trusted local Git and original reader code, not arbitrary source
that should be treated as hostile. Child environments contain no inherited API
credentials, and bytecode writes are disabled. The manifest is operator-owned:
review and trust every `reader_root` and its original source/dependencies before
invocation. HEAD/clean-source checks establish identity, not code safety; selecting
an untrusted reader can execute its Python code. Evidence and metrics paths are
read as data, never evaluated as extraction code. Shared installed dependencies are
not a hermetic historical dependency lock. The destination is a new mode-0700
directory; existing output is refused. Inputs are SHA-256 checked before and after
validation. Keep generated output private: it contains source paths and identifiers.

## Manifest contract

The following is a **shape example**, not runnable evidence or a real result:

```json
{
  "schema": "operatebench.scorecard-manifest.v1",
  "runs": [
    {
      "trial_id": "operator-assigned-trial-label",
      "execution_run_id": "exact-id-from-the-original-ledger",
      "source_commit": "full-original-source-git-commit",
      "reader_root": "original-source-checkout",
      "model": "exact-ledger-model",
      "profile": null,
      "scenario": "V1",
      "rubric": "original-rubric-label",
      "ledger": "trial/execution_ledger.ndjson",
      "artifact": "trial/episode_artifact.json",
      "metrics": "original-report.json",
      "metrics_keys": ["cells", "operator-cell-label"],
      "partial": null,
      "preserve_report_milestones": false
    }
  ]
}
```

All relative paths resolve against the manifest's directory. `metrics_keys` is an
optional sequence of exact dictionary keys/list indexes selecting the original
report object containing `evaluation`. No expressions or extraction code execute.
The **entire** evaluation must equal the original reader-validated artifact.
Provider artifacts require successful original bundle validation against their
ledger. Deterministic controls have null execution IDs/ledgers and a `model`
equal to the artifact's agent ID. Scenario, model, execution ID and request profile
are checked against retained evidence. `trial_id` and `rubric` are explicit
operator assignment labels. The reader checks their structure, not who assigned
them or when. Store the manifest with the original assignment/protocol records.

Each row supplies its own `reader_root` and exact `source_commit`. The checkout
must have that HEAD and a clean `src/`. A fresh subprocess imports only that
checkout's `src` via PYTHONPATH. The output records the actual reader engine
version, SHA-256 of artifact/ledger/bundle reader files, source commit, tracking
tool SHA-256, input hashes and individual validation outcomes. These checks are
artifact reading and ledger/bundle auditing, **not causal replay**. Historical
cohorts must use their matching original readers; never repin them to current code
or rerun/regrade frozen trials to make a scorecard work.

Use null references for genuinely absent retained outputs, including excluded
trials. An explicitly provided path that does not exist is an error, not silently
converted to an unknown row. An invalid reader, mismatched report or identity
aborts generation; it is not quietly dropped from a denominator. Trial IDs,
non-null execution IDs and repeated artifact/ledger bytes are globally unique in
one manifest. Copied bundles under different paths/labels are rejected.

## Output semantics

The stable JSON schema is `operatebench.scorecard.v1`; CSV has one row per
assignment and encodes **each cell as JSON** (literal `null`, Boolean `false`, and
separate nested cost/dimension objects survive round trips). Markdown contains the
summary and the same per-row records. Files are `scorecard.json`, `scorecard.csv`
and `scorecard.md`.

- **Assigned** includes exclusions and missing output. Summary counts are raw
  counts, not conditional percentages. There is no implicit comparable-cohort
  pooling, dimension average, action-count progress or efficiency-per-success.
- **Execution** retains terminal status/exclusion code separately from call
  terminal reasons, attempt fault codes and the original call cap. A local
  `provider_budget` stop is not a provider server error. The tracker does not
  guess reservation-admission versus model-call exhaustion from cohort names.
- **Business completion observed** is the original legitimate-completion Boolean
  for a retained valid complete artifact, otherwise null. **Completion verified**
  is an evidence indicator: false when completion was not verified, including
  absent evidence. It does **not** mean unknown business completion was disproved.
- **Evidence eligible**, original `reliable`, dimensions, and business outcome are
  separate. Excluded/partial/missing rows have null reliability/dimensions, never
  fabricated false/zero grades. Retained noncompletion can legitimately be false.
- **Costs** preserve original ledger totals verbatim, including measured dollars,
  reserved dollars and forfeited/uncertain reservations. They are never added into
  a claimed provider bill. `cost_scope` distinguishes a complete ledger from a
  validated prefix; unretained tail spend remains unknown. No ledger means null,
  not zero. No accounting or budget-settlement logic is changed.
- **Latency** sums retained provider-attempt latencies only if all are known; any
  unknown attempt makes the sum null. It has the retained ledger scope, not whole
  episode wall time. `wall_time_seconds` is always null: neither simulated duration
  nor provider-attempt latency is relabelled as wall time.
- **Milestones** default to null. Opt-in `preserve_report_milestones` retains the
  selected report's `milestones_retrospective` vector, checking only its Boolean
  earned/receipt-list shape and consistency. Supply only previously causally
  validated report vectors; the tracker does not independently prove receipt
  joins or fabricate milestone predicates. Provenance explicitly says
  report-supplied, retrospective, not recomputed or causally revalidated. No
  Maintenance-specific chain is silently generalized to other domains.

## New partial sidecars

An optional `partial` reference is a **NON-SCORED diagnostic prefix**, never a
complete artifact, metrics source or reliable/pass signal. The v1 diagnostic
reader validates complete newline-terminated JSON records, header schema/non-score
marker, record shape/order and identity anchored to the retained ledger/artifact.
It ignores only the final unterminated line, and reports that truncation. It
reports record counts and closure presence, **not** state/receipt semantic truth,
causal completion, milestone credit or grades. No closure can make it eligible.
This is deliberately labelled framing/shape/identity validation, not replay or
cryptographic/causal sidecar validation. A standalone sidecar without a retained
identity anchor is refused. No old sidecar is synthesized for frozen trials.

## Verification

```sh
PYTHONPATH="$PWD/src:$PWD" "$PYTHON" -m pytest tests/test_track_scorecard.py
"$PYTHON" -m ruff check tools/track_scorecard.py tests/test_track_scorecard.py
"$PYTHON" -m mypy tools/track_scorecard.py
```

Tests use small offline synthetic controls and real artifact/ledger readers. They
cover two successful reference controls, observed noncompletion versus unknown,
excluded/partial/missing evidence, unchanged input bytes, duplicate identities and
bundles, model call budgets versus provider errors, missing latency and incomplete
ledger scopes. Synthetic fixtures are not scientific model results. Regenerating
private historical scorecards is optional and requires an external manifest and
matching original source; it never modifies prior files. Future weights,
statistics, new milestone collection and budget/performance changes are out of
scope.
