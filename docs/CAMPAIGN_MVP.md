# Maintenance V1 campaign MVP

This source-checkout CLI runs one sequential, balanced campaign using the existing
Maintenance V1 runner and strict grader. It does not change Artifact 8, ledger 3,
replay, or grading. It is a descriptive case study, not a leaderboard or an
estimate of statistical superiority. V2, resume, parallel execution and inference
intervals are outside this MVP.

## Offline quickstart

From the repository root, with the project dependencies installed into `.venv`
(`python3 -m venv .venv` then `.venv/bin/python -m pip install -e .`), copy this
single block. The fresh private output is outside the checkout. `env -i` excludes
ambient credentials and provider routing; both real SDKs use HTTPX MockTransport,
public placeholder keys, and a process-local network tripwire. No authorization
record or real credential file is needed or read.

The CLI and library use the same scoped tripwire during offline execution,
including setup and report publication. It denies ordinary Python IPv4/IPv6
connect/send and DNS operations and restores every patched function on exit,
including interruption or setup failure; it installs no permanent audit hook.
This is a process-local accidental-network guard, not a hostile-code/native-code
sandbox. While active it also denies other threads' internet socket operations;
do not embed an offline campaign alongside concurrent networking work.

```sh
campaign_output="$(mktemp -d /tmp/operatebench-campaign.XXXXXXXX)"
env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$PWD/src:$PWD" \
  .venv/bin/python -m tools.campaign run \
  --config examples/campaign-offline.json --output "$campaign_output/run"
```

The supplied config explicitly selects **two historical offline profiles**, Haiku
4.5 and Luna 5.6, in round-robin order for two repeats (four distinct trials).
These pins are not claims of current flagship status, availability or current
pricing. The synthetic provider follows the reference agent; its successful
scores are integration evidence, **not model capability measurements**. Costs and
usage in this mode are simulated, not provider billing evidence.

Each trial has explicit pinned model/settings, output-token, cumulative-token,
call, money, request-time and wall-time bounds. The config also supplies the
aggregate money ceiling and explicit rates/provenance. The aggregate journal is
authoritative for full wire-adjusted exposure; the legacy ledger retains its
initial-reservation projection. Initial reservations and any serialized SDK wire
supplement are admitted before HTTP dispatch. Unknown liability remains held in
both the trial and aggregate accounts, without charging it twice. SDK retries
are disabled.

## Evidence and reports

The output contains `plan.json`, immutable sequenced `status/` events, the
`successor-events.jsonl` accounting journal, per-trial evidence directories, and
`report.json` / `report.md`. The plan seals source, spec, ordered assignments,
profiles, rates and bounds before execution. Outputs are exclusive (0700 directory,
0600 files); an existing output directory is refused. No campaign is resumed.

Reports retain every assignment, including `not_started`, and show assigned,
started, scored and verified-reliable counts. Raw success fractions use all three
denominators; a zero denominator is undefined, not zero. Per-profile and total
counters include failures; units distinguish HTTP attempts, ACT decisions, and
RETRIEVE batches. Missing counters are marked and aggregate counters explicitly
count incomplete-counter trials. Terminal and strict evaluation dimensions remain
separate. No driver-side success rubric is added.

Known measured cost, pending exposure and unknown exposure are separate; exposure
is not actual cost. Partial and unfinished starts remain NON-SCORED, with null
reliability. Provider failures, unknown usage, interruptions and infrastructure
faults stop the campaign; a fully scored business failure may continue. Budget
exhaustion is not evidence of model inability. Later assignments remain
`not_started`, not failed model trials. Hard process/storage failures may leave an
unfinished start or unreadable evidence rather than a final report; retained
records are never repaired into success.

Regenerate to stdout without changing the campaign (same source files required):

```sh
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m tools.campaign report \
  --output "$campaign_output/run" --json
```

Regeneration validates plan and status hashes, legal status transitions, ledger
identity, terminal evidence hashes, and scored bundle audit/replay. Reports retain
SHA-256 digests of plan, status, accounting and trial evidence inputs. These are
local integrity/provenance checks, **not signatures against an attacker who can
rewrite the entire directory**. Keep the original report/input digests separately
if comparing later reconstructions. A partial cannot become a score merely by
editing its status. Torn or inconsistent records fail closed.

Campaign evidence/report schema is now `operatebench.campaign.v2`; the input
configuration remains `operatebench.campaign.v1`. The reader explicitly refuses
legacy v1 evidence (no accounting binding), rather than upgrading it to a trusted
success. Preserve old evidence unchanged and use its original source for historical
inspection only; it does not gain the v2 integrity guarantee.

Every terminal campaign status binds the durable accounting sequence length and
hash-chain tip. Reconstruction requires that exact prefix, so truncating only the
accounting file—even to a valid genesis or earlier prefix—is refused. Later valid
accounting records are allowed, retaining their pending/unknown liabilities.
Audit opens the journal read-only and still forbids resume.

An unfinished `started` trial has no terminal accounting checkpoint. Its financial
status is explicitly `unknown_incomplete_start`: measured cost and total exposure
are null, never an inferred zero. Campaign `financial_complete` is false and
`accounting` values are null; `observed_accounting` and per-trial observed measured/
exposure fields retain the available prefix. Pending/unknown exposure fields are
observed liabilities, not proof that no additional liability exists. Partial ledger
counters are labeled `lower_bound`; missing counters are `unknown`, and both count
in `incomplete_counter_trials`. Aggregate counters are lower bounds if any started
trial has incomplete counters. These safeguards detect local inconsistency, not
coherent rewrites of all evidence and its bindings.

Exit status: 0 means all assignments scored (not necessarily reliable); 1 means a
readable incomplete campaign; 2 means refusal or evidence failure. `--help` lists
`run`, `report`, and inert `proposal`. Proposal output is never authorization.

## Live execution is a separate decision

This implementation/test exercise authorizes no provider spending. Current model
availability, profile suitability, rates, aggregate budget and trial count require
separate owner approval. Historical canary approvals/journals are not reusable.
The new optional authority path binds a fresh exact plan, clean source, config,
profiles and output; its private one-shot consumption markers are fsynced before
credential contents are read. It does not create approvals. Dummy authority
rehearsals accept only public placeholders and offline mode. There are no paid
launch instructions or credential-store changes in this quickstart. Moving to a
new campaign after a stop requires fresh approval and reconciliation of prior
unknown liabilities; a fresh directory is not permission to reset spending.

Live config rates must be at least the respective profile module's fixed
historical `FIXED_CANARY_INPUT_USD_PER_MTOK` and
`FIXED_CANARY_OUTPUT_USD_PER_MTOK` floors. Underpricing is refused before authority
or credential access. Passing this conservative historical minimum is **not
current-price verification**: fresh owner-reviewed current rates must separately
be approved before any live launch, and higher applicable rates must be used.
The offline example's simulated rates are not live price guidance. The exact
declared rates and source pins are hash-bound in the plan; all execution config,
including per-trial limits, aggregate ceiling and accounting namespace, is taken
from that detached canonical plan, not the mutable caller config.
