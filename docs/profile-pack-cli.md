# International profile packs: development CLI

These two additive packs are development-only synthetic processes, not official
benchmark admission, Artifact 8 evidence, legal advice or new model ratings.
Runtime and grader semantics still require independent review and remediation;
a passing local check is not a claim of legal or semantic correctness. Existing
Maintenance, Commerce and Property Compliance IDs and aliases remain available.
No provider, credential or paid authority is added by this command lane.

## Bounded profile scope

| Process | Profile | Implemented fixture scope |
|---|---|---|
| Return/refund | `UK` | Ordinary informed distance cancellation in the selected UK profile |
| Return/refund | `DE` | Ordinary informed German distance withdrawal, not EU-wide coverage |
| Return/refund | `US_CA` | California purchase-time merchant policy for ordinary goods; unsupported statutory facts refer |
| Return/refund | `AU_VIC` | Selected Australian consumer-law major/minor fault stipulations in Victoria, not an automatic online cooling-off right |
| Property compliance | `EN_PRIVATE_ELECTRICAL` | Selected England private-rental electrical checks and remedial evidence |
| Property compliance | `SC_PRIVATE_ELECTRICAL` | Selected Scotland private-rental electrical/equipment checks |
| Property compliance | `AU_VIC_RENTAL_CHECKS` | Victoria rental electrical/gas checks, not Australia-wide rental law |
| Property compliance | `NZ_HEALTHY_HOMES` | Selected New Zealand Healthy Homes components and statement; no invented statutory certificate expiry |

There are two processes with four profiles each, not eight independent national
benchmarks. Scenario counts are fixture coverage, not process or country counts.
Standalone GB annual gas and NZ smoke profiles remain research-only. Technical
standards, exceptional cohorts and unsupported facts must not be inferred from
these fixtures. See the detailed scope, research sources and unimplemented
branches in [Commerce profiles](commerce-return-refund-profiles.md) and
[Compliance profiles](contributions/prospire/property_compliance_profiles.md).

## Public contract v2

The two profile domains expose `public_contract_version=2` in model-visible
policy, a domain-local marker, not an engine, package, provider mapping, fixture
or evaluator version. Producing source commit and implementation bindings remain required.
Compliance publishes canonical exact required payload fields, including empty
`{}`, no additional fields, non-empty string values and separate `evidence_refs`;
the same guidance accompanies PAYLOAD refusals. Commerce clarifies review ordering,
pending versus unknown payment, settlement versus finality, citations and one-time
notices. Refusal codes distinguish duplicate entitlement, notice/finality readiness,
already-notified status and review ordering/repetition from substantive mismatch.
These observation/diagnostic changes leave accepted business effects, action guards
and evaluator safety unchanged: a rejected proposal is not an applied harmful
effect but still makes the episode unsafe; successful recovery does not erase it.

Historical tapes, observations, findings and source bindings stay unchanged:
no relabelling as v2, rewriting, rescoring or in-place repair. Shared engine or
fixture versions do not imply identical domain implementation. Exact historical
replay needs the producing source and original public contract; current development
replay checks implementation binding and full episode agreement. New-build tests
prove only new-build consistency, not old-record replay compatibility. The marker
neither migrates artifacts nor adds a historical-runtime loader. Changed prompts
may change decisions without changing accepted business semantics: cross-version
results are not interchangeable or pooled. Model benefit requires separately
authorized fresh runs; offline payload-shape probes show boundary behavior, not
model improvement, and authorize neither provider execution nor evidence rewriting.

## Offline source-checkout commands

Run from the repository root after installing the project dependencies. The
operation-type alias is the pack ID without its final `.v1`.

```bash
uv run operatebench list-packs --json

uv run operatebench validate --pack commerce.return_refund.profiles.v1 \
    examples/operatebench/commerce_return_refund_profiles/operation.yaml
uv run operatebench run --pack commerce.return_refund.profiles.v1 \
    --spec examples/operatebench/commerce_return_refund_profiles/operation.yaml \
    --scenario UK_NORMAL --agent reference-mock --output commerce-profile-run.json
uv run operatebench replay --pack commerce.return_refund.profiles.v1 \
    --spec examples/operatebench/commerce_return_refund_profiles/operation.yaml \
    --run commerce-profile-run.json
uv run operatebench check --pack commerce.return_refund.profiles.v1 \
    --spec examples/operatebench/commerce_return_refund_profiles/operation.yaml

uv run operatebench validate --pack lettings.property_compliance.profiles.v1 \
    examples/contributions/prospire/property_compliance_profiles/operation.yaml
uv run operatebench run --pack lettings.property_compliance.profiles.v1 \
    --spec examples/contributions/prospire/property_compliance_profiles/operation.yaml \
    --scenario EN_normal --agent reference-mock --output compliance-profile-run.json
uv run operatebench replay --pack lettings.property_compliance.profiles.v1 \
    --spec examples/contributions/prospire/property_compliance_profiles/operation.yaml \
    --run compliance-profile-run.json
uv run operatebench check --pack lettings.property_compliance.profiles.v1 \
    --spec examples/contributions/prospire/property_compliance_profiles/operation.yaml
```

Use a fresh output filename for each run: records are write-once. `reference`
runs the fixed observation-driven agent directly. `reference-mock` feeds that
reference's decisions through the actual ModelAgent tool parser using an offline
transport; it demonstrates interface execution, not independent model ability.
`validate` only checks the specification. `check` executes authored reference
expectations and negative-control failure/must-pass expectations; it does not
establish completeness of the grader.

The separate `operatebench.development-run.v1` record binds the spec, scenario,
profile, runtime/evaluator identities and decision tape and includes the episode
and grade. Replay reconstructs execution from recorded decisions, without the
original solver or provider, and checks the reconstructed result. Wrong pack,
changed profile/spec, altered observations or inconsistent tape are refused.
These integrity checks are not cryptographic third-party attestation or official
evidence admission. New runs after runtime/domain changes need new records;
historical records stay immutable.
Core `Escalate` lowering is not supported in this development lane: profiles use
explicit domain review actions and `Wait`, not a hidden Maintenance fallback.

Exit codes retain the shared CLI contract: `0` passed, `1` refused input, `2`
executed but failed its contract, `64` invalid command usage. Append `--json` for
machine-readable command results.
