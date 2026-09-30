# Public candidate verification — local successor

The bounded offline evidence below describes the historical 0.12.0 base only.
Successful hosted CI for the corrected runtime source is recorded under
Candidate acceptance boundary. It precedes this documentation correction;
historical digests and PASS statements do not attest to the documentation successor.

This technical preview separates accepted base evidence from candidate acceptance.
The verified base completed technical acceptance on 2026-09-21 and was staged
privately. That historical staging was not publication authorization.
Exact source identities, command outputs and artifact hashes are retained in the
controlled acceptance record, rather than embedded self-referentially here.

## Current identities

| Identity | Current value |
| --- | --- |
| Distribution | 0.1.0 |
| Lifecycle engine | 0.13.0 |
| Artefact contract | 8 |
| Execution ledger | 3 |
| Result dimensions | 11 |
| Model protocol | `operatebench.model.v4` |
| Responses request mapping | `lifecycle_openai_responses_model_request_v9` |

## Bounded offline evidence

The historical 0.12.0 maintenance check was exercised from source with provider credential
environment variables absent and networking isolated, using:

```sh
python -m operatebench.cli check --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml --json
```

It passed 11 checks: the reference agent on `V1`, `V2` and `V3`, and eight
targeted negatives on `V1`, with each expected failure closure satisfied.
V1 and V3 require reliable references. V2 is the explicitly authored permanent
notification-dispatch fault control: check PASS requires legitimate human transfer,
`reliable=false`, exactly recovery and obligations failing with
`REQUIRED_NOTIFICATION_UNDELIVERED` and `OBLIGATION_MESSAGE_NOT_ESTABLISHED`,
and every other dimension passing. This does not mean three reliable references.
This is deterministic construct evidence, not model evidence, matched-control
validation, a published benchmark result or a model ranking. This maintenance
check made no model or provider call; that statement is scoped to this exercised
gate, not to the host or all project activity. Private provider-backed diagnostics
historically occurred and are not evidence certifying this successor.

## Candidate acceptance boundary

**Changed-candidate hosted CI: pending.** This candidate is the documentation
successor, whose package checks and final acceptance are also outstanding.

The historical 0.12.0 verified-base full suites completed on Python 3.11 and 3.14
for the 2026-09-21 historical base. That is scoped base evidence, not a claim that
the changed candidate has completed its full matrix.

The pre-documentation correction runtime source passed
[hosted CI run 36725044261](https://github.com/DwellyOrg/OperateBench/actions/runs/36725044261):
`test (py3.11)`, `test (py3.14)`, `independent secret scan` and `build distribution`
all succeeded. Its Python 3.11 result was 12,204 passed, 369 skipped and 7 subtests,
with 92% branch-instrumented coverage. The workflow runs pytest on both Python
versions, with branch coverage for `operatebench` and `boundarybench` on 3.11
and JUnit diagnostics on both. It also checks lint, types, licensing, contribution isolation,
public-release constraints and deterministic Maintenance gates; the build job
checks sdist/wheel metadata, exact distribution membership and frozen bytes,
and scans the built distributions for credentials and private provenance.
These are provider/model-offline checks; dependency bootstrap uses the network.

The unchanged historical [manifest](PUBLICATION_MANIFEST.json) contains the per-gate census:
G1–G5, G7 and G10 have scoped verified-base technical PASS; G6 retains its
candidate-CI and coverage requirement; G8 full manual claims review and G9 complete
external-link rechecking remain open. Independent AI review is not independent
human review. G11 independent human assessment and know-how/rights confirmation,
and G12 explicit publication authority, remain mandatory. The manifest
is historical evidence, not a current all-pass record.

An exact-source controlled acceptance record binds the candidate source/tree,
changed-path review, command outcomes, artifact hashes, hosted CI run and gate
dispositions. It can close technical requirements without another manifest edit;
CI cannot close human requirements. The
[CI workflow](https://github.com/DwellyOrg/OperateBench/blob/main/.github/workflows/ci.yml)
defines the checks; the run linked above binds their observed success to the
pre-documentation correction source.
Historical results are not rerun, regraded or relabeled by this record.

No rankings, paid calls, remote creation or public visibility changes are
authorized by this document. Technical passes do not authorize a tag, release or
package-index publication.
