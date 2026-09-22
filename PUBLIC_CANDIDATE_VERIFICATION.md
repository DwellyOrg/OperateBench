# Public candidate verification — local successor

This technical preview separates accepted base evidence from candidate acceptance.
The verified base completed technical acceptance on 2026-09-21 and was staged
privately under the approved project identity. This is not public publication.
Exact source identities, command outputs and artifact hashes are retained in the
controlled acceptance record, rather than embedded self-referentially here.

## Current identities

| Identity | Current value |
| --- | --- |
| Distribution | 0.1.0 |
| Lifecycle engine | 0.12.0 |
| Artefact contract | 8 |
| Execution ledger | 3 |
| Result dimensions | 11 |
| Model protocol | `operatebench.model.v4` |
| Responses request mapping | `lifecycle_openai_responses_model_request_v9` |

## Bounded offline evidence

The current maintenance check was exercised from source with provider credential
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

**Changed-candidate hosted CI: pending.** The verified-base full suites completed
on Python 3.11 and 3.14 with zero failures, errors or skips. That is scoped base
evidence, not a claim that the changed candidate has completed its full matrix.
Fresh tests, package checks and branch coverage must be recorded for the candidate;
current counts come from its actual CI output, not copied historical totals.

The [manifest](PUBLICATION_MANIFEST.json) contains the per-gate census:
G1–G5, G7 and G10 have scoped verified-base technical PASS; G6 retains its
candidate-CI and coverage requirement; G8 full manual claims review and G9 complete
external-link rechecking remain open. Independent AI review is not independent
human review. G11 independent human assessment and know-how/rights confirmation,
and G12 explicit publication authority, remain mandatory.

The owner has confirmed the licence split, project name and author/copyright
attribution. This is not independent legal advice or approval of domain know-how
for publication. Internal storage is not publication; final public visibility
requires its own controlled decision.

An exact-source controlled acceptance record binds the candidate source/tree,
changed-path review, command outcomes, artifact hashes, hosted CI run and gate
dispositions. It can close technical requirements without another manifest edit;
CI cannot close human requirements. See the
[CI workflow](https://github.com/DwellyOrg/OperateBench/actions/workflows/ci.yml)
for actual hosted results. No successful run is asserted by this link.
Historical results are not rerun, regraded or relabeled by this record.

No rankings, paid calls, remote creation or public visibility changes are
authorized by this document. Technical passes do not authorize a tag, release or
package-index publication.
