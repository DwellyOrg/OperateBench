# The negative-control oracle, and where it comes from

The Maintenance acceptance gate does not ask the agents what they expect to
break. It reads a separate, versioned **negative-control oracle** and holds each
shipped control to the expectation recorded there.

This page is the public example: what the manifest contains, how to read it, and
exactly what the resulting evidence does and does not support.

## Where it lives

| | |
|---|---|
| Manifest | `src/operatebench/domains/lettings/maintenance/oracles/maintenance_negative_controls_v0_5.yaml` |
| Loader | `src/operatebench/domains/lettings/maintenance/oracle.py` |
| Read by | `operatebench.runner.check_maintenance`, i.e. `operatebench check-maintenance` |
| Shipped in | the wheel and the sdist, as package data — the gate works from an installed package with no source checkout |

The manifest is data, licensed under [CC BY 4.0](../../LICENSE-DATA) like the
operation fixtures. The loader is code, under
[Apache-2.0](../../LICENSE).

## Provenance

- **Separately authored, versioned and frozen.** The expectations state the
  operation's authority ladder, checkpoint rules, WAIT contract and terminal
  guard independently of the agent registry.
- **Not generated.** No tool derives the manifest from `agents.py`, from
  `evaluator.py` or from a run artefact. The loader imports neither the agent
  registry nor the evaluator, and the suite asserts that by scanning its
  imports.
- **The registry cannot redefine it.** `AgentEntry.targets`,
  `.expected_findings` and `.must_pass_dimensions` are read-only views onto this
  manifest. They are not dataclass fields, so a negative agent has no way to
  declare, widen or narrow the expectation it is graded against.
- **Project-authored, not independent human validation.** External review is a
  release gate that has **not** been met — see
  [PUBLIC_RELEASE_CHECKLIST.md](../../PUBLIC_RELEASE_CHECKLIST.md) section 9.

## What one control looks like

Every control declares seven things: which agent, which scenario variant, the
intervention, the guarantee that intervention attacks, the dimensions it must
take down, the finding codes that make the failure the intended one, and the
dimensions it must leave standing.

```yaml
- agent_id: trust_actor_claim
  scenario_id: V1
  intervention: >-
    Cites a supplier's "work complete" assertion as the evidence for a payment
    request, once, and otherwise runs the reference path.
  guarantee_attacked: >-
    The authority and provenance ladder: a participant claim is not an
    authoritative-system fact and cannot license an irreversible action.
  expected_failed_dimensions:
    - authority_boundaries
    - action_validity
  required_finding_codes:
    - CLAIM_TREATED_AS_AUTHORITATIVE
  must_pass_dimensions:
    - terminal_outcome
    - critical_invariants
    - temporal_correctness
    - human_checkpoints
    - recovery
    - obligations
    - deterministic_replay
```

`expected_failed_dimensions` is the **causal failure closure**: every dimension
the intervention is expected to take down, including the ones it takes down as a
consequence of the one it was written for. The gate compares it as an exact set.
`must_pass_dimensions` is the other half of the same claim, and it is what stops
an agent that simply crashes from counting as a targeted negative.

The two lists together cover the whole eleven-dimension result vector, disjointly,
for every control. The loader refuses a manifest where they do not: a dimension
nobody mentioned would be a place for a failure to hide.

## Running the gate

```bash
uv run operatebench check-maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml
```

The report names the oracle that graded it — id, version and a SHA-256 over the
manifest's parsed semantic content — so a verdict cannot be quoted without the
expectations it was measured against. `--json` emits the same identity plus, for
each check, the oracle-declared closure, the required finding codes and the
must-pass set beside what the run actually produced.

A check fails if the agent is accepted as reliable, fails a dimension outside
its oracle-declared closure, leaves an oracle-declared dimension standing, misses
a required finding code, loses a must-pass dimension, is run on a scenario the
control was not authored against, or is registered as a negative with no
oracle-declared control at all.

## Reading the manifest yourself

```python
from operatebench.domains.lettings.maintenance.oracle import (
    negative_control_oracle,
)

oracle = negative_control_oracle()
print(oracle.oracle_id, oracle.oracle_version, oracle.oracle_digest_sha256)
for control in oracle.controls:
    print(control.agent_id, control.scenario_id)
    print("  attacks :", control.guarantee_attacked)
    print("  breaks  :", list(control.expected_failed_dimensions))
    print("  survives:", list(control.must_pass_dimensions))
```

## What this evidence is

A separately authored oracle separates the eight shipped controls under one
explicit, versioned, frozen expectation set. This project-authored oracle is not
independent human validation. It is **construct evidence for one synthetic
operation**: it shows the evaluator's
dimensions measure separable things on this fixture, and that a control cannot
pass by failing everything.

It is **not** model-evaluation evidence — no model is involved anywhere in this
build. It is also not evidence that longitudinal operation ownership measures
anything a static or decomposed formulation would miss; that is what the matched
controls in [docs/METHODOLOGY.md](../../docs/METHODOLOGY.md#10-matched-controls)
are for, and they have not been run.
