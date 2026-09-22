# Contributing an operation

An OperationSpec is the expensive artifact in this project. One semantic mistake
in a scenario propagates into every result measured against it, and the more
scenarios there are, the harder that is to notice. This document describes how a
new synthetic operation gets in, and what it has to survive first.

> **Preview status.** Proposal self-service is open for invited design partners
> through the [Partner Contribution Kit](PARTNER_CONTRIBUTION_KIT.md); executable
> operation packs remain guided. A proposal is intake for semantic review, not
> permission to open a fixture or implementation pull request.

## Hard rules

Partner code, tests, fixtures and documentation must be confined respectively
to `src/operatebench/contributions/<owner_id>/`,
`tests/contributions/<owner_id>/`, `examples/contributions/<owner_id>/`, and
`docs/contributions/<owner_id>/`. A partner PR must never edit either central
`sdk/builtins.py` or `resources/operation_pack_registry.yaml`; maintainers perform
static registration separately. After review, the maintainer registration PR
updates the reserved identity manifest and builtins atomically. The
contributor keeps copyright under the repository licence and DCO, with no
copyright assignment. Registration remains distinct from evidence admission.

Generated, Git-ignored real `__pycache__` directories are not part of the
authored contribution surface, but the isolation scan still enumerates each
cache's immediate entries. It permits only regular, non-symlink CPython cache
filenames and rejects every subdirectory or other entry; bytecode outside a
cache is also forbidden. These opaque generated files are not treated as
authored input by the library scan. Normal imports still require scanned source,
while the policy rejects dynamic loaders. The trusted CLI gate that CI runs
before pytest rejects every Git-tracked `.pyc`/`.pyo` and every tracked path
containing an exact `__pycache__` component in source or test contribution trees.
Untracked generated canonical caches remain admissible for local development.

The isolation checker is a review policy, not a filesystem sandbox. Packaged
resources use its finite static pattern: import exactly `files, as_file` from
`operatebench._resource_access`, then call
`files(__package__).joinpath("fixture.yaml")`. The literal name must identify one
existing regular, non-symlink public resource file directly inside the same
package; nested paths, traversal, dynamic names and chained Traversable operations
are rejected. Module imports, aliases, indirect calls, keyword or different
package arguments, and dynamic loaders are also rejected.
Imports of `sys` are forbidden in contribution source and tests because
interpreter-global module state is outside the owner isolation boundary. This
is a policy restriction, not a sandbox claim.

Absolute OperateBench imports are also a finite, context-aware interface. Pack
source may import only `operatebench._resource_access`,
`operatebench._write_once`, `operatebench.core.errors`,
`operatebench.core.evaluation`, `operatebench.core.outcomes`,
`operatebench.core.protocol`, `operatebench.core.read_contract`,
`operatebench.core.retrieval`, the public `operatebench.sdk` root, and
`operatebench.sdk.errors`. The resource helper retains the exact `files,
as_file` rule above; the write-once helper must import exactly
`_write_once_bytes, _WriteOnceFailure`. Operation tests may import only their
own canonical operation package plus `operatebench.cli`,
`operatebench.core.errors`, and the public `operatebench.sdk` root. Standard
library and third-party imports remain subject to the other isolation rules.
Repository-layout aliases such as `src.operatebench.*`, foreign contribution
packages, and every other `operatebench.*` module are rejected.

In particular, contribution source and owner-scoped tests may not import
`operatebench.sdk.builtins`, `operatebench.sdk.registry`, or
`operatebench.core.engine`. Built-in registration is mutable, maintainer-owned
shared state: contributors test their pack's public behavior, while maintainers
test exact built-in registration and reserved-manifest agreement centrally in
`tests/test_operation_pack_registry.py`. Partners never receive a registration
exception. These static import checks enforce the review boundary; they do not
claim to sandbox classes exposed by the supported public modules.

Contribution manifests declare the exact licensing split: source and test code
use `code_license: Apache-2.0`; fixtures and documentation use
`content_license: CC-BY-4.0`. These fields describe licensing only. They do not
assign ownership; contributors retain copyright, and the DCO remains the commit
certification policy.

1. **Fully synthetic.** Every actor, identifier, amount, date, address,
   document, message, threshold, policy value and event plan is invented for the
   benchmark. Privacy status is `SYNTHETIC_ONLY`, and there is no other accepted
   value for a public operation.
2. **No production records.** No customer data, contacts, real property or
   account identifiers, proprietary message text, internal prices, real approval
   thresholds or copied production code. Contributions must use wholly invented
   topology and content rather than reconstructing an organisation's workflow.
3. **Not anybody's official policy.** An operation must not be presented as the
   real policy or workflow of Dwelly or of any other organisation, and must
   carry a disclaimer saying so.
4. **Executable before persuasive.** A prose design that nothing runs is not a
   contribution. The fixture is the contract; where a design document and the
   fixture disagree, the fixture is what executes.
5. **Deterministic.** No wall-clock dependence, no randomness without a pinned
   seed, no network, no live provider call anywhere in the operation or its
   evaluation.
6. **No LLM judge in core correctness.** Surface wording may be generated or
   varied; authority, state, timing and evidence predicates stay exact.

## What an operation must contain

To be a Lifecycle operation rather than a task with timestamps, a spec needs all
of:

- more than one simulated day of event time;
- at least three actor roles, including one authoritative system whose facts
  outrank actor claims;
- at least one correct `WAIT` with a declared wake condition and a fallback;
- at least one scheduled wake-up or follow-up, and at least one deadline;
- at least one exogenous event the agent did not cause and cannot prevent;
- at least one human checkpoint that is required, plus a reachable path where
  escalation would be unnecessary;
- an authority boundary — a threshold, mandate or permission the agent must not
  cross;
- at least one claim that is plausible and false, or true but not yet
  authoritative;
- at least one loop, reopen or further-work path;
- more than one legitimate trajectory;
- a guarded terminal, with a late event that must not resurrect it;
- a deterministic evaluator bundle and enumerated admissible outcomes per
  settled decision state.

An operation missing several of these is likely a Boundary Track case — which is
a fine thing to be, and is contributed differently.

## Intake and QC

The development path uses stages 1 and 2 below. Stages 3 through 7 describe
scientific QC gates for a separate future evidence-promotion process that is
unavailable in SDK v2; they are not prerequisites completed or recorded by
development registration. Within that future process, a spec that fails one
gate does not proceed to the next.

The proposer can complete stage 1 without code. Early design partners supply
operation structure, failure modes and an independent semantic reviewer;
maintainers help encode proposals that pass this gate through the guided
[Operation Pack SDK](OPERATION_PACK_SDK.md). The SDK standardises development
commands and scaffolding; it does not itself satisfy any future evidence gate.

### 1. Proposal and semantic review

Start with the no-code form described in
[PARTNER_CONTRIBUTION_KIT.md](PARTNER_CONTRIBUTION_KIT.md): the synthetic case,
its actors, authority sources, decisions, and — most importantly — **the failure
modes it is meant to expose and why an ordinary task formulation would miss
them**. Reviewed by a maintainer plus at least one reviewer who did not author
the proposal.

Rejected here: cases that reduce to a single boundary decision; cases whose
difficulty is perception or retrieval rather than operation ownership; cases
that restate an existing scenario with different nouns.

### 2. Static validation, then execution

Two separate gates, in this order, because they answer different questions and
only one of them runs the case.

**Static validation.** `operatebench validate` is static schema, authority and
reducer-vocabulary validation. It loads the fixture, holds it to the strict
schema, checks that every declared event type has an executable reducer, checks
that every authored event's actor holds the authority its type requires, and
reports the spec's identity and digest. **It does not execute scenarios and it
does not prove terminal reachability.** A spec can pass `validate` and still
have a variant that never reaches the terminal it declares.

```bash
operatebench validate examples/operatebench/maintenance_v0_1.yaml
```

A vocabulary that accepts what the engine cannot run is a schema promising more
than the build honours. Declared-but-unreduced event types are rejected, not
warned about.

**Execution.** Whether a variant actually runs, reaches its declared terminal
and grades as expected is settled by the explicitly selected pack:

```bash
operatebench run --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml \
    --scenario V1 --agent reference --output v1_reference.json
operatebench replay --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml \
    --run v1_reference.json
operatebench check --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml
```

Repeat `run` and `replay` for every declared variant — in this build `V1`,
`V2` and `V3`. Exit codes: `0` yes; `1` the input was refused, by name;
`2` it ran and the answer is no; `64` the command line itself was invalid.
`run-maintenance` and `check-maintenance` remain compatibility aliases over the
same adapter.

The registry snapshots reviewed metadata and rechecks it before dispatch as
defense in depth. The static source checker rejects dynamic reflection through
`getattr`, `setattr`, `delattr`, `.__getattribute__`, `.__setattr__`, and
`.__delattr__` in contribution source and tests. Ordinary direct attributes and
dictionaries remain available. Maintainer review remains the authority;
arbitrary Python is not sandboxed. The registry is a fixed set of registered Python packs, not runtime plugin
discovery. A new vertical owns its strict loader, runtime wiring, evaluator,
oracle and replay behind the SDK contract. Registration provides development
execution only and is not scientific admission. Development registration
neither requires nor records independent outcome-oracle, privacy,
negative-control, or matched-control review. Those reviews belong to the future
evidence-promotion process, which is unavailable in SDK v2.

### 3. Independent oracle

The admissible outcomes at each settled decision state are authored **by someone
other than the fixture author**, from the case description rather than from the
fixture, and then compared. Disagreements are resolved before the scenario
proceeds, and the resolution is recorded — a disagreement usually means the case
is ambiguous, not that one reviewer was careless.

The oracle enumerates admissible outcomes and forbidden ones explicitly.
Multiple valid trajectories are expected; "the reference path" is one admissible
trajectory, never the definition of correctness.

### 4. Privacy classification

A reviewer independent of the author confirms, item by item, that actors,
identifiers, messages, documents, amounts, dates, policies, thresholds and event
plans are invented; that no production or personal data is present; that the
disclaimer and `data_provenance` fields are accurate; and that privacy status is
`SYNTHETIC_ONLY`.

Where the operation models disclosure, the spec must also declare
**actor-visible fields and disclosure rules**, so that cross-actor leakage and
over-sharing at human checkpoints are evaluable rather than merely discouraged.

### 5. Negative controls and their oracle

Every scenario ships with targeted negative agents **and** a separately authored
negative-control oracle. The oracle is a machine-readable manifest, written from
the case description rather than from the agents, that declares for each
control: the agent id, the scenario variant, the intervention, the guarantee it
attacks, the causal failure closure it must produce, the finding codes that make
that failure the intended one, and the dimensions that must remain passing. The
shipped example is
`src/operatebench/domains/lettings/maintenance/oracles/maintenance_negative_controls.yaml`;
see [../examples/operatebench/negative_control_oracle.md](../examples/operatebench/negative_control_oracle.md).

An expectation declared next to the agent it grades is not structurally
separate. The manifest must be loadable on its own, must not be generated from the
registry or the evaluator, and the registry must have no way to override it.

The generic gate dispatches to the selected pack, which owns its separately
authored oracle. There is deliberately no CLI option to replace that oracle.

```bash
operatebench check --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml
```

The legacy `check-maintenance` spelling remains an alias.

It is two-directional:

- V1/V3 references are reliable and V2 matches its explicit fault-control expectation;
- each negative control fails exactly the closure oracle-declared for it, for the
  oracle-declared finding codes;
- every dimension the manifest lists as must-pass still passes;
- a negative with no oracle-declared control fails the gate as ungraded.

A negative agent that fails everything proves nothing and is not accepted as
evidence that the evaluator separates behaviours.

### 6. Matched controls

A scenario is not complete without its controls: a static/decomposed variant, a
Boundary-only variant of its consequential decisions, and a time-removed
variant. These are what make it possible to say whether the longitudinal
environment added anything — see
[METHODOLOGY.md](METHODOLOGY.md#10-matched-controls).

For future evidence promotion, authoring the controls is part of authoring the
evidence candidate, not a later research step.

### 7. Sealed or public

Each variant is designated:

- **public** — shipped and readable, for development, reproduction and
  debugging;
- **sealed** — held out for evaluation, rotated on a published cadence, with
  identity and version published but semantics withheld while sealed.

Sealed variants are handled under the same review process; the difference is
distribution, not scrutiny. Rotation is a version event and invalidates
comparability with the previous set — see
[VERSIONING.md](VERSIONING.md#scenario-lifecycle-and-sealed-variants).

## Statistical hygiene

Variants of one authored case are **one Semantic Scenario**, and results cluster
under it. Adding variants to an existing scenario increases coverage of that
case; it does not increase the number of semantic units, and contributions that
present it that way will be asked to relabel.

Three variants of one scenario are not three scenarios. Three scenarios by one
author are not three independently authored scenarios.

## Capability tags

Tag each scenario with the capabilities it exercises, so coverage is inspectable
and controls can be matched: `state_dependency`, `insufficient_information`,
`authority_change`, `wait_required`, `late_event`, `reopen`,
`human_checkpoint`, `duplicate_event`, `post_commit_failure`,
`disclosure_boundary`.

## What a submission looks like

The list below describes a future evidence-promotion submission, not development
registration or the initial proposal. A proposer starts with the worksheet and
issue form; implementation begins only after semantic triage.

- the executable fixture;
- a short design rationale — the case, its failure modes, why it needs operation
  ownership;
- the independent admissible-outcome oracle (stage 3);
- the independent negative-control oracle manifest (stage 5);
- reference and targeted negative agents, and evidence that they run — the exact
  pack-selected `run`, `replay` and `check` invocations and their output,
  not only that `validate` accepted the fixture;
- matched controls;
- the privacy classification record;
- capability tags, provenance and version identity.

## Licensing of contributions

Fixtures, schemas and documentation are contributed under
[CC BY 4.0](../LICENSE-DATA); code under the
[Apache License 2.0](../LICENSE). Contributors keep their copyright — there is
no copyright assignment — and licence their contribution under the terms above.
See [../CONTRIBUTING.md](../CONTRIBUTING.md).
