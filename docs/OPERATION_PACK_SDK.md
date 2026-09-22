# Operation Pack SDK v2

The SDK is the guided implementation step after an operation proposal passes
semantic triage. It gives every reviewed domain pack the same offline commands
without pretending that workflow semantics can be expressed by one generic DSL.

> **Status: development dispatch only.** Registering a pack makes these commands
> available. It does not make the pack official, sealed, comparable across
> domains, leaderboard-eligible, or model evidence. Artifact 9, CommonOutcomeV1,
> matched controls and independent review remain separate gates.

## Contributor path

After proposal approval, create an inert scaffold:

```bash
operatebench init-operation \
  --pack-id partner.example_company.return_refund.synthetic \
  --operation-id example_company_return_refund_synthetic_v1 \
  --owner-id example_company \
  --owner-display-name "Example Company Ltd" \
  --contribution-kind partner \
  --destination return_refund
```

`init-operation` creates one new directory. It never overwrites or merges an
existing path, follows a symlink, imports generated Python, or changes the
registry. The generated code intentionally refuses every command until its
domain semantics, evaluator, agents, independent oracle, replay and tests have
been implemented.

Use a valid Python package name for the destination (for example,
`return_refund`, not `return-refund`) so the generated relative imports work
while running `test_pack.py` before the package is moved into the repository.
For the no-symlink/no-overwrite guarantee, `init-operation` also requires the
host to expose descriptor-relative directory operations and no-follow directory
opens. It refuses without writing anything on a platform that lacks them.

The scaffold command above creates a hypothetical, unregistered candidate. Its
proposed pack ID is not expected to resolve through the registry. After the
implementation is complete, maintainers choose and statically register its
canonical identity; registration still does not admit the pack as evidence.

Every contribution uses four matching owner namespaces:
`src/operatebench/contributions/<owner_id>/<operation>/`,
`tests/contributions/<owner_id>/<operation>/`,
`examples/contributions/<owner_id>/<operation>/`, and
`docs/contributions/<owner_id>/<operation>.md`. A partner pull request may
change only those four paths beneath its own `owner_id`; it must never edit
`src/operatebench/sdk/builtins.py` or
`src/operatebench/resources/operation_pack_registry.yaml`. Static registration
is a separate, maintainer-only admission change that updates those two files
atomically after review. The declarative owner and operation manifests
are checked without importing Python, and wholly synthetic tests and fixtures
are mandatory. Contribution metadata binds `default_spec` directly to that
owner's declared `examples/contributions/<owner_id>/<operation>/operation.yaml`;
partner registration requires the same normalized owner path.

`contribution_kind` records authorship authority, not registration or admission.
An owner whose exact display name and maintainer authority already appear in the
reserved registry may author a new, uniquely named `maintainer` contribution.
That contribution remains absent from `BUILTIN_PACKS`, undispatchable, and
ineligible as evidence until a separate maintainer registration change reserves
its exact identities and adds it to the built-in tuple. An owner without that
reserved authority cannot self-declare `maintainer`.

The contributor retains ownership under the project licence and DCO; no
copyright assignment is required. Registration records technical inclusion
only; evidence admission, legal approval, publication approval and human review
remain outside this status. This private development repository does not claim
that the workflow is already public or self-service proven.

The repository's completed Commerce pack demonstrates that transition.
From the root of a source checkout, the shipped runnable example is:

```bash
# validate is static only: no scenario runs and no terminal is proven reachable.
uv run operatebench validate \
  --pack commerce.return_refund.synthetic.v1 \
  examples/operatebench/commerce_return_refund_v0_1.yaml

uv run operatebench run \
  --pack commerce.return_refund.synthetic.v1 \
  --spec examples/operatebench/commerce_return_refund_v0_1.yaml \
  --scenario V1 --agent reference --output return-v1.json

uv run operatebench replay \
  --pack commerce.return_refund.synthetic.v1 \
  --spec examples/operatebench/commerce_return_refund_v0_1.yaml \
  --run return-v1.json

uv run operatebench check \
  --pack commerce.return_refund.synthetic.v1 \
  --spec examples/operatebench/commerce_return_refund_v0_1.yaml
```

Exit codes are: `0` the command ran and its contract passed; `1` an ordinary
runtime, domain, input-content, or filesystem operation was refused by name; `2`
the command ran and its contract failed; `64` the command line itself was
malformed. Argparse errors and command-specific missing combinations use `64`
without a traceback, so they cannot be mistaken for validation evidence.
`check` accepts no caller-supplied oracle path: a caller must not be able to
soften a shipped negative-control contract.

The [worked property-compliance reference](contributions/prospire/property_compliance.md)
is the maintainer-authored, isolated, synthetic, non-evidence canonical
scaffold-to-static-registration example. Its invented workflow is a contribution
pattern, not policy or real compliance evidence.

## Static trust boundary

Source, import and declared-path isolation is enforceable ownership and CI
policy, not a sandbox against arbitrary reviewed Python; public-fork CI has no
secrets and uses ephemeral hosted runners. These controls do not make malicious
Python safe to execute on a privileged or local host.

Generated, Git-ignored real `__pycache__` directories are outside the authored
contribution surface, but the isolation checker enumerates every immediate
cache entry. Only regular, non-symlink CPython cache filenames are allowed;
subdirectories and every other entry are rejected. The library checker does not
treat these opaque files as authored input. Normal imports still require scanned
source and dynamic loaders are forbidden. Before pytest, CI runs the trusted CLI
gate, which rejects every Git-tracked `.pyc`/`.pyo` and every tracked path with
an exact `__pycache__` component under contribution source or tests. Untracked
generated canonical caches remain admissible locally; `.pyc`/`.pyo` files
outside a cache remain forbidden by the library scan.

For partner packs, the runtime registry also checks that the declared module is
already a real Python module and that its `PACK` global and exact class-name
global bind the submitted object and class. It stores detached immutable metadata
snapshots for listing and owner queries, and rechecks exact current metadata against
the snapshot before dispatch. These runtime checks are defense in depth, not
unforgeable in-process provenance or attestation; they do not defeat arbitrary
in-process Python, `object.__setattr__`, or compromise of private process memory.
The static source checker plus maintainer review is the admission authority;
arbitrary Python remains unsandboxed.

The registry is built from a fixed tuple of reviewed Python objects. It has no
runtime `register` method, entry-point discovery, module-string import, or spec
field that selects executable code. This is not a sandbox: registered Python is
trusted repository code and receives normal process authority. External input is
the strict synthetic fixture, not the implementation.

Each pack declares a unique canonical `pack_id` and `operation_type`. The pack
must reject a fixture whose declared operation type differs before execution or
replay. Artifact 8 does not persist `pack_id`; this type binding prevents a CLI
selection from becoming an unrecorded label attached to another pack's run.

Aliases are CLI conveniences only. They are unique, immutable and never written
as operation identity.

## What a pack owns

SDK v2 is deliberately a high-level command contract:

- `validate(ValidateRequest) -> CommandResult`;
- `run(RunRequest) -> CommandResult`;
- `replay(ReplayRequest) -> CommandResult`;
- `check(CheckRequest) -> CommandResult`.

The pack owns its strict loader, state model, `OperationDomain`, deterministic
evaluator, agents, independent oracle, artefact writer and replay semantics.
The generic CLI owns only pack selection, rendering and the four exit
categories: 0 for success, 1 for an error, 2 for contract failure and 64 for
usage. A `CommandResult` carries JSON-safe payload, UTF-8-safe text lines
and one `contract_passed` bit; malformed input is raised as an
`OperateBenchError`, never encoded as a failed benchmark result.

This high-level seam is intentional. The current `runner.py` and Artifact 8
validator still bind Maintenance-specific domain, evaluator, agent, terminal,
dimension and state semantics. Changing them in the SDK PR would be a runtime or
artefact-methodology change, not contributor tooling. The Maintenance adapter
therefore delegates to those existing implementations unchanged, while a new
incubator pack may own a development serializer and runner.

## Development registration

A static registry entry makes a pack discoverable and runnable as trusted
repository code. It means only that maintainers admit the pack to the
development/incubator dispatch surface; it is not a privacy review, an outcome
review, or evidence admission.

A candidate pack is development-registered only after all of the following are
true:

1. its proposal has passed semantic and synthetic-data triage;
2. all four commands are deterministic and offline;
3. reference agents pass every declared scenario;
4. targeted negatives match a separately authored oracle exactly, including
   required passing dimensions;
5. wrong-pack and wrong-operation-type inputs are refused before execution;
6. run records replay from pack-owned live semantics rather than agreeing with
   their own recorded state;
7. its metadata explicitly remains `evidence_eligible=false`;
8. the complete repository quality and packaging gates pass.

## Evidence admission

SDK v2 does not implement an evidence-admission route. The registry rejects
`evidence_eligible=true` fail-closed, including for trusted built-ins. Static
registration therefore cannot assert or imply independent privacy review,
independent outcome review, semantic-oracle adequacy for research use, model
comparability, or leaderboard eligibility.

Any future admission mechanism must be separate and versioned, and must
represent actual independent privacy and outcome-review gates plus controls that
establish the semantic oracle used for evidence. Until such a mechanism ships,
all SDK packs are non-evidence development/incubator packs. The commerce
return/refund pack is registered with `status=incubator` and
`evidence_eligible=false` by default.

The bounded SDK v2 architecture check now ships as the
[synthetic commerce return/refund pack](operations/COMMERCE_RETURN_REFUND.md).
It adds a second, non-lettings domain through its own package plus the fixed
built-in tuple, without edits to Core, the generic CLI or the Maintenance
runner. That demonstrates this contribution seam for one additional domain; it
does not prove that every Core outcome or future workflow is domain-neutral.

## Known Core limitation

The Core `Escalate` handling still contains Maintenance-specific action and
checkpoint vocabulary. SDK v2 does not claim that every Core outcome is already
domain-neutral. An incubator pack may avoid that outcome and model a domain
handoff through its own exact actions while the leak remains visible and tracked;
copying Maintenance nouns into another domain would conceal rather than solve it.

## Legacy compatibility

The existing commands remain supported as aliases over the Maintenance adapter:

```bash
operatebench validate examples/operatebench/maintenance_v0_1.yaml
operatebench run-maintenance --spec ... --scenario V1 --agent reference --output run.json
operatebench replay --spec ... --run run.json
operatebench check-maintenance --spec ...
```

The adapter uses the existing strict loader, runner, evaluator, oracle, Artifact
8 writer and replay code. The SDK does not move the engine version, artefact
version, fixture digest, evaluation dimensions, or result semantics.
