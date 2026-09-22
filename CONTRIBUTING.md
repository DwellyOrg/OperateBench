# Contributing to OperateBench

OperateBench is a technical preview. The construct — whether operation ownership
measures something a task benchmark misses — is not yet validated, so the most
valuable contributions right now are the ones that test it rather than expand
it.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Most useful contributions, roughly in order

1. **Evidence that the construct does not work.** A case where the longitudinal
   framing adds no signal over a static or Boundary-only version, an ambiguity
   in the oracle, or a scenario that a plain tool-use agent passes for the wrong
   reason. Negative evidence is a result, and this project would rather publish
   it than discover it later.
2. **Evaluator defects.** A trajectory graded reliable that should not be, or
   the reverse. Include the run artifact and the versions.
3. **Determinism and replay failures.** A run that does not replay to the same
   state, event sequence and result vector.
4. **Corrections to [docs/RELATED_WORK.md](docs/RELATED_WORK.md).** A benchmark
   we mischaracterised, a neighbour we missed, or a claim we have not earned.
5. **Documentation that is wrong, unclear or overclaims.** Particularly any
   sentence that sounds like a novelty, ranking or production-representativeness
   claim.
6. **New operations.** Expensive and high-risk; see
   [docs/CONTRIBUTING_OPERATIONS.md](docs/CONTRIBUTING_OPERATIONS.md).
   Proposal self-service is open for invited design partners through the
   [Partner Contribution Kit](docs/PARTNER_CONTRIBUTION_KIT.md). After semantic
   triage, the [Operation Pack SDK](docs/OPERATION_PACK_SDK.md) provides a guided
   scaffold and common development commands. Do not submit a fixture or
   implementation before semantic triage; a registered development pack is not
   official benchmark evidence.

## What we will decline, and why

- **Breadth before validity.** More scenarios over one unvalidated construct
  multiply a single semantic mistake. Scenario count is not currently a goal.
- **LLM judges in core correctness.** Authority, state, timing and evidence
  predicates stay deterministic. A judge model may diagnose surface quality only.
- **Leaderboard, ranking or scoring aggregation.** Not supported by the
  evidence, and not accepted as a feature.
- **Any real production data.** No customer records, contacts, real
  identifiers, proprietary text or real thresholds — in fixtures, tests,
  examples, issues or pull requests.
- **Wall-clock dependence, randomness without a pinned seed, or network access**
  anywhere in operation execution or evaluation.
- **Retroactive renaming of BoundaryBench artifacts.** They keep their original
  identities. See [docs/VERSIONING.md](docs/VERSIONING.md).

## Before you open a pull request

- Open an issue first for anything beyond a typo or an obvious defect. Design
  disagreement is cheaper before the code exists.
- Keep the change scoped. A fixture change, a runtime change and an evaluator
  change have different comparability consequences and belong in different pull
  requests.
- State the version consequence explicitly. In this preview, fixture changes move
  `operation.operation_version` and its spec digest; runtime or evaluator changes
  move the unified `engine_version` (`OPERATEBENCH_VERSION`), which is *not* the
  distribution version and does not move with it — a runtime change inside an
  unchanged wheel still moves it. Separate runtime/evaluator identities are a
  target design, not fields contributors can report today. See
  [docs/VERSIONING.md](docs/VERSIONING.md).

## Development

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

Work from a source checkout. `git clone https://github.com/DwellyOrg/OperateBench`
is the intended path **once this candidate is published**; until then it does not
resolve, and the commands below are run from the directory that holds
`pyproject.toml`.

```bash
cd operatebench                # the checkout root: the directory holding pyproject.toml
uv sync --frozen --all-groups

# tests and quality gates — no credential or provider/model call after bootstrap
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run mypy tools

# static validation: schema, actor authority and reducer vocabulary only.
# It does not execute scenarios and does not prove terminal reachability.
uv run operatebench validate examples/operatebench/maintenance_v0_1.yaml

# registered packs and the generic development path
uv run operatebench list-packs
uv run operatebench run --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml \
    --scenario V1 --agent reference --output v1_reference.json
uv run operatebench replay --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml --run v1_reference.json
uv run operatebench check --pack maintenance \
    --spec examples/operatebench/maintenance_v0_1.yaml

# run-maintenance/check-maintenance remain compatibility aliases.

# the public-release gate: metadata, licensing, private content, links, artefacts
uv run python -m tools.check_public_release
```

`mypy` is scoped to `src` and `tools`. The offline suite predates strict typing
and is not strict-clean, so `mypy .` fails on a clean checkout; fixing that is
work on the suite rather than a configuration change, and pretending otherwise
would ship a CI step that cannot pass.

Continuous integration runs the same commands on Python 3.11 and 3.14, plus a
package build. The runner needs network access to fetch actions, interpreters and
locked dependencies; benchmark execution and tests use no credentials and make
no live provider/model call. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

A change to operation semantics, runtime semantics or the evaluator must keep
`check-maintenance` green: V1/V3 references reliable and V2 matching its explicit fault-control expectation, and
every negative agent matching the separate oracle's declared failure closure,
required findings and must-pass dimensions.

## Reporting a defect

Include: what you ran; `operation.operation_version`, the spec digest, unified
`engine_version`, `artifact_version`, scenario/variant ids and the run artifact
if available; then what you expected and what happened. State/trajectory digests
are already in the artifact. Separate runtime/evaluator versions and bundle
digests do not exist in this preview and are not required.

For security, privacy or data-exposure issues — including any suspicion that
non-synthetic data reached this repository — **do not open a public issue**. See
[SECURITY.md](SECURITY.md).

## Licensing and copyright

Code, including imported or executable schemas, is contributed under the
[Apache License 2.0](LICENSE); non-code fixtures and documentation are under
[CC BY 4.0](LICENSE-DATA). `REUSE.toml` is the authoritative path/class map.

You retain copyright in your contributions. There is no copyright assignment and
no transfer of rights to PROSPIRE TECHNOLOGIES LTD. By opening a pull request you confirm that you
have the right to contribute the material and that you licence it under the
terms above.

Every commit in a pull request must certify the [Developer Certificate of
Origin 1.1](DCO) with a `Signed-off-by: Name <email>` trailer. Create it with
`git commit -s`; amend an existing commit with `git commit --amend -s`. The
PR-only DCO gate checks every commit introduced by the pull request. This is a
DCO process, not a contributor licence agreement.

By contributing, you also confirm the content is synthetic or your own, and
contains no personal data, customer data or proprietary material belonging to
anyone else.

## Governance, honestly stated

Dwelly is the initial steward and currently makes the final call on scope and
releases. There is no consortium, foundation or formal governance structure, and
building one is explicitly deferred until the construct is validated. If that
changes, it will be documented here rather than assumed.
