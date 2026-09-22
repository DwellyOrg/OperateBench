# Repository governance

This file records the intended code-review policy for `main`. The ruleset itself
is forge-side configuration: this file does not prove that a ruleset is enabled
or that the forge currently enforces any setting below.

## Main-branch change policy

- Changes to `main` are made only by pull request.
- A pull request requires one independent approval; stale approvals are
  dismissed after new changes, and all review conversations must be resolved.
- Required check contexts use these stable aggregate job names: `test (py3.11)`,
  `test (py3.14)`, `build distribution`, and `dco`.
- Force-push and branch deletion are prohibited. Bypass is limited to a named
  emergency role, not all maintainers or repository administrators by default.

The aggregate test jobs transitively run Ruff lint and formatting, mypy for
shipped source and tools, REUSE licensing checks, pytest, the public scanner,
and the causal operation gate. `build distribution` transitively runs the build,
Twine metadata checks, and the public scanner over built artifacts. These steps
are not separately require-able check contexts; the stable aggregate job names
above are the contexts the forge can require.

`CODEOWNERS` identifies the current owner of sensitive source, tools,
methodology, publication, governance, configuration, and workflow surfaces. It
does not substitute for the independent approval requirement.

The approval-, CODEOWNERS-, and bypass-dependent ruleset MUST NOT be activated
while only `@khanukov` exists and no distinct emergency role exists. Activation
prerequisites are both: (a) at least one distinct reviewer or team with repository
access and a verified approval path; and (b) a named, distinct emergency team or
custom role with a verified bypass path. Until both prerequisites exist, forge
enforcement is blocked and the repository remains unprotected; partial activation
is not safe.

## Candidate markers

Signed annotated tags may mark an internal or public candidate only after
exact-candidate verification and separate authorization. A tag is a candidate
marker, not publication by itself, and this policy does not authorize pushing a
tag or changing repository visibility.