# Public release checklist

This checklist governs the first public release of OperateBench. Run technical
items in order and keep public technical evidence in
[PUBLIC_CANDIDATE_VERIFICATION.md](PUBLIC_CANDIDATE_VERIFICATION.md). Human
decisions belong only in controlled records outside public Git.

**Status of this candidate.** The [publication manifest](PUBLICATION_MANIFEST.json)
records scoped verified-base technical evidence, not blanket pending or automatic
acceptance of a changed candidate. G1–G5, G7 and G10 have verified-base technical
PASS; G6 still requires candidate hosted CI and branch coverage. G8 full manual
claims review and G9 complete external-link rechecking remain open. G11 and G12
remain human gates. Independent AI review is not independent human approval.
Mechanical licensing is not legal or publication authorization; the owner has
separately confirmed licences, project name and author/copyright attribution.
An exact-source controlled acceptance record can close candidate technical items
without a post-review manifest edit. CI cannot close the human requirements.

The unchecked boxes below are intentionally retained as a reusable release-time
runbook, not as a second status ledger. Release operators must run and check the
applicable pending items against the exact release candidate, record fresh
evidence where required, and obtain the human approvals before publication.

Rules for using it:

- Every item is **blocking** unless it says otherwise. There is no partial
  release.
- Record evidence — command, output reference, reviewer, date — for each item.
  An item marked done without evidence is not done.
- A failed item stops the release. Fix, then re-run every item downstream of the
  fix rather than assuming they still hold.
- The last item is a human decision that nobody may infer from the others.

---

## 1. Content and scope

- [ ] The published tree contains only the included classes in the publication
      manifest, and nothing from the excluded classes.
- [ ] No internal document, filename, path, hostname, repository reference or
      identifier appears anywhere in the tree or its history.
- [ ] No absolute local path appears in any file.
- [ ] Unbuilt capabilities are not implied to exist (additional domains,
      matched controls, multi-operation evaluation, model evaluation).
- [ ] Sealed scenario variants are excluded; their identity and version are
      published, their semantics are not.

## 2. Clean history

- [ ] The published repository is produced from the deliberate curated snapshot;
      existing internal storage is not publication and imports no internal history.
- [ ] History contains no credential, no internal document and no personal data
      at any commit — verified against history, not only the working tree.
- [ ] BoundaryBench artifacts, where included, retain their original names,
      identifiers and digests, and are not relabelled or renumbered.
- [ ] The export method and source commit are recorded in the publication
      manifest.

## 3. Licensing and attribution

The owner has confirmed the Apache-2.0 / CC BY 4.0 split, the project name and
author/copyright attribution. This is not a claim of independent legal advice.
PROSPIRE TECHNOLOGIES LTD legal authorization/counsel and domain know-how/rights
assessment must not be inferred from mechanical checks or licence confirmation.
Publication remains a separate human decision.

- [ ] `LICENSE` is the exact Apache License 2.0 text.
- [ ] `LICENSE-DATA` correctly describes the CC BY 4.0 arrangement for
      fixtures, schemas published as data and documentation.
- [ ] `NOTICE` names Dwelly as initial steward, disclaims endorsement of results,
      and states the synthetic-content position.
- [ ] No contributor copyright transfer or assignment is asserted anywhere.
- [ ] `CITATION.cff` parses as valid CFF, and its version, date and repository
      URL match the release.
- [ ] Third-party names and trademarks are used for identification only.
- [ ] **PROSPIRE TECHNOLOGIES LTD has legally authorized the Apache-2.0 /
      CC BY 4.0 split**, the
      trademark and stewardship language in `NOTICE`, and the absence of any
      copyright-assignment claim. *(Not a self-service item. Blocking. Recorded
      only in the controlled record system outside public Git.)*

## 4. Package and CLI

- [ ] `uv sync --frozen --all-groups` succeeds on Python 3.11 and 3.14.
- [ ] `uv build` produces an sdist and a wheel; metadata checks pass.
- [ ] The package installs into a clean environment on both Python versions.
- [ ] `README.md` documents a source-checkout install that has been run
      end to end. It claims no package-index publication, because there is
      none, and carries no placeholder install line.
- [ ] From a clean checkout, with no credential set and no network access:
      `operatebench validate`, `run-maintenance`, `replay` and
      `check-maintenance` all behave as documented, including exit codes
      (`0` yes, `1` input refused by name, `2` ran and the answer is no).
      `validate` is static schema, authority and reducer-vocabulary validation
      only: it executes no scenario and proves no terminal reachable, so it is
      never recorded as evidence that a variant runs.
- [ ] Terminal reachability is evidenced by execution, not by `validate`:
      `run-maintenance` plus `replay` on **each** shipped variant (`V1`, `V2`,
      `V3`), with the reached terminal recorded.
- [ ] `check-maintenance` passes: V1/V3 references reliable and V2 matching its explicit fault-control expectation,
      every negative agent matching the separate oracle's failure closure,
      required findings and must-pass dimensions.
- [ ] Every published run artifact replays to the same state, event sequence and
      result vector.
- [ ] CI is green on both Python versions, requests no secrets, declares
      `permissions: contents: read`, and runs only commands that work against
      the current `pyproject.toml` and `uv.lock`.
- [ ] `uv run python -m tools.check_public_release` passes, and its checks are
      also asserted by `tests/test_public_release.py` so the command and the
      suite cannot drift apart.

## 5. Secrets, privacy and synthetic content

- [ ] Secret scan over the working tree: no key, token, credential file or
      environment file. Run by `uv run python -m tools.check_public_release`,
      which allowlists — by exact literal, never by prefix — the historical
      synthetic test keys whose whole purpose is to prove a credential is
      refused, redacted or never written.
- [ ] Secret scan over history, once history exists. The working-tree scan
      deliberately does not read `.git`; this is a separate gate with a separate
      tool, and it cannot be run before the initial commit.
- [ ] No credential is readable from any manifest, ledger, log, error message or
      artifact.
- [ ] PII scan over tree and history: no personal data, customer data, real
      contact, address or account identifier.
- [ ] Every published fixture is classified `SYNTHETIC_ONLY`, with an accurate
      disclaimer and provenance statement.
- [ ] A reviewer independent of the fixture author has confirmed, item by item,
      that actors, identifiers, amounts, dates, messages, documents, policies,
      thresholds and event plans are invented.
- [ ] No fixture is presented as Dwelly policy or as production behaviour.

## 6. Claims review

Scan the whole tree — README, docs, code comments, fixture text, CI, manifest —
and confirm none of the following appears in any form:

- [ ] "first" (first longitudinal / first real-world workflow / first enterprise
      / first long-horizon benchmark), "broadest", "most realistic",
      "industry standard", "state of the art".
- [ ] Production-representative or official-policy claims for any fixture.
- [ ] Any leaderboard, ranking, composite capability score, or comparison of
      named models.
- [ ] Any claim that reference or negative agent results say something about
      model capability.
- [ ] Any claim of validated construct, proven incremental signal, or
      release-grade artifact immutability.
- [ ] "Longitudinal" or "long-horizon" used without the immediate qualifier that
      distinguishes our meaning from the established one.
- [ ] Any suggestion that OperateBench is a rename of BoundaryBench, or that a
      BoundaryBench run is Lifecycle evidence.

Positively present:

- [ ] Prior art conceded in `docs/RELATED_WORK.md`, with sources.
- [ ] Limitations stated in `docs/METHODOLOGY.md`, including the kill criteria.
- [ ] Technical-preview status stated on the first screen of `README.md`.

## 7. Links and references

- [ ] Every relative link and anchor resolves within the published tree —
      checked mechanically by `uv run python -m tools.check_public_release`, not
      by eye.
- [ ] Every external link resolves, and cites the work it claims to cite:
      arXiv identifiers, titles and repository URLs verified individually by
      bounded HTTP GET, with any inaccessible source recorded rather than
      dropped.
- [ ] No link points at an internal or private resource.
- [ ] The repository URL, package name and any project website referenced are
      the approved public identities.

## 8. Documentation coherence

- [ ] Commands in `README.md` and `CONTRIBUTING.md` match the shipped CLI
      exactly.
- [ ] The result vector in `README.md` matches what the evaluator actually
      reports.
- [ ] The scenario table matches the shipped fixture's variants and terminals.
- [ ] `docs/VERSIONING.md` matches the version fields the artifacts actually
      carry, and says plainly where the policy specifies more than the build
      emits.
- [ ] The negative-agent table in `docs/METHODOLOGY.md` matches the shipped
      registry, agent by agent and dimension by dimension.
- [ ] The distribution version, both package versions and the Boundary Track
      compatibility statement agree across `pyproject.toml`, `README.md`,
      `CITATION.cff` and `PUBLICATION_MANIFEST.json`.
- [ ] Contact addresses in `SECURITY.md`, `CODE_OF_CONDUCT.md` and
      `CITATION.cff` are monitored.
- [ ] Issue and pull-request channels exist and are staffed before the
      repository is public.

## 9. Independent review

- [ ] A reviewer independent of the authors has reviewed fixture semantics and
      the independent oracle.
- [ ] A reviewer independent of the authors has reviewed the privacy
      classification.
- [ ] A reviewer independent of the authors has reviewed the public claims
      against the evidence, with authority to block.
- [ ] Review findings are either resolved or published as known limitations —
      not silently carried.

## 10. Publication decision

- [ ] Every technical item above has public evidence; human gates are complete
      in the controlled record system outside public Git.
- [ ] Every blocking gate is closed for the exact candidate, with technical
      evidence in the manifest or an exact-source controlled acceptance record.
      Human requirements need their own explicit controlled approvals; neither
      verified-base passes nor CI can supply them. A manifest edit is not required
      to record exact-source closure.
- [ ] The exact commit to be published is identified by hash.
- [ ] A named approver has **explicitly** authorised final public visibility for
      the exact reviewed commit. Approval of an earlier draft, of the project, or
      of a different commit does not count.
- [ ] A rollback plan exists: who can make the repository private again, and how
      a correction note is published if something is found after release. Note
      that anything published may already have been mirrored, cached or indexed
      — un-publishing is not undoing.

## After release

- [ ] Watch for reports of non-synthetic content; treat any as the highest
      priority (see `SECURITY.md`).
- [ ] Publish a correction note for any defect found post-release, per
      `docs/VERSIONING.md`.
- [ ] Do not add scenarios or claims to the public repository until the matched
      controls and independently authored scenarios in
      `docs/METHODOLOGY.md` are actually done.
