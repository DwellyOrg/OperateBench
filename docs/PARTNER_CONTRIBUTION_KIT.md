# Partner contribution kit

OperateBench design partners can propose a fully synthetic business operation
without first learning the runtime or writing benchmark code. This page is the
single starting point for that proposal.

> **Current stage: proposal self-service; executable packs remain guided.**
> Invited partners may submit an operation proposal using the issue form. After
> triage, the [Operation Pack SDK](OPERATION_PACK_SDK.md) supplies a safe
> incubator scaffold and common development commands. An accepted proposal or a
> registered development pack is not an official benchmark case or model evidence.

## The 30-minute path

1. Read the fit and data-safety checks below.
2. Draft the answers in the
   [operation proposal worksheet](templates/OPERATION_PROPOSAL_WORKSHEET.md).
3. Compare it with the
   [completed synthetic example](../examples/partner_operation_proposal.md).
4. Open the repository's **Operation proposal** issue form and paste the
   answers. Do not attach internal documents, screenshots, exports, transcripts,
   prompts, policies, or customer records.

If any answer might disclose confidential or personal information, stop. Contact
the maintainers through the private security route in [SECURITY.md](../SECURITY.md)
before putting it in GitHub.

## Is this an operation?

A strong proposal describes one case that remains open while time and the world
move independently. It normally has all of the following:

- at least three actor roles;
- an authoritative system whose facts outrank an actor's claim;
- a correct `WAIT`, a wake condition, and a fallback;
- a scheduled follow-up and a deadline;
- an exogenous event the agent cannot cause or prevent;
- a required human checkpoint and a path where escalation would be unnecessary;
- an authority boundary the agent must not cross;
- a plausible claim that is false or not yet authoritative;
- a loop, retry, reopen, or further-work path;
- at least two legitimate trajectories;
- guarded finality and a late event that must not resurrect the case;
- five to eight targeted failure modes that an ordinary one-shot task would
  hide.

If the case is one decision with extra timestamps, it is probably a Boundary
Track proposal rather than a Lifecycle operation. That is useful, but it should
be labelled honestly.

## Transfer structure, never records

The contribution should preserve operational topology and failure modes while
inventing every concrete detail.

| Safe to contribute | Do not contribute |
|---|---|
| Invented actor roles | Names, emails, phone numbers, addresses, account IDs |
| Invented system names | Real vendor or internal system names when confidential |
| Invented messages | Copied emails, chats, tickets, calls, prompts, or documents |
| Invented amounts and thresholds | Real prices, approval limits, SLAs, or policy values |
| Generalised failure topology | Production logs, screenshots, exports, or incident records |
| A synthetic policy written for the benchmark | Your organisation's actual policy or proprietary workflow text |

Every public proposal and pack must be `SYNTHETIC_ONLY` and must state that it is
not the contributing organisation's official policy. Sanitising names in a real
record is not enough: reconstruct the case with invented content.

## What the proposal must establish

The issue form asks for:

1. the operation and why a normal task benchmark misses it;
2. actors, authority, and authoritative systems;
3. lifecycle states and at least two legitimate paths;
4. waits, wake conditions, fallbacks, follow-ups, and deadlines;
5. human checkpoints and authority boundaries;
6. actor claims versus authoritative facts;
7. loops, reopen/recovery, guarded finality, and late events;
8. five to eight causal failure modes;
9. a proposed static/Boundary/time-removed control plan;
10. capability tags and an explicit synthetic-data attestation.

The proposal is semantic input, not an executable fixture. Reviewers may reject
it early if it reduces to a task, duplicates an existing scenario with new
nouns, depends mainly on perception/retrieval difficulty, or cannot be shared
safely.

## Development roles and future evidence QC

| Role | Responsibility |
|---|---|
| Partner subject-matter expert | Explains topology, legitimate outcomes, and realistic failures without sharing production material |
| Pack engineer | Encodes the synthetic state machine, events, agents, and deterministic checks |
| Independent outcome-oracle author | Defines admissible and forbidden outcomes without reading the authored reference path |
| Independent negative-control reviewer | Confirms each negative agent attacks one declared guarantee and leaves unrelated dimensions passing |
| Privacy reviewer | Confirms every concrete detail is invented and the disclaimer is accurate |
| OperateBench maintainer | Owns scope, compatibility, CI, and admission decisions |

The partner subject-matter expert and pack engineer support development. The
independent roles and the separation rule below are gates only for a future
evidence-admission process: one person may not author the fixture and
independently certify the same oracle.

## What happens after submission

1. **Fit review** — operation versus task, novelty relative to existing cases,
   and safety of the proposed abstraction.
2. **Synthetic design workshop** — one semantic scenario, variants, actors,
   events, obligations, valid outcomes, and failure modes are frozen.
3. **Guided implementation** — after approval, generate the complete inert
   repository contribution from a fresh checkout with:

   ```console
   operatebench init-company-operation --repository-root . --owner-id example_partner --owner-display-name "Example Ledger Partner Ltd" --operation-name invoice_review --pack-id partner.example_partner.invoice_review.synthetic --operation-id example_partner_invoice_review_synthetic_v1
   ```

   The command derives every source, test, example, and documentation path from
   the repository root and validated identity components; it has no destination
   option. It creates `OWNER.yaml` for a new owner or reuses an exact existing
   declaration without changing its bytes. Creation is one multi-root
   transaction with safe rollback, not a claim of portable atomic rename. Then
   co-author the strict loader, state
   machine, evaluator, reference/negative agents, oracle and replay. The
   scaffold never registers or runs itself.

   During pre-beta development, every owner-scoped public source, test, example, and
   documentation tree is restricted to finite text resources with suffixes
   `.py`, `.yaml`, `.yml`, `.json`, `.md`, or `.txt`. Hidden entries, symlinks,
   special files, native/binary payloads, images, CSV files, and non-canonical
   bytecode caches are rejected. Canonical generated CPython bytecode caches are
   tolerated inside source and test owner trees for local Python compatibility.
   The shared source and test roots likewise tolerate only an immediate canonical
   CPython `__pycache__` alongside their inert initializer and owner directories.
   Shared example and documentation roots reject caches and contain owner directories
   only; root-level maintainer README files are not allowed.
4. **Development registration** — passing conformance and static registration
   makes a pack usable and discoverable for development. Development
   registration neither requires nor records independent outcome-oracle,
   privacy, negative-control, or matched-control review. It does not make the
   pack official or sealed research evidence.
5. **Future evidence admission** — independent outcome-oracle,
   negative-control, privacy, and matched-control review belong to a separate
   evidence-admission and promotion process that is unavailable in SDK v2. The
   roles and separation expectations above remain the scientific QC gates for
   that future process.

See [Contributing an operation](CONTRIBUTING_OPERATIONS.md) for the planned
seven-stage scientific QC process.

## Contribution and claim boundary

- Code and tests are Apache-2.0; fixtures and documentation are CC BY 4.0.
- Contributors retain copyright. Every commit uses the DCO `Signed-off-by`
  trailer described in [CONTRIBUTING.md](../CONTRIBUTING.md).
- `OWNER.yaml` declares `code_license: Apache-2.0` for source/tests and
  `content_license: CC-BY-4.0` for fixtures/docs; neither field assigns copyright.
- No proposal authorises provider calls, model evaluation, publication, a
  leaderboard, or a claim about the contributing company.
- A development pack is not official benchmark evidence until the separately
  specified identity, matched-control, review, and release gates pass.
- Self-service is not considered proven until a real partner completes the cold
  path from proposal through pack PR without maintainer coding.
