<!-- SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Property compliance: four bounded jurisdiction profiles

**Synthetic-only development benchmark, version 0.1.0.** This is not legal
advice, an official certificate, permission to occupy or let, evidence admission,
a real contractor-register check, or proof that a human reviewed a property.
All properties, credentials, reports, communications and receipts are fixtures.
Source research was frozen on 26 September 2026; deployment needs renewed legal
and technical validation. The legacy synthetic compliance contribution is unchanged.

## Identity and interface

- Pack: `lettings.property_compliance.profiles.v1`
- Operation type: `lettings.property_compliance.profiles`
- Operation ID: `lettings_property_compliance_profiles_v1`
- Module: `operatebench.contributions.prospire.property_compliance_profiles`
- [Fixture](../../../examples/contributions/prospire/property_compliance_profiles/operation.yaml)
- [Manifest](../../../src/operatebench/contributions/prospire/property_compliance_profiles/CONTRIBUTION.yaml)
- [Executable tests](../../../tests/contributions/prospire/property_compliance_profiles/test_pack.py)

The package exports `load_spec(path)`, `build_domain(spec, scenario_id)`,
`build_agent(agent_id)`, `evaluate_episode(episode, spec, scenario_id)`,
`profile_identity(spec, scenario_id)`, `REFERENCE_SCENARIOS` and
`NEGATIVE_CONTROLS`. The immutable Spec exposes `operation_id`, `content_digest`,
`profiles` and `scenarios`. JSON syntax is used in the YAML fixture. Duplicate
keys, nonfinite constants, oversized input and any canonical-content change
from the reviewed digest are refused: this is a finite fixture, not a general
legal-policy authoring engine. Profile identity returns `profile_id`,
`profile_version`, `profile_digest` and `jurisdiction`; each profile version is
`2026-09-26.v1`.

```python
from operatebench.core.engine import Engine
from operatebench.contributions.prospire import property_compliance_profiles as api

spec = api.load_spec(
    "examples/contributions/prospire/property_compliance_profiles/operation.yaml"
)
episode = Engine(
    api.build_domain(spec, "EN_normal"),
    api.build_agent("reference"),
    identity={
        "operation_id": spec.operation_id,
        "operation_instance_id": "opaque-example-instance",
        "agent_id": "reference",
    },
).run()
grade = api.evaluate_episode(episode, spec, "EN_normal")
assert grade["reliable"]
```

Registered as `lettings.property_compliance.profiles.v1` with status `incubator`
and `evidence_eligible=False`; see the shared [profile CLI](../../profile-pack-cli.md).
Contribution-import admission remains integration-owned. No paid provider is
needed; tests replay persisted Core JSON decision tapes through a fresh domain,
not as official evidence.

## Implemented profiles and public source map

Every profile in the fixture embeds `source_claims` and `sources` entries with
claim ID, official URL, access timestamp and retrieved quote. The table maps
those claims to the finite implementation, not to every possible legal duty.
England legislation takes precedence over a contradictory guidance FAQ: remedial
confirmation plus the original report goes to the council **without requiring
a council request**. Original-report request duties are distinct.

| Profile and bounded population | Source claims / official source | Executable distinctions |
| --- | --- | --- |
| `EN_PRIVATE_ELECTRICAL`: specified private residential tenancies in England; not social housing | EN01, EN04–05: [revised regulation 3](https://www.legislation.gov.uk/uksi/2020/312/regulation/3); EN02–03, EN06, EN08: [ministry guidance](https://www.gov.uk/government/publications/electrical-safety-standards-in-the-private-and-social-rented-sectors-guidance/electrical-safety-standards-in-the-private-and-social-rented-sectors-guidance) | Five-year maximum and earlier report due date; qualified, correctly addressed installation report; C3 does not require repair; linked qualified remedy confirmation, no periodic reset; tenant and council remedial delivery receipts; inspection-triggered authored shorter remedy deadline. |
| `SC_PRIVATE_ELECTRICAL`: private residential Repairing Standard; not agricultural extension or separate fire/CO certification | SC01: [statutory guidance](https://www.gov.scot/publications/repairing-standard-statutory-guidance-private-landlords/); SC02–05, SC07–08: [Annex D3](https://www.gov.scot/publications/repairing-standard-statutory-guidance-private-landlords/pages/16/); SC11: [enforcement/access route](https://www.gov.scot/publications/repairing-standard-statutory-guidance-private-landlords/pages/7/) | Installation plus landlord equipment; accredited scheme **or competence checklist** for installation; trained PAT route only for equipment; separate equipment due dates; no imported English statutory 28-day remedy clock; refused access routes to a synthetic Right Of Entry review request, not a Tribunal decision. |
| `AU_VIC_RENTAL_CHECKS`: ordinary newly entered post-29 March 2021 Victorian agreements with gas in scope | VI01–03: [CAV safety-check duties](https://www.consumer.vic.gov.au/housing/renting/repairs-alterations-safety-and-pets/gas-electrical-and-water-safety-standards/rental-providers-gas-and-electrical-safety); VI04–06: [CAV repair urgency](https://www.consumer.vic.gov.au/housing/renting/repairs-alterations-safety-and-pets/repairs/repairs-in-rental-properties) | Gas and electrical records have independent two-year clocks; Type A servicing gasfitter and electrician/REC capabilities are separate; expired gas cannot be renewed by current electrical evidence; dangerous faults require remedy rather than borrowed UK defect classifications. |
| `NZ_HEALTHY_HOMES`: ordinary non-exempt residential rentals | NZ01–03: [compliance](https://www.tenancy.govt.nz/healthy-homes/healthy-homes-compliance); NZ04–07: [statement](https://www.tenancy.govt.nz/healthy-homes/compliance-statement); NZ08: [ongoing upkeep](https://www.tenancy.govt.nz/healthy-homes/healthy-homes-standards-what-a-landlord-needs-to-know/keeping-your-property-up-to-standard) | Five evidenced standards (heating, insulation, ventilation, moisture/drainage, draught stopping), landlord/agent signature rather than assessor logo alone; null statutory `next_due`; deterioration/replacement reopens evidence, repair verification and statement delivery. |

The full quotes include contextual and research-only branches. Presence of a
quote does not imply an executable scenario for every sentence. Qualification
labels are synthetic evidence attestations, not reconstructed BS/AS/NZS technical
standards or actual licence lookups. Current fixture dates are explicitly pinned;
calendar expiry checks use inclusive due dates. This is not a general local-time,
DST, leap-day anniversary or legal time-computation service.

## Core process, observation and grading

A genuine Core OperationDomain builds temporal events and triggers. The agent
retrieves the current compliance record through `get_compliance_record`, sees
public policy and source rules, and proposes actions: `request_inspection`,
`accept_report`, `request_remediation`, `send_bundle`, `request_review` and
`provisionally_close`. It then waits for inspector, contractor, verifier,
messenger, reviewer or clock events. A contractor's repair claim is not a
qualified verification; a send request is not a delivery receipt. Replaced
reports remain in history and reopen acceptance, work, delivery and finality.
Stale versions and stale reads are rejected. Access refusal does not extend
expiry or authorize forced entry.

Only ACT, WAIT and COMPLETE are supported. ASK and Core ESCALATE are unsupported;
use a domain `request_review` ACT followed by WAIT. A reviewer acknowledgement
means synthetic workflow referral, not human approval or resolved compliance.
Thirty-minute quiet periods, 120-minute no-response thresholds, and service
objectives outside sourced numeric deadlines are **benchmark policy**, not law.
Required deliveries are already triggered in the authored cases; this slice is
not an implementation of every possible request/tenancy trigger.

Terminals include `VERIFIED_CURRENT`, `ACCESS_BLOCKED_REFERRED` and
`EVIDENCE_REFERRED`. Current means only that the scoped fixture gates are met.
Historical lateness remains a scoring failure even if later evidence is current.
The grader returns JSON-safe `reliable`, `dimensions`, `findings` and
`terminal_outcome`. It authenticates Core provenance, binds the exact fixture,
scenario and profile, and independently walks causal evidence/actions rather
than trusting reducer readiness or terminal labels. Rejected unsafe proposals
remain failures after subsequent recovery. Dimensions are provenance, scope,
qualification, verification, delivery, temporal, versioning, access, retrieval,
decision validity and terminal correctness.

The reference and negative agents operate on observations, not scenario IDs,
expected outcomes or hidden event queues. Tests round-trip the model-visible
projection through JSON for every reference scenario and reject hidden fields.
Additional corrupt-reducer tests check that unqualified acceptance and unjustified
referral are caught independently. Delivered initial and replacement report values
are compared to the frozen authored fixture, not merely to their self-asserted
hashes (`AUTHORED_REPORT_MISMATCH`); this grader-only authority is never exposed
to the agent. Runtime and grader separately enforce the public synthetic
30-minute quiet interval, bound to the provisional-close timestamp and report
generation. Old-generation timers cannot close a reopened case; invalid cause
or elapsed time produces `INVALID_QUIET_CONTRACT` in the grader. This does not
promise knowledge of unknown future reports. The grader also reconstructs full
verification payloads and delivery bundles, recipients, request times and
receipt times, comparing them to persisted final evidence
(`CAUSAL_STATE_MISMATCH`). Runtime ingestion, delivery and readiness gates require
qualified, successful, property/defect-linked confirmation with an unchanged
periodic due date; false qualification or standards assertions cannot authorize
a bundle or `VERIFIED_CURRENT`. Corruption tests use genuine Core executions and
paired legitimate domain subclasses, not blanket subclass rejection.

## Finite coverage and controls

There are 25 scenarios. Each prefix `EN`, `SC`, `AU`, `NZ` has `_normal`,
`_remediation`, `_access`, `_reopen` and `_qualification`. Additional cases are
`SC_checklist`, `AU_gas_expired`, `NZ_assessor_only`, `EN_wrong_property` and
`EN_no_response`. The normal cases also prevent an always-refuse strategy from
passing. Alternate council-first delivery is legitimate, not a negative control.

`NEGATIVE_CONTROLS` is a tuple of mappings with `agent_id`, `scenario_id`,
`failed_dimensions`, `finding_codes` and `must_pass` (all other dimensions).
The 14 independently authored rows are:

| Agent | Scenario | Required finding |
| --- | --- | --- |
| `delay_remedy` | `EN_remediation` | `LATE_REMEDIATION` |
| `skip_reads` | `EN_normal` | `ACTION_WITHOUT_RETRIEVAL` |
| `premature_complete` | `EN_normal` | `REPORTS_MISSING` |
| `trust_bad_report` | `EN_wrong_property` | `PROPERTY_SCOPE` |
| `trust_bad_report` | `SC_qualification` | `QUALIFICATION` |
| `trust_bad_report` | `AU_qualification` | `QUALIFICATION` |
| `trust_bad_report` | `AU_gas_expired` | `EXPIRY` |
| `trust_bad_report` | `NZ_assessor_only` | `SIGNATURE` |
| `trust_repair_claim` | `EN_remediation` | `VERIFICATION` |
| `wrong_recipient` | `NZ_normal` | `RECIPIENT` |
| `skip_council` | `EN_remediation` | `DELIVERY` |
| `stale_version` | `EN_reopen` | `VERSION` |
| `force_access` | `SC_access` | `ACCESS` |
| `unnecessary_review` | `AU_normal` | `UNNECESSARY_REVIEW` |

Tests assert exact failing dimensions and finding codes, all required passing
dimensions, and eventual replay-final recovery. They also cover strict fixture
loading, wrong-case/forged episode refusal, expiry boundaries, independent
Victorian clocks, NZ null expiry and persisted decision-tape replay.

## Research-only and excluded branches

GB annual gas (England and Scotland), early-check anniversary preservation,
and standalone NZ smoke alarms are **not implemented profiles**. Also excluded:
England social-housing transition, new-build/full-rewire EIC substitution and
all tenancy/request trigger permutations; Scottish new-equipment purchase
exemption and separate alarm/CO standards; Victorian legacy/gas rollover and
rooming-house cohorts, detailed entry law and technical standards; NZ exemptions,
capacity calculations, licensed-work certificate subtypes and smoke/access law.
No global exemption or safety determination follows from exclusion. Ambiguous
technical/cohort rules require qualified review, not invented rules. Unknown
profiles cannot be loaded in this pinned fixture; this is refusal of unsupported
configuration, not an implemented universal out-of-profile workflow.

The finite synthetic evidence does not establish live operational robustness,
legal compliance, production authority or real model performance. Shared SDK
registration, canonical development run records and broader integration testing
remain outside this contribution's ownership.
