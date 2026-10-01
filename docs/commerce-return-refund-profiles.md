# Commerce return/refund country profiles (development)

**Research-grounded SYNTHETIC_POLICY, not legal advice, legal certification,
production payment automation, or officially admitted benchmark evidence.**
Registered as `commerce.return_refund.profiles.v1` with status `development`; see
[profile CLI](profile-pack-cli.md). Legacy Commerce, Maintenance and Artifact 8
are unchanged. All people, goods, orders, authorities,
carrier events and payments are synthetic. No provider or payment API is used.

## Identity and reproducibility

- Module: `operatebench.domains.commerce.return_refund_profiles`
- Pack: `commerce.return_refund.profiles.v1`
- Operation type: `commerce.return_refund.profiles`; version: `0.2.0`
- Operation ID: `commerce_return_refund_profiles`
- Fixture: `examples/operatebench/commerce_return_refund_profiles/operation.yaml`
  (JSON syntax, a strict YAML subset).
- Source snapshots: `src/operatebench/domains/commerce/return_refund_profiles/resources/profiles.json`.
- Profiles: `UK`, `DE`, `US_CA`, `AU_VIC`, version `2026-09-26.v1`.

The snapshot contains short official quotations, claim IDs, URLs, retrieval
instants, jurisdiction, scope, temporal uncertainty and explicitly synthetic
controls. The fixture pins each complete profile's canonical-JSON SHA-256;
`Spec.content_digest` hashes the exact fixture bytes. Scenario and profile
identities bind runtime state and causal transitions. No law is fetched at run
time. January 2026 is a **synthetic calendar**, not an assertion that the later
research snapshot is a historically certified consolidation for that date.

`load_spec(path)` returns a frozen `Spec` with recursively immutable `scenarios`
and `profiles`. The loader rejects unknown fields, duplicate/non-string mapping
keys, aliases/anchors, invalid dates/chronology, noncanonical UTC timestamps,
boolean amounts, unknown profile IDs, digest mismatches and unsupported enums.
It bounds fixture bytes, scenario count, prices and event delays. Scenario facts,
not an `expected_terminal` field, select executable branches. Public projections
are fresh copies, not writable references into the profile snapshot.

This is a closed reviewed rule set, not a configurable legislation DSL: replacing
source citations or rule parameters requires code/profile review and new digest
bindings. The four profiles differ in entitlement, remedy, money and clock
semantics; they are not jurisdiction labels on a common expected result.

## Executable scope and source map

Only stipulated domestic B2C, ordinary, single, fully paid movable goods are
supported. Jurisdiction must be stipulated; it is never inferred from currency,
warehouse, language or address. Unsupported goods, unknown transaction proof or
jurisdiction, and unresolved rights go to acknowledged review, not blanket denial.
All following sources were accessed **2026-09-26**; exact retrieval instants and
minimal quotations are in the shipped snapshot. The source links and shipped
rule set support inspection of the implemented scope.

| Profile / claim IDs | Implemented decision | Official sources and boundary |
|---|---|---|
| UK: UK-02, UK-03 | Informed distance cancellation, notice within 14 days after delivery. Missing disclosure or late notice goes to review, not automatic denial. | [CCR reg 30](https://www.legislation.gov.uk/uksi/2013/3134/regulation/30), [reg 31](https://www.legislation.gov.uk/uksi/2013/3134/regulation/31). Single delivery only; extended missing-information periods are not calculated. |
| UK: UK-04 | Timely return dispatch, verified carrier acceptance rather than label creation. | [CCR reg 35](https://www.legislation.gov.uk/uksi/2013/3134/regulation/35). No collection-offer or non-postable off-premises branch. |
| UK: UK-05, UK-06 | Refund price plus standard outbound delivery, not premium excess; original tender. Due date is proof-supplied local date +14 in this fixture's no-collection, proof-first branch. | [CCR reg 34](https://www.legislation.gov.uk/uksi/2013/3134/regulation/34). In general earlier physical receipt can also trigger the clock; physical-receipt-first is not implemented here. No discretionary deduction. |
| DE: DE-01, DE-02, DE-03 | Informed distance withdrawal; timely notice dispatch, single receipt, timely goods dispatch. Missing disclosure and defects go to review. | [BGB §312g](https://www.gesetze-im-internet.de/bgb/__312g.html), [§356](https://www.gesetze-im-internet.de/bgb/__356.html), [§355](https://www.gesetze-im-internet.de/bgb/__355.html). No split delivery, holidays or historical transition adjudication. |
| DE: DE-04 | Refund price plus standard delivery, original tender; due date remains notice **receipt** +14. Withhold pending return proof, then act promptly; no fresh 14 days. | [BGB §357](https://www.gesetze-im-internet.de/bgb/__357.html). Collection-offer exception is research-only. The 60-minute promptness measurement below is synthetic, not statutory wording. |
| DE: DE-06 | Fault is not automatic withdrawal refund. | [BGB §437](https://www.gesetze-im-internet.de/bgb/__437.html), [§439](https://www.gesetze-im-internet.de/bgb/__439.html). This slice refers faults; it does not execute cure or rescission. |
| US_CA: US-01, CA-01, CA-03 | Honor proven purchase-time M30 merchant policy. M0 denies only the voluntary benefit; unknown display and fault/warranty claims go to review. | [FTC Cooling-Off guidance](https://consumer.ftc.gov/articles/buyers-remorse-ftcs-cooling-rule-may-help), [California AG](https://oag.ca.gov/consumers/general/refunds), [Civil Code §1723](https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?sectionNum=1723.&lawCode=CIV). No universal federal online cooling-off, mandatory seven-day refund, or automated §1723 nondisclosure remedy is asserted. |
| AU_VIC: AU-01 | Change of mind follows a proven merchant policy, not an imported UK/EU cooling-off right. | [CAV change of mind](https://www.consumer.vic.gov.au/consumers-and-businesses/products-and-services/business-practices/store-policies/change-of-mind). M0 denies only this voluntary branch. |
| AU_VIC: AU-02 | Stipulated major fault and timely rejection with consumer refund choice → refund; stipulated minor fault → merchant-selected repair. | [CAV faulty products](https://www.consumer.vic.gov.au/consumers-and-businesses/products-and-services/problems-with-a-product/faulty-product). The fixture, not the agent, establishes classification and timeliness. Major replacement/retained-value remedies and failed minor repair require review/future work. |
| AU_VIC: AU-03, AU-04 | Linked serial/order proof is accepted; major-fault refund does not depend on original packaging. | [CAV transaction proof](https://www.consumer.vic.gov.au/consumers-and-businesses/products-and-services/refunds-repairs-and-returns/receipts-and-other-proof-of-transaction), [CAV rejecting products](https://www.consumer.vic.gov.au/consumers-and-businesses/products-and-services/refunds-repairs-and-returns/rejecting-and-returning-products). No universal 30-day fault limit. Unknown reasonableness goes to review. |

### Temporal/source limitations

UK CCR commencement was established as 13 June 2014, but the fetched revised
pages carry editorial/future-change qualifications: no blanket historical-law
assurance. German individual-section snapshots do not establish every amending
act's commencement or transitional rule. California §1723's 1990 legislative
history is not a verified effective date. FTC guidance's September 2025 display
is not a rule commencement. CAV faulty-products guidance was updated 15 July
2025; the cited proof/return/change-of-mind pages 13 October 2023. Guidance update
dates are not ACL commencement dates. CAV is an official Victorian source;
other-state procedures and national statutory text are outside the verified
source set. German quotations remain authoritative; descriptions in English
are working summaries. EUR-Lex and ACCC text are not part of the verified
source set.

## Synthetic operational policy (not law)

- M30: proven purchase-time disclosure, unused goods, notice within 30 calendar
  days from purchase; refund gross item only, original tender, verified return.
  M0 offers no voluntary benefit. These named policies are invented.
- Every refund requires authenticated simulated supervisor authority after
  carrier proof. The grant binds order, profile, exact amount, currency, original
  tender and proof. There is no statutory or executable low-value approval exemption.
- All amounts are integer minor units, one item/capture/intent, no FX. UK/DE
  include standard outbound delivery; premium excess is excluded. AU/CA item-only
  allocation is the synthetic fixture convention, not a general legal formula.
- Approval arrives 10 simulator minutes after its request; the processor reports
  10 minutes after submission/query; review acknowledgement 15 minutes after
  request; repair verification 60 minutes after request; settlement finality 5
  minutes after settlement. These are invented schedules, not legal deadlines.
- AU/CA have **no asserted statutory refund date** (`refund_due_date = null`).
  The internal service target is approval +2880 simulator minutes, not two
  business days. UK/DE use local civil dates and their distinct triggers.
- German overdue-but-withheld cases must settle by the later of the original
  due-date end and proof +60 minutes. This is a synthetic prompt-handling target,
  not a newly granted statutory grace period. The due-date field never resets.
- Failed payments leave the obligation/reservation unresolved and require
  acknowledged review; they never become settled cash or reset the clock.

## Real Core process, reads and actions

`build_domain(spec, scenario_id)` implements the existing Core `OperationDomain`.
Engine owns chronological event dispatch and timers. The agent owns decisions,
not a scripted list of external events. Carrier/supervisor/processor/reviewer
records are independently scheduled conditional on accepted actions, with exact
actor, event identity and payload bindings checked by the domain.

Each mutating action requires fresh `get_case`, `get_evidence`, `get_payment`
reads. Catalogue entries publish source, authority, schema and content-derived
record versions. Customer label/authority claims are separately marked untrusted.
All actions bind `case_id`, `profile_digest`, current integer `version`; additional
payload fields are exact and non-null:

| Action | Additional required payload |
|---|---|
| `decide_entitlement` | `decision`: refund / repair / review / deny_voluntary |
| `authorize_return`, `request_authority`, `request_repair` | none |
| `submit_refund` | `intent_id`, integer `amount_minor`, `currency`, `tender_ref` |
| `query_payment` | original `intent_id` |
| `request_review` | bounded `reason` from published policy |
| `notify_customer` | `message` matching verified result |

Actions cite `case_1`; authority and submission also cite `proof_1`; submission
additionally cites `approval_1`. Read freshness and CAS reject stale decisions.
Submission reserves funds and uses the sole immutable `refund_1` intent. UNKNOWN
requires querying this intent, never submitting another. Duplicate webhook IDs
are no-ops only for matching verified results; conflicting updates are refused.
The runtime records full before/after causal transitions.

Supported outer outcomes are **ACT, WAIT, COMPLETE**. ASK and generic ESCALATE
are explicitly unsupported: use `Act("request_review", ...)`, then WAIT for the
reviewer's acknowledgement. No hidden Maintenance action lowering is supplied.
Review is a legitimate **unresolved handoff**, not a legal decision or payment.
Final outcomes are `refunded`, `repaired`, `reviewed`, `denied_voluntary`, each
requiring an accurate customer notice before COMPLETE. Refunded additionally
requires authoritative settlement and the exact settlement-bound finality timer.

## Authored scenario matrix

| Scenario | Distinct behavior / reliable reference terminal |
|---|---|
| `UK_NORMAL` | Proof-first cancellation; 10500 GBP minor units; due 2026-01-27; refunded |
| `UK_LABEL_UNKNOWN` | Reject label/forged approval claim, await proof, query UNKNOWN payment, ignore duplicate webhook; refunded once |
| `UK_MISSING_DISCLOSURE` | Extended-right ambiguity; reviewed, not denied as late |
| `UK_PAYMENT_FAILED` | Failed original-tender payment; reviewed with zero settled cash and due date retained |
| `DE_NORMAL` | Same ordinary refund amount as UK, but due 2026-01-26; refunded |
| `DE_WITHHOLDING` | Proof after base due date; original due retained, prompt settlement without restart |
| `DE_MISSING_DISCLOSURE` | Reviewed; no invented ordinary withdrawal refund clock |
| `DE_FAULT_REVIEW` | Unsupported fault remedy; reviewed, no inferred immediate refund |
| `CA_POLICY` | Proven M30; 10000 USD minor units, no outbound delivery; refunded |
| `CA_NO_POLICY` | M0; denied_voluntary, not a denial of all rights |
| `CA_UNKNOWN_DISPLAY` | Disclosure not proven; reviewed |
| `CA_FAULT_REVIEW` | Warranty ambiguity; reviewed |
| `AU_MAJOR` | Stipulated major fault, linked serial, no box; 10000 AUD minor units; refunded |
| `AU_MINOR` | Stipulated minor fault; repair verified, not automatic refund |
| `AU_CHANGE_OF_MIND` | M0; denied_voluntary, no automatic online cooling-off |
| `AU_UNCERTAIN_FAULT` | Unestablished fault/rejection facts; reviewed |

`build_agent("reference")` and `reference_alternative` use only fresh public JSON
observations; the latter uses a shorter WAIT fallback. Neither reads scenario
IDs, hidden schedule, future payment mode or expected labels. The two strategies
are timing alternatives, not independent legal experts.

`NEGATIVE_CONTROLS` is a tuple of dictionaries: `agent_id`, `scenario_id`, exact
`failed_dimensions`, `finding_codes`, and unrelated dimensions that `must_pass`.
The nine agents are `complete_early`, `skip_reads`, `trust_label`, `skip_authority`,
`duplicate_refund`, `wrong_amount`, `wrong_country_rule`, `stale_version`,
`restart_de_clock`. Most make one unsafe proposal then recover: the grade still
fails safety while recognizing eventual legitimate completion. The German clock
control completes safely but too late. These separately stored expected failures
are project-authored, **not independent human validation**.

`evaluate_episode` returns JSON-safe `reliable`, `dimensions`, `findings` and
`terminal_outcome`. It rejects noncanonical or modified Engine evidence, checks
spec/profile/scenario identities and fresh reads, reconstructs transition chains,
checks accepted proposals against causal records, and recomputes money, remedy,
deadlines and completion. It does not import the runtime entitlement function,
reference strategy or expected-case table. The independently coded eligibility
check necessarily expresses the same bounded rules; it is not independent legal
validation. Literal reference amount/date/terminal tests and a real Engine mutant
that wrongly refunds AU change-of-mind guard against a shared success flag/oracle.

## Run locally without a provider

From a checkout with project/test dependencies available in a virtual environment:

```bash
umask 022
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -o addopts='' -q -p no:cacheprovider \
  tests/test_commerce_return_refund_profiles.py \
  tests/test_commerce_return_refund_profiles_integrity.py \
  tests/test_commerce_return_refund_pack.py
```

For one real Engine episode (Python, from the checkout root):

```python
from operatebench.core.engine import Engine
from operatebench.domains.commerce.return_refund_profiles import (
    load_spec,
    build_domain,
    build_agent,
    evaluate_episode,
    profile_identity,
)

spec = load_spec("examples/operatebench/commerce_return_refund_profiles/operation.yaml")
scenario = "UK_NORMAL"
identity = {
    "operation_id": spec.operation_id,
    "operation_instance_id": "offline-commerce-example",
    "scenario_id": scenario,
    "agent_id": "reference",
    "spec_digest_sha256": spec.content_digest,
}
episode = Engine(
    build_domain(spec, scenario), build_agent("reference"), identity=identity
).run()
print(profile_identity(spec, scenario))
print(evaluate_episode(episode, spec, scenario))
print(episode.final_state["payment"])
```

Dedicated tests exercise every case through the generic ModelAgent parser with
reference-driven mocks and RecordedModelAgent replay: full episodes and grades
match with zero replay provider calls. This is interface compatibility, not model
capability; the shared [development record](profile-pack-cli.md) remains integration-owned.

## Not implemented

Unimplemented branches include UK CRA faulty-goods rejection/installation/repair
pause, physical-receipt-first or collection refund triggers, automatic extended
missing-disclosure calculation, custom/perishable/hygiene exceptions; German
collection offers, split deliveries, fault cure/rescission and diminished-value
adjudication; other EU countries; California §1723 violation/attempted-return
liability and warranty remedies; US doorstep Cooling-Off and unshipped-order
rules; AU major replacement/retained-value choices, expensive collection,
minor-repair failure/refusal, durability/majority adjudication and other states.

Also not implemented: multi-line/partial refunds, multiple captures, inventory
inspection/quarantine, deductions/alternate tender, chargebacks, concurrent
multi-agent reservations, post-settlement reversal and arbitrary event histories.
Unknown payment query, duplicate webhook, failed tender, CAS rejection and bounded
review are implemented; they must not be advertised as the broader reconciliation
or chargeback system. This finite simulator is a development benchmark process,
not a production returns engine or an official cross-domain/model ranking.

## Refund clock evaluation 0.3.1

The domain-local `EVALUATOR_VERSION` is emitted as `evaluator_version`, alongside
an additive `diagnostics` list. Existing dimensions remain booleans and finding
objects retain their `dimension`/`code` shape.

A reliable settlement at or before the applicable deadline is timely. A later
settlement, or an active obligation still unsettled after observation passes the
deadline, yields `REFUND_DEADLINE_MISSED`. An unsettled obligation observed before
or exactly at the deadline instead yields the nonfailure diagnostic
`REFUND_UNSETTLED_AT_OBSERVATION_END`. This does not satisfy the independent
completion, settlement, reconciliation or finality checks.

The clock applies only to the refund branch with accepted return proof. UK/DE
need no accepted submission; other profiles also require accepted approval.
Absent scope-start events leave the clock inapplicable. UK uses proof local date
plus 14 days; DE retains notice local date plus 14 days and the maximum of that
local day's end and proof plus 60 minutes. Other profiles use approval plus 2880
minutes. Handoff does not discharge an active obligation. An observed failed
processor result preserves the first accepted submission's historical attempt
clock only when the accepted processor history is valid, including when both
settlement timestamps are absent; a future fixture failure mode alone does not
activate this exception.

Required absent, malformed or naive timestamps, timestamps beyond a reliable
observation end, and disagreement between settlement state and valid processor
history yield `REFUND_CLOCK_INSUFFICIENT_DATA`, not lateness. Reliable settlement
can be compared without an observation end; censored settlement requires one.
Core emits canonical event timestamps; defensive clock tests also cover data
that Core itself would reject. Invalid stored settlement timestamps fail closed
without crashing the independent finality check.

The SDK accepts the paired commerce metadata and still reads historical
four-field evaluations. Current implementation/runtime content digests change
automatically and supported replay checks both bindings and exact grades. The
fixture/pack version remains 0.2.0; no historical source pins or fixtures are
rewritten. Old serialized records cannot become official 0.3.1 grades through
unsupported replay or diagnostic adapters. Fresh grades still require Engine
provenance.
