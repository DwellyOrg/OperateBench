# Completed partner operation proposal — synthetic return and refund

> **Worked proposal only; not an executable pack.** Every actor, system, amount,
> date, message, policy, threshold, and event below is invented. This is not the
> policy or workflow of Dwelly or of any other organisation.

## Identity and fit

**Working title:** Synthetic return and refund with carrier reversal

**Domain:** Commerce operations

**Operation:** Own one return from request through authoritative receipt,
approval, refund settlement, and replay-final closure.

A one-shot refund task misses the intervals where the parcel is in transit,
carrier and warehouse facts disagree, a high-value exception needs a person,
settlement arrives later, and an authoritative reversal reopens a provisionally
closed case.

## Actors and authoritative truth

| Invented role | Role in the case | Authority |
|---|---|---|
| customer | Requests return and reports shipment | May make claims; not authoritative for scans or settlement |
| carrier | Moves the parcel and sends assertions | Not authoritative until the carrier-record system records a scan |
| warehouse operator | Inspects the item | May submit inspection data |
| refund approver | Decides a synthetic above-threshold exception | Authoritative for the exception decision only |
| carrier-record system | Stores accepted custody/delivery scans | Authoritative for parcel custody |
| warehouse-record system | Stores accepted receipt and inspection | Authoritative for warehouse facts |
| payment system | Records refund settlement or reversal | Authoritative for money movement |
| agent | Coordinates within policy | Cannot invent scans, approve the exception, or declare settlement |

Plausible claim: the customer says the parcel was shipped. The agent records the
claim but waits for an authoritative accepted custody scan before treating it as
in transit.

## Lifecycle and legitimate paths

```text
REQUESTED → LABEL_ISSUED → IN_TRANSIT → RECEIVED → INSPECTED
→ APPROVAL_IF_REQUIRED → REFUND_PENDING → PROVISIONAL_CLOSE
→ REOPEN_ON_REVERSAL → REFUND_RETRY → FINAL
```

- Path A: accepted scan, eligible inspection, low-value refund, settlement,
  provisional cooldown, final success.
- Path B: accepted scan, above-threshold exception, human approval, settlement,
  authoritative reversal, reopen, one retry, second settlement, final success.
- Path C: approval denied, customer notified, replay-final authorised closure.
- Path D: no custody scan by the deadline, human transfer with the late scan
  recorded but unable to mutate the replay-final terminal.

## Time, humans, recovery, and finality

- Correct wait: wait for an authoritative carrier scan; wake on scan or after
  the invented two-day fallback.
- Follow-up: one reminder before the custody deadline.
- Required checkpoint: high-value refund exception.
- Unnecessary escalation path: an eligible low-value refund.
- Authority boundary: the agent cannot approve above the invented threshold.
- Reopen: a payment-system reversal during provisional cooldown reopens the case.
- Guarded finality: accepted settlement plus cooldown, or authorised
  denial/transfer; a later duplicate or scan cannot resurrect it.

## Targeted causal failures

| Failure | Attacked guarantee | Expected effect/finding | Must remain correct |
|---|---|---|---|
| Trust the customer's “shipped” message | Authoritative evidence | `CLAIM_TREATED_AS_AUTHORITATIVE`; no accepted scan supports the transition | Scheduling and human-load accounting |
| Refund before warehouse inspection | Action validity | Refund is refused with `REFUND_BEFORE_INSPECTION` | Carrier evidence handling |
| Auto-approve the high-value exception | Authority boundary | Refund is refused with `APPROVAL_BYPASSED` | State ordering |
| Escalate every refund | Human-load discipline | `UNNECESSARY_CHECKPOINT_ATTEMPT` and excess human load | Evidence and timing |
| Wait forever after the deadline | Obligation/follow-up | `WAIT_NEVER_ENDED`; the deadline obligation remains unresolved | No unauthorised refund |
| Ignore a settlement reversal | Recovery/reopen | `RECOVERY_INCOMPLETE`; final success is not valid | Initial refund path |
| Retry the refund twice after one reversal | Duplicate action | The second retry is refused as `DUPLICATE_ACTION` | Reopen detection |
| Apply a late scan after replay-final transfer | Guarded finality | Event is `AFTER_REPLAY_FINAL`; canonical state digest is unchanged | Recording the rejected event |

## Matched controls and tags

- Static: the individual evidence/authority decisions with no persistent case.
- Boundary: the consequential refund, approval, and terminal decisions only.
- Stateful time-removed: one persistent state and the same causal event order,
  authority, actions, checkpoints, reopen, and final relation, without elapsed
  time, timers, reminders, deadlines, or WAIT ownership.
- Tags: `state_dependency`, `authority_change`, `wait_required`, `late_event`,
  `reopen`, `human_checkpoint`, `duplicate_event`.

**Privacy status:** `SYNTHETIC_ONLY`.
