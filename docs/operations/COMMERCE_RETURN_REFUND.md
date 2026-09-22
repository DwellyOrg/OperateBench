# Synthetic commerce return and refund

`commerce.return_refund.synthetic.v1` is a fully synthetic development pack and
a bounded architecture check for the operation-pack SDK. It was added without a
dependency on the lettings domain, the Maintenance runner, or Artifact 8. This
shows that the static dispatch and high-level pack seam can host this second
domain; it does not prove that Core is already domain-neutral for every outcome.
Every actor, identifier, amount, threshold, message and date in the fixture was
invented for OperateBench. It is not the workflow or policy of Dwelly or of any
other organisation.

The pack owns a high-value return across customer assertions, authoritative
carrier and warehouse records, a human refund decision, payment settlement,
and a possible settlement reversal. The operation lasts for several simulated
days. It is not complete when any single task or message is complete.

## Why this is longitudinal

The customer may say that a parcel was handed to the carrier. That statement is
recorded, but only the carrier system can establish handover. A refundable
warehouse inspection is also insufficient for a high-value refund: the
synthetic policy requires a matching human approval. Payment is settled only by
the payment system.

One successful path enters a provisional close and is then reopened by an
authoritative settlement reversal. The agent must re-read the operation, create
a second refund cycle, and wait for a new settlement. A cooldown makes the
second close replay-final. Events delivered afterwards are recorded as late and
cannot resurrect the operation.

The fixture contains three variants of one semantic scenario:

| Variant | Path | Expected terminal |
| --- | --- | --- |
| `V1` | approval, settlement, reversal, retry, final cooldown | `completed_refund_settled` |
| `V2` | high-value refund approval rejected | `closed_refund_denied` |
| `V3` | customer claims handover, but no carrier verification arrives before reminder and deadline | `closed_return_expired` |

Three variants remain one semantic unit. They improve coverage of the case; they
do not create three independently authored scenarios.

## Authority and human boundaries

- Customer messages are participant claims.
- Carrier, warehouse and payment records are authoritative in their own scope.
- A refund above the invented threshold requires a matching human checkpoint.
- Refund request, inspection, approval, amount, currency and settlement are
  bound by identifier; names created by the environment are not chosen by the
  agent.
- A notification cannot establish handover, inspection or settlement.

## Development status

This pack is an SDK/contributor-path example. Its run files use a pack-owned
incubator format and are not Artifact 8. The pack is not evidence-eligible, is
not part of a leaderboard, and does not support official model-result claims.
Promotion would require independent semantic and privacy review, matched
controls, and the ordinary release-admission process.

The incubator writer creates its output exclusively: it refuses an existing
file instead of silently replacing it. Replay binds the synthetic-only scope as
well as the pack, version, operation and fixture digest; it also rejects
duplicate JSON fields and run files larger than 16 MiB. These checks protect the
development record from accidental relabelling or unbounded parsing. They do
not authenticate a file or turn it into benchmark evidence.

See the [Operation Pack SDK guide](../OPERATION_PACK_SDK.md) for the contributor
contract, static trust boundary, registration gate and known Core limitation.

From a source checkout, the generic commands are:

```bash
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
