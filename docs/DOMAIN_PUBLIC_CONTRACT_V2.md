# Profile-domain public contract v2

The property-compliance profiles and commerce return/refund profiles expose
`public_contract_version=2` in their model-visible policy. This domain-local
marker is not an engine, package, provider mapping, fixture or evaluator version.
The producing source commit and implementation binding remain necessary identity.

Compliance publishes exact required payload fields from its canonical schemas,
including the empty `{}` case, no additional fields, non-empty string values,
and separate `evidence_refs`. The same guidance accompanies PAYLOAD refusals.
Commerce clarifies review ordering, pending versus unknown payment status,
settlement versus finality, evidence citations and one-time notices. Its refusal
codes now distinguish duplicate entitlement, notice readiness, finality readiness,
already-notified status and review ordering/repetition from substantive mismatch.

These changes alter observations and refusal diagnostics, not accepted business
effects, action guards or evaluator safety rules. A rejected proposal is not an
applied harmful effect; it still makes the episode unsafe under existing grading.
Successful recovery does not erase the rejected proposal.

## Historical meaning and replay

Earlier source builds emitted different observations and refusal diagnostics.
Historical tapes, observations, findings and source bindings must remain unchanged;
no record is relabelled as v2, rewritten, rescored or repaired in place. A shared
engine or fixture version does not establish identical domain implementation.
Exact replay requires the producing implementation and original public contract;
current development-record replay checks implementation binding and full episode
agreement. New-build replay tests establish new-build consistency only, not replay
compatibility for old records. Historical records require their original source.

The v2 marker does not migrate old artifacts or add a historical-runtime loader.
Changed prompts may change model decisions even where accepted business semantics
are unchanged. Cross-version model results are not interchangeable or pooled as
one condition; assessing model benefit requires separately authorized fresh runs.
Offline payload-shape probes demonstrate boundary behavior, not that clarified
wording improves model performance. These tests authorize neither provider
execution nor historical evidence rewriting.
