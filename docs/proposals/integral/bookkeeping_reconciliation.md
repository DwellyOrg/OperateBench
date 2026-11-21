# Pending Integral feedback: synthetic bookkeeping reconciliation proposal

Maintainer-authored proposal for Integral feedback; pending acceptance. This is
not their contribution or endorsement, not an official pack, and not a tax-law test.
No Integral representative has agreed to author, own or review this work. The
`integral` workspace names the intended feedback audience, not an owner or an
authorship authority. This fully invented, test-only reference implementation is
non-evidence, not any company's actual workflow or policy or accounting advice.
No legal deadlines, proprietary data or real thresholds are used.

One invented company, month, transaction and receipt use integer minor amounts
in invented SYN currency. Connector import status, counts and claimed amounts
have no authority to reconcile the ledger. The ledger actor owns transaction
facts; the customer owns receipt versions; the accountant actor alone approves
an exact report identity. The case snapshot preserves those distinct sources;
retrieving that snapshot does not promote embedded claims into ledger facts.
Report identity hashes exact ledger and receipt versions and content. Upload or
import success cannot replace acceptance or human approval.

Events span two synthetic days. The agent requests the specific missing or
conflicted document and transaction, waits for clarification with a 120-minute
fallback, then obtains an accountant-acknowledged unresolved handoff if needed.
A matching report receives asynchronous accountant approval before provisional
publication. A 30-minute synthetic quiet interval guards finality. A late ledger
memo correction creates a new report, invalidates approval, and reopens review;
old quiet timers cannot close the corrected report. Source history and prior
reports remain visible. Events after guarded FINAL are audited without mutation.
Unknown or incomplete states cannot be promoted to success.

The proposal is test-only, outside canonical contribution and admission trees,
without an owner manifest, contribution manifest, pack adapter or registry entry.
Its Python lives under `tests/proposals/integral` and is excluded from the wheel.
The generic synthetic operation and pack IDs do not identify Integral policy.
From a source checkout with Python 3.11+ and the project's development dependencies
installed:

```sh
PYTHONPATH=src python -m pytest -v -s -o addopts= \
  tests/test_bookkeeping_reconciliation_engine.py \
  tests/proposals/integral/bookkeeping_reconciliation
PYTHONPATH=src python -m tools.check_contribution_isolation
```

Public inspiration is limited to service categories described on these public pages: [ongoing bookkeeping and reconciliation](https://www.integral.de/de/bookkeeping),
[automated imports and document questions](https://www.integral.de/de/automate-bookkeeping),
[centralized documents](https://www.integral.de/de/documents), and
[status visibility](https://www.integral.de/de/deliverables).
These are marketing descriptions, not audited workflow evidence. Version hashes,
amounts, actors, thresholds, approval and finality controls are invented test
assumptions, not reconstructed processes. The AI preview supplies no runtime
contract and is unused. The site operator identified by its
[imprint](https://www.integral.de/de/imprint) has not contributed to, reviewed or
endorsed this work. No marketing text is copied.

Promotion to `contributions/integral` requires a real authorized Integral
participant to substantively review, adopt or edit the implementation and
sign their contribution under the project DCO. Only then may an accurately
attributed partner owner and contribution declaration be proposed through the
normal contribution checks. Authorship authority is distinct from registration:
see [Operation Pack SDK](../../OPERATION_PACK_SDK.md). Separate maintainer review
and registration remain necessary for any official pack; evidence admission is
also separate. None of those steps has happened or is implied by this workspace.

Source/tests: Apache-2.0; fixture/docs: CC-BY-4.0. Existing maintainer copyright
(PROSPIRE TECHNOLOGIES LTD) remains covered by repository `REUSE.toml`; no Integral
copyright, co-author, DCO signer or endorsement is claimed.
