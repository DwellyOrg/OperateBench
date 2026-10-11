# Synthetic bookkeeping reconciliation development slice

Maintainer-authored under `prospire`, fully invented, development-only and
non-evidence. No outside contribution, review, affiliation or endorsement is
claimed. This is not any company's actual workflow or policy, accounting advice,
a tax-law test, or an official benchmark pack. No legal deadlines are used.

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

The candidate is development-only and unregistered. From a source checkout
with Python 3.11+ and the project's development dependencies installed:

```sh
PYTHONPATH=src python -m pytest -v -s -o addopts= \
  tests/test_bookkeeping_reconciliation_engine.py \
  tests/contributions/prospire/bookkeeping_reconciliation
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

Source/tests: Apache-2.0; fixture/docs: CC-BY-4.0, per existing owner declaration.
