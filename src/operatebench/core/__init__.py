"""Domain-neutral Core: time, events, outcomes, ledger, evaluation, engine.

Nothing in this subpackage imports a domain pack. The dependency runs the other
way: a domain pack implements :mod:`operatebench.core.protocol` and Core runs it.
The import-direction test in the suite is what keeps that true.
"""

from __future__ import annotations
