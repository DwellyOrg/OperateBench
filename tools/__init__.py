"""Repository and release tooling carried by the source distribution.

These modules are excluded from the installable wheel and included in the sdist
so an extracted source release can run the same checks as the repository tests.
The command and tests share one implementation rather than drifting apart.
"""

from __future__ import annotations
