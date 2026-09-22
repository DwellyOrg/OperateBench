"""Provider integrations for the provider-neutral adapter contract.

Everything under this package talks to a real service. The provider-neutral
contract in :mod:`boundarybench.adapter` deliberately knows nothing about any of
them: a provider module imports the contract, never the other way round, so an
SDK's types, retry behaviour and error taxonomy stay on one side of the
boundary and the runner keeps a single shape to execute.
"""

from __future__ import annotations

__all__: list[str] = []
