"""The model boundary: a provider-neutral agent, its transport, and playback.

Three modules, one seam.

:mod:`operatebench.agents.transport` is the whole provider surface OperateBench
knows about — a request, a response, and one method. It names no vendor, imports
no SDK and opens no socket; a concrete provider is something a caller injects.

:mod:`operatebench.agents.model` turns a transport into an
:class:`~operatebench.core.protocol.OperationAgent`. It parses a response into
the five Core outcomes and gives every way that can fail a stable name.

:mod:`operatebench.agents.playback` is why a model episode can be reproduced
without asking the provider anything twice: every decision is recorded against
the digest of the exact observation that produced it, and a recorded tape
re-executes the episode with the transport removed. For a model run it also
rebuilds each request from the provider identity the record names, so the model
a run is attributed to is part of what produced its decisions rather than a
string beside them.
"""

from __future__ import annotations

__all__: list[str] = []
