"""The provider kernel: what talking to a provider is, with no track in it.

Two tracks in this distribution ask a model for one structured answer. The
Boundary Track projects a :class:`~boundarybench.adapter.TurnRequest` into a
request body and reads an ``AdapterCall`` back; the Lifecycle Track will project
a ``ModelRequest`` and read a ``ModelResponse``. Between those two projections
sits everything that is *not* either track's question — which endpoint the
credential is sent to, whether the body matches the settings a run recorded, how
many times a turn may retry and how long it waits, what a reservation is
resolved to, what an attempt is recorded as, whether the bytes that came back are
the response contract that was asked for. Written twice, those would drift, and
the direction that matters is the silent one: two retry loops or two readings of
one usage block produce real evidence that looks exactly like correct evidence.

So they are written once, here, and neither track owns them.

**This package imports no track, and that is enforced.** Nothing under it may
import :mod:`boundarybench` or any Lifecycle module — agents, core, domains,
runner, artefact, controls, CLI — because a kernel that reached into one track
could not be composed by the other. The single sideways import allowed is
:mod:`operatebench.jsonsafe`, the canonical UTF-8 JSON primitive both tracks
already hash their artefacts with. ``tests/test_provider_kernel_boundary.py``
scans the source for it: an import direction is not something a runtime test can
observe after the import has happened.

**A track imports upward and supplies its own projection.** The Boundary
adapter builds the body, hands it to the shared exchange in
:mod:`operatebench.providers.openai_responses`, and parses the answer with its
own parser. The exchange sees a payload and a deadline; it has never heard of a
``TurnRequest``, a scaffold, an observed case or an ``AdapterCall``.

**Nothing here is a benchmark result, and nothing here is a live call.** No
module in this package opens a socket on import, pre-authorises a configuration
or reads a credential into anywhere durable.
"""

from __future__ import annotations

__all__: list[str] = []
