"""The single place the package version is stated.

A leaf module, so anything that has to record the running build's version — the
run manifest, most of all — can import it without waiting for the package's own
``__init__`` to finish importing everything else.
"""

from __future__ import annotations

__version__ = "0.0.15"

__all__ = ["__version__"]
