"""Private NON-SCORED execution evidence, separate from complete artifacts.

Correction to excluded-run observability (HM02), not to execution or scoring:
engine, complete-artifact, decision-tape and provider-ledger versions stay fixed.
The new sidecar owns its separate v1 schema; it is never a replay/grading input.
All six fixed provider operators inherit this writer through the central recorder.

Rows are incremental; no whole trajectory or decision tape is copied per turn.
Append work is O(total serialized bytes), with one fsync per durable row. Memory
retains one run's rows until the evidence object is released, including on the
caller-visible ProviderFailure. There is no automatic file deletion or byte cap:
operators own private-directory retention. Agent checkpoints/decisions are bounded
by max_invocations * (max_turns_per_invocation +
max_retrieval_batches_per_invocation), plus one final failure checkpoint;
receipts follow the existing bounded episode's transitions, not a second loop.
State snapshots cost their own size, not the size of the trajectory so far.

State checkpoints precede each agent call and follow a caught provider failure.
A process crash retains completed fsynced rows, potentially followed by an
incomplete last line: consume only complete JSON lines, never infer a terminal
or score from a prefix. The last state may predate later receipt rows when a
crash occurs between transitions and the next checkpoint. File fsync does not
promise directory-entry survival after machine/power loss. Write failures close
the writer permanently; unsupported JSON is rejected without repr/string repair.
Provider error text is not persisted; structured decisions and domain state may
still contain sensitive operational data and this is not a redaction format.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from operatebench.jsonsafe import canonical_json_bytes, to_json
from operatebench.version import OPERATEBENCH_VERSION


class PartialExecutionEvidence:
    """Diagnostic evidence only: never an EpisodeOutcome or grading input."""

    def __init__(
        self,
        identity: Mapping[str, Any],
        *,
        path: Path | None = None,
        dir_fd: int | None = None,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self._closed = False
        self.path = path
        self._fd: int | None = None
        if path is not None:
            self._fd = os.open(
                path if dir_fd is None else path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=dir_fd,
            )
        self.append(
            {
                "kind": "header",
                "schema": "operatebench.partial-execution.v1",
                "scoring": "NON-SCORED",
                "engine_version": OPERATEBENCH_VERSION,
                "identity": dict(identity),
                "source_files_sha256": {
                    name: hashlib.sha256(
                        (Path(__file__).parent / name).read_bytes()
                    ).hexdigest()
                    for name in (
                        "runner.py",
                        "core/engine.py",
                        "core/ledger.py",
                        "agents/playback.py",
                        "agents/evidence.py",
                        "agents/transport.py",
                        "partial_evidence.py",
                    )
                },
            }
        )

    def append(self, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise ValueError("partial evidence is closed")
        projected = to_json(row)
        assert isinstance(projected, dict)
        if self._fd is not None:
            data = canonical_json_bytes(projected, "partial execution evidence") + b"\n"
            try:
                view = memoryview(data)
                while view:
                    written = os.write(self._fd, view)
                    if written <= 0:
                        raise OSError("partial evidence write failed")
                    view = view[written:]
                os.fsync(self._fd)
            except BaseException:
                self.close()
                raise
        self.rows.append(projected)

    def decision(self, row: Mapping[str, Any]) -> None:
        self.append({"kind": "decision", "decision": dict(row)})

    def receipt(self, row: Mapping[str, Any]) -> None:
        self.append({"kind": "receipt", "receipt": dict(row)})

    def state(self, state: Mapping[str, Any], *, at: str) -> None:
        self.append({"kind": "state", "at": at, "last_state": dict(state)})

    def finish(
        self,
        classification: str,
        exclusion_code: str | None = None,
        *,
        ledger_terminal: Mapping[str, Any] | None = None,
        ledger_digest_sha256: str | None = None,
    ) -> None:
        if self._closed:
            return
        # Only closed classifications cross this diagnostic boundary. Never keep
        # an arbitrary exception fault/detail, even when no ledger is attached.
        if type(classification) is not str or classification not in (
            "scored",
            "excluded",
            "aborted",
        ):
            classification = "aborted"
        if exclusion_code is not None and (
            type(exclusion_code) is not str
            or exclusion_code
            not in (
                "aborted",
                "provider_budget",
                "provider_deadline",
                "provider_protocol",
                "provider_transport",
            )
        ):
            exclusion_code = "unknown"
        self.append(
            {
                "kind": "closure",
                "classification": classification,
                "exclusion_code": exclusion_code,
                "ledger_terminal": ledger_terminal,
                "ledger_digest_sha256": ledger_digest_sha256,
            }
        )
        self.close()

    def close(self) -> None:
        self._closed = True
        if self._fd is not None:
            fd, self._fd = self._fd, None
            os.close(fd)
