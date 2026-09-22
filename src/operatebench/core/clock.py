"""Simulated time. There are no wall-clock sleeps anywhere in this build.

An operation that lives for eleven simulated days must execute in milliseconds
and must produce the same trajectory every time, so time here is a number the
engine moves forward when it delivers the next authored or scheduled event —
never something the process waits for. :class:`SimulatedClock` refuses to move
backwards, which is what makes "the next event" a total order rather than a
suggestion.

One timestamp format, stated once: ``YYYY-MM-DDTHH:MM:SSZ``. Offsets, naive
local times and fractional seconds are refused rather than normalised, because
a benchmark artefact whose meaning depends on the reader's timezone is not a
frozen artefact.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from operatebench.core.errors import ClockRewindError, MalformedTimestampError

#: The one accepted instant format. Also the one emitted format.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def parse_timestamp(text: object, context: str) -> int:
    """Read a canonical instant into whole simulated seconds since the epoch.

    ``context`` names the artefact position being read, so the operator learns
    which authored event carried the bad instant rather than that some string
    somewhere did not parse.
    """
    if not isinstance(text, str):
        raise MalformedTimestampError(
            f"{context}: timestamp must be a string in {TIMESTAMP_FORMAT!r} form, "
            f"got {type(text).__name__}"
        )
    try:
        moment = datetime.strptime(text, TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise MalformedTimestampError(
            f"{context}: {text!r} is not a canonical UTC instant of the form "
            f"2031-03-03T09:00:00Z ({exc})"
        ) from exc
    return int((moment - _EPOCH).total_seconds())


def format_timestamp(seconds: int) -> str:
    """The inverse of :func:`parse_timestamp`, and the only way instants are written."""
    return (_EPOCH + timedelta(seconds=seconds)).strftime(TIMESTAMP_FORMAT)


def shift_minutes(text: str, minutes: int, *, context: str = "timestamp") -> str:
    """Move an instant forward by whole simulated minutes.

    Forward only. Every offset in a scenario plan is a delay from something that
    already happened, so a negative offset is an authoring mistake rather than a
    way to schedule an event into the past.
    """
    if minutes < 0:
        raise ClockRewindError(
            f"{context}: offset of {minutes} minute(s) would move simulated time "
            "backwards; scenario offsets are delays, not rewinds"
        )
    return format_timestamp(parse_timestamp(text, context) + minutes * 60)


class SimulatedClock:
    """The episode's monotonic simulated instant.

    Deliberately not a counter the engine may set freely: ``advance_to`` is the
    only mutator and it refuses to rewind, so every record written during an
    episode carries a non-decreasing instant by construction rather than by
    convention.
    """

    def __init__(self, start: str, *, context: str = "clock start") -> None:
        self._start_seconds = parse_timestamp(start, context)
        self._seconds = self._start_seconds

    @property
    def now(self) -> str:
        """The current simulated instant."""
        return format_timestamp(self._seconds)

    @property
    def elapsed_minutes(self) -> int:
        """Whole simulated minutes since the episode began."""
        return (self._seconds - self._start_seconds) // 60

    def advance_to(self, moment: str, *, context: str = "clock advance") -> None:
        """Move simulated time to ``moment``. Standing still is allowed."""
        target = parse_timestamp(moment, context)
        if target < self._seconds:
            raise ClockRewindError(
                f"{context}: cannot advance to {moment} from {self.now}; simulated "
                "time never moves backwards"
            )
        self._seconds = target


__all__ = [
    "TIMESTAMP_FORMAT",
    "ClockRewindError",
    "MalformedTimestampError",
    "SimulatedClock",
    "format_timestamp",
    "parse_timestamp",
    "shift_minutes",
]
