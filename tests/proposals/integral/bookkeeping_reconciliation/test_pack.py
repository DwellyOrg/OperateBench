"""Synthetic proposal refusal and strict input checks; no partner authorship."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from operatebench.core.errors import SpecSchemaError
from tests.proposals.integral.bookkeeping_reconciliation.operation import (
    ISSUERS,
    BookkeepingDomain,
)
from tests.proposals.integral.bookkeeping_reconciliation.spec import load_spec

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "examples/proposals/integral/bookkeeping_reconciliation/operation.yaml"


@pytest.mark.parametrize(
    "action,payload,code",
    [
        ("publish_report", {"report_id": "import-1"}, "UNVERIFIED_SOURCES"),
        ("draft_report", {"source_version": "0"}, "UNVERIFIED_SOURCES"),
        (
            "request_receipt",
            {"transaction_id": "other", "document_id": "receipt-1"},
            "REQUEST_SCOPE",
        ),
        ("unknown", {}, "PAYLOAD"),
    ],
)
def test_refusal_preserves_state(action, payload, code):
    domain = BookkeepingDomain(load_spec(FIXTURE), "match")
    state = domain.initial_state()
    before = deepcopy(state)
    verdict = domain.apply_action(
        state, action, payload, (), SimpleNamespace(now="2026-09-02T09:00:00Z")
    )
    assert not verdict.accepted and verdict.code == code
    assert state == before


def test_unknown_state_never_final():
    domain = BookkeepingDomain(load_spec(FIXTURE), "match")
    state = domain.initial_state()
    before = deepcopy(state)
    assert not domain.apply_terminal(state, (), None).accepted
    assert state == before and not domain.is_replay_final(state)


def test_wrong_operation_refused(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(
        FIXTURE.read_text().replace(
            "accounting.bookkeeping_reconciliation.synthetic.v1", "other"
        )
    )
    with pytest.raises(SpecSchemaError):
        load_spec(path)


def test_old_approval_cannot_sign_corrected_report():
    domain = BookkeepingDomain(load_spec(FIXTURE), "reopen")
    state = domain.initial_state()
    state["report"] = {"report_id": "corrected-report"}
    state["review_requested"] = "corrected-report"
    before = deepcopy(state)
    event = SimpleNamespace(
        event_type="approval",
        actor_id="accountant",
        payload={"report_id": "old-report", "decision": "approved"},
    )
    verdict = domain.reduce_event(state, event, None)
    assert not verdict.accepted and verdict.code == "APPROVAL_VERSION"
    assert state == before


def test_old_or_early_quiet_event_cannot_finalize_correction():
    domain = BookkeepingDomain(load_spec(FIXTURE), "reopen")
    state = domain.initial_state()
    state.update(
        report={"report_id": "new-report"},
        provisional=True,
        provisional_at="2026-09-02T10:00:00Z",
    )
    context = SimpleNamespace(now="2026-09-02T10:10:00Z", record=lambda *args: None)
    before = deepcopy(state)
    for report_id in ("old-report", "new-report"):
        event = SimpleNamespace(
            event_type="quiet", actor_id="clock", payload={"report_id": report_id}
        )
        domain.reduce_event(state, event, context)
        assert state == before
    assert not domain.apply_terminal(state, (), context).accepted
    assert state == before


IMPORT_CLAIM = {
    "source_id": "connector-import-1",
    "version": 1,
    "status": "ok",
    "count": 1,
    "claimed_amount_minor": 12000,
}


@pytest.mark.parametrize("kind,actor", ISSUERS.items())
@pytest.mark.parametrize("payload", [None, [], [("report_id", "bogus")]])
def test_non_mapping_event_payload_is_atomic(kind, actor, payload):
    domain = BookkeepingDomain(load_spec(FIXTURE), "match")
    state = domain.initial_state()
    before = deepcopy(state)
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    event = SimpleNamespace(event_type=kind, actor_id=actor, payload=payload)
    verdict = domain.reduce_event(state, event, context)
    assert not verdict.accepted
    assert verdict.code == ("IMPORT_PAYLOAD" if kind == "imported" else "EVENT_PAYLOAD")
    assert state == before
    assert records == []


@pytest.mark.parametrize(
    "payload",
    [
        {"report_id": "bogus", "extra": True},
        {},
        {"report_id": None},
        {"report_id": 1},
        {"report_id": ""},
        {"report_id": "   "},
    ],
)
def test_malformed_quiet_payload_is_atomic(payload):
    domain = BookkeepingDomain(load_spec(FIXTURE), "match")
    state = domain.initial_state()
    before = deepcopy(state)
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    event = SimpleNamespace(event_type="quiet", actor_id="clock", payload=payload)
    verdict = domain.reduce_event(state, event, context)
    assert not verdict.accepted and verdict.code == "QUIET_PAYLOAD"
    assert state == before
    assert records == []


@pytest.mark.parametrize("report_id", ["old-report", "new-report"])
def test_valid_stale_or_early_quiet_timer_is_recorded_without_state_change(report_id):
    domain = BookkeepingDomain(load_spec(FIXTURE), "reopen")
    state = domain.initial_state()
    state.update(
        report={"report_id": "new-report"},
        provisional=True,
        provisional_at="2026-09-02T10:00:00Z",
    )
    before = deepcopy(state)
    records = []
    context = SimpleNamespace(
        now="2026-09-02T10:10:00Z", record=lambda *args: records.append(args)
    )
    payload = {"report_id": report_id}
    event = SimpleNamespace(event_type="quiet", actor_id="clock", payload=payload)
    assert domain.reduce_event(state, event, context).accepted
    assert state == before
    assert records == [
        ("bookkeeping_event", {"kind": "quiet", "source": "clock", "data": payload})
    ]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        list(IMPORT_CLAIM.items()),
        {},
        *[
            {k: v for k, v in IMPORT_CLAIM.items() if k != missing}
            for missing in IMPORT_CLAIM
        ],
        IMPORT_CLAIM | {"extra": "claim"},
        *[IMPORT_CLAIM | {"source_id": value} for value in (None, 1, True, "", "   ")],
        *[
            IMPORT_CLAIM | {"status": value}
            for value in (None, 1, True, [], "", "unknown")
        ],
        *[
            IMPORT_CLAIM | {key: value}
            for key in ("version", "count", "claimed_amount_minor")
            for value in (None, True, False, "1", 1.0, float("nan"), float("inf"), -1)
        ],
        IMPORT_CLAIM | {"version": 0},
    ],
)
@pytest.mark.parametrize("existing_import", [None, IMPORT_CLAIM])
def test_malformed_connector_claim_is_atomic(payload, existing_import):
    domain = BookkeepingDomain(load_spec(FIXTURE), "match")
    state = domain.initial_state()
    state["import"] = deepcopy(existing_import)
    before = deepcopy(state)
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    event = SimpleNamespace(event_type="imported", actor_id="connector", payload=payload)
    verdict = domain.reduce_event(state, event, context)
    assert not verdict.accepted and verdict.code == "IMPORT_PAYLOAD"
    assert state == before
    assert records == []


@pytest.mark.parametrize("status,count", [("ok", 1), ("failed", 0)])
def test_valid_connector_claim_remains_only_a_claim(status, count):
    domain = BookkeepingDomain(load_spec(FIXTURE), "match")
    state = domain.initial_state()
    before = deepcopy(state)
    payload = IMPORT_CLAIM | {"status": status, "count": count, "claimed_amount_minor": 0}
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    event = SimpleNamespace(event_type="imported", actor_id="connector", payload=payload)
    assert domain.reduce_event(state, event, context).accepted
    assert state == before | {"import": payload}
    assert records == [
        (
            "bookkeeping_event",
            {
                "kind": "imported",
                "source": "connector",
                "data": payload,
            },
        )
    ]
    after = deepcopy(state)
    verdict = domain.apply_action(
        state, "draft_report", {"source_version": "0"}, (), context
    )
    assert not verdict.accepted
    assert verdict.code == (
        "IMPORT_NOT_LEDGER" if status == "failed" else "UNVERIFIED_SOURCES"
    )
    assert state == after
    assert len(records) == 1
