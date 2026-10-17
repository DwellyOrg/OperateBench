"""Synthetic reducer copy work and retained-version isolation; no timing thresholds."""

import copy

import pytest

from operatebench.providers.cost import CostCapExceededError
from tools.aggregate_budget import Budget
from tools.three_flow_budget import CampaignBudget


class Visited:
    def __init__(self, visits):
        self.visits = visits

    def __deepcopy__(self, memo):
        self.visits.append(self)
        return Visited(self.visits)


def row(state="reserved"):
    return dict(
        cell="a",
        state=state,
        ib=10,
        ob=2,
        ir="2",
        **{"or": "5"},
        bound="3/100000",
        initial_bound="3/100000",
        initial_input_bound=10,
        nested={"values": [1, {"x": [2]}]},
    )


@pytest.mark.parametrize("cls,kind", [(Budget, "dispatch"), (CampaignBudget, "raw_rpc")])
def test_copy_work_visits_only_changed_row(cls, kind):
    budget = object.__new__(cls)
    budget.cap = None
    visits = []
    old = {
        str(i): dict(row("settled"), actual="0", probe=Visited(visits))
        for i in range(100)
    }
    old["target"] = dict(
        row("sent" if kind == "raw_rpc" else "reserved"), probe=Visited(visits)
    )
    event = {"kind": kind, "id": "target", "ib": 11, "ordinal": 1}
    result = budget._reduce(old, event)
    assert len(visits) == 1
    assert result is not old
    assert result["target"] is not old["target"]
    result["target"]["nested"]["values"][1]["x"].append(3)
    assert old["target"]["nested"]["values"][1]["x"] == [2]


@pytest.mark.parametrize("cls", [Budget, CampaignBudget])
@pytest.mark.parametrize(
    "state,event,cap,reject",
    [
        ("reserved", {"kind": "dispatch", "ib": 11}, None, False),
        (
            "reserved",
            {"kind": "cancel", "proof": "guarded-wire-not-consumed"},
            None,
            False,
        ),
        ("reserved", {"kind": "unknown"}, None, False),
        ("sent", {"kind": "settle", "input": 1, "output": 1}, None, False),
        ("reserved", {"kind": "settle", "input": 1, "output": 1}, None, True),
        ("reserved", {"kind": "dispatch", "ib": 9}, None, True),
        ("reserved", {"kind": "dispatch", "ib": 11}, 0, True),
        ("sent", {"kind": "settle", "input": 100, "output": 1}, None, True),
        ("reserved", {"kind": "cancel", "proof": "bad"}, None, True),
        ("settled", {"kind": "unknown"}, None, True),
        ("reserved", {"kind": "invalid"}, None, True),
        ("sent", {"kind": "raw_rpc", "ordinal": 1}, None, False),
        ("sent", {"kind": "raw_rpc", "ordinal": 2}, None, True),
        ("reserved", {"kind": "raw_rpc", "ordinal": 1}, None, True),
    ],
)
def test_retained_nested_rows_unchanged(cls, state, event, cap, reject):
    if event["kind"] == "raw_rpc" and cls is Budget:
        reject = True
    budget = object.__new__(cls)
    budget.cap = cap
    old = {"target": row(state), "other": row("cancelled")}
    snapshot = copy.deepcopy(old)
    event = dict(event, id="target")
    if reject:
        with pytest.raises(CostCapExceededError if cap == 0 else ValueError):
            budget._reduce(old, event)
    else:
        result = budget._reduce(old, event)
        result["target"]["nested"]["values"].append(99)
    assert old == snapshot


@pytest.mark.parametrize("cls", [Budget, CampaignBudget])
def test_reserve_does_not_copy_existing_rows(cls):
    budget = object.__new__(cls)
    budget.cap = None
    visits = []
    old = {"other": dict(row("cancelled"), probe=Visited(visits))}
    result = budget._reduce(
        old,
        dict(
            kind="reserve",
            id="new",
            cell="new",
            ib=1,
            ob=2,
            ir="2",
            **{"or": "5"},
            digest="synthetic",
        ),
    )
    assert visits == []
    assert "new" not in old
    assert result["new"]["state"] == "reserved"


@pytest.mark.parametrize("cls", [Budget, CampaignBudget])
def test_changed_row_detaches_aliases_in_retained_versions(cls):
    budget = object.__new__(cls)
    budget.cap = None
    shared = row()
    old = {"target": shared, "alias": shared}
    result = budget._reduce(old, {"kind": "dispatch", "id": "target", "ib": 11})
    assert old["target"]["state"] == old["alias"]["state"] == "reserved"
    assert result["alias"]["state"] == "reserved"
    shared["nested"]["values"].append(99)
    assert result["target"]["nested"]["values"] == [1, {"x": [2]}]


@pytest.mark.parametrize("cls", [Budget, CampaignBudget])
@pytest.mark.parametrize("sync_index", [1, 2])
@pytest.mark.parametrize(
    "kind", ["reserve", "dispatch", "cancel", "unknown", "settle", "raw_rpc"]
)
def test_fsync_failure_never_publishes_or_mutates_retained_records(
    tmp_path, monkeypatch, cls, sync_index, kind
):
    if kind == "raw_rpc" and cls is Budget:
        pytest.skip("raw RPC events belong to CampaignBudget")
    import os
    from decimal import Decimal

    tmp_path.chmod(0o700)
    kwargs = (
        {"namespace": "offline-test-cow", "cap": Decimal("1000")}
        if cls is Budget
        else {"campaign_id": "synthetic-cow", "cap": Decimal("1000")}
    )
    budget = cls(tmp_path, create=True, **kwargs)
    try:
        budget.records = {
            "target": row("sent" if kind in ("settle", "raw_rpc") else "reserved")
        }
        retained = budget.records
        snapshot = copy.deepcopy(retained)
        seq, previous = budget.seq, budget.previous
        calls = 0
        real = os.fsync

        def fail(fd):
            nonlocal calls
            calls += 1
            assert budget.records is retained
            assert retained == snapshot
            if calls == sync_index:
                raise OSError("synthetic fsync fault")
            return real(fd)

        monkeypatch.setattr(os, "fsync", fail)
        event = {
            "kind": kind,
            "id": "target",
            "ib": 11,
            "input": 1,
            "output": 1,
            "proof": "guarded-wire-not-consumed",
            "ordinal": 1,
        }
        if kind == "reserve":
            event.update(
                id="new", cell="new", ob=2, ir="2", **{"or": "5"}, digest="synthetic"
            )
        with pytest.raises(OSError, match="synthetic fsync"):
            if kind == "raw_rpc":
                budget.raw_rpc("a", "synthetic-endpoint", "0" * 64)
            else:
                budget._event(event)
        assert budget.broken
        assert budget.records is retained
        assert retained == snapshot
        assert (budget.seq, budget.previous) == (seq, previous)
    finally:
        budget.close()
