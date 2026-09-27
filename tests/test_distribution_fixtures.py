"""Isolation of the release suite's immutable preparation, not scanner-result caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.distribution_fixtures import ArchiveSeed, materialize
from tools import check_public_release as checks

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shape", ["canonical", "b3"])
def test_hostile_archive_copy_cannot_poison_seed_or_later_positive(
    tmp_path: Path, request: pytest.FixtureRequest, shape: str
) -> None:
    seed: ArchiveSeed = request.getfixturevalue(f"{shape}_archive_seed")
    assert type(seed) is tuple
    assert all(type(item) is tuple and type(item[1]) is bytes for item in seed)
    before = tuple(hashlib.sha256(raw).hexdigest() for _, raw in seed)
    attacked = materialize(seed, tmp_path / "attacked")
    untouched = materialize(seed, tmp_path / "untouched")
    assert all(
        a.stat().st_ino != b.stat().st_ino
        for a, b in zip(attacked, untouched, strict=True)
    )
    attacked[0].write_bytes(b"hostile replacement, not a ZIP archive")
    assert checks.check_built_distributions(ROOT, distribution_dir=attacked[0].parent)
    later = materialize(seed, tmp_path / "later")
    for index, (_, raw) in enumerate(seed):
        assert untouched[index].read_bytes() == later[index].read_bytes() == raw
        assert later[index].stat().st_ino not in {
            attacked[index].stat().st_ino,
            untouched[index].stat().st_ino,
        }
    assert tuple(hashlib.sha256(raw).hexdigest() for _, raw in seed) == before
    assert checks.check_built_distributions(ROOT, distribution_dir=later[0].parent) == []
    if shape == "b3":
        from tests.test_artifact8_evidence_freeze import _audit_distribution

        _audit_distribution(*later)


def test_seed_reuse_does_not_cache_the_scanners_source_oracle(
    tmp_path: Path, canonical_archive_seed: ArchiveSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel, _ = materialize(canonical_archive_seed, tmp_path / "dist")
    original = Path.read_bytes
    source = ROOT / "src" / "operatebench" / "version.py"
    reads = []

    def changed(path: Path) -> bytes:
        if path == source:
            reads.append(path)
            return original(path) + b"\n# changed source oracle\n"
        return original(path)

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", changed)
        problems = checks.check_built_distributions(ROOT, distribution_dir=wheel.parent)
    assert reads
    assert any(
        "bytes differ" in problem or "source differs" in problem for problem in problems
    )
    assert checks.check_built_distributions(ROOT, distribution_dir=wheel.parent) == []
