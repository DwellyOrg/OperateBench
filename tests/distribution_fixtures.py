"""Pristine distribution preparation for the release tests only.

Session state is bytes, never writable paths, archive metadata or scanner verdicts
for mutable inputs. Tests explicitly request private materializations. A test that
changes source or constructor behavior must call the uncached constructor instead;
these fixtures describe only the unchanged checkout, not arbitrary source roots.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

ArchiveSeed = tuple[tuple[str, bytes], tuple[str, bytes]]
DistributionFactory = Callable[[Path], tuple[Path, Path]]
ROOT = Path(__file__).resolve().parents[1]


def _capture(wheel: Path, sdist: Path) -> ArchiveSeed:
    return (wheel.name, wheel.read_bytes()), (sdist.name, sdist.read_bytes())


def materialize(seed: ArchiveSeed, directory: Path) -> tuple[Path, Path]:
    """Exclusive fresh files, not copies/hardlinks of shared writable seed paths."""
    directory.mkdir()
    paths = []
    for name, raw in seed:
        path = directory / name
        with path.open("xb") as stream:
            stream.write(raw)
        paths.append(path)
    return paths[0], paths[1]


@pytest.fixture(scope="session")
def canonical_archive_seed(tmp_path_factory: pytest.TempPathFactory) -> ArchiveSeed:
    from tests.test_public_release import _canonical_distribution

    directory = tmp_path_factory.mktemp("canonical-seed") / "dist"
    seed = _capture(*_canonical_distribution(directory))
    shutil.rmtree(directory)
    return seed


@pytest.fixture()
def canonical_distribution(canonical_archive_seed: ArchiveSeed) -> DistributionFactory:
    return partial(materialize, canonical_archive_seed)


@pytest.fixture(scope="session")
def b3_archive_seed(
    canonical_archive_seed: ArchiveSeed, tmp_path_factory: pytest.TempPathFactory
) -> ArchiveSeed:
    from tests.test_artifact8_evidence_freeze import _write_test_distribution

    directory = tmp_path_factory.mktemp("b3-seed") / "dist"
    _write_test_distribution(
        directory, canonical_distribution=partial(materialize, canonical_archive_seed)
    )
    seed = _capture(next(directory.glob("*.whl")), next(directory.glob("*.tar.gz")))
    shutil.rmtree(directory)
    return seed


@pytest.fixture()
def b3_distribution(b3_archive_seed: ArchiveSeed) -> DistributionFactory:
    return partial(materialize, b3_archive_seed)


@pytest.fixture(scope="session")
def built_archive_seed(tmp_path_factory: pytest.TempPathFactory) -> ArchiveSeed:
    """One real offline build pair; never substitute synthetic archives for Hatchling."""
    directory = tmp_path_factory.mktemp("built-seed") / "dist"
    subprocess.run(
        ["uv", "build", "--offline", "--out-dir", str(directory)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    seed = _capture(next(directory.glob("*.whl")), next(directory.glob("*.tar.gz")))
    # Establish source provenance on the actual built wheel, not a synthetic oracle.
    with zipfile.ZipFile(io.BytesIO(seed[0][1])) as wheel:
        for name in wheel.namelist():
            if name.startswith(("operatebench/", "boundarybench/")):
                assert wheel.read(name) == (ROOT / "src" / name).read_bytes()
    shutil.rmtree(directory)
    return seed


@pytest.fixture()
def built_distribution(
    built_archive_seed: ArchiveSeed, tmp_path: Path
) -> tuple[Path, Path]:
    return materialize(built_archive_seed, tmp_path / "dist")


@pytest.fixture(scope="session")
def extracted_sdist_scan(
    built_archive_seed: ArchiveSeed, tmp_path_factory: pytest.TempPathFactory
) -> tuple[int, str, str]:
    """One untouched extraction/CLI execution serves three success assertion groups."""
    extracted = tmp_path_factory.mktemp("sdist-scan")
    with tarfile.open(fileobj=io.BytesIO(built_archive_seed[1][1]), mode="r:gz") as sdist:
        sdist.extractall(extracted, filter="data")
    root = next(extracted.iterdir())
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "HTTP_PROXY": "http://" + "127.0.0.1:9",
            "HTTPS_PROXY": "http://" + "127.0.0.1:9",
            "NO_PROXY": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "tools.check_public_release", "--surface", "sdist"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    shutil.rmtree(extracted)
    return result.returncode, result.stdout, result.stderr
