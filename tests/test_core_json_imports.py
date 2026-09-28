"""The Core import closure uses canonical JSON, not Boundary compatibility."""

import subprocess
import sys

import pytest


def test_fresh_core_engine_does_not_import_boundary():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import operatebench.core.engine; "
            "assert not any(n == 'boundarybench' or "
            "n.startswith('boundarybench.') for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_shim_public_objects_and_exception_catches_are_identical():
    import boundarybench.jsonsafe as shim
    import operatebench.jsonsafe as canonical

    assert set(shim.__all__) == set(canonical.__all__)
    for name in shim.__all__:
        assert getattr(shim, name) is getattr(canonical, name)
    for producer, catcher in ((canonical, shim), (shim, canonical)):
        with pytest.raises(catcher.NonJsonValueError):
            producer.canonical_json_bytes({"bad": float("nan")}, "test")
        cyclic = []
        cyclic.append(cyclic)
        with pytest.raises(catcher.CyclicStructureError):
            producer.canonical_json_bytes(cyclic, "test")
        assert (
            producer.canonical_json_bytes({"b": 1, "a": "é"}, "test")
            == b'{"a":"\xc3\xa9","b":1}'
        )
