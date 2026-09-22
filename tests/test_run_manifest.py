"""Run identity: the immutable manifest written before any episode runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import ADAPTER_CONTRACT_VERSION
from boundarybench.jsonsafe import NonJsonValueError
from boundarybench.runmanifest import (
    MANIFEST_FILENAME,
    RunLimits,
    RunManifestError,
    RunManifestFormatError,
    RunManifestMismatchError,
    RunPathError,
    build_run_manifest,
    canonical_json,
    load_run_manifest,
    open_run_session,
    resolve_run_paths,
    run_identity_digest,
    write_run_manifest,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST


def _build(**overrides: object):
    kwargs: dict[str, object] = {
        "suite": validate_suite(SUITE_MANIFEST),
        "scaffold": load_scaffold(STANDARD_SCAFFOLD),
        "provider": "test-double",
        "model": "scripted-test-double",
        "implementation": "scripted_test_double",
        "adapter_version": ADAPTER_CONTRACT_VERSION,
        "adapter_settings": {"temperature": 0, "seed": 7},
        "trials": 2,
        "limits": RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    }
    kwargs.update(overrides)
    return build_run_manifest(**kwargs)  # type: ignore[arg-type]


def test_run_id_is_content_addressed_and_ignores_the_wall_clock() -> None:
    """Two identical plans are the same run, whenever they were started.

    Nothing about *when* is recorded at all, so "whenever" is not a field that
    has to be excluded from identity — there is no wall-clock fact to exclude.
    """
    first = _build()
    second = _build()

    assert first.run_id == second.run_id
    assert len(first.run_id) == 64
    assert "created_at" not in first.as_dict()


@pytest.mark.parametrize(
    "overrides",
    [
        {"trials": 3},
        {"model": "other-model"},
        {"provider": "other-provider"},
        {"implementation": "other_impl"},
        {"adapter_version": "0.2.0-dev.1"},
        {"adapter_settings": {"temperature": 1, "seed": 7}},
        {
            "limits": RunLimits(
                max_turns=13,
                max_messages=64,
                episode_timeout_seconds=30.0,
            )
        },
        {
            "limits": RunLimits(
                max_turns=12,
                max_messages=65,
                episode_timeout_seconds=30.0,
            )
        },
        {
            "limits": RunLimits(
                max_turns=12,
                max_messages=64,
                episode_timeout_seconds=31.0,
            )
        },
    ],
)
def test_anything_that_changes_execution_changes_the_run_id(
    overrides: dict[str, object],
) -> None:
    assert _build(**overrides).run_id != _build().run_id


def test_episode_plan_covers_every_variant_for_every_trial() -> None:
    manifest = _build()
    suite = validate_suite(SUITE_MANIFEST)

    expected = [
        (variant.variant_id, trial)
        for cube in suite.cubes
        for variant in cube.cube.variants
        for trial in (1, 2)
    ]
    assert [(e.variant_id, e.trial_index) for e in manifest.episode_plan] == expected
    assert len({e.episode_id for e in manifest.episode_plan}) == len(expected)
    assert manifest.retry_policy == {"retries": 0, "policy": "none"}


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"api_key": "x"}, "api_key"),
        ({"API_KEY": "x"}, "API_KEY"),
        ({"apiKey": "x"}, "apiKey"),
        ({"auth_token": "x"}, "auth_token"),
        ({"secret": "x"}, "secret"),
        ({"Authorization": "x"}, "Authorization"),
        ({"nested": {"bearer_token": "x"}}, "bearer_token"),
        ({"list": [{"client_secret": "x"}]}, "client_secret"),
        ({"password": "x"}, "password"),
        ({"private_key": "x"}, "private_key"),
    ],
)
def test_secret_bearing_settings_are_refused_not_redacted(
    settings: dict[str, object], expected: str
) -> None:
    """A secret must never reach the manifest, so it is refused at the door."""
    with pytest.raises(RunManifestError) as excinfo:
        _build(adapter_settings=settings)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"trials": 0}, "trials must be a positive integer"),
        ({"trials": True}, "trials must be a positive integer"),
        ({"trials": 1.5}, "trials must be a positive integer"),
        (
            {
                "limits": RunLimits(
                    max_turns=0,
                    max_messages=64,
                    episode_timeout_seconds=1.0,
                )
            },
            "max_turns must be a positive integer",
        ),
        (
            {
                "limits": RunLimits(
                    max_turns=True,
                    max_messages=64,
                    episode_timeout_seconds=1.0,
                )
            },
            "max_turns must be a positive integer",
        ),
        (
            {
                "limits": RunLimits(
                    max_turns=4,
                    max_messages=64,
                    episode_timeout_seconds=0,
                )
            },
            "episode_timeout_seconds must be a positive, finite number",
        ),
        (
            {
                "limits": RunLimits(
                    max_turns=4,
                    max_messages=64,
                    episode_timeout_seconds=float("inf"),
                )
            },
            "episode_timeout_seconds must be a positive, finite number",
        ),
        ({"adapter_settings": {"x": object()}}, "not JSON-safe"),
        (
            {
                "limits": RunLimits(
                    max_turns=4,
                    max_messages=64,
                    episode_timeout_seconds="30",
                )
            },
            "episode_timeout_seconds must be a positive, finite number",
        ),
        (
            {
                "limits": RunLimits(
                    max_turns=4,
                    max_messages=64,
                    episode_timeout_seconds=float("nan"),
                )
            },
            "episode_timeout_seconds must be a positive, finite number",
        ),
    ],
)
def test_invalid_run_parameters_are_refused(
    kwargs: dict[str, object], expected: str
) -> None:
    with pytest.raises(RunManifestError) as excinfo:
        _build(**kwargs)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({7: "x"}, id="non-string-key"),
        pytest.param({"temperature": float("nan")}, id="nan"),
        pytest.param({"temperature": float("inf")}, id="infinity"),
    ],
)
def test_settings_json_safety_is_refused_by_the_shared_validator(
    settings: dict[Any, Any],
) -> None:
    """Settings that canonical JSON cannot carry fail as one named domain error.

    The wording belongs to :mod:`boundarybench.jsonsafe`, which is the single
    validator every artefact boundary shares; asserting the raised class and its
    cause pins the contract without pinning one module's prose, and without a
    second settings-only validator existing to keep an old sentence alive.
    """
    with pytest.raises(RunManifestFormatError) as excinfo:
        _build(adapter_settings=settings)

    assert isinstance(excinfo.value.__cause__, NonJsonValueError)
    assert "adapter_settings" in str(excinfo.value)


def test_manifest_records_the_provenance_needed_to_reproduce_it(
    tmp_path: Path,
) -> None:
    manifest = _build()
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    payload = manifest.as_dict()

    assert payload["suite"]["suite_content_digest"] == suite.content_digest
    assert payload["suite"]["suite_id"] == suite.manifest.suite_id
    assert payload["suite"]["benchmark_version"] == suite.manifest.benchmark_version
    assert payload["scaffold"]["content_digest"] == scaffold.content_digest
    assert payload["scaffold"]["scaffold_version"] == scaffold.scaffold_version
    assert payload["adapter"]["provider"] == "test-double"
    assert payload["contract_versions"]["runner"]
    assert payload["contract_versions"]["evaluator"]
    assert payload["contract_versions"]["package"]
    assert "no model" in payload["scope"].lower()
    assert payload["status"] == "SYNTHETIC_INFRASTRUCTURE_SMOKE"


def test_the_recorded_package_version_is_the_version_that_ships() -> None:
    """``contract_versions.package`` has to name a build someone can install.

    The manifest exists so a run can be reproduced. If the package version it
    records is not a version the installed distribution actually ships, the
    recorded provenance points at a build that was never published, and the one
    field that says *which BoundaryBench ran* cannot be acted on.

    The distribution is ``operatebench``; ``boundarybench`` is a package inside
    it and keeps its own historical version, which is deliberately *not* the
    distribution's — see ``docs/VERSIONING.md``. Asking the metadata for a
    distribution named ``boundarybench`` therefore always raised
    ``PackageNotFoundError``, and the skip that answered it meant this check
    never ran anywhere: not in development, not against an installed project,
    not against the built wheel. So the metadata is queried under the name that
    is actually installed, and the recorded package version is checked against
    the module the distribution ships rather than skipped.
    """
    from importlib.metadata import version

    import boundarybench
    import operatebench
    from boundarybench.version import __version__
    from operatebench.version import __version__ as distribution_version

    assert version("operatebench") == distribution_version
    # Both packages ship inside that one distribution, so the version the
    # manifest records is a version an installer can actually put on disk.
    assert operatebench.__name__ == "operatebench"
    assert boundarybench.__version__ == __version__
    assert _build().as_dict()["contract_versions"]["package"] == __version__
    # The two lines are separate on purpose: a package version that had been
    # renumbered into the distribution's line would silently change what a
    # historical manifest's provenance means.
    assert __version__ != distribution_version


# -- persistence -------------------------------------------------------------


def test_manifest_is_written_once_atomically_and_never_overwritten(
    tmp_path: Path,
) -> None:
    manifest = _build()
    root = tmp_path / "run"

    with open_run_session(root, manifest) as session:
        paths, resumed = session.paths, session.resumed

    assert resumed is False
    assert paths.root == root
    assert paths.manifest_path == root / "run_manifest.json"
    assert paths.ledger_path == root / "episodes.jsonl"
    first_bytes = paths.manifest_path.read_bytes()
    assert not list(root.glob("*.tmp"))
    assert not list(root.glob(".run_manifest.json.*"))

    with open_run_session(root, manifest) as second:
        assert second.resumed is True
        assert second.paths == paths

    assert paths.manifest_path.read_bytes() == first_bytes


def test_resume_refuses_a_manifest_for_a_different_request(tmp_path: Path) -> None:
    root = tmp_path / "run"
    with open_run_session(root, _build()):
        pass

    with (
        pytest.raises(RunManifestMismatchError) as excinfo,
        open_run_session(root, _build(trials=3)),
    ):
        pass  # pragma: no cover - the mismatch must refuse this

    assert "already holds configuration" in str(excinfo.value)


def test_resume_refuses_a_tampered_manifest(tmp_path: Path) -> None:
    """Editing a recorded field breaks the hash it was recorded under."""
    root = tmp_path / "run"
    manifest = _build()
    with open_run_session(root, manifest) as session:
        paths = session.paths
    payload = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    payload["limits"]["max_turns"] = 99
    paths.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with (
        pytest.raises(RunManifestMismatchError) as excinfo,
        open_run_session(root, manifest),
    ):
        pass  # pragma: no cover - the tamper must refuse this

    assert "does not hash to its recorded configuration_id" in str(excinfo.value)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda p: p.pop("trials"), "missing required field(s) ['trials']"),
        (lambda p: p.update(extra=1), "unknown field(s) ['extra']"),
        (lambda p: p.update(trials=True), "'trials' must be a positive integer"),
        (lambda p: p.update(trials="2"), "'trials' must be a positive integer"),
        (
            lambda p: p.update(configuration_id="short"),
            "'configuration_id' must be a lowercase",
        ),
        # ``run_id`` is not merely a renamed field: a manifest that states one is
        # a manifest from the schema that conflated configuration with execution,
        # and it is refused as unknown rather than read under a guessed meaning.
        (lambda p: p.update(run_id="0" * 64), "unknown field(s) ['run_id']"),
        (lambda p: p.update(schema_version=6), "unsupported schema_version 6"),
        # The three identity fields are checked against the provider the same
        # file records, so each of them names itself when it disagrees.
        (lambda p: p.update(status="OFFICIAL"), "its 'status' is fixed"),
        (lambda p: p.update(track="official"), "'track' must be one of"),
        (
            lambda p: p.update(retry_policy={"retries": 1, "policy": "none"}),
            "'retry_policy' must be exactly",
        ),
        (lambda p: p.update(episode_plan=[]), "'episode_plan' must not be empty"),
        (lambda p: p.update(adapter={"provider": "x"}), "missing"),
        (lambda p: p["limits"].update(max_turns=0), "must be a positive integer"),
        (lambda p: p.update(created_at="whenever"), "unknown field(s)"),
        (
            lambda p: p.update(scope="An official benchmark result."),
            "its 'scope' is fixed",
        ),
        (lambda p: p.update(suite="the spike suite"), "'suite' must be a JSON object"),
        (lambda p: p["suite"].update(suite_id=""), "'suite_id' must be a non-empty"),
        (lambda p: p.update(episode_plan={}), "'episode_plan' must be a list"),
        (lambda p: p.update(episode_plan=["first"]), "episode_plan[0] must be a JSON"),
        (
            lambda p: p["episode_plan"][0].update(episode_id="renamed#t001"),
            "is not the id of this plan entry",
        ),
        (
            lambda p: p["episode_plan"].append(dict(p["episode_plan"][0])),
            "duplicate episode_id",
        ),
    ],
)
def test_malformed_stored_manifests_are_refused(
    tmp_path: Path, mutate, expected: str
) -> None:
    root = tmp_path / "run"
    with open_run_session(root, _build()) as session:
        paths = session.paths
    payload = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    paths.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunManifestError) as excinfo:
        load_run_manifest(paths.manifest_path)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"trials": 1, "trials": 2}', "duplicate JSON key 'trials'"),
        ('{"trials": NaN}', "not a finite JSON value"),
        ("[]", "must be a JSON object"),
        ("{", "is not valid JSON"),
    ],
)
def test_unparseable_stored_manifests_are_refused(
    tmp_path: Path, body: str, expected: str
) -> None:
    path = tmp_path / "run_manifest.json"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(RunManifestFormatError) as excinfo:
        load_run_manifest(path)

    assert expected in str(excinfo.value)


def test_a_stored_manifest_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    path.write_bytes(b'{"run_id": "\xff\xfe"}')

    with pytest.raises(RunManifestFormatError) as excinfo:
        load_run_manifest(path)

    assert "not valid UTF-8" in str(excinfo.value)


def test_a_manifest_path_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunManifestFormatError) as excinfo:
        load_run_manifest(tmp_path / "absent.json")

    assert "cannot read run manifest" in str(excinfo.value)


def test_json_safe_settings_are_recorded_normalised_and_sorted() -> None:
    """Settings are part of run identity, so their recorded form is canonical."""
    manifest = _build(
        adapter_settings={
            "top_p": 0.9,
            "stop": ["END", "STOP"],
            "max_output": 512,
            "stream": False,
            "seed": None,
            "nested": {"b": "two", "a": "one"},
        }
    )

    settings = manifest.as_dict()["adapter_settings"]

    assert list(settings) == [
        "max_output",
        "nested",
        "seed",
        "stop",
        "stream",
        "top_p",
    ]
    assert list(settings["nested"]) == ["a", "b"]
    assert settings["stop"] == ["END", "STOP"]
    assert settings["top_p"] == 0.9
    assert settings["seed"] is None
    assert settings["stream"] is False


def test_symlinked_output_paths_are_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(RunPathError) as excinfo, open_run_session(link, _build()):
        pass  # pragma: no cover - the unsafe path must refuse this

    assert "symlink" in str(excinfo.value)


def test_symlinked_run_files_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").symlink_to(tmp_path / "elsewhere.json")

    with pytest.raises(RunPathError) as excinfo, open_run_session(root, _build()):
        pass  # pragma: no cover - the unsafe path must refuse this

    assert "symlink" in str(excinfo.value)


def test_output_path_that_is_a_file_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RunPathError) as excinfo, open_run_session(root, _build()):
        pass  # pragma: no cover - the unsafe path must refuse this

    assert "not a directory" in str(excinfo.value)


def test_a_stored_retry_policy_naming_another_policy_is_refused(tmp_path: Path) -> None:
    """The stored policy is checked, not just the stored retry count.

    ``identity_payload`` reprojects the pinned :data:`RETRY_POLICY` rather than
    echoing what the file said, so a stored ``"policy"`` that is merely
    *ignored* would still rehash to the recorded ``run_id`` — the manifest would
    claim a retry policy the run never had and validate perfectly. The check on
    the retries count cannot stand in for it: this edit leaves ``retries`` at
    zero and changes only the name of the policy.
    """
    root = tmp_path / "run"
    with open_run_session(root, _build()) as session:
        paths = session.paths
    payload = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    payload["retry_policy"] = {"retries": 0, "policy": "exponential_backoff"}
    paths.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunManifestFormatError) as excinfo:
        load_run_manifest(paths.manifest_path)

    assert "'retry_policy' must be exactly" in str(excinfo.value)


# -- the canonical encoding --------------------------------------------------

#: Text a run's settings realistically carry once anything is written for a
#: non-English market: prose, an accent and a currency symbol. Every artefact
#: this build ships is pure ASCII, which leaves the encoding choice unobserved.
UNICODE_PAYLOAD: dict[str, str] = {
    "prompt": "Répondez en français ☕",
    "scope": "Tarifs en €",
}

#: :data:`UNICODE_PAYLOAD` in the documented canonical form — sorted keys,
#: compact separators, non-ASCII left unescaped — written out by hand rather
#: than produced by :func:`canonical_json`, so a change to the production
#: encoder cannot move this expectation with it.
UNICODE_CANONICAL_JSON = '{"prompt":"Répondez en français ☕","scope":"Tarifs en €"}'

#: SHA-256 over :data:`UNICODE_CANONICAL_JSON`'s UTF-8 bytes, computed out of
#: band. A literal, for the same reason.
UNICODE_PAYLOAD_DIGEST = (
    "a30a0c4989228d479ac95ff7622f78526a025c82d49461e3e22ba3266d9e6592"
)


def test_the_canonical_encoding_leaves_non_ascii_text_unescaped() -> None:
    """``ensure_ascii=False`` is part of run identity, not a formatting detail.

    On ASCII input both spellings agree, so nothing observes the choice today.
    The first run configured with a non-English prompt would get one run id
    under each spelling, and a manifest written by one build would fail to
    rehash under the other.
    """
    assert not UNICODE_CANONICAL_JSON.isascii()
    assert canonical_json(UNICODE_PAYLOAD) == UNICODE_CANONICAL_JSON
    assert run_identity_digest(UNICODE_PAYLOAD) == UNICODE_PAYLOAD_DIGEST


def test_a_configuration_id_is_the_utf8_hash_of_its_unescaped_payload() -> None:
    """The same encoding, reached through the whole manifest rather than a vector.

    The expectation is re-encoded here by hand, so the production encoder is
    never asked to confirm itself.
    """
    manifest = _build(adapter_settings={"system_prefix": "Répondez en français ☕"})
    encoded = json.dumps(
        manifest.configuration_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    assert not encoded.isascii()
    assert (
        manifest.configuration_id == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    )


def test_the_execution_digest_is_the_hash_of_exactly_its_three_bound_facts() -> None:
    """The second digest, encoded by hand for the same reason as the first.

    It covers the configuration it belongs to, the execution id and the creation
    instant, and nothing else — so this states that list rather than trusting
    :meth:`RunManifest.execution_payload` to state it.
    """
    manifest = _build()
    encoded = json.dumps(
        {
            "configuration_id": manifest.configuration_id,
            "created_at_utc": manifest.created_at_utc,
            "execution_id": manifest.execution_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    assert (
        manifest.execution_digest == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_refuses_a_number_json_cannot_carry(value: float) -> None:
    """``NaN`` and the infinities are not JSON, and are refused rather than spelt.

    Python writes them as bare ``NaN``/``Infinity`` tokens, which no conforming
    reader accepts — and this build's own readers reject on sight. A digest
    taken over one would name a document nothing can read back.
    """
    with pytest.raises(ValueError) as excinfo:
        canonical_json({"value": value})

    assert "is not a finite number" in str(excinfo.value)


# -- atomic publication ------------------------------------------------------


def test_the_scratch_manifest_is_created_inside_the_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publication is an ``os.link``, which cannot cross a filesystem.

    A scratch file made in the system temporary directory would fail outright
    whenever ``--output-dir`` is on another mount, and even where it works it
    ends the run's one write in a directory the operator never named.
    """
    created: list[Path] = []
    real_mkstemp = tempfile.mkstemp

    def spy(*args: Any, **kwargs: Any) -> tuple[int, str]:
        descriptor, name = real_mkstemp(*args, **kwargs)
        created.append(Path(name))
        return descriptor, name

    monkeypatch.setattr(tempfile, "mkstemp", spy)
    root = tmp_path / "run"
    root.mkdir()
    paths = resolve_run_paths(root)

    write_run_manifest(paths, _build())

    assert created
    assert [path.parent for path in created] == [root]
    assert not list(root.glob(f".{MANIFEST_FILENAME}.*"))


def test_publication_tolerates_a_scratch_file_that_is_already_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup is best-effort; the manifest is published by the link, not by it.

    A temporary-file sweeper, or a second cleanup, can remove the scratch file
    between the link and the unlink. Raising there would mask a manifest that is
    already on disk and turn a completed publication into a run failure.
    """
    root = tmp_path / "run"
    root.mkdir()
    paths = resolve_run_paths(root)
    manifest = _build()
    real_link = os.link

    def link_then_sweep(source: Any, target: Any, **kwargs: Any) -> None:
        real_link(source, target)
        os.unlink(source)

    monkeypatch.setattr(os, "link", link_then_sweep)

    write_run_manifest(paths, manifest)

    assert load_run_manifest(paths.manifest_path).run_id == manifest.run_id


# -- limits and paths at their boundaries ------------------------------------


def test_an_episode_timeout_of_exactly_one_second_is_accepted(tmp_path: Path) -> None:
    """One second is a budget, not a rejected boundary.

    ``episode_timeout_seconds`` is refused at zero and below. One is the
    smallest whole budget above that, it is a value an operator would plausibly
    pass to bound a smoke run, and it has to survive the round trip through the
    stored manifest as well as the build.
    """
    manifest = _build(
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=1)
    )
    assert manifest.limits.episode_timeout_seconds == 1.0

    with open_run_session(tmp_path / "run", manifest) as session:
        paths = session.paths

    stored = load_run_manifest(paths.manifest_path)
    assert stored.limits.episode_timeout_seconds == 1.0
    assert stored.run_id == manifest.run_id


def test_a_relative_output_path_is_resolved_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--output-dir runs/today`` is an ordinary way to name a run directory.

    Safety is decided on the absolute path, because an ancestor link is only
    visible there; a relative path therefore has to be resolved first rather
    than rejected or crashed on.
    """
    monkeypatch.chdir(tmp_path)
    manifest = _build()

    with open_run_session(Path("run"), manifest) as session:
        paths = session.paths

    assert paths.root == Path("run")
    assert (tmp_path / "run" / MANIFEST_FILENAME).exists()
    assert load_run_manifest(tmp_path / "run" / MANIFEST_FILENAME).run_id == (
        manifest.run_id
    )


def test_a_relative_path_through_a_symlinked_ancestor_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ancestor check has to survive the resolution, not be skipped by it."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "linked").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RunPathError) as excinfo:
        resolve_run_paths(Path("linked") / "run")

    assert "symlink" in str(excinfo.value)
    assert not (elsewhere / "run").exists()
