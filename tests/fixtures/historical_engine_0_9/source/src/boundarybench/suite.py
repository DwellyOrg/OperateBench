"""Suite manifests: the collection-level contract over several Cubes.

One Cube proves the machinery runs. A suite proves it generalises, which is only
meaningful if the properties that make it a generalisation are *enforced* rather
than described. A suite manifest therefore declares the shape of the collection
it expects — how many Cubes of each type, how many of each fact location, how
the expected dispositions balance across every variant — and validation fails
when the artefacts on disk disagree.

The manifest names sibling files, so path handling is treated as untrusted
input: absolute paths, ``..`` segments and symlinks in referenced path segments
are refused rather than resolved, and every referenced artefact must live under
the suite directory. The caller may itself reach that directory through a
symlink; the containment boundary is the directory containing the suite file.

The collection's identity is pinned, not merely reported. ``suite_content_digest``
is declared in the versioned artefact and verified on every validation, so a
change to the canonicalisation, to the hashed payload or to any member Cube
cannot quietly rewrite what the suite *is* while the tests stay green; it has to
be re-pinned in the same reviewed commit.

Nothing here scores a model. The reported constant-strategy results are
degenerate baselines that show what an agent gets for free by always answering
the same way; they are not a floor on autonomy and must not be presented as one.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from boundarybench.compiler import CompileError, Cube, compile_cube
from boundarybench.jsonsafe import JsonSafetyError, canonical_json_bytes
from boundarybench.loader import load_card, read_yaml_mapping
from boundarybench.manifest import (
    ManifestError,
    SemanticManifest,
    card_fingerprint,
    load_manifest,
)
from boundarybench.manifest import verify_cube as _verify_cube
from boundarybench.schema import (
    CUBE_TYPES,
    DISPOSITIONS,
    ELICITATION_ACTIONS,
    FACT_LOCATIONS,
    PROTOCOL_ACTIONS,
    SchemaError,
)
from boundarybench.solvers import SolverReport, check_solvers

#: The suite file format this build reads.
SUITE_SCHEMA_VERSION = 1

#: The benchmark version this suite ships under. Explicitly a prerelease: there
#: is no stable benchmark to version yet, and a bare ``0.2.0`` would imply one.
BENCHMARK_VERSION = "0.2.0-dev.1"

#: The only lifecycle status a suite may claim in this build.
SUITE_STATUSES: frozenset[str] = frozenset({"METHODOLOGY_SPIKE"})
#: Only fully synthetic collections are admissible.
SUITE_PRIVACY_STATUSES: frozenset[str] = frozenset({"SYNTHETIC_ONLY"})

#: What the report says about itself, so no consumer has to infer the scope.
SUITE_SCOPE = (
    "Synthetic methodology spike. Not a model benchmark release: no model runs, "
    "no ranking and no public claim are supported by this suite."
)
#: Said in full every time the constant strategies are reported.
CONSTANT_STRATEGY_NOTE = (
    "Degenerate baselines: the score an agent gets by always answering the same "
    "way, shown so a constant strategy cannot be mistaken for competence. This "
    "is not a floor on autonomy."
)

#: A prerelease is mandatory, so a suite cannot claim a stable version number.
_BENCHMARK_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+-[0-9A-Za-z]+(\.[0-9A-Za-z]+)*$")
_SUITE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "suite_id",
    "benchmark_version",
    "status",
    "privacy_status",
    "disclaimer",
    "suite_content_digest",
    "constraints",
    "entries",
)
_ENTRY_KEYS: tuple[str, ...] = ("cube_id", "card", "semantic_manifest")
_CONSTRAINT_SECTIONS: tuple[str, ...] = (
    "cube_types",
    "fact_locations",
    "expected_dispositions",
)
_CONSTRAINT_VOCABULARY: Mapping[str, frozenset[str]] = {
    "cube_types": CUBE_TYPES,
    "fact_locations": FACT_LOCATIONS,
    "expected_dispositions": DISPOSITIONS,
}


class SuiteError(ValueError):
    """Base class for every suite-level failure."""


class SuiteFormatError(SuiteError):
    """The suite manifest is unreadable, malformed or internally inconsistent."""


class SuiteDigestError(SuiteFormatError):
    """The declared ``suite_content_digest`` is not what the artefacts hash to.

    A :class:`SuiteFormatError`, because a pin that disagrees with its own
    collection makes the manifest internally inconsistent — the same class of
    failure as a malformed field, and handled by the same callers. The declared
    and computed digests are carried as attributes so a caller (or a test) can
    act on the values without parsing the message.
    """

    def __init__(self, message: str, *, declared: str, computed: str) -> None:
        super().__init__(message)
        self.declared = declared
        self.computed = computed


class SuitePathError(SuiteError):
    """A referenced artefact path is unsafe or unusable."""


class SuiteConstraintError(SuiteError):
    """The artefacts are individually valid but the collection is not what it claims."""


class SuiteCubeError(SuiteError):
    """A referenced construct card does not load or does not compile."""


class SuiteManifestError(SuiteError):
    """A referenced semantic manifest is malformed or no longer describes its card."""


# -- path safety -------------------------------------------------------------


def _resolve_artefact(directory: Path, value: str, context: str) -> Path:
    """Resolve one relative artefact path, refusing anything that could escape.

    A suite manifest is a reviewed file that names its siblings. Anything that
    reaches beyond the suite directory — an absolute path, a ``..`` segment, or
    a symlink in any manifest-authored path segment — is refused outright rather
    than resolved, so what a reviewer reads in the manifest is what validation
    actually loads. The directory containing the suite file is the containment
    boundary even when the caller reached that directory through a symlink.
    """
    if value != value.strip():
        raise SuitePathError(f"{context}: path {value!r} has surrounding whitespace")

    pure = PurePosixPath(value)
    if pure.is_absolute() or Path(value).is_absolute() or value.startswith("\\"):
        raise SuitePathError(
            f"{context}: path {value!r} must be relative to the suite directory"
        )
    # Split the string exactly as authored. ``PurePosixPath`` normalises
    # ``./x``, ``x/`` and ``a//x`` into the same parts as ``x``, so trusting it
    # would let one artefact enter a suite under several distinct spellings —
    # and the duplicate-card, duplicate-manifest and cross-Cube guards all
    # compare the strings a reviewer actually reads.
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SuitePathError(
            f"{context}: path {value!r} must be a canonical relative path, with "
            "no empty, '.' or '..' segments"
        )

    current = directory
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise SuitePathError(
                f"{context}: path {value!r} passes through symlink {part!r}; a "
                "suite may only reference real files under its own directory"
            )
    if not current.exists():
        raise SuitePathError(f"{context}: {value!r} does not exist under {directory}")
    if not current.is_file():
        raise SuitePathError(f"{context}: {value!r} is not a file")
    return current


# -- parsed manifest ---------------------------------------------------------


@dataclass(frozen=True)
class SuiteEntry:
    """One Cube in the suite: its declared id and its two resolved artefacts."""

    cube_id: str
    card: str
    semantic_manifest: str
    card_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SuiteConstraints:
    cube_types: Mapping[str, int]
    fact_locations: Mapping[str, int]
    expected_dispositions: Mapping[str, int]

    def __post_init__(self) -> None:
        # The suite content digest is taken over exactly these tallies, so a
        # section that stayed editable afterwards would leave a report quoting
        # constraints its own digest was never computed over. Frozen here rather
        # than only in the parser, so a directly constructed SuiteConstraints
        # carries the same immutable semantics as an authored one. Counts are
        # ints, so proxying each section is the whole of "recursively immutable".
        for name in _CONSTRAINT_SECTIONS:
            section: Mapping[str, int] = getattr(self, name)
            object.__setattr__(self, name, MappingProxyType(dict(section)))

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            "cube_types": dict(self.cube_types),
            "fact_locations": dict(self.fact_locations),
            "expected_dispositions": dict(self.expected_dispositions),
        }


@dataclass(frozen=True)
class SuiteManifest:
    schema_version: int
    suite_id: str
    benchmark_version: str
    status: str
    privacy_status: str
    disclaimer: str
    #: The collection's pinned identity, as declared in the versioned artefact.
    #: :func:`validate_suite` refuses the suite when it disagrees with what the
    #: artefacts on disk actually hash to.
    suite_content_digest: str
    constraints: SuiteConstraints
    entries: tuple[SuiteEntry, ...]
    directory: Path
    path: Path

    @property
    def normalized_disclaimer(self) -> str:
        """Whitespace-normalised, so reflowing the YAML block is free."""
        return " ".join(self.disclaimer.split())


# -- parsing -----------------------------------------------------------------


def _text(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise SuiteFormatError(f"{context}: {key!r} must be a non-empty string")
    return value


def _member(
    raw: Mapping[str, Any], key: str, allowed: frozenset[str], context: str
) -> str:
    value = raw[key]
    if not isinstance(value, str) or value not in allowed:
        raise SuiteFormatError(
            f"{context}: {key!r} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def _sha256(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise SuiteFormatError(
            f"{context}: {key!r} must be a lowercase 64-character SHA-256 hex "
            f"digest, got {value!r}"
        )
    return value


def _count(raw: Mapping[str, Any], key: str, context: str) -> int:
    value = raw[key]
    # bool is an int subclass, so `1-ACT: true` must not read as a count of 1.
    if type(value) is not int or value < 0:
        raise SuiteFormatError(
            f"{context}: count for {key!r} must be a non-negative integer, got {value!r}"
        )
    return value


def _parse_constraints(raw: Mapping[str, Any], context: str) -> SuiteConstraints:
    section = raw["constraints"]
    if not isinstance(section, Mapping):
        raise SuiteFormatError(f"{context}: 'constraints' must be a mapping")
    missing = [name for name in _CONSTRAINT_SECTIONS if name not in section]
    if missing:
        raise SuiteFormatError(f"{context}: constraints is missing {missing}")
    unknown = sorted(set(map(str, section)) - set(_CONSTRAINT_SECTIONS))
    if unknown:
        raise SuiteFormatError(f"{context}: constraints has unknown sections {unknown}")

    parsed: dict[str, dict[str, int]] = {}
    for name in _CONSTRAINT_SECTIONS:
        body = section[name]
        body_context = f"{context}: constraints.{name}"
        if not isinstance(body, Mapping) or not body:
            raise SuiteFormatError(f"{body_context} must be a non-empty mapping")
        allowed = _CONSTRAINT_VOCABULARY[name]
        unknown_keys = sorted(str(key) for key in body if key not in allowed)
        if unknown_keys:
            raise SuiteFormatError(
                f"{body_context}: unknown key(s) {unknown_keys}; allowed: "
                f"{sorted(allowed)}"
            )
        parsed[name] = {str(key): _count(body, key, body_context) for key in body}
    return SuiteConstraints(
        cube_types=parsed["cube_types"],
        fact_locations=parsed["fact_locations"],
        expected_dispositions=parsed["expected_dispositions"],
    )


def _parse_entries(
    raw: Mapping[str, Any], directory: Path, context: str
) -> tuple[SuiteEntry, ...]:
    section = raw["entries"]
    if isinstance(section, (str, bytes)) or not isinstance(section, Sequence):
        raise SuiteFormatError(f"{context}: 'entries' must be a list")
    if not section:
        raise SuiteFormatError(f"{context}: 'entries' must not be empty")

    entries: list[SuiteEntry] = []
    seen_ids: set[str] = set()
    seen_cards: set[str] = set()
    seen_manifests: set[str] = set()
    for index, item in enumerate(section):
        entry_context = f"{context}: entries[{index}]"
        if not isinstance(item, Mapping):
            raise SuiteFormatError(f"{entry_context} must be a mapping")
        missing = [key for key in _ENTRY_KEYS if key not in item]
        if missing:
            raise SuiteFormatError(f"{entry_context} is missing {missing}")
        unknown = sorted(str(key) for key in item if key not in _ENTRY_KEYS)
        if unknown:
            raise SuiteFormatError(f"{entry_context} has unknown field(s) {unknown}")

        cube_id = _text(item, "cube_id", entry_context)
        card = _text(item, "card", entry_context)
        manifest = _text(item, "semantic_manifest", entry_context)
        if cube_id in seen_ids:
            raise SuiteFormatError(f"{entry_context}: duplicate cube_id {cube_id!r}")
        if card in seen_cards:
            raise SuiteFormatError(f"{entry_context}: duplicate card path {card!r}")
        if manifest in seen_manifests:
            raise SuiteFormatError(
                f"{entry_context}: duplicate semantic_manifest path {manifest!r}"
            )
        seen_ids.add(cube_id)
        seen_cards.add(card)
        seen_manifests.add(manifest)

        entries.append(
            SuiteEntry(
                cube_id=cube_id,
                card=card,
                semantic_manifest=manifest,
                card_path=_resolve_artefact(directory, card, f"{entry_context}.card"),
                manifest_path=_resolve_artefact(
                    directory, manifest, f"{entry_context}.semantic_manifest"
                ),
            )
        )
    return tuple(entries)


def load_suite(path: str | Path) -> SuiteManifest:
    """Read and strictly validate a suite manifest, resolving its artefacts."""
    path = Path(path)
    context = str(path)
    try:
        raw = read_yaml_mapping(path, what="suite manifest")
    except SchemaError as exc:
        raise SuiteFormatError(str(exc)) from exc

    offenders = sorted(repr(key) for key in raw if not isinstance(key, str))
    if offenders:
        raise SuiteFormatError(
            f"{context}: mapping keys must be strings, got {offenders}"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise SuiteFormatError(f"{context}: missing required field(s) {missing}")
    unknown = sorted(set(raw) - set(_REQUIRED_KEYS))
    if unknown:
        raise SuiteFormatError(
            f"{context}: unknown field(s) {unknown}; allowed: {sorted(_REQUIRED_KEYS)}"
        )

    version = raw["schema_version"]
    if type(version) is not int or version != SUITE_SCHEMA_VERSION:
        raise SuiteFormatError(
            f"{context}: unsupported schema_version {version!r}; this build "
            f"reads {SUITE_SCHEMA_VERSION}"
        )

    suite_id = _text(raw, "suite_id", context)
    if not _SUITE_ID_PATTERN.fullmatch(suite_id):
        raise SuiteFormatError(
            f"{context}: suite_id {suite_id!r} must use only lowercase letters, "
            "digits, underscores and hyphens"
        )

    benchmark_version = _text(raw, "benchmark_version", context)
    if not _BENCHMARK_VERSION_PATTERN.fullmatch(benchmark_version):
        raise SuiteFormatError(
            f"{context}: benchmark_version {benchmark_version!r} must be an "
            "explicit prerelease such as '0.2.0-dev.1'; there is no stable "
            "benchmark version to claim"
        )

    return SuiteManifest(
        schema_version=version,
        suite_id=suite_id,
        benchmark_version=benchmark_version,
        status=_member(raw, "status", SUITE_STATUSES, context),
        privacy_status=_member(raw, "privacy_status", SUITE_PRIVACY_STATUSES, context),
        disclaimer=_text(raw, "disclaimer", context),
        suite_content_digest=_sha256(raw, "suite_content_digest", context),
        constraints=_parse_constraints(raw, context),
        entries=_parse_entries(raw, path.parent, context),
        directory=path.parent,
        path=path,
    )


# -- validation report -------------------------------------------------------


def _disposition_counts(cube: Cube) -> dict[str, int]:
    counts = dict.fromkeys(sorted(DISPOSITIONS), 0)
    for variant in cube.variants:
        counts[variant.expected_disposition] += 1
    return counts


@dataclass(frozen=True)
class SuiteCubeReport:
    """One validated Cube, as the suite sees it."""

    cube_id: str
    card: str
    semantic_manifest: str
    cube: Cube
    manifest: SemanticManifest
    card_fingerprint_sha256: str

    @property
    def cube_type(self) -> str:
        return self.cube.card.cube_type

    @property
    def fact_location(self) -> str:
        return self.cube.card.fact_location

    @property
    def expected_dispositions(self) -> dict[str, int]:
        return _disposition_counts(self.cube)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cube_id": self.cube_id,
            "card": self.card,
            "semantic_manifest": self.semantic_manifest,
            "cube_type": self.cube_type,
            "fact_location": self.fact_location,
            "privacy_status": self.cube.card.privacy_status,
            "card_fingerprint_sha256": self.card_fingerprint_sha256,
            "variant_count": len(self.cube.variants),
            "expected_dispositions": self.expected_dispositions,
            "constant_strategy_scores": dict(self.cube.constant_strategy_scores),
            "variants": [
                {"variant_id": v.variant_id, "content_digest": v.content_digest}
                for v in self.cube.variants
            ],
        }


@dataclass(frozen=True)
class SuiteReport:
    """A suite that passed schema, path, manifest, compile and constraint gates."""

    manifest: SuiteManifest
    cubes: tuple[SuiteCubeReport, ...]
    content_digest: str

    #: Validation raises on any failure, so a report that exists has passed.
    ok: bool = True

    @property
    def cube_count(self) -> int:
        return len(self.cubes)

    @property
    def variant_count(self) -> int:
        return sum(len(cube.cube.variants) for cube in self.cubes)

    @property
    def cube_types(self) -> dict[str, int]:
        return self._tally(cube.cube_type for cube in self.cubes)

    @property
    def fact_locations(self) -> dict[str, int]:
        return self._tally(cube.fact_location for cube in self.cubes)

    @property
    def expected_dispositions(self) -> dict[str, int]:
        counts = dict.fromkeys(sorted(DISPOSITIONS), 0)
        for cube in self.cubes:
            for disposition, count in cube.expected_dispositions.items():
                counts[disposition] += count
        return counts

    @property
    def constant_strategy_scores(self) -> dict[str, int]:
        """What a suite-wide always-ACT and always-STOP agent would score."""
        totals = {"always_act": 0, "always_stop": 0}
        for cube in self.cubes:
            for name, score in cube.cube.constant_strategy_scores.items():
                totals[name] += score
        return totals

    @property
    def degenerate_strategies_correct(self) -> bool:
        """Each constant strategy scores exactly its own disposition's variants."""
        dispositions = self.expected_dispositions
        scores = self.constant_strategy_scores
        return (
            scores["always_act"] == dispositions["ACT"]
            and scores["always_stop"] == dispositions["STOP"]
            and scores["always_act"] + scores["always_stop"] == self.variant_count
        )

    @staticmethod
    def _tally(values: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "suite_id": self.manifest.suite_id,
            "benchmark_version": self.manifest.benchmark_version,
            "status": self.manifest.status,
            "privacy_status": self.manifest.privacy_status,
            "scope": SUITE_SCOPE,
            "disclaimer": self.manifest.normalized_disclaimer,
            "suite_content_digest_sha256": self.content_digest,
            "cube_count": self.cube_count,
            "variant_count": self.variant_count,
            "cube_types": self.cube_types,
            "fact_locations": self.fact_locations,
            "expected_dispositions": self.expected_dispositions,
            "constant_strategy_scores": self.constant_strategy_scores,
            "constant_strategy_note": CONSTANT_STRATEGY_NOTE,
            "degenerate_strategies_correct": self.degenerate_strategies_correct,
            "declared_constraints": self.manifest.constraints.as_dict(),
            "cubes": [cube.as_dict() for cube in self.cubes],
        }


def suite_content_digest(
    manifest: SuiteManifest, cubes: Sequence[SuiteCubeReport]
) -> str:
    """Hash the normalised suite metadata, card fingerprints and variant digests.

    Everything that changes what the collection *is* is covered: its identity and
    version, the constraints it commits to, which Cubes it contains, each card's
    semantic fingerprint and each member variant's id *and* content digest in
    compiled order. Both halves are named explicitly rather than leaning on the
    id being inside its own content digest: that transitive argument only holds
    for variants the compiler produced, so it is not a collection-level identity
    contract. YAML formatting and comment changes are not covered.

    The declared ``suite_content_digest`` is the one field deliberately left out:
    a payload that hashed the pin would have no fixed point, so the pin could
    never be stated. Everything the pin exists to protect is still in here.

    Public because ``SuiteReport.content_digest`` travels inside the report it
    describes. A caller about to execute against a suite has to derive the
    identity again from the objects it is actually holding — the alternative is
    reading back a field that a rebuilt report supplied for itself.
    """
    payload = {
        "schema_version": manifest.schema_version,
        "suite_id": manifest.suite_id,
        "benchmark_version": manifest.benchmark_version,
        "status": manifest.status,
        "privacy_status": manifest.privacy_status,
        "disclaimer": manifest.normalized_disclaimer,
        "constraints": manifest.constraints.as_dict(),
        "cubes": [
            {
                "cube_id": cube.cube_id,
                "card": cube.card,
                "semantic_manifest": cube.semantic_manifest,
                "cube_type": cube.cube_type,
                "fact_location": cube.fact_location,
                "card_fingerprint_sha256": cube.card_fingerprint_sha256,
                "variants": [
                    {
                        "variant_id": variant.variant_id,
                        "content_digest": variant.content_digest,
                    }
                    for variant in cube.cube.variants
                ],
            }
            for cube in cubes
        ],
    }
    try:
        return hashlib.sha256(
            canonical_json_bytes(payload, f"suite {manifest.suite_id}")
        ).hexdigest()
    except JsonSafetyError as exc:
        raise SuiteFormatError(
            f"{manifest.path}: this suite cannot be digested: {exc}"
        ) from exc


def _verify_declared_digest(manifest: SuiteManifest, computed: str) -> None:
    """Refuse a suite whose pinned identity is not the identity it actually has.

    Reporting a computed digest proves nothing on its own: a change to the
    canonicalisation or to the payload fields silently rewrites what the
    collection *is* while every test stays green. Declaring the digest in the
    versioned artefact and checking it here makes that drift a failure, so a
    genuine semantic revision has to be re-pinned in the same reviewed commit.
    """
    declared = manifest.suite_content_digest
    if declared != computed:
        raise SuiteDigestError(
            f"{manifest.path}: declared suite_content_digest {declared} does not "
            f"match the computed digest {computed}. The suite's identity has "
            "drifted; a semantic revision requires an explicit, reviewed re-pin.",
            declared=declared,
            computed=computed,
        )


def _reject_cross_cube_duplicates(cubes: Sequence[SuiteCubeReport]) -> None:
    """Two entries must not ship the same task twice under different names."""
    seen_ids: dict[str, str] = {}
    seen_digests: dict[str, str] = {}
    for cube in cubes:
        for variant in cube.cube.variants:
            owner = seen_ids.get(variant.variant_id)
            if owner is not None:
                raise SuiteFormatError(
                    f"duplicate variant id {variant.variant_id!r} in {cube.card!r} "
                    f"and {owner!r}"
                )
            seen_ids[variant.variant_id] = cube.card
            owner = seen_digests.get(variant.content_digest)
            if owner is not None:
                raise SuiteFormatError(
                    f"duplicate variant content digest {variant.content_digest} "
                    f"shared by {cube.card!r} and {owner!r}; two entries describe "
                    "the same task"
                )
            seen_digests[variant.content_digest] = cube.card


def _enforce_privacy_claim(
    manifest: SuiteManifest, cubes: Sequence[SuiteCubeReport]
) -> None:
    """A suite must not claim a privacy status stronger than its members carry.

    The only admissible suite status in this build is ``SYNTHETIC_ONLY``, and at
    collection level that is a claim about every card in it. ``SYNTHETIC_POLICY``
    is admissible for an individual Cube — it says the *policy* is invented but
    stops short of the whole-artefact claim — so a suite carrying one while
    advertising ``SYNTHETIC_ONLY`` would overstate its own members.
    """
    claimed = manifest.privacy_status
    overstated = [
        f"{cube.card} is {cube.cube.card.privacy_status}"
        for cube in cubes
        if cube.cube.card.privacy_status != claimed
    ]
    if overstated:
        raise SuiteConstraintError(
            f"{manifest.path}: suite claims privacy_status {claimed!r}, but "
            f"{overstated}; a suite may not claim a stronger privacy status than "
            "its member cards carry"
        )


def _check_action_surface(manifest: SuiteManifest, cube: SuiteCubeReport) -> None:
    """Prove every action this Cube offers is one its environment will perform.

    ``allowed_actions`` is what a run puts in front of a model — the tool
    schemas, ``available_actions`` and the names dispatch accepts are all
    projected from it — so an entry the environment cannot execute is an
    invitation to fail for a reason the model was never shown. The shape of that
    defect: the maintenance Cube's fact arrives through ``ask_user``, and a
    ``call_tool`` the card has no use for is offered anyway and then refused from
    hidden metadata.

    Two ways a card can offer one. An elicitation channel the Cube's query
    registry offers *nothing* on can only ever be answered "this case resolves no
    query on that channel"; and a workflow action that is not irreversible is one
    ``Environment.act`` refuses outright. Checked at suite level, where a run's
    action surface is fixed, rather than per card: what this forbids is a
    *suite* that can produce an unanswerable turn.

    The channels are read from the compiled registry rather than from the facts,
    because a query is not the same thing as a fact: a channel that offers three
    questions whose authored answers are ``not_recorded`` is a channel the
    environment executes and the model can use, and refusing it would forbid
    exactly the cases this phase exists to author.
    """
    card = cube.cube.card
    channels = set(cube.cube.query_registry.affordances())
    offered = [
        action for action in card.allowed_actions if action not in PROTOCOL_ACTIONS
    ]
    unusable = sorted(
        set(offered) - set(card.irreversible_actions)
        | (set(card.allowed_actions) & ELICITATION_ACTIONS - channels)
    )
    if unusable:
        raise SuiteConstraintError(
            f"{manifest.path}: {cube.cube_id} offers action(s) {unusable} that its "
            "environment cannot perform: an elicitation channel no fact is "
            "declared to arrive through, or a workflow action that is not "
            "irreversible. Every action a Cube allows is offered to the model as "
            "a tool, so one the case would refuse measures whether the model "
            "guesses rather than whether it follows the case"
        )


def enforce_suite_constraints(manifest: SuiteManifest, report: SuiteReport) -> None:
    """Prove the tallies a suite commits to are the tallies its cubes contain.

    Public for the same reason as :func:`suite_content_digest`: the constraints
    are a declaration, and checking them at load time only binds the objects
    that were loaded. Anything that executes against a suite it did not load
    itself has to re-derive them from the cubes in hand.
    """
    declared = manifest.constraints
    for name, expected, observed in (
        ("cube_types", declared.cube_types, report.cube_types),
        ("fact_locations", declared.fact_locations, report.fact_locations),
        (
            "expected_dispositions",
            declared.expected_dispositions,
            report.expected_dispositions,
        ),
    ):
        # A declared zero and an absent key mean the same thing, so compare only
        # the non-zero tallies rather than the incidental key sets.
        wanted = {key: count for key, count in expected.items() if count}
        actual = {key: count for key, count in observed.items() if count}
        if wanted != actual:
            raise SuiteConstraintError(
                f"{manifest.path}: declared {name} {wanted} but the suite contains "
                f"{actual}"
            )
    for cube_report in report.cubes:
        _check_action_surface(manifest, cube_report)
    if not report.degenerate_strategies_correct:  # pragma: no cover
        # Unreachable while every variant carries exactly one binary disposition;
        # kept so a wider ontology cannot silently break the baseline accounting.
        raise SuiteConstraintError(
            f"{manifest.path}: constant strategy scores "
            f"{report.constant_strategy_scores} do not account for the "
            f"{report.variant_count} variants"
        )


def validate_suite(path: str | Path) -> SuiteReport:
    """Validate a suite end to end: schema, paths, manifests, compile, constraints."""
    manifest = load_suite(path)

    cubes: list[SuiteCubeReport] = []
    for entry in manifest.entries:
        # Every per-artefact failure is re-raised as a SuiteError, so a caller
        # validating a collection has one exception family to handle rather than
        # three leaking out of the card, compiler and manifest layers.
        try:
            cube = compile_cube(load_card(entry.card_path))
        except (SchemaError, CompileError) as exc:
            raise SuiteCubeError(f"{manifest.path}: {entry.card}: {exc}") from exc
        if cube.cube_id != entry.cube_id:
            raise SuiteConstraintError(
                f"{manifest.path}: entry declares cube_id {entry.cube_id!r} but "
                f"{entry.card!r} compiles to {cube.cube_id!r}"
            )
        try:
            semantic = load_manifest(entry.manifest_path)
            fingerprint = _verify_cube(
                cube,
                semantic,
                card_name=entry.card_path.name,
                context=f"{manifest.path}: {entry.semantic_manifest}",
            )
        except ManifestError as exc:
            raise SuiteManifestError(str(exc)) from exc
        cubes.append(
            SuiteCubeReport(
                cube_id=cube.cube_id,
                card=entry.card,
                semantic_manifest=entry.semantic_manifest,
                cube=cube,
                manifest=semantic,
                card_fingerprint_sha256=fingerprint,
            )
        )

    _reject_cross_cube_duplicates(cubes)
    _enforce_privacy_claim(manifest, cubes)
    computed = suite_content_digest(manifest, cubes)
    _verify_declared_digest(manifest, computed)
    report = SuiteReport(manifest=manifest, cubes=tuple(cubes), content_digest=computed)
    enforce_suite_constraints(manifest, report)
    return report


# -- causal CI over the collection -------------------------------------------


@dataclass(frozen=True)
class SuiteCausalResult:
    cube_id: str
    card: str
    report: SolverReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "cube_id": self.cube_id,
            "card": self.card,
            "ok": self.report.ok,
            "outcomes": [
                {
                    "name": outcome.name,
                    "kind": outcome.kind,
                    "passed_variants": outcome.passed_variants,
                    "total_variants": outcome.total_variants,
                    "satisfied": outcome.satisfied,
                    "failures": list(outcome.failures),
                }
                for outcome in self.report.outcomes
            ],
        }


@dataclass(frozen=True)
class SuiteCheckReport:
    suite: SuiteReport
    results: tuple[SuiteCausalResult, ...]

    @property
    def ok(self) -> bool:
        return self.suite.ok and all(result.report.ok for result in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "suite": self.suite.as_dict(),
            "causal_ci": [result.as_dict() for result in self.results],
        }


def check_suite(report: SuiteReport) -> SuiteCheckReport:
    """Run every Cube's causal CI gate and collect the verdicts."""
    return SuiteCheckReport(
        suite=report,
        results=tuple(
            SuiteCausalResult(
                cube_id=cube.cube_id, card=cube.card, report=check_solvers(cube.cube)
            )
            for cube in report.cubes
        ),
    )


__all__ = [
    "BENCHMARK_VERSION",
    "CONSTANT_STRATEGY_NOTE",
    "SUITE_SCHEMA_VERSION",
    "SUITE_SCOPE",
    "SuiteCausalResult",
    "SuiteCheckReport",
    "SuiteConstraintError",
    "SuiteConstraints",
    "SuiteCubeError",
    "SuiteCubeReport",
    "SuiteDigestError",
    "SuiteEntry",
    "SuiteError",
    "SuiteFormatError",
    "SuiteManifest",
    "SuiteManifestError",
    "SuitePathError",
    "SuiteReport",
    "card_fingerprint",
    "check_suite",
    "enforce_suite_constraints",
    "load_suite",
    "suite_content_digest",
    "validate_suite",
]
