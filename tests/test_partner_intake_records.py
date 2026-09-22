"""Contract tests for maintainer-owned managed-intake partner records."""

from __future__ import annotations

import json
import keyword
import re
import subprocess
import sys
import unicodedata
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tests" / "fixtures" / "partner-intake"
COMPANIES = REGISTRY / "companies"
SCHEMA_PATH = REGISTRY / "record-v1.schema.json"

HIGHER_PERMISSIONS = (
    "private_repository_read",
    "executable_repository_pr",
    "provider_model_execution",
    "development_registration",
    "evidence_admission",
    "publication_publicity",
)

OWNER_COMPONENT = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")
GITHUB_ACTOR = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def _load(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    loaded = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate
    )
    assert isinstance(loaded, dict)
    return loaded


def _records() -> list[tuple[Path, dict[str, Any]]]:
    return [(path, _load(path)) for path in sorted(COMPANIES.glob("*.json"))]


def _policy_violations(record: dict[str, Any]) -> list[str]:
    """Evaluate access invariants without relying on the JSON Schema."""
    provenance = record["provenance"]
    access = record["access"]
    violations: list[str] = []

    if type(provenance["partner_confirmed"]) is not bool:
        violations.append("partner confirmation provenance is not a JSON boolean")

    if provenance["partner_confirmed"] and access["partner_confirmation"] != "APPROVED":
        violations.append("confirmed partner lacks approved confirmation")

    if not provenance["partner_confirmed"]:
        if access["partner_confirmation"] == "APPROVED":
            violations.append("unconfirmed partner has approved confirmation")
        for permission in HIGHER_PERMISSIONS:
            if access[permission] != "HOLD":
                violations.append(f"unconfirmed partner has elevated {permission}")

    if (
        access["partner_confirmation"] == "APPROVED"
        and not provenance["partner_confirmed"]
    ):
        violations.append("approved confirmation lacks provenance")

    partner_access_approved = (
        provenance["partner_confirmed"]
        and access["partner_confirmation"] == "APPROVED"
        and access["repository_access_approval"] == "APPROVED"
    )
    for permission in ("private_repository_read", "executable_repository_pr"):
        if access[permission] == "GO":
            actors = access["github"]["actors"]
            if access["github"]["status"] != "APPROVED":
                violations.append(f"{permission} lacks GitHub approval")
            if not actors:
                violations.append(f"{permission} lacks a GitHub actor")
            if len(actors) != len(set(actors)) or any(
                GITHUB_ACTOR.fullmatch(actor) is None for actor in actors
            ):
                violations.append(f"{permission} has invalid GitHub actors")
            if not partner_access_approved:
                violations.append(f"{permission} lacks partner/access approval")

    if (
        access["executable_repository_pr"] == "GO"
        and access["private_repository_read"] != "GO"
    ):
        violations.append("executable PR lacks repository read permission")

    if access["publication_publicity"] == "GO":
        if provenance["publication_consent"] != "GRANTED":
            violations.append("publication lacks consent")
        if not partner_access_approved:
            violations.append("publication lacks partner/access approval")

    return violations


def _provenance_date_violations(record: dict[str, Any]) -> list[str]:
    """Require provenance dates to denote canonical Gregorian calendar dates."""
    value = record["identity"]["identity_checked_on"]
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        return ["identity_checked_on is not a valid calendar date"]
    if parsed.isoformat() != value:
        return ["identity_checked_on is not a canonical ISO date"]
    return []


def _fictional_example_record() -> dict[str, Any]:
    return _load(COMPANIES / "fictional_example.json")


def _confirmed_record() -> dict[str, Any]:
    record = deepcopy(_fictional_example_record())
    record["provenance"].update(
        partner_confirmed=True,
        publication_consent="GRANTED",
    )
    record["access"].update(
        private_repository_read="GO",
        executable_repository_pr="GO",
        publication_publicity="GO",
        repository_access_approval="APPROVED",
        partner_confirmation="APPROVED",
    )
    record["access"]["github"] = {
        "actors": ["fictional-example-partner"],
        "status": "APPROVED",
    }
    return record


def _access_approved_record() -> dict[str, Any]:
    record = deepcopy(_fictional_example_record())
    record["provenance"]["partner_confirmed"] = True
    record["access"].update(
        repository_access_approval="APPROVED",
        partner_confirmation="APPROVED",
    )
    record["access"]["github"] = {
        "actors": ["fictional-example-partner"],
        "status": "APPROVED",
    }
    return record


def test_schema_meta_validates_and_every_company_record_conforms() -> None:
    schema = _load(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    records = _records()
    assert records, "managed-intake registry must contain at least one company"
    for path, record in records:
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        assert errors == [], f"{path}: " + "; ".join(error.message for error in errors)
        assert _provenance_date_violations(record) == [], path


def test_provenance_dates_are_semantically_valid_and_canonical() -> None:
    for invalid_date in ("2026-02-29", "2026-02-31", "2026-13-01", "20260903"):
        record = _fictional_example_record()
        record["identity"]["identity_checked_on"] = invalid_date
        assert _provenance_date_violations(record), invalid_date

    leap_day = _fictional_example_record()
    leap_day["identity"]["identity_checked_on"] = "2028-02-29"
    assert _provenance_date_violations(leap_day) == []


def test_schema_rejects_non_calendar_and_noncanonical_dates() -> None:
    validator = Draft202012Validator(_load(SCHEMA_PATH))
    for invalid_date in (
        "0000-01-01",
        "2026-02-29",
        "2026-02-30",
        "2026-02-31",
        "2026-04-31",
        "2026-13-01",
        "1900-02-29",
        "20260903",
        "2026-9-03",
        "2026-09-3",
        " 2026-09-03",
        "2026-09-03\n",
    ):
        record = _fictional_example_record()
        record["identity"]["identity_checked_on"] = invalid_date
        assert not validator.is_valid(record), invalid_date


def test_schema_accepts_calendar_date_boundaries() -> None:
    validator = Draft202012Validator(_load(SCHEMA_PATH))
    for valid_date in (
        "2026-09-03",
        "2028-02-29",
        "2000-02-29",
        "2026-04-30",
        "2026-01-31",
        "9999-12-31",
    ):
        record = _fictional_example_record()
        record["identity"]["identity_checked_on"] = valid_date
        assert validator.is_valid(record), valid_date


def test_registry_identifiers_names_and_contacts_are_unique_and_canonical() -> None:
    records = _records()
    company_ids = [record["company_id"] for _, record in records]
    owner_ids = [
        unicodedata.normalize("NFC", record["owner"]["owner_id"]) for _, record in records
    ]
    display_names = [
        unicodedata.normalize("NFC", record["owner"]["display_name"]).casefold()
        for _, record in records
    ]
    contacts = [
        unicodedata.normalize("NFC", record["contact"]["email"]).casefold()
        for _, record in records
    ]

    assert len(company_ids) == len(set(company_ids))
    assert len(owner_ids) == len(set(owner_ids))
    assert len(display_names) == len(set(display_names))
    assert len(contacts) == len(set(contacts))
    for (_, record), owner_id in zip(records, owner_ids, strict=True):
        assert record["company_id"] == owner_id
        assert OWNER_COMPONENT.fullmatch(owner_id)
        assert not keyword.iskeyword(owner_id)
        assert owner_id not in WINDOWS_RESERVED


def test_digests_are_canonical_lowercase_hex() -> None:
    for path, record in _records():
        kit = record["starter_kit"]
        assert SHA256.fullmatch(kit["sha256"]), path


def test_permissions_fail_closed_while_access_prerequisites_are_pending() -> None:
    for path, record in _records():
        assert _policy_violations(record) == [], path


def test_schema_rejects_inconsistent_permission_elevations() -> None:
    validator = Draft202012Validator(_load(SCHEMA_PATH))
    mutations: list[tuple[str, dict[str, Any]]] = []

    false_with_approval = _fictional_example_record()
    false_with_approval["access"]["partner_confirmation"] = "APPROVED"
    mutations.append(("false provenance with approved confirmation", false_with_approval))

    for permission in HIGHER_PERMISSIONS:
        unconfirmed_elevation = _fictional_example_record()
        unconfirmed_elevation["access"][permission] = "GO"
        mutations.append((f"unconfirmed {permission} elevation", unconfirmed_elevation))

    def elevated(permission: str) -> dict[str, Any]:
        record = _access_approved_record()
        record["access"][permission] = "GO"
        return record

    for label, change in (
        ("without an actor", ("actors", [])),
        ("with duplicate actors", ("actors", ["partner", "partner"])),
        ("with an invalid actor", ("actors", ["-partner"])),
        ("without GitHub approval", ("status", "PENDING")),
    ):
        read = elevated("private_repository_read")
        read["access"]["github"][change[0]] = change[1]
        mutations.append((f"repository read {label}", read))

    read_without_access_approval = elevated("private_repository_read")
    read_without_access_approval["access"]["repository_access_approval"] = "PENDING"
    mutations.append(
        ("repository read without access approval", read_without_access_approval)
    )

    read_without_confirmation = elevated("private_repository_read")
    read_without_confirmation["access"]["partner_confirmation"] = "PENDING"
    mutations.append(("repository read without confirmation", read_without_confirmation))

    pr_without_read = elevated("executable_repository_pr")
    mutations.append(("executable PR without repository read", pr_without_read))

    publication_without_consent = _confirmed_record()
    publication_without_consent["provenance"]["publication_consent"] = "PENDING"
    mutations.append(("publication without consent", publication_without_consent))

    publication_without_access_approval = _confirmed_record()
    publication_without_access_approval["access"]["repository_access_approval"] = (
        "PENDING"
    )
    mutations.append(
        ("publication without access approval", publication_without_access_approval)
    )

    publication_without_confirmation = _confirmed_record()
    publication_without_confirmation["access"]["partner_confirmation"] = "PENDING"
    mutations.append(
        ("publication without confirmation", publication_without_confirmation)
    )

    for label, record in mutations:
        assert not validator.is_valid(record), label
        assert _policy_violations(record), label


def test_true_confirmation_provenance_requires_approved_confirmation() -> None:
    validator = Draft202012Validator(_load(SCHEMA_PATH))
    for confirmation in ("PENDING", "REJECTED"):
        for permission in (None, *HIGHER_PERMISSIONS):
            record = _access_approved_record()
            record["access"]["partner_confirmation"] = confirmation
            if permission is not None:
                record["access"][permission] = "GO"
            label = f"true provenance with {confirmation} and {permission or 'no'} GO"
            assert not validator.is_valid(record), label
            assert _policy_violations(record), label


def test_partner_confirmed_requires_an_exact_json_boolean() -> None:
    record = _fictional_example_record()
    record["provenance"]["partner_confirmed"] = 0
    assert not Draft202012Validator(_load(SCHEMA_PATH)).is_valid(record)
    assert _policy_violations(record)


def test_confirmed_partner_lifecycle_can_validate() -> None:
    record = _confirmed_record()
    assert Draft202012Validator(_load(SCHEMA_PATH)).is_valid(record)
    assert _policy_violations(record) == []


def test_fictional_example_record_pins_the_fictional_example_only() -> None:
    expected = {
        "schema_version": 1,
        "company_id": "fictional_example",
        "owner": {
            "owner_id": "fictional_example",
            "display_name": "Fictional Example Research Ltd",
        },
        "identity": {
            "brand": "Fictional Example",
            "website_url": "https://fictional.example/",
            "legal_entity": "Fictional Example Research Ltd",
            "identity_source_url": "https://fictional.example/de/imprint",
            "identity_checked_on": "2026-09-03",
            "registered_office": "Example City, Fictional Jurisdiction",
            "commercial_register": "FICTIONAL-REGISTER-0001",
        },
        "contact": {
            "email": "contact@fictional.invalid",
            "contact_source": "owner_supplied",
        },
        "provenance": {
            "partner_confirmed": False,
            "publication_consent": "PENDING",
        },
        "starter_kit": {
            "filename": "fictional-example-kit.zip",
            "sha256": "beeb4bb8795fe955c36470dab1b2876bc1ed685ce8dbd94f3fb855936207f62b",
        },
        "access": {
            "managed_intake": "GO",
            "proposal_return_bundle": "GO",
            "private_repository_read": "HOLD",
            "executable_repository_pr": "HOLD",
            "provider_model_execution": "HOLD",
            "development_registration": "HOLD",
            "evidence_admission": "HOLD",
            "publication_publicity": "HOLD",
            "github": {"actors": [], "status": "PENDING"},
            "repository_access_approval": "PENDING",
            "partner_confirmation": "PENDING",
        },
    }
    assert _load(COMPANIES / "fictional_example.json") == expected


def test_private_registry_is_scanned_by_the_public_candidate_gate() -> None:
    from tools import check_public_release

    relative_paths = {
        path.relative_to(ROOT).as_posix()
        for path in check_public_release.candidate_files()
    }
    expected_paths = {
        path.relative_to(ROOT).as_posix()
        for path in REGISTRY.rglob("*")
        if path.is_file()
    }
    assert expected_paths <= relative_paths

    subprocess.run(
        [sys.executable, "-m", "tools.check_public_release"],
        cwd=ROOT,
        check=True,
    )
