"""Finite imports needed by genuine Core-based contribution domains.

Census: property_compliance_profiles source/tests, integration contract v1.
No provider, playback implementation, dynamic module, registry or private symbol
access is granted by these additions.
"""

import ast
import shutil
from pathlib import Path

import pytest

from tests.test_contribution_isolation import _owner
from tools.check_contribution_isolation import (
    ContributionError,
    _check_ast,
    check_contributions,
)

PROFILE_MODULE = "operatebench.contributions.prospire.property_compliance_profiles"
COMMAND_IMPORT = "from operatebench.sdk.profile_packs.pack import COMPLIANCE_COMMANDS"
SHIM_PATH = "src/operatebench/contributions/prospire/property_compliance_profiles/pack.py"


@pytest.fixture
def profile_tree(tmp_path):
    root = Path(__file__).parents[1]
    for relative in (
        "src/operatebench/contributions",
        "src/operatebench/resources",
        "tests/contributions",
        "examples/contributions",
        "docs/contributions",
    ):
        shutil.copytree(root / relative, tmp_path / relative)
    assert len(check_contributions(tmp_path)) == 2
    return tmp_path


@pytest.mark.parametrize(
    "expression",
    [
        'COMPLIANCE_COMMANDS.factories.load_spec.__globals__["COMMERCE_COMMANDS"]',
        'COMPLIANCE_COMMANDS.run.__func__.__globals__["run_mock_model"]',
        "COMPLIANCE_COMMANDS.factories",
        "COMPLIANCE_COMMANDS",
        'getattr(COMPLIANCE_COMMANDS, "factories")',
        "vars(COMPLIANCE_COMMANDS)",
    ],
)
def test_profile_shim_full_checker_refuses_traversal(profile_tree, expression):
    path = profile_tree / SHIM_PATH
    original = path.read_text()
    target = "        return COMPLIANCE_COMMANDS.run(request)"
    assert original.count(target) == 1
    path.write_text(original.replace(target, f"        exposed = {expression}\n{target}"))
    with pytest.raises(ContributionError, match="profile command shim"):
        check_contributions(profile_tree)


def test_profile_shim_full_checker_refuses_import_alias(profile_tree):
    path = profile_tree / SHIM_PATH
    path.write_text(
        path.read_text().replace(COMMAND_IMPORT, COMMAND_IMPORT + " as commands")
    )
    with pytest.raises(ContributionError, match="profile command shim"):
        check_contributions(profile_tree)


@pytest.mark.parametrize(
    "old,new",
    [
        ("COMPLIANCE_COMMANDS.run(request)", "COMPLIANCE_COMMANDS.check(request)"),
        ("request: RunRequest", "request: RunRequest = COMPLIANCE_COMMANDS.factories"),
        ("    def run(", "    @COMPLIANCE_COMMANDS.run\n    def run("),
        (
            'display_name="Property compliance jurisdiction profiles"',
            "display_name=COMPLIANCE_COMMANDS.factories.pack_id",
        ),
        (
            "PACK = ComplianceProfilesPack()",
            "exposed = COMPLIANCE_COMMANDS.factories\nPACK = ComplianceProfilesPack()",
        ),
    ],
)
def test_profile_shim_full_checker_requires_exact_structure(profile_tree, old, new):
    path = profile_tree / SHIM_PATH
    original = path.read_text()
    assert original.count(old) == 1
    path.write_text(original.replace(old, new))
    with pytest.raises(ContributionError, match="profile command shim"):
        check_contributions(profile_tree)


@pytest.mark.parametrize("owner,path", [("other", "pack.py"), ("prospire", "extra.py")])
def test_profile_shim_requires_exact_owner_and_filename(owner, path):
    source = (Path(__file__).parents[1] / SHIM_PATH).read_text()
    with pytest.raises(ContributionError, match="unauthorized location"):
        _check_ast(
            Path(path),
            ast.parse(source),
            owner,
            PROFILE_MODULE,
            PROFILE_MODULE + ".pack",
            is_test=False,
        )


def test_profile_command_shim_exact_import_allowed():
    _check_ast(
        Path("pack.py"),
        ast.parse((Path(__file__).parents[1] / SHIM_PATH).read_text()),
        "prospire",
        PROFILE_MODULE,
        PROFILE_MODULE + ".pack",
        is_test=False,
    )


@pytest.mark.parametrize(
    "module,suffix,is_test,snippet",
    [
        (PROFILE_MODULE, ".nested.pack", False, COMMAND_IMPORT),
        (PROFILE_MODULE, ".agents", False, COMMAND_IMPORT),
        (PROFILE_MODULE, ".pack", True, COMMAND_IMPORT),
        ("operatebench.contributions.northstar.sample", ".pack", False, COMMAND_IMPORT),
        (PROFILE_MODULE, ".pack", False, COMMAND_IMPORT + ", COMMERCE_COMMANDS"),
        (
            PROFILE_MODULE,
            ".pack",
            False,
            COMMAND_IMPORT.replace("COMPLIANCE_COMMANDS", "*"),
        ),
        (PROFILE_MODULE, ".pack", False, "import operatebench.sdk.profile_packs.pack"),
        (
            PROFILE_MODULE,
            ".pack",
            False,
            "from operatebench.sdk.profile_packs import pack",
        ),
        (
            PROFILE_MODULE,
            ".pack",
            False,
            COMMAND_IMPORT + "\ngetattr(COMPLIANCE_COMMANDS, 'factories')",
        ),
    ],
)
def test_profile_command_shim_does_not_open_namespace(module, suffix, is_test, snippet):
    with pytest.raises(ContributionError):
        _check_ast(
            Path("pack.py"),
            ast.parse(snippet),
            module.split(".")[2],
            module,
            module + suffix,
            is_test=is_test,
        )


SOURCE_IMPORTS = {
    "operatebench.core.engine": (
        "EpisodeOutcome",
        "has_canonical_episode_outcome_provenance",
    ),
    "operatebench.core.outcomes": ("Act", "Complete", "Wait"),
    "operatebench.core.protocol": (
        "EpisodePlan",
        "PlannedEvent",
        "PlannedTrigger",
        "Verdict",
        "model_projection",
    ),
    "operatebench.core.read_contract": ("ActionEvidenceContract",),
    "operatebench.core.retrieval": ("ToolResult", "RetrieveBatch", "RetrievalRequest"),
    "operatebench.core.retrieval_evidence": (
        "PUBLIC_RECORD_VERSION_ALGORITHM",
        "public_record_version",
    ),
}
TEST_IMPORTS = {
    "operatebench.agents.playback": (
        "RecordedOutcomeAgent",
        "RecordingAgent",
        "tape_from_records",
    ),
    "operatebench.core.engine": ("Engine",),
    "operatebench.core.protocol": ("Verdict", "model_projection"),
    "operatebench.core.outcomes": ("Act",),
}


@pytest.mark.parametrize(
    "location,imports", [("source", SOURCE_IMPORTS), ("test", TEST_IMPORTS)]
)
def test_exact_required_imports_are_allowed(tmp_path, location, imports):
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(
        "".join(
            f"from {module} import {', '.join(names)}\n"
            for module, names in imports.items()
        )
    )
    assert check_contributions(tmp_path) == ("northstar:northstar_sample",)


@pytest.mark.parametrize("module", tuple(SOURCE_IMPORTS))
@pytest.mark.parametrize("location", ["source", "test"])
def test_one_extra_symbol_refuses_whole_import(tmp_path, module, location):
    source = _owner(tmp_path, "northstar")
    path = (
        source / "extra.py"
        if location == "source"
        else tmp_path / "tests/contributions/northstar/sample/test_pack.py"
    )
    path.write_text(f"from {module} import {SOURCE_IMPORTS[module][0]}, __dict__\n")
    with pytest.raises(ContributionError):
        check_contributions(tmp_path)


@pytest.mark.parametrize(
    "snippet",
    [
        "from operatebench.core.engine import _EPISODE_OUTCOME_PROVENANCE_KEY",
        "from operatebench.agents.model import ModelAgent",
        "from operatebench.agents.playback import RecordingAgent",
        "from operatebench.sdk.registry import register",
        "from operatebench.core import engine",
        "import operatebench.core.engine",
        "from operatebench.core.outcomes import Escalate",
    ],
)
def test_development_additions_do_not_grant_escape_hatches(tmp_path, snippet):
    source = _owner(tmp_path, "northstar")
    (source / "extra.py").write_text(snippet + "\n")
    with pytest.raises(ContributionError):
        check_contributions(tmp_path)
