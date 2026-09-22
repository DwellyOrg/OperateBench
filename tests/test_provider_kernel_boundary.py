"""The provider kernel is track-neutral, and the tracks re-export it.

Two properties, and neither of them is visible from behaviour alone.

**The kernel imports no track.** ``operatebench.providers`` is the shared
provider surface, written so that more than one track can compose it. A single
import of ``boundarybench`` — or of a Lifecycle module — from inside it would
make the second composition impossible, and nothing would fail until somebody
tried it. The check is a source scan rather
than a convention, for the same reason
``tests/test_operatebench_packaging.py`` scans Core: an import direction is not
something a runtime test can observe once the import has already happened.

The one sideways reach the kernel is allowed is the shared canonical-JSON
primitive, which is a serialisation utility rather than a track construct and is
named here explicitly rather than left to a prefix rule.

**The historical names still resolve, to the same objects.** Nothing about this
extraction may move a symbol out from under a caller. ``is`` rather than ``==``
throughout: a re-export that produced an equal-but-distinct class would break
``isinstance`` and the exception taxonomy silently, which is exactly the failure
a compatibility shim exists to prevent.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL = REPO_ROOT / "src" / "operatebench" / "providers"

#: Track packages the kernel may not reach into. ``boundarybench`` is the whole
#: Boundary Track; the rest are the Lifecycle modules that own an operation, an
#: agent, a run or an artefact.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "boundarybench",
    "operatebench.agents",
    "operatebench.artifact",
    "operatebench.cli",
    "operatebench.controls",
    "operatebench.core",
    "operatebench.decision_points",
    "operatebench.domains",
    "operatebench.runner",
)

#: The only modules outside ``operatebench.providers`` the kernel may import
#: from its own distribution. One entry, and it is the canonical UTF-8 JSON
#: primitive both tracks already hash their artefacts with.
ALLOWED_SIBLINGS: frozenset[str] = frozenset({"operatebench.jsonsafe"})


def _kernel_modules() -> list[Path]:
    return sorted(KERNEL.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_the_kernel_package_exists_and_carries_modules() -> None:
    assert KERNEL.is_dir()
    assert (KERNEL / "__init__.py").is_file()
    # More than the package marker: a boundary test that passed over an empty
    # directory would prove nothing at all.
    assert len(_kernel_modules()) > 1


@pytest.mark.parametrize("path", _kernel_modules(), ids=lambda path: path.name)
def test_no_kernel_module_imports_a_track(path: Path) -> None:
    offenders = {
        module
        for module in _imported_modules(path)
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_PREFIXES
        )
    }
    assert offenders == set(), f"{path.name} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", _kernel_modules(), ids=lambda path: path.name)
def test_every_first_party_kernel_import_is_the_kernel_or_the_json_primitive(
    path: Path,
) -> None:
    for module in _imported_modules(path):
        if not module.startswith(("operatebench", "boundarybench")):
            # Standard library and the pinned provider SDKs are exactly what a
            # kernel module is expected to reach for.
            continue
        assert (
            module == "operatebench.providers"
            or module.startswith("operatebench.providers.")
            or module in ALLOWED_SIBLINGS
        ), f"{path.name} imports {module}"


# -- the compatibility re-exports --------------------------------------------


def test_the_adapter_contract_re_exports_the_kernel_taxonomy() -> None:
    import boundarybench.adapter as adapter
    from operatebench.providers import faults as kernel

    for name in (
        "AdapterError",
        "AdapterProviderError",
        "AdapterBusyError",
        "SingleFlight",
        "PROVIDER_FAULTS",
        "RETRYABLE_PROVIDER_FAULTS",
        "ATTEMPT_OUTCOMES",
        "ATTEMPT_SETTLEMENTS",
        "TURN_TERMINAL_REASONS",
    ):
        assert getattr(adapter, name) is getattr(kernel, name), name


def test_the_adapter_contract_re_exports_the_kernel_measurements() -> None:
    import boundarybench.adapter as adapter
    from operatebench.providers import telemetry as kernel

    for name in ("TurnDeadline", "AdapterUsage", "ProviderAttempt", "ProviderTelemetry"):
        assert getattr(adapter, name) is getattr(kernel, name), name


def test_the_boundary_exception_taxonomy_is_unchanged() -> None:
    from boundarybench.adapter import (
        AdapterBusyError,
        AdapterError,
        AdapterOutputLimitError,
        AdapterProtocolError,
        AdapterProviderError,
    )
    from boundarybench.budget import CostCapExceededError, CostReservationBreachedError
    from boundarybench.providers.common import ProviderConfigurationError
    from boundarybench.providers.openai_responses import OpenAIConfigurationError

    for subclass in (
        AdapterProtocolError,
        AdapterProviderError,
        AdapterOutputLimitError,
        AdapterBusyError,
        CostCapExceededError,
        CostReservationBreachedError,
        ProviderConfigurationError,
    ):
        assert issubclass(subclass, AdapterError), subclass
    assert issubclass(OpenAIConfigurationError, ProviderConfigurationError)
    assert not issubclass(AdapterOutputLimitError, AdapterProtocolError)


def test_providers_common_re_exports_the_moved_kernel_symbols() -> None:
    import boundarybench.providers.common as common
    from operatebench.providers import config, executor, faults, response, usage

    expected = {
        "Fault": faults,
        "fault_detail": faults,
        "classify_http_status": faults,
        "ProviderRetryPolicy": executor,
        "check_retry_policy": executor,
        "TurnExecutor": executor,
        "SDK_MAX_RETRIES": executor,
        "MINIMUM_BACKOFF_MULTIPLIER": executor,
        "check_model": config,
        "check_endpoint": config,
        "check_environment": config,
        "check_payload_fields": config,
        "normalized_endpoint": config,
        "resolve_api_key": config,
        "MODEL_PATTERN": config,
        "ProviderConfigurationError": config,
        "TokenUsage": usage,
        "is_token_count": usage,
        "checked_token_usage": usage,
        "MAX_RESPONSE_DEPTH": response,
        "check_response_contract": response,
        "check_response_collection": response,
        "check_response_object": response,
    }
    for name, module in expected.items():
        assert getattr(common, name) is getattr(module, name), name


def test_providers_wire_re_exports_the_moved_wire_module() -> None:
    import boundarybench.providers.wire as shim
    from operatebench.providers import wire as kernel

    for name in kernel.__all__:
        assert getattr(shim, name) is getattr(kernel, name), name


def test_the_budget_and_pricing_primitives_the_kernel_owns_are_the_same_objects() -> None:
    import boundarybench.budget as budget
    import boundarybench.pricing as pricing
    from operatebench.providers import cost, usage

    for name in (
        "BudgetError",
        "CostCapExceededError",
        "CostReservationBreachedError",
        "REQUEST_OVERHEAD_TOKEN_ALLOWANCE",
        "conservative_input_token_bound",
        "usd_text",
    ):
        assert getattr(budget, name) is getattr(cost, name), name
    assert pricing.MAX_EXACT_TOKEN_COUNT is usage.MAX_EXACT_TOKEN_COUNT


def test_the_shared_json_primitive_is_one_object_under_both_names() -> None:
    import boundarybench.jsonsafe as shim
    import operatebench.jsonsafe as primitive

    for name in primitive.__all__:
        assert getattr(shim, name) is getattr(primitive, name), name
