"""The provider registry, and the CLI once it can name more than one provider.

Two things are under test and they are the same thing from two sides. The
registry is the single place that says what a provider *is* — its credential
variable, its endpoint, its pricing key, how to build its adapter and what
settings a run of it records. The CLI is the caller that must not hold a second
opinion about any of that.

No test here contacts a provider. The ones that check refusal paths prove the
stronger property: that a refusal happens before a run directory exists, so an
operator who mistyped something is left with an error message rather than a
directory that looks like evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boundarybench.cli import ADAPTERS, main
from boundarybench.providers.registry import (
    PROVIDER_ADAPTERS,
    ProviderChoice,
    provider_choice,
)
from tests.conftest import SUITE_MANIFEST

#: Clearly synthetic, purely local configuration identities.
SYNTHETIC_CONFIGURATION = "0" * 64

#: The four credentialed providers this build ships, and the exact model each
#: one is pinned to for the methodology spike.
EXPECTED = (
    ("anthropic", "_".join(("ANTHROPIC", "API", "KEY")), "claude-haiku-4-5-20251001"),
    ("openai", "OPENAI_API_KEY", "gpt-5.6-luna"),
    ("xai", "XAI_API_KEY", "grok-4.5"),
    ("mistral", "MISTRAL_API_KEY", "mistral-small-2603"),
)

NEW_PROVIDERS = ("openai", "xai", "mistral")


# -- the registry -------------------------------------------------------------


def test_every_credentialed_provider_is_registered_once() -> None:
    assert set(PROVIDER_ADAPTERS) == {name for name, _var, _model in EXPECTED}
    for name in PROVIDER_ADAPTERS:
        assert provider_choice(name).name == name


@pytest.mark.parametrize(("name", "variable", "model"), EXPECTED)
def test_each_choice_states_its_credential_variable_and_prices_its_model(
    name: str, variable: str, model: str
) -> None:
    """A provider's credential variable and its price table are the same fact.

    The registry is where a run learns both, so a provider whose adapter reads
    one variable while the CLI checks another, or whose identity says one thing
    while its price is looked up under something else, is impossible to build.
    """
    from boundarybench.pricing import price_for

    choice = provider_choice(name)
    assert isinstance(choice, ProviderChoice)
    assert choice.api_key_variable == variable
    # The provider key the registry uses to price is the same string the adapter
    # identity records, so a run cannot be priced under a provider it did not
    # name.
    assert choice.identity(model).provider == choice.provider
    price = price_for(provider=choice.provider, model=model)
    assert price.provider == choice.provider


@pytest.mark.parametrize(("name", "_variable", "model"), EXPECTED)
def test_settings_are_buildable_offline_and_carry_no_credential(
    name: str, _variable: str, model: str
) -> None:
    """A preflight computes identity and settings without a credential.

    Which is what makes a configuration identity discoverable offline: an
    operator has to be able to compute the value they are being asked to
    authorise before they authorise it.
    """
    import json

    choice = provider_choice(name)
    settings = choice.settings(model)
    rendered = json.dumps(settings, sort_keys=True)
    assert settings["sdk_max_retries"] == 0
    assert settings["max_output_tokens"] == 1024
    assert "api_key" not in rendered.lower()
    assert choice.api_key_variable not in rendered


def test_every_provider_pins_the_same_output_ceiling() -> None:
    """Two models cut off at different lengths are not answering one question.

    The ceiling is a comparability property of the matrix rather than a
    per-provider preference, so it is asserted across the registry rather than
    left to four constants that happen to agree today.
    """
    ceilings = {
        name: provider_choice(name).max_output_tokens for name in PROVIDER_ADAPTERS
    }
    assert len(set(ceilings.values())) == 1, ceilings


def test_an_unknown_provider_is_refused() -> None:
    with pytest.raises(KeyError):
        provider_choice("cohere")


# -- the CLI ------------------------------------------------------------------


def test_the_cli_offers_every_registered_adapter_and_the_fake() -> None:
    assert set(ADAPTERS) == {"fake-scripted", *PROVIDER_ADAPTERS}


def _argv(output: Path, adapter: str, model: str, *extra: str) -> list[str]:
    return [
        "run-suite",
        str(SUITE_MANIFEST),
        "--adapter",
        adapter,
        "--model",
        model,
        "--output-dir",
        str(output),
        *extra,
    ]


@pytest.mark.parametrize("name", NEW_PROVIDERS)
def test_a_missing_credential_costs_an_error_and_not_a_half_written_run(
    name: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal happens before ``open_run_session`` creates anything.

    This is the property that matters most about the ordering: a directory left
    behind by a refused invocation is a thing an audit can mistake for a run.
    """
    from boundarybench.providers.common import (
        AUDITED_CONFIGURATIONS_VARIABLE,
        LIVE_CONFIGURATION_VARIABLE,
    )

    choice = provider_choice(name)
    model = {n: m for n, _v, m in EXPECTED}[name]
    monkeypatch.delenv(choice.api_key_variable, raising=False)
    monkeypatch.setenv(AUDITED_CONFIGURATIONS_VARIABLE, SYNTHETIC_CONFIGURATION)
    monkeypatch.setenv(LIVE_CONFIGURATION_VARIABLE, SYNTHETIC_CONFIGURATION)
    root = tmp_path / "run"

    code = main(_argv(root, name, model))

    assert code != 0
    assert choice.api_key_variable in capsys.readouterr().err
    assert not root.exists(), "a refused invocation must leave no run directory"


@pytest.mark.parametrize("name", NEW_PROVIDERS)
def test_a_provider_run_requires_an_explicit_model(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing here picks a default: a default would pick a moving alias."""
    root = tmp_path / "run"
    code = main(
        [
            "run-suite",
            str(SUITE_MANIFEST),
            "--adapter",
            name,
            "--output-dir",
            str(root),
        ]
    )
    assert code != 0
    assert "--model" in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize("name", NEW_PROVIDERS)
def test_arbitrary_settings_are_refused_for_a_real_provider(
    name: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator-supplied setting is free text that lands in a run manifest."""
    choice = provider_choice(name)
    model = {n: m for n, _v, m in EXPECTED}[name]
    monkeypatch.setenv(choice.api_key_variable, "not-a-real-key")
    root = tmp_path / "run"
    code = main(_argv(root, name, model, "--setting", "note=hello"))
    assert code != 0
    assert "--setting" in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize("name", NEW_PROVIDERS)
def test_an_unauthorised_configuration_is_refused_before_the_credential(
    name: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from boundarybench.providers.common import (
        AUDITED_CONFIGURATIONS_VARIABLE,
        LIVE_CONFIGURATION_VARIABLE,
    )

    choice = provider_choice(name)
    model = {n: m for n, _v, m in EXPECTED}[name]
    monkeypatch.setenv(choice.api_key_variable, "not-a-real-key")
    monkeypatch.delenv(AUDITED_CONFIGURATIONS_VARIABLE, raising=False)
    monkeypatch.delenv(LIVE_CONFIGURATION_VARIABLE, raising=False)
    root = tmp_path / "run"

    code = main(_argv(root, name, model))

    assert code != 0
    assert "authorise" in capsys.readouterr().err.lower()
    assert not root.exists()


@pytest.mark.parametrize("name", NEW_PROVIDERS)
def test_a_redirected_endpoint_is_refused_before_a_run_directory(
    name: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A base URL is where the credential is sent, so a redirect is refused."""
    from boundarybench.providers.common import (
        AUDITED_CONFIGURATIONS_VARIABLE,
        LIVE_CONFIGURATION_VARIABLE,
    )

    choice = provider_choice(name)
    model = {n: m for n, _v, m in EXPECTED}[name]
    monkeypatch.setenv(choice.api_key_variable, "not-a-real-key")
    monkeypatch.setenv(AUDITED_CONFIGURATIONS_VARIABLE, SYNTHETIC_CONFIGURATION)
    monkeypatch.setenv(LIVE_CONFIGURATION_VARIABLE, SYNTHETIC_CONFIGURATION)
    variable = choice.rejected_variables[0]
    monkeypatch.setenv(variable, "https://elsewhere.invalid")
    root = tmp_path / "run"

    code = main(_argv(root, name, model))

    assert code != 0
    assert variable in capsys.readouterr().err
    assert not root.exists()


# -- preflight ----------------------------------------------------------------


@pytest.mark.parametrize(("name", "_variable", "model"), EXPECTED)
def test_preflight_computes_a_plan_offline_for_every_provider(
    name: str,
    _variable: str,
    model: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preflight contacts nothing and still reports the configuration identity.

    Which is the whole point of it: the identity an operator is asked to
    authorise has to be computable without spending anything to find it out.
    """
    import json

    choice = provider_choice(name)
    monkeypatch.setenv(choice.api_key_variable, "not-a-real-key")
    for variable in choice.rejected_variables:
        monkeypatch.delenv(variable, raising=False)

    code = main(
        [
            "preflight",
            str(SUITE_MANIFEST),
            "--adapter",
            name,
            "--model",
            model,
            "--output-dir",
            str(tmp_path / f"{name}-run"),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter"]["provider"] == choice.provider
    assert payload["adapter"]["model"] == model
    assert len(payload["configuration_id"]) == 64
    # Nothing was created: a preflight plans, it does not open a run.
    assert not (tmp_path / f"{name}-run").exists()


def test_preflight_configuration_identities_differ_across_providers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four providers, four identities.

    A shared configuration identity would let an audit of one provider authorise
    a run against another, which is the exact authority the gate withholds.
    """
    import json

    seen: dict[str, str] = {}
    for name, _variable, model in EXPECTED:
        choice = provider_choice(name)
        monkeypatch.setenv(choice.api_key_variable, "not-a-real-key")
        for variable in choice.rejected_variables:
            monkeypatch.delenv(variable, raising=False)
        code = main(
            [
                "preflight",
                str(SUITE_MANIFEST),
                "--adapter",
                name,
                "--model",
                model,
                "--output-dir",
                str(tmp_path / f"{name}-run"),
                "--json",
            ]
        )
        assert code == 0
        seen[name] = json.loads(capsys.readouterr().out)["configuration_id"]
    assert len(set(seen.values())) == 4, seen


@pytest.mark.parametrize(("name", "_variable", "model"), EXPECTED)
def test_preflight_reports_the_price_policy_of_the_provider_it_planned_for(
    name: str,
    _variable: str,
    model: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The policy in the plan is this provider's policy, digest and all.

    A preflight's ``pricing_policy`` block is what an operator audits the plan's
    money against: its version, its published source, the date this repository
    read it, and the digest a run records in its configuration identity. A block
    assembled from a build-wide default would name one vendor's page as the
    provenance of another vendor's rates — internally consistent, shaped
    correctly, and wrong in the one place an audit checks. It would also
    contradict the ``cost_bound`` printed beside it, which is computed from the
    provider's own rate.
    """
    import json

    from boundarybench.pricing import pricing_policy_payload

    choice = provider_choice(name)
    monkeypatch.setenv(choice.api_key_variable, "not-a-real-key")
    for variable in choice.rejected_variables:
        monkeypatch.delenv(variable, raising=False)

    code = main(
        [
            "preflight",
            str(SUITE_MANIFEST),
            "--adapter",
            name,
            "--model",
            model,
            "--output-dir",
            str(tmp_path / f"{name}-run"),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pricing_policy"] == pricing_policy_payload(choice.provider)
    assert payload["pricing_policy"]["provider"] == choice.provider
    assert model in payload["pricing_policy"]["models"]


@pytest.mark.parametrize("name", NEW_PROVIDERS)
def test_a_cost_cap_against_an_unpriced_model_fails_closed(
    name: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cap this build cannot measure spend against is not a control."""
    choice = provider_choice(name)
    monkeypatch.setenv(choice.api_key_variable, "not-a-real-key")
    code = main(
        [
            "preflight",
            str(SUITE_MANIFEST),
            "--adapter",
            name,
            "--model",
            "a-model-nobody-priced",
            "--output-dir",
            str(tmp_path / "run"),
            "--max-cost-usd",
            "1.00",
        ]
    )
    assert code != 0
    assert "price" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(("name", "variable", "model"), EXPECTED)
def test_the_human_preflight_names_this_providers_own_credential_variable(
    name: str,
    variable: str,
    model: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line an operator reads before authorising a run has to be true.

    The credential line was rendered from a module-level constant rather than
    from the configuration being planned, so a preflight of an OpenAI, xAI or
    Mistral plan printed ``ANTHROPIC_API_KEY present`` while its JSON reported
    the right variable. An operator following the human output would export the
    wrong variable, or — worse — believe the wrong one had been checked.

    The value is never printed, in either form, and that is asserted here rather
    than assumed.
    """
    choice = provider_choice(name)
    monkeypatch.setenv(choice.api_key_variable, "not-a-real-key")
    for rejected in choice.rejected_variables:
        monkeypatch.delenv(rejected, raising=False)

    code = main(
        [
            "preflight",
            str(SUITE_MANIFEST),
            "--adapter",
            name,
            "--model",
            model,
            "--output-dir",
            str(tmp_path / f"{name}-run"),
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert f"{variable} present (value never read)" in out
    assert "credential" in out
    assert variable == choice.api_key_variable
    # No other provider's variable is named, so the line cannot be read as a
    # claim about a credential this plan never checked.
    for other, other_variable, _model in EXPECTED:
        if other != name:
            assert other_variable not in out
    assert "not-a-real-key" not in out
