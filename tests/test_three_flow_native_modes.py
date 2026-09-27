"""Pinned offline template selection; no model responses or serving attestation."""

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from tools.three_flow_campaign import RIVER_MODELS
from tools.three_flow_river_assets import family_render

ASSET_DIRECTORY = os.environ.get("THREE_FLOW_RIVER_ASSETS")
ASSETS = Path(ASSET_DIRECTORY) if ASSET_DIRECTORY else None


@pytest.mark.parametrize("model", [m for m in RIVER_MODELS if "DeepSeek" not in m])
def test_native_off_minimum_render(model):
    if ASSETS is None or not ASSETS.exists():
        pytest.skip("explicit pinned local assets unavailable")
    legacy = family_render(model, {"synthetic": True}, ASSETS)
    selected = family_render(
        model, {"synthetic": True}, ASSETS, mode_profile="off-or-minimum-v1"
    )
    assert selected != legacy
    if "GLM-5.3" in model:
        assert "Reasoning Effort: Low" in selected
        assert selected.endswith("<|assistant|><think>")
    elif "Kimi" in model:
        assert selected.endswith("<|im_assistant|>assistant<|im_middle|><think></think>")
    else:
        assert selected.rstrip().endswith("</think>")


@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parent / "fixtures/deepseek_chat_goldens.json").read_text()
    ),
)
def test_deepseek_executed_official_chat_parity(case):
    from tools.three_flow_river_native import render_messages

    model = next(
        m for m in RIVER_MODELS if ("V4.1" if case["family"] == "v41" else "V4-") in m
    )
    if ASSETS is None or not ASSETS.exists():
        pytest.skip("explicit pinned local assets unavailable")
    actual = render_messages(
        model, case["messages"], case["tools"], ASSETS, mode_profile="off-or-minimum-v1"
    )
    assert actual == case["expected"]
    assert "Reasoning Effort: 75" not in actual
    assert "secret" not in actual


def test_kimi_off_parser():
    from operatebench.agents.transport import ToolCall
    from tools.three_flow_river_assets import frame
    from tools.three_flow_river_kimi import parse_kimi

    call = ToolCall("done", {"reason": "synthetic"})
    raw = frame(call).removeprefix("</think>")
    assert parse_kimi(raw, reasoning_prefilled=False) == (call,)
    assert parse_kimi("<think>" + raw + "</think>", reasoning_prefilled=False) == ()
    assert parse_kimi("<think>hidden</think>" + raw, reasoning_prefilled=False) == (call,)
    assert parse_kimi("plain text", reasoning_prefilled=False) == ()
    for malformed in (
        "<think>unfinished",
        "</think>" + raw,
        "<think><think>x</think>" + raw,
    ):
        with pytest.raises(ValueError):
            parse_kimi(malformed, reasoning_prefilled=False)


@pytest.mark.parametrize("model", RIVER_MODELS)
def test_production_native_mode_wire(tmp_path, monkeypatch, model):
    try:
        version("river-client")
    except PackageNotFoundError:
        pytest.skip("optional river-client SDK unavailable; requires Python >=3.12")
    import tools.three_flow_river_mock as mock
    from tests.test_three_flow_river import fixture
    from tools.three_flow_river import sdk
    from tools.three_flow_runtime import run_trial

    if ASSETS is None or not ASSETS.exists():
        pytest.skip("explicit pinned local assets unavailable")
    # Synthetic output for the selected mode; do not change the legacy mock.
    for name in ("frame", "synthetic_frame"):
        original = getattr(mock, name)

        def selected(*args, original=original):
            raw = original(*args)
            return raw if "GLM-5.3" in model else raw.removeprefix("</think>")

        monkeypatch.setattr(mock, name, selected)
    budget, channel, transport, trial = fixture(
        tmp_path, ASSETS, model, mode_profile="off-or-minimum-v1"
    )
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert channel.submissions
        pb, _ = sdk()
        for request in channel.submissions:
            restored = pb.InferenceGenerateRequest.FromString(request.SerializeToString())
            assert restored == request
            prompt = request.prompts[0]
            rendered = transport.tokenizer.decode(list(prompt.input_ids))
            assert transport.tokenizer.encode(rendered) == list(prompt.input_ids)
            assert (
                prompt.max_tokens == 32768
            )  # synthetic explicit fixture, not production cap
            assert list(prompt.stop) == (["<|im_end|>"] if "Kimi" in model else [])
            assert (prompt.temperature, prompt.top_p, prompt.top_k) == (0, 1, -1)
            if "GLM-5.3" in model:
                assert rendered.endswith("<|assistant|><think>")
                assert "Reasoning Effort: Low" in rendered
            else:
                assert rendered.rstrip().endswith("</think>")
                assert "Reasoning Effort: 75" not in rendered
        assert transport.settings["reasoning_prefilled"] == ("GLM-5.3" in model)
        assert transport.settings["thinking"] == ("GLM-5.3" in model)
        assert transport.settings["deepseek_reasoning_effort"] is None
        assert transport.settings["mode_profile"] == "off-or-minimum-v1"
        assert transport.request_mapping.endswith("-off-or-minimum-v1")
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", [m for m in RIVER_MODELS if "Kimi" not in m])
def test_selected_parser_nonaction_and_delimiter_controls(model):
    from operatebench.agents.transport import ToolCall
    from tools.three_flow_river_assets import synthetic_frame
    from tools.three_flow_river_native import NativeSyntaxError, decode_response

    schemas = {"done": {"type": "object", "properties": {"reason": {"type": "string"}}}}
    call = ToolCall("done", {"reason": "ok"})
    raw = synthetic_frame(model, call).removeprefix("</think>")

    def parse(text, prefilled=False):
        return decode_response(
            model,
            text,
            schemas,
            reasoning_prefilled=prefilled,
            mode_profile="off-or-minimum-v1",
        ).tool_calls

    assert parse(raw) == (call,)
    assert parse("plain text") == ()
    assert parse("<think>" + raw + "</think>") == ()
    assert parse("<think>thought</think>" + raw) == (call,)
    assert parse("thought</think>" + raw, True) == (call,)
    for broken in (
        "</think>" + raw,
        "<think>unfinished",
        "<think><think>nested</think>" + raw,
    ):
        with pytest.raises(NativeSyntaxError):
            parse(broken)
    with pytest.raises(NativeSyntaxError):
        parse(raw, True)
