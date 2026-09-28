"""Independent native frames: narrowly allow verified terminal omission only."""

# Fullwidth delimiters are literal native protocol bytes.
# ruff: noqa: RUF001

import os
import socket
import sys
from pathlib import Path

import pytest

from operatebench.providers.faults import AdapterProviderError
from tools.three_flow_river import RiverCampaignTransport, sdk
from tools.three_flow_river_native import NativeTokenizer

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="River SDK")

CASES = [
    (
        "Qwen/Qwen3.6-35B-A3B-FP8",
        248046,
        "<|im_end|>",
        "<tool_call>\n<function=complete>\n<parameter=reason>\n"
        "offline fixture\n</parameter>\n</function>\n</tool_call>",
    ),
    (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        1,
        "<｜end▁of▁sentence｜>",
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="complete">\n'
        '<｜DSML｜ parameter name="reason" string="true">offline fixture'
        "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    ),
]


@pytest.mark.parametrize("model,eos,marker,raw", CASES)
@pytest.mark.parametrize(
    "variant",
    ["omitted", "included", "wrong_text", "double_eos", "usage", "cap", "malformed"],
)
def test_terminal_omission(model, eos, marker, raw, variant, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    import grpc

    monkeypatch.setattr(grpc, "secure_channel", forbidden)
    monkeypatch.setattr(grpc, "insecure_channel", forbidden)
    assets = os.environ.get("THREE_FLOW_RIVER_ASSETS")
    if assets is None:
        pytest.skip("explicit pinned assets required")
    t = object.__new__(RiverCampaignTransport)
    t.model = model
    t.pb, _ = sdk()
    t.tokenizer = NativeTokenizer(model, Path(assets))
    t.max_output_tokens = 30000
    t.reasoning_prefilled = False
    t.mode_profile = "off-or-minimum-v1"
    assert t.tokenizer.decode([eos]) == marker
    if variant == "malformed":
        raw = raw.replace("complete", "not_a_tool")
    ids = [*t.tokenizer.encode(raw), eos]
    text = raw + marker if variant == "included" else raw
    if variant == "wrong_text":
        text += "unexpected"
    if variant == "double_eos":
        ids.append(eos)
    response = t.pb.InferenceResponse(
        results=[t.pb.InferenceResult(text=text, token_ids=ids)],
        usage=t.pb.Usage(
            prompt_tokens=1, completion_tokens=len(ids), total_tokens=1 + len(ids)
        ),
    )
    if variant == "usage":
        response.usage.completion_tokens -= 1
    if variant in ("wrong_text", "double_eos", "usage"):
        with pytest.raises(AdapterProviderError) as exc:
            t.response(response, [7])
        assert exc.value.fault == "provider_response_invalid"
    else:
        result = t.response(
            response, [7], max_output_tokens=len(ids) if variant == "cap" else 30000
        )
        assert result.output_tokens == len(ids)
        assert result.stop_reason == ("max_tokens" if variant == "cap" else "completed")
        if variant in ("cap", "malformed"):
            assert not result.tool_calls
        else:
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "complete"
            assert result.tool_calls[0].arguments == {"reason": "offline fixture"}
