"""Injected official River SDK gRPC stub, durable shared campaign accounting.

No client/credential discovery, default channel, admission or paid entrypoint.
Optional SDK imports occur only when constructing this transport on Python >=3.12.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from operatebench.agents.lifecycle_contract import (
    INSTRUCTIONS_CONTRACT,
    check_model_request,
    elide_null_optionals,
    outcome_tools,
)
from operatebench.agents.model import MAX_OUTPUT_TOKENS, MODEL_PROTOCOL_VERSION
from operatebench.agents.transport import ModelRequest, ModelResponse, ToolCall
from operatebench.providers.faults import AdapterProviderError, SingleFlight
from tools.aggregate_budget import SharedGuard
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_errors import grpc_error, river_failed
from tools.three_flow_river_assets import catalog, family_render, family_tokenizer
from tools.three_flow_river_kimi import parse_kimi
from tools.three_flow_river_native import InterfaceAmbiguityError, decode_response

RPC_PREFIX = "/river.api.v1.RiverService/"
GRPC_OPTIONS = (("grpc.enable_retries", 0),)
SDK_VERSION = "0.12.0"


def sdk() -> tuple[Any, Any]:
    if version("river-client") != SDK_VERSION:
        raise ValueError("River SDK requires reviewed version 0.12.0")
    return (
        importlib.import_module("river_client._proto.river_pb2"),
        importlib.import_module("river_client._proto.river_pb2_grpc"),
    )


def invalid() -> AdapterProviderError:
    return AdapterProviderError(
        "provider_response_invalid", "invalid River envelope; liability retained"
    )


class ExplicitRiverChannel:
    """Optional caller-supplied TLS factory; no defaults or credential discovery.

    Construction grants no authority. The operator must admit rates/settings and
    authorization before supplying this capability. Tests supply fake factories.
    """

    def __init__(
        self, *, api_key: str, channel_factory: Any, credentials_factory: Any
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("explicit credential capability required")
        self._metadata = (("x-api-key", api_key),)
        self._channel = channel_factory(
            "api.river.ai:443", credentials_factory(), options=GRPC_OPTIONS
        )

    def unary_unary(self, path: str, **kwargs: Any) -> Any:
        rpc = self._channel.unary_unary(path, **kwargs)

        def invoke(request: Any, *, timeout: float | None) -> Any:
            return rpc(request, timeout=timeout, metadata=self._metadata)

        return invoke

    def close(self) -> None:
        self._channel.close()


class BudgetChannel:
    """Generated stub's channel: reserve/fsync immediately before actual submit.

    The explicit inner channel must have transport retries disabled. No SDK
    high-level retry/session/heartbeat machinery is used. Polls share the same
    outstanding reservation and are recorded as network RPCs, not generations.
    """

    def __init__(self, inner: Any, guard: SharedGuard) -> None:
        if not isinstance(guard.budget, CampaignBudget):
            raise ValueError("shared durable campaign journal required")
        self.inner, self.guard = inner, guard
        self.observer: Any = None
        self.captures: list[dict[str, Any]] = []
        self.expected: bytes | None = None
        self.reservation: Any = None
        self.request_id: str | None = None
        self.local_failure: str | None = None
        self.pb, _ = sdk()

    def unary_unary(
        self,
        path: str,
        *,
        request_serializer: Any,
        response_deserializer: Any,
        **_kwargs: Any,
    ) -> Any:
        name = path.removeprefix(RPC_PREFIX)

        # The generated stub defines all methods. Only these two are capabilities.
        def invoke(request: Any, *, timeout: float | None) -> Any:
            if name not in ("InferenceGenerate", "RetrieveFuture"):
                raise ValueError("non-inference River RPC forbidden")
            raw = request_serializer(request)
            if name == "InferenceGenerate":
                if raw != self.expected or self.reservation is not None:
                    raise ValueError("changed or repeated River submission")
                prompt = request.prompts[0]
                self.guard._max_output_tokens = prompt.max_tokens
                try:
                    self.reservation = self.guard.authorize(
                        input_tokens_upper_bound=len(prompt.input_ids)
                    )
                except OSError:
                    self.local_failure = "budget_journal_error"
                    raise
                try:
                    self.guard.budget.dispatch(
                        self.guard.cell,
                        {
                            "rpc": path,
                            "model": request.base_model,
                            "max_output_tokens": prompt.max_tokens,
                            "protobuf_sha256": hashlib.sha256(raw).hexdigest(),
                        },
                    )
                except Exception:
                    # Trusted local seam, never inferred from provider text.
                    self.local_failure = "dispatch_journal_error"
                    raise
            elif self.request_id is None or request.request_id != self.request_id:
                raise ValueError("poll differs from acknowledged generation")
            row = {
                "rpc": path,
                "request_base64": base64.b64encode(raw).decode(),
                "request_sha256": hashlib.sha256(raw).hexdigest(),
                "model_submission": name == "InferenceGenerate",
            }
            self.captures.append(row)
            if self.observer is not None:
                try:
                    self.observer({"kind": "wire_request", **row})
                except Exception:
                    self.local_failure = "wire_evidence_error"
                    raise
            rpc = self.inner.unary_unary(
                path,
                request_serializer=request_serializer,
                response_deserializer=response_deserializer,
                _registered_method=True,
            )
            if hasattr(self.guard.budget, "raw_rpc"):
                self.guard.budget.raw_rpc(self.guard.cell, path, row["request_sha256"])
            try:
                response = rpc(request, timeout=timeout)
            except importlib.import_module("grpc").RpcError as exc:
                row["response_projection"] = grpc_error(exc, name)
                if self.observer is not None:
                    self.observer({"kind": "wire_response", **row})
                raise
            expected_type = (
                self.pb.AsyncResponse
                if name == "InferenceGenerate"
                else self.pb.RetrieveFutureResponse
            )
            if not isinstance(response, expected_type):
                raise invalid()
            if name == "RetrieveFuture" and response.WhichOneof("response") == "failed":
                row["response_projection"] = river_failed(response.failed)
                if self.observer is not None:
                    self.observer({"kind": "wire_response", **row})
                raise AdapterProviderError(
                    "provider_server_error", "native generation failed"
                )
            response_raw = response.SerializeToString(deterministic=True)
            known = expected_type()
            known.CopyFrom(response)
            known.DiscardUnknownFields()
            if known.SerializeToString(deterministic=True) != response_raw:
                raise invalid()
            row.update(
                response_base64=base64.b64encode(response_raw).decode(),
                response_sha256=hashlib.sha256(response_raw).hexdigest(),
            )
            if self.observer is not None:
                try:
                    self.observer({"kind": "wire_response", **row})
                except Exception:
                    # In particular, do not poll an ack that was not persisted.
                    self.local_failure = "wire_evidence_error"
                    raise
            if name == "InferenceGenerate":
                if not response.request_id:
                    raise invalid()
                self.request_id = response.request_id
            return response

        return invoke

    def unary_stream(self, *_args: Any, **_kwargs: Any) -> Any:
        def forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("streaming/training River RPC forbidden")

        return forbidden

    def close(self) -> None:
        self.inner.close()


class RiverCampaignTransport:
    @property
    def local_failure(self) -> str | None:
        return self.wire.local_failure or (
            "budget_journal_error" if self.guard.budget.broken else None
        )

    provider = "river"
    api = "grpc_inference_generate"
    request_mapping = "three-flow-river-native-v1"

    def __init__(
        self,
        *,
        model: str,
        assets: Path,
        channel: Any,
        guard: SharedGuard,
        max_output_tokens: int,
        channel_options: tuple[tuple[str, int], ...],
        network_timeout: float | None = 120.0,
        mode_profile: str = "legacy-v1",
    ) -> None:
        if mode_profile not in ("legacy-v1", "off-or-minimum-v1"):
            raise ValueError("unknown reasoning mode profile")
        self.mode_profile = mode_profile
        self.reasoning_prefilled = mode_profile == "legacy-v1" or "GLM-5.3" in model
        if mode_profile != "legacy-v1":
            self.request_mapping += "-" + mode_profile
        if not isinstance(channel, ExplicitRiverChannel) and not getattr(
            channel, "offline_mock", False
        ):
            raise ValueError("explicit retry-disabled channel capability required")
        if channel_options != GRPC_OPTIONS:
            raise ValueError("explicit channel must disable gRPC retries")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("explicit positive output setting required")
        if guard._max_output_tokens != max_output_tokens:
            raise ValueError("guard output setting differs")
        rows = [r for r in catalog()["models"] if r["model"] == model]
        if len(rows) != 1:
            raise ValueError("exact River roster model required")
        self.model, self.assets, self.guard = model, assets, guard
        self.network_timeout = network_timeout
        self.max_output_tokens = max_output_tokens
        self.tokenizer = family_tokenizer(assets, rows[0])
        self.pb, generated = sdk()
        self.wire = BudgetChannel(channel, guard)
        self.stub = generated.RiverServiceStub(self.wire)
        self.flight = SingleFlight(adapter="three-flow River")
        self.last_turn: dict[str, Any] | None = None
        self.settings = {
            "sdk": "river-client",
            "sdk_version": SDK_VERSION,
            "sdk_surface": "official generated RiverServiceStub/protobuf",
            "endpoint": "api.river.ai:443",
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "max_output_tokens": max_output_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "stop": ["<|im_end|>"] if "Kimi" in model else [],
            "thinking": True,
            "deepseek_reasoning_effort": 75 if "V4.1" in model else None,
            "max_retries": 0,
            "grpc_options": list(map(list, GRPC_OPTIONS)),
            "rpc_timeout_seconds": network_timeout,
            "generation_wall_limit": None,
            "credential_source": "explicit injected channel; no SDK Client constructor",
            "tokenizer_revision": rows[0]["revision"],
            "tokenizer_model": rows[0].get("tokenizer_model", model),
            "assets": rows[0]["assets"],
            "tokenizer_runtime": version("tiktoken" if "Kimi" in model else "tokenizers"),
            "template_runtime": version("Jinja2"),
            "instruction_sha256": hashlib.sha256(
                INSTRUCTIONS_CONTRACT.encode()
            ).hexdigest(),
            "returned_model_attested": False,
            "identity_basis": "requested model; protobuf has no returned model field",
            "provider_stop_reason_available": False,
            "stop_classification": (
                "local token cap or completed response; raw reason absent"
            ),
            "usage_policy": "full prompt and all generated token IDs; no cache discount",
            "model_contract_verified": False,
            "admission": False,
            "evidence_origin": "SDK_MOCK_NOT_LLM"
            if getattr(channel, "offline_mock", False)
            else "PROVIDER_CANDIDATE",
        }
        if mode_profile != "legacy-v1":
            self.settings.update(
                mode_profile=mode_profile,
                reasoning_mode="MINIMUM" if self.reasoning_prefilled else "OFF",
                reasoning_prefilled=self.reasoning_prefilled,
                thinking=self.reasoning_prefilled,
                deepseek_reasoning_effort=None,
            )
            if "DeepSeek" in model:
                self.settings["encoder"] = {
                    "thinking_mode": "chat",
                    "reasoning_effort": None,
                }
            else:
                self.settings["template_kwargs"] = (
                    {"reasoning_effort": "low"}
                    if self.reasoning_prefilled
                    else {"thinking": False}
                    if "Kimi" in model
                    else {"enable_thinking": False}
                )

    def response(
        self, response: Any, ids: list[int], *, max_output_tokens: int | None = None
    ) -> ModelResponse:
        if max_output_tokens is None:
            max_output_tokens = self.max_output_tokens
        if not isinstance(response, self.pb.InferenceResponse) or not response.HasField(
            "usage"
        ):
            raise invalid()
        u = response.usage
        if len(response.results) != 1:
            raise invalid()
        sample = response.results[0]
        output = list(sample.token_ids)
        text_matches = self.tokenizer.decode(output) == sample.text
        terminal = {
            "Qwen/Qwen3.6-35B-A3B-FP8": (248046, "<|im_end|>"),
            "deepseek-ai/DeepSeek-V4.1-Flash": (1, "<｜end▁of▁sentence｜>"),  # noqa: RUF001
            "nvidia/Kimi-K2.6-NVFP4": (163586, "<|im_end|>"),
            "nvidia/GLM-5.2-NVFP4-262K": (154829, "<|observation|>"),
            "zai-org/GLM-5.3-Flash": (154829, "<|observation|>"),
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": (11, "<|im_end|>"),
        }.get(self.model)
        if not text_matches and terminal and output and output[-1] == terminal[0]:
            text_matches = (
                self.tokenizer.decode(output[-1:]) == terminal[1]
                and self.tokenizer.decode(output[:-1]) == sample.text
            )
        if (
            u.prompt_tokens != len(ids)
            or u.completion_tokens != len(output)
            or u.total_tokens != u.prompt_tokens + u.completion_tokens
            or u.training_tokens != 0
            or len(output) > max_output_tokens
            or not text_matches
            or sample.retained_kv
            or not 0 <= sample.cached_prompt_tokens <= len(ids)
            or (sample.prompt_tokens and sample.prompt_tokens != len(ids))
            or (sample.prompt_token_ids and list(sample.prompt_token_ids) != ids)
        ):
            raise invalid()
        if len(output) >= max_output_tokens:
            return ModelResponse(
                model=self.model, stop_reason="max_tokens", output_tokens=len(output)
            )
        try:
            calls = (
                parse_kimi(sample.text, reasoning_prefilled=self.reasoning_prefilled)
                if "Kimi" in self.model
                else decode_response(
                    self.model,
                    sample.text,
                    outcome_tools(),
                    reasoning_prefilled=self.reasoning_prefilled,
                    mode_profile=self.mode_profile,
                ).tool_calls
            )
            calls = tuple(
                ToolCall(c.name, elide_null_optionals(c.name, c.arguments)) for c in calls
            )
        except ValueError:
            calls = ()  # Malformed native model output remains a model decision error.
        return ModelResponse(
            model=self.model,
            stop_reason="completed",
            tool_calls=calls,
            output_tokens=len(output),
        )

    def prepare_request(self, request: ModelRequest) -> ModelRequest:
        if not getattr(self.guard, "diagnostic", False):
            return request
        rendered = family_render(
            self.model, dict(request.prompt), self.assets, mode_profile=self.mode_profile
        )
        bound = len(self.tokenizer.encode(rendered))
        return replace(
            request, max_output_tokens=cast(Any, self.guard).output_bound(bound)
        )

    def send(self, request: ModelRequest) -> ModelResponse:
        with self.flight:
            self.last_turn = None
            check_model_request(
                replace(request, max_output_tokens=MAX_OUTPUT_TOKENS), model=self.model
            )
            diagnostic = getattr(self.guard, "diagnostic", False)
            if (
                not diagnostic and request.max_output_tokens != self.max_output_tokens
            ) or not (
                type(request.max_output_tokens) is int
                and 0 < request.max_output_tokens <= self.max_output_tokens
            ):
                raise ValueError("request output differs from bound settings")
            started = time.monotonic()
            start = len(self.wire.captures)
            self.wire.reservation = None
            self.wire.request_id = None
            try:
                rendered = family_render(
                    self.model,
                    dict(request.prompt),
                    self.assets,
                    mode_profile=self.mode_profile,
                )
                ids = self.tokenizer.encode(rendered)
                proto = self.pb.InferenceGenerateRequest(
                    base_model=self.model,
                    prompts=[
                        self.pb.InferencePrompt(
                            input_ids=ids,
                            max_tokens=request.max_output_tokens,
                            temperature=0,
                            top_p=1,
                            top_k=-1,
                            stop=self.settings["stop"],
                        )
                    ],
                )
                self.wire.expected = proto.SerializeToString()
                ack = self.stub.InferenceGenerate(proto, timeout=self.network_timeout)
                while True:
                    answer = self.stub.RetrieveFuture(
                        self.pb.RetrieveFutureRequest(request_id=ack.request_id),
                        timeout=self.network_timeout,
                    )
                    kind = answer.WhichOneof("response")
                    if kind == "inference":
                        break
                    if (
                        kind != "try_again"
                        or answer.try_again.request_id != ack.request_id
                    ):
                        raise invalid()
                    time.sleep(0.01)
                result = self.response(
                    answer.inference, ids, max_output_tokens=request.max_output_tokens
                )
                usage = answer.inference.usage
                measured = self.guard.settle(
                    self.wire.reservation,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
                self.last_turn = {
                    "requested_output_bound": request.max_output_tokens,
                    "input_bound": len(ids),
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "measured_usd": str(measured),
                    "latency_seconds": time.monotonic() - started,
                    "network_rpcs": len(self.wire.captures) - start,
                    "model_submissions": 1,
                    "provider_request_id": ack.request_id,
                    "provider_stop_reason": None,
                    "local_stop_classification": result.stop_reason,
                    "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "wire": self.wire.captures[start:],
                }
                return result
            except BaseException as exc:
                if self.guard.ticket is not None and not self.guard.budget.broken:
                    self.guard.forfeit(self.wire.reservation)
                self.last_turn = {
                    "classification": "infrastructure_failure"
                    if self.local_failure is not None
                    else exc.fault
                    if isinstance(exc, AdapterProviderError)
                    else "internal_error",
                    "measured_usd": None,
                    "latency_seconds": time.monotonic() - started,
                    "network_rpcs": len(self.wire.captures) - start,
                    "wire": self.wire.captures[start:],
                }
                if isinstance(exc, InterfaceAmbiguityError):
                    self.last_turn["classification"] = "provider_response_invalid"
                    raise AdapterProviderError(
                        "provider_response_invalid", "native interface domain exclusion"
                    ) from exc
                if isinstance(exc, TimeoutError):
                    self.last_turn["classification"] = "provider_timeout"
                    raise AdapterProviderError(
                        "provider_timeout", "native RPC timed out"
                    ) from None
                grpc = importlib.import_module("grpc")
                if isinstance(exc, grpc.RpcError):
                    projection = grpc_error(
                        exc,
                        "RetrieveFuture" if self.wire.request_id else "InferenceGenerate",
                    )
                    self.last_turn["error_projection"] = projection
                    fault = {
                        "INVALID_ARGUMENT": "provider_request_rejected",
                        "UNAUTHENTICATED": "provider_authentication",
                        "PERMISSION_DENIED": "provider_authentication",
                        "RESOURCE_EXHAUSTED": "provider_rate_limited",
                        "DEADLINE_EXCEEDED": "provider_timeout",
                        "UNAVAILABLE": "provider_network_error",
                    }.get(projection["grpc_status"], "provider_server_error")
                    self.last_turn["classification"] = fault
                    if projection["grpc_status"] == "INVALID_ARGUMENT":
                        self.last_turn.update(
                            diagnostic_category="provider_request_refusal",
                            safe_status="INVALID_ARGUMENT",
                            automatic_resubmission=False,
                        )
                    raise AdapterProviderError(fault, "native RPC failed") from None
                raise
            finally:
                self.wire.expected = None

    def close(self) -> None:
        self.wire.close()
