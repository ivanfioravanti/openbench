"""MLX provider implementation."""

import os
from typing import Any

import httpx
from typing_extensions import override

from inspect_ai.model import GenerateConfig
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.model._openai import OpenAIAsyncHttpxClient
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.tool import ToolChoice, ToolInfo


class MLXAPI(OpenAICompatibleAPI):
    """MLX provider for `mlx_lm.server` OpenAI-compatible endpoints."""

    DEFAULT_BASE_URL = "http://localhost:8080/v1"
    DEFAULT_API_KEY = "openbench"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        if base_url and port:
            raise ValueError("base_url and port cannot both be provided.")

        model_name_clean = model_name.replace("mlx/", "", 1)

        base_url = base_url or os.environ.get("MLX_BASE_URL")
        if not base_url:
            base_url = (
                f"http://localhost:{port}/v1"
                if port is not None
                else self.DEFAULT_BASE_URL
            )

        api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or self.DEFAULT_API_KEY
        )

        timeout_seconds = getattr(config, "timeout", None)
        if timeout_seconds is not None and "http_client" not in model_args:
            model_args["http_client"] = OpenAIAsyncHttpxClient(
                timeout=httpx.Timeout(timeout=timeout_seconds)
            )

        self._last_response: dict[str, Any] | None = None

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="mlx",
            service_base_url=base_url,
            **model_args,
        )

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        result = await super().generate(input, tools, tool_choice, config)

        if isinstance(result, tuple):
            output_or_exception, call = result
            if isinstance(output_or_exception, ModelOutput):
                output_or_exception = self._attach_mlx_metadata(output_or_exception)
            return output_or_exception, call

        if isinstance(result, ModelOutput):
            return self._attach_mlx_metadata(result)

        return result

    @override
    def on_response(self, response: dict[str, Any]) -> None:
        self._last_response = response

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name

    def _attach_mlx_metadata(self, output: ModelOutput) -> ModelOutput:
        metadata = self._metadata_from_response(self._last_response)
        if not metadata:
            return output

        output.metadata = {**(output.metadata or {}), **metadata}
        return output

    @staticmethod
    def _metadata_from_response(
        response: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(response, dict):
            return None

        metadata: dict[str, Any] = {}
        usage = response.get("usage")
        if isinstance(usage, dict):
            metadata["mlx_usage"] = usage

        standard_keys = {
            "id",
            "object",
            "created",
            "model",
            "choices",
            "usage",
            "service_tier",
            "system_fingerprint",
        }
        extra = {k: v for k, v in response.items() if k not in standard_keys}
        if extra:
            metadata["mlx_response_extra"] = extra

        return metadata or None
