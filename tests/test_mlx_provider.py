"""Unit tests for MLX provider defaults and response metadata extraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inspect_ai.model import GenerateConfig
from inspect_ai.model._chat_message import ChatMessageUser
from inspect_ai.model._model_output import ModelOutput

from openbench.model._providers.mlx import MLXAPI


class TestMLXProviderDefaults:
    """Test MLX provider defaults."""

    def test_default_base_url(self):
        """Provider should default to localhost:8080/v1."""
        with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
            provider = MLXAPI(model_name="mlx/test-model")
            assert provider.base_url == "http://localhost:8080/v1"

    def test_port_builds_base_url(self):
        """Provider should build localhost base URL when port is provided."""
        with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
            provider = MLXAPI(model_name="mlx/test-model", port=9000)
            assert provider.base_url == "http://localhost:9000/v1"

    def test_uses_env_base_url(self, monkeypatch: pytest.MonkeyPatch):
        """Provider should use MLX_BASE_URL when base_url is omitted."""
        monkeypatch.setenv("MLX_BASE_URL", "http://localhost:9999/v1")
        with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
            provider = MLXAPI(model_name="mlx/test-model")
            assert provider.base_url == "http://localhost:9999/v1"

    def test_timeout_from_config(self):
        """Client is created with timeout from GenerateConfig."""
        with patch(
            "openbench.model._providers.mlx.OpenAIAsyncHttpxClient"
        ) as mock_client:
            with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
                config = GenerateConfig(timeout=120)
                MLXAPI(model_name="mlx/test-model", config=config)
                mock_client.assert_called_once()
                timeout_obj = mock_client.call_args.kwargs["timeout"]
                assert getattr(timeout_obj, "read", None) == 120

    def test_base_url_and_port_conflict(self):
        """Provider should reject conflicting base_url and port."""
        with pytest.raises(ValueError, match="cannot both be provided"):
            MLXAPI(
                model_name="mlx/test-model",
                base_url="http://localhost:8080/v1",
                port=8080,
            )


@pytest.mark.asyncio
async def test_extracts_mlx_stats_into_metadata():
    """MLX provider should attach usage and extra server fields as metadata."""
    with patch("inspect_ai.model._providers.openai_compatible.AsyncOpenAI"):
        provider = MLXAPI(model_name="mlx/test-model")

    provider._last_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": "test-model",
        "choices": [],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
        "stats": {
            "generation_tps": 123.4,
            "prompt_tps": 456.7,
        },
        "server": "mlx_lm.server",
    }

    output = ModelOutput.from_content(model="test-model", content="ok")
    model_call = MagicMock()

    with patch(
        "inspect_ai.model._providers.openai_compatible.OpenAICompatibleAPI.generate",
        new=AsyncMock(return_value=(output, model_call)),
    ):
        result, _ = await provider.generate(
            input=[ChatMessageUser(content="hello")],
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(),
        )

    assert isinstance(result, ModelOutput)
    assert result.metadata is not None
    assert result.metadata["mlx_usage"]["prompt_tokens"] == 11
    assert result.metadata["mlx_response_extra"]["stats"]["generation_tps"] == 123.4
    assert result.metadata["mlx_response_extra"]["server"] == "mlx_lm.server"
