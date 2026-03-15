"""MLX Local provider — in-process inference on Apple Silicon via mlx-lm.

Supports batched generation: when Inspect AI dispatches multiple concurrent
``generate()`` calls they are transparently grouped and processed in a single
``BatchGenerator`` pass on the Metal GPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typing_extensions import override

from inspect_ai._util.content import ContentText
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    ModelOutput,
    ModelUsage,
    StopReason,
)
from inspect_ai.tool import ToolChoice, ToolInfo

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 8192

# How long the batch scheduler waits for new work before running a step (seconds).
_BATCH_DRAIN_TIMEOUT = 0.005

# Sentinel used to tell the background thread to shut down.
_SHUTDOWN = object()


# ── helpers ───────────────────────────────────────────────────────────


def _rstrip_until(text: str, stop_seqs: list[str]) -> str:
    """Strip the first matching stop sequence from *text*."""
    for seq in stop_seqs:
        idx = text.find(seq)
        if idx >= 0:
            text = text[:idx]
    return text


def _chat_messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert Inspect AI ChatMessage objects to simple role/content dicts
    suitable for ``tokenizer.apply_chat_template``."""
    result: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, ChatMessageSystem):
            result.append({"role": "system", "content": msg.text})
        elif isinstance(msg, ChatMessageUser):
            if isinstance(msg.content, str):
                text = msg.content
            else:
                parts: list[str] = []
                for part in msg.content:
                    if isinstance(part, ContentText):
                        parts.append(part.text)
                    elif hasattr(part, "text"):
                        parts.append(part.text)
                    else:
                        logger.warning(
                            "mlxlocal: skipping non-text content part (%s) — "
                            "multimodal input is not yet supported.",
                            type(part).__name__,
                        )
                text = "\n".join(parts)
            result.append({"role": "user", "content": text})
        elif isinstance(msg, ChatMessageAssistant):
            result.append({"role": "assistant", "content": msg.text or ""})
        elif isinstance(msg, ChatMessageTool):
            tool_text = msg.text or ""
            if msg.tool_call_id:
                tool_text = f"[Tool Result: {msg.tool_call_id}]\n{tool_text}"
            result.append({"role": "user", "content": tool_text})
        else:
            result.append({"role": "user", "content": str(msg)})
    return result


def _map_finish_reason(reason: str | None) -> StopReason:
    if reason == "stop":
        return "stop"
    elif reason == "length":
        return "max_tokens"
    return "stop"


# ── batch request / result dataclasses ────────────────────────────────


@dataclass
class _BatchRequest:
    """A single generation request submitted to the batch scheduler."""

    prompt_tokens: list[int]
    max_tokens: int
    sampler: Any
    stop_seqs: list[str]
    # Filled in by the scheduler after insertion:
    future: asyncio.Future[_BatchResult] | None = field(default=None)
    loop: asyncio.AbstractEventLoop | None = field(default=None)
    wall_start: float = field(default_factory=time.perf_counter)


@dataclass
class _BatchResult:
    """Result returned from the batch scheduler for a single request."""

    text: str
    finish_reason: str
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    wall_time: float


# ── batch scheduler (runs on a dedicated thread) ──────────────────────


class _BatchScheduler:
    """Drives ``BatchGenerator`` from a background thread.

    Callers submit ``_BatchRequest`` objects via ``submit()`` and ``await``
    the attached ``asyncio.Future`` to get a ``_BatchResult``.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._queue: queue.Queue[_BatchRequest | object] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="mlxlocal-batch", daemon=True
        )
        self._thread.start()

    # -- public API (called from the asyncio event loop) ----------------

    def submit(self, request: _BatchRequest) -> None:
        """Enqueue a generation request.  The caller awaits ``request.future``."""
        self._queue.put(request)

    def shutdown(self) -> None:
        """Signal the background thread to exit."""
        self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=30)

    # -- background thread logic ----------------------------------------

    def _run(self) -> None:
        """Main loop executed on the background thread."""
        from mlx_lm.generate import BatchGenerator

        try:
            self._loop(BatchGenerator)
        except Exception:
            logger.exception("mlxlocal batch scheduler crashed")

    def _loop(self, BatchGeneratorCls: type) -> None:
        gen: Any | None = None
        # uid → (_BatchRequest, accumulated tokens list)
        inflight: dict[int, tuple[_BatchRequest, list[int]]] = {}

        while True:
            # 1. Drain all pending requests from the queue.
            new_requests: list[_BatchRequest] = []
            try:
                # Block briefly if there's nothing in-flight, so we don't
                # busy-wait.  If there *is* in-flight work, use a short
                # timeout so we keep calling gen.next() promptly.
                timeout = _BATCH_DRAIN_TIMEOUT if inflight else None
                item = self._queue.get(timeout=timeout)
                if item is _SHUTDOWN:
                    break
                new_requests.append(item)  # type: ignore[arg-type]
            except queue.Empty:
                pass

            # Drain any additional items without blocking.
            while True:
                try:
                    item = self._queue.get_nowait()
                    if item is _SHUTDOWN:
                        # Resolve any in-flight requests with an error.
                        self._cancel_inflight(inflight, "scheduler shutting down")
                        if gen is not None:
                            gen.close()
                        return
                    new_requests.append(item)  # type: ignore[arg-type]
                except queue.Empty:
                    break

            # 2. Insert new prompts into the BatchGenerator.
            if new_requests:
                if gen is None:
                    gen = BatchGeneratorCls(
                        self._model,
                        stop_tokens=self._tokenizer.eos_token_ids,
                    )

                prompts = [r.prompt_tokens for r in new_requests]
                max_tokens_list = [r.max_tokens for r in new_requests]
                samplers = [r.sampler for r in new_requests]

                uids = gen.insert(
                    prompts,
                    max_tokens=max_tokens_list,
                    samplers=samplers,
                )
                for uid, req in zip(uids, new_requests):
                    inflight[uid] = (req, [])

            # 3. Run one generation step if there's anything in-flight.
            if gen is not None and inflight:
                responses = gen.next()
                if not responses:
                    # All done — shouldn't happen if inflight is non-empty
                    # but guard defensively.
                    gen.close()
                    gen = None
                    continue

                # gen.stats() can raise ZeroDivisionError if called
                # before any generation time has elapsed.  We only need
                # stats when resolving a finished uid, so grab them lazily.
                stats = None

                def _get_stats() -> Any:
                    nonlocal stats
                    if stats is None:
                        try:
                            stats = gen.stats()  # type: ignore[union-attr]
                        except ZeroDivisionError:
                            # Return a dummy stats object when times are 0.
                            stats = type(
                                "_Stats",
                                (),
                                {
                                    "prompt_tokens": 0,
                                    "prompt_tps": 0.0,
                                    "prompt_time": 0.0,
                                    "generation_tokens": 0,
                                    "generation_tps": 0.0,
                                    "generation_time": 0.0,
                                    "peak_memory": 0.0,
                                },
                            )()
                    return stats

                for r in responses:
                    if r.uid not in inflight:
                        continue
                    req, tokens = inflight[r.uid]

                    if r.finish_reason is None:
                        # Still generating — accumulate token.
                        tokens.append(r.token)

                        # Check for stop sequences in partial text.
                        if req.stop_seqs:
                            partial = self._tokenizer.decode(tokens)
                            for seq in req.stop_seqs:
                                if seq in partial:
                                    # Stop early — resolve this uid.
                                    text = _rstrip_until(partial, req.stop_seqs)
                                    gen.remove([r.uid])
                                    del inflight[r.uid]
                                    self._resolve(
                                        req,
                                        text,
                                        "stop",
                                        _get_stats(),
                                        len(tokens),
                                    )
                                    break
                    else:
                        # Generation finished for this uid.
                        # Don't append the final token for "stop" (it's EOS).
                        if r.finish_reason != "stop":
                            tokens.append(r.token)
                        text = self._tokenizer.decode(tokens)
                        if req.stop_seqs:
                            text = _rstrip_until(text, req.stop_seqs)
                        del inflight[r.uid]
                        self._resolve(
                            req,
                            text,
                            r.finish_reason,
                            _get_stats(),
                            len(tokens),
                        )

                # If nothing left in-flight and nothing queued, close the
                # generator to free memory.
                if not inflight and self._queue.empty():
                    gen.close()
                    gen = None

        # Clean shutdown.
        if gen is not None:
            gen.close()

    def _resolve(
        self,
        req: _BatchRequest,
        text: str,
        finish_reason: str,
        stats: Any,
        generation_tokens: int,
    ) -> None:
        """Resolve a request's future from the background thread."""
        wall_time = time.perf_counter() - req.wall_start
        result = _BatchResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=stats.prompt_tokens,
            generation_tokens=generation_tokens,
            prompt_tps=stats.prompt_tps if stats.prompt_time > 0 else 0.0,
            generation_tps=stats.generation_tps if stats.generation_time > 0 else 0.0,
            peak_memory_gb=stats.peak_memory,
            wall_time=wall_time,
        )
        assert req.loop is not None and req.future is not None
        req.loop.call_soon_threadsafe(req.future.set_result, result)

    @staticmethod
    def _cancel_inflight(
        inflight: dict[int, tuple[_BatchRequest, list[int]]],
        reason: str,
    ) -> None:
        for _uid, (req, _tokens) in inflight.items():
            exc = RuntimeError(reason)
            assert req.loop is not None and req.future is not None
            req.loop.call_soon_threadsafe(req.future.set_exception, exc)
        inflight.clear()


# ── Inspect AI ModelAPI implementation ────────────────────────────────


class MLXLocalAPI(ModelAPI):
    """Direct in-process MLX inference provider with batched generation.

    Usage::

        bench eval mmlu --model mlxlocal/mlx-community/Llama-3.2-3B-Instruct-4bit --limit 5

    The model is loaded once into unified memory via ``mlx_lm.utils.load()``
    and kept alive for the duration of the evaluation run.  Concurrent
    ``generate()`` calls are automatically batched on the GPU for higher
    throughput.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        adapter_path: str | None = model_args.pop("adapter_path", None)
        trust_remote_code: bool = model_args.pop("trust_remote_code", False)

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        try:
            from mlx_lm.utils import load as mlx_load
        except ImportError as exc:
            raise ImportError(
                "The mlxlocal provider requires mlx-lm.  "
                "Install it with:  uv pip install mlx-lm"
            ) from exc

        tokenizer_config: dict[str, Any] = {}
        if trust_remote_code:
            tokenizer_config["trust_remote_code"] = True

        # Resolve local filesystem paths (expand ~ and relative paths)
        # so that mlx-lm's _download() recognises them as local directories
        # instead of treating them as HuggingFace repo IDs.
        resolved_name = model_name
        if (
            model_name.startswith("~")
            or model_name.startswith("/")
            or model_name.startswith(".")
        ):
            resolved_path = Path(os.path.expanduser(model_name)).resolve()
            if resolved_path.is_dir():
                resolved_name = str(resolved_path)
            else:
                logger.warning(
                    "mlxlocal: path '%s' (resolved to '%s') does not exist; "
                    "falling back to HuggingFace Hub lookup.",
                    model_name,
                    resolved_path,
                )

        logger.info("mlxlocal: loading model '%s' …", resolved_name)
        loaded = mlx_load(
            resolved_name,
            adapter_path=adapter_path,
            tokenizer_config=tokenizer_config or None,
        )
        self._mlx_model = loaded[0]
        self._tokenizer = loaded[1]
        logger.info("mlxlocal: model loaded.")

        self._use_chat_template: bool = self._tokenizer.chat_template is not None
        self._scheduler = _BatchScheduler(self._mlx_model, self._tokenizer)

    # ------------------------------------------------------------------
    # ModelAPI interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        request_dict: dict[str, Any] = {}
        response_dict: dict[str, Any] = {}

        try:
            # 1. Convert chat messages → prompt string → token ids
            messages = _chat_messages_to_dicts(input)
            if self._use_chat_template:
                prompt_str: str = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_str = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

            # Tokenize
            add_special = not self._use_chat_template
            prompt_tokens: list[int] = self._tokenizer.encode(
                prompt_str, add_special_tokens=add_special
            )

            # 2. Build sampler from GenerateConfig
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(
                temp=config.temperature if config.temperature is not None else 0.0,
                top_p=config.top_p if config.top_p is not None else 1.0,
            )

            # 3. Determine max tokens
            max_tokens = config.max_tokens or DEFAULT_MAX_TOKENS

            # 4. Gather stop sequences
            stop_seqs: list[str] = list(config.stop_seqs) if config.stop_seqs else []

            # 5. Seed if requested
            if config.seed is not None:
                import mlx.core as mx

                mx.random.seed(config.seed)

            request_dict = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stop": stop_seqs or None,
                "seed": config.seed,
            }

            # 6. Submit to batch scheduler and await result
            loop = asyncio.get_running_loop()
            future: asyncio.Future[_BatchResult] = loop.create_future()

            batch_req = _BatchRequest(
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                sampler=sampler,
                stop_seqs=stop_seqs,
                future=future,
                loop=loop,
            )
            self._scheduler.submit(batch_req)
            result: _BatchResult = await future

            response_dict = {
                "text": result.text,
                "finish_reason": result.finish_reason,
                "prompt_tokens": result.prompt_tokens,
                "generation_tokens": result.generation_tokens,
                "prompt_tps": result.prompt_tps,
                "generation_tps": result.generation_tps,
                "peak_memory_gb": result.peak_memory_gb,
            }

            # 7. Build ModelOutput
            choice = ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=result.text,
                    model=self.model_name,
                    source="generate",
                ),
                stop_reason=_map_finish_reason(result.finish_reason),
            )
            usage = ModelUsage(
                input_tokens=len(prompt_tokens),
                output_tokens=result.generation_tokens,
                total_tokens=len(prompt_tokens) + result.generation_tokens,
            )
            output = ModelOutput(
                model=self.model_name,
                choices=[choice],
                usage=usage,
                metadata={
                    "prompt_tps": result.prompt_tps,
                    "generation_tps": result.generation_tps,
                    "peak_memory_gb": result.peak_memory_gb,
                },
            )

            call = ModelCall.create(
                request=request_dict,
                response=response_dict,
                time=result.wall_time,
            )

            return output, call

        except Exception as exc:
            call = ModelCall.create(
                request=request_dict,
                response=response_dict,
                time=0.0,
            )
            return exc, call

    # ------------------------------------------------------------------
    # ModelAPI overrides
    # ------------------------------------------------------------------

    @override
    def should_retry(self, ex: Exception) -> bool:
        return False

    @override
    def connection_key(self) -> str:
        return "mlxlocal"

    @override
    def max_tokens(self) -> int | None:
        return DEFAULT_MAX_TOKENS

    @override
    def collapse_user_messages(self) -> bool:
        return True

    @override
    def collapse_assistant_messages(self) -> bool:
        return True

    @override
    async def aclose(self) -> None:
        self._scheduler.shutdown()
