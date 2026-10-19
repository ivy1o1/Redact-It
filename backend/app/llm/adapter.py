"""HTTP adapter for a local llama.cpp ``llama-server``.

Only the OpenAI-compatible surface of llama-server is used (``/health``,
``/v1/models``, ``/v1/chat/completions``). No cloud provider, no external SDK,
no telemetry.

Safety properties enforced here:
  * the target host must be loopback unless explicitly allowed in config, so
    document text can never be routed off the machine;
  * prompts are size-bounded by the caller and re-checked before sending;
  * connect and read timeouts are explicit and separately configurable;
  * every failure mode maps to a specific, actionable :class:`ModelError`.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import Settings
from ..errors import (
    ModelAuthError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from ..logging_setup import get_logger
from ._loopback import assert_local_target

logger = get_logger("llm.adapter")

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


class ModelHealth:
    """Result of probing the local server."""

    __slots__ = ("reachable", "status", "models", "base_url", "detail")

    def __init__(
        self,
        *,
        reachable: bool,
        status: str,
        models: list[str],
        base_url: str,
        detail: str | None = None,
    ) -> None:
        self.reachable = reachable
        self.status = status
        self.models = models
        self.base_url = base_url
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "status": self.status,
            "models": self.models,
            "base_url": self.base_url,
            "detail": self.detail,
        }


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a completion.

    Grammar-constrained decoding normally returns bare JSON, but fences and
    leading prose still occur when ``LLM_USE_JSON_SCHEMA`` is off.
    """
    if not text or not text.strip():
        raise ModelResponseError("The local model returned an empty completion.")

    candidate = _FENCE.sub("", text.strip()).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    start = candidate.find("{")
    if start == -1:
        raise ModelResponseError(
            "The local model did not return a JSON object. Got prose instead. "
            "Retry, or set INTERPRETATION_MODE=deterministic to analyse without "
            "the model.",
            details={"preview_length": len(candidate)},
        )

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        ch = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                fragment = candidate[start : index + 1]
                try:
                    parsed = json.loads(fragment)
                except json.JSONDecodeError as exc:
                    raise ModelResponseError(
                        "The local model returned malformed JSON. Retry, or set "
                        "INTERPRETATION_MODE=deterministic to analyse without the model.",
                        details={"parse_error": str(exc)[:200]},
                    ) from exc
                if isinstance(parsed, dict):
                    return parsed
                break

    raise ModelResponseError(
        "The local model returned an unbalanced JSON fragment.",
        details={"preview_length": len(candidate)},
    )


class LlamaCppClient:
    """Minimal async client for a local llama-server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------- lifecycle
    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(
                connect=self._settings.llm_connect_timeout_seconds,
                read=self._settings.llm_timeout_seconds,
                write=self._settings.llm_timeout_seconds,
                pool=self._settings.llm_connect_timeout_seconds,
            )
            headers = {"Accept": "application/json"}
            api_key = self._settings.llm_api_key
            if api_key is not None and api_key.get_secret_value():
                headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers=headers,
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # --------------------------------------------------------------- probing
    async def health(self) -> ModelHealth:
        """Probe ``/health`` and ``/v1/models`` without sending any content."""
        base_url = self._settings.llm_base_url
        assert_local_target(base_url, allow_non_loopback=self._settings.llm_allow_non_loopback)

        try:
            response = await self._ensure_client().get(self._settings.health_url)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            return ModelHealth(
                reachable=False,
                status="unreachable",
                models=[],
                base_url=base_url,
                detail=f"{type(exc).__name__}: connection refused or host down",
            )
        except httpx.TimeoutException:
            return ModelHealth(
                reachable=False,
                status="timeout",
                models=[],
                base_url=base_url,
                detail="the model server did not answer /health in time",
            )
        except httpx.HTTPError as exc:
            return ModelHealth(
                reachable=False,
                status="error",
                models=[],
                base_url=base_url,
                detail=f"{type(exc).__name__}",
            )

        if response.status_code == 503:
            return ModelHealth(
                reachable=True,
                status="loading",
                models=[],
                base_url=base_url,
                detail="the server is still loading the model",
            )
        if response.status_code != 200:
            return ModelHealth(
                reachable=True,
                status="unhealthy",
                models=[],
                base_url=base_url,
                detail=f"/health returned HTTP {response.status_code}",
            )

        models: list[str] = []
        try:
            listing = await self._ensure_client().get(self._settings.models_url)
            if listing.status_code == 200:
                payload = listing.json()
                entries = payload.get("data") or payload.get("models") or []
                for entry in entries:
                    if isinstance(entry, dict):
                        identifier = entry.get("id") or entry.get("name")
                        if isinstance(identifier, str) and identifier:
                            models.append(identifier)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("model_listing_failed", extra={"error_type": type(exc).__name__})

        return ModelHealth(
            reachable=True, status="ok", models=models, base_url=base_url
        )

    # ------------------------------------------------------------- inference
    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run one deterministic chat completion and return parsed JSON."""
        base_url = self._settings.llm_base_url
        assert_local_target(base_url, allow_non_loopback=self._settings.llm_allow_non_loopback)

        url = self._settings.chat_completions_url
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.llm_temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens or self._settings.llm_max_tokens,
            "stream": False,
        }

        if schema is not None and self._settings.llm_use_json_schema:
            # Grammar-constrained decoding. This is what makes a 1B model usable:
            # the enum of detector names cannot be violated at the token level.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "redaction_plan", "schema": schema, "strict": True},
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        client = self._ensure_client()
        last_error: Exception | None = None

        for attempt in range(1, self._settings.llm_max_attempts + 1):
            try:
                response = await client.post(url, json=payload)
            except httpx.ConnectTimeout as exc:
                raise ModelUnavailableError(
                    base_url, reason="connection timed out"
                ) from exc
            except httpx.ConnectError as exc:
                raise ModelUnavailableError(
                    base_url, reason="connection refused"
                ) from exc
            except httpx.ReadTimeout as exc:
                raise ModelTimeoutError(
                    f"The local model did not finish within "
                    f"{self._settings.llm_timeout_seconds:.0f}s. The document may be large or "
                    "the model busy. Retry, reduce LLM_MAX_PROMPT_CHARS, or "
                    "raise LLM_TIMEOUT_SECONDS.",
                    details={"timeout_seconds": self._settings.llm_timeout_seconds},
                ) from exc
            except httpx.TimeoutException as exc:
                raise ModelTimeoutError(
                    "The local model request timed out.", details={"attempt": attempt}
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "llm_transport_error",
                    extra={"error_type": type(exc).__name__, "attempt": attempt},
                )
                continue

            data = self._handle_status(response, base_url)
            content = self._extract_content(data)
            try:
                parsed = extract_json_object(content)
            except ModelResponseError:
                if attempt < self._settings.llm_max_attempts:
                    logger.warning("llm_json_retry", extra={"attempt": attempt})
                    continue
                raise
            logger.info(
                "llm_completion_ok",
                extra={
                    "attempt": attempt,
                    "keys": sorted(parsed.keys())[:12],
                    "usage_total": (data.get("usage") or {}).get("total_tokens"),
                },
            )
            return parsed

        raise ModelResponseError(
            "The local model server could not be reached after "
            f"{self._settings.llm_max_attempts} attempts.",
            details={"error_type": type(last_error).__name__ if last_error else None},
        )

    # ---------------------------------------------------------------- helpers
    def _handle_status(self, response: httpx.Response, base_url: str) -> dict[str, Any]:
        status = response.status_code

        if status in (401, 403):
            raise ModelAuthError(
                f"The local model server rejected authentication (HTTP {status}). "
                "If llama-server was started with --api-key, set LLM_API_KEY to the "
                "same value.",
                details={"status_code": status, "base_url": base_url},
            )

        if status == 503:
            raise ModelLoadingError(base_url)

        if status == 404:
            raise ModelNotFoundError(self._settings.llm_model, base_url)

        if status >= 400:
            try:
                body = response.json()
            except ValueError:
                body = None
            message = ""
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message", ""))[:300]
                elif isinstance(error, str):
                    message = error[:300]
            if "json_schema" in message.lower() or "response_format" in message.lower():
                raise ModelResponseError(
                    "This llama-server build rejected the constrained-JSON request. "
                    "Set LLM_USE_JSON_SCHEMA=false, or update llama.cpp.",
                    details={"status_code": status, "message": message},
                )
            raise ModelResponseError(
                f"The local model server returned HTTP {status}. {message}".strip(),
                details={"status_code": status, "message": message},
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ModelResponseError(
                "The local model server returned a non-JSON response body.",
                details={"status_code": status},
            ) from exc
        if not isinstance(data, dict):
            raise ModelResponseError(
                "The local model server returned a JSON value that is not an object.",
                details={"status_code": status},
            )
        return data

    def _extract_content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelResponseError(
                "The local model response contained no choices.",
                details={"keys": sorted(data.keys())[:12]},
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise ModelResponseError("Malformed choice in the local model response.")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ModelResponseError("Malformed message in the local model response.")
        content = message.get("content")
        if not isinstance(content, str):
            raise ModelResponseError("The local model response had no text content.")
        if first.get("finish_reason") == "length":
            logger.warning("llm_completion_truncated")
        return content


__all__ = ["LlamaCppClient", "ModelHealth", "extract_json_object"]
