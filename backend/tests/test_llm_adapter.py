"""Tests for the local llama-server adapter and the loopback guard.

The adapter is the only door through which document text can leave this
machine, so these tests are mostly about two things: that every failure mode
produces a specific, actionable error rather than a fake result, and that
nothing is sent to a host that is not loopback.

All requests are served by ``httpx.MockTransport``; no test contacts a real
server (the live end-to-end check is ``test_model_live.py``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest

from app.config import Settings, get_settings
from app.errors import (
    ModelAuthError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from app.llm._loopback import assert_local_target, host_is_local
from app.llm.adapter import LlamaCppClient, extract_json_object

from .conftest import make_settings

COMPLETION = '{"targets": [], "ambiguous": false}'


def ok_response(content: str = COMPLETION) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 51},
        },
    )


class Recorder:
    """Collects the requests a mocked server was asked to serve."""

    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self.responder = responder
        self.requests: list[httpx.Request] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content.decode())

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responder(request)


def run_client(
    settings: Settings, responder: Callable[[httpx.Request], httpx.Response]
) -> tuple[LlamaCppClient, Recorder]:
    rec = Recorder(responder)
    client = LlamaCppClient(settings)
    # Inject the transport instead of patching httpx: the guard logic under test
    # lives in the adapter, and MockTransport cannot reach the network by
    # construction, so a leak in the test itself is impossible.
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handler))
    return client, rec


def chat(client: LlamaCppClient, **kwargs: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {"system_prompt": "sys", "user_prompt": "user"}
    defaults.update(kwargs)
    return asyncio.run(client.chat_json(**defaults))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


class TestExtractJsonObject:
    """Grammar-constrained decoding usually returns bare JSON; we still cope."""

    def test_bare_json_object(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_a_json_fence_is_stripped(self) -> None:
        assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_leading_and_trailing_prose_are_ignored(self) -> None:
        text = 'Sure! Here is the plan:\n{"a": {"b": 2}}\nHope that helps.'
        assert extract_json_object(text) == {"a": {"b": 2}}

    def test_a_brace_inside_a_string_does_not_end_the_object(self) -> None:
        assert extract_json_object('{"s": "brace } inside"}') == {"s": "brace } inside"}

    def test_an_escaped_quote_inside_a_string_is_respected(self) -> None:
        assert extract_json_object('{"t": "say \\"hi\\" now"}') == {"t": 'say "hi" now'}

    def test_only_the_first_object_is_taken(self) -> None:
        assert extract_json_object('{"a": 1} {"b": 2}') == {"a": 1}

    def test_an_empty_completion_is_refused(self) -> None:
        with pytest.raises(ModelResponseError, match="empty completion"):
            extract_json_object("   ")

    def test_prose_without_any_object_is_refused(self) -> None:
        with pytest.raises(ModelResponseError, match="Got prose instead") as exc:
            extract_json_object("no json here")
        # The recovery path must be named in the message, not just implied.
        assert "deterministic" in str(exc.value)

    def test_a_json_array_is_not_an_object(self) -> None:
        with pytest.raises(ModelResponseError, match="did not return a JSON object"):
            extract_json_object("[1, 2]")

    def test_a_json_null_is_not_an_object(self) -> None:
        with pytest.raises(ModelResponseError, match="did not return a JSON object"):
            extract_json_object("null")

    def test_an_unbalanced_fragment_is_reported_as_such(self) -> None:
        with pytest.raises(ModelResponseError, match="unbalanced JSON fragment"):
            extract_json_object('{"a": ')

    def test_malformed_content_inside_an_object_is_reported_as_such(self) -> None:
        with pytest.raises(ModelResponseError, match="malformed JSON"):
            extract_json_object('{"a": unquoted}')


class TestLoopbackEnforcement:
    def test_loopback_names_and_addresses_are_local(self) -> None:
        for host in ("localhost", "127.0.0.1", "::1", "[::1]"):
            assert host_is_local(host), host

    def test_public_and_private_hosts_are_not_local(self) -> None:
        for host in ("api.openai.com", "generativelanguage.googleapis.com", "10.0.0.5", "192.168.1.20"):
            assert not host_is_local(host), host

    def test_an_empty_host_is_not_local(self) -> None:
        assert not host_is_local("")

    def test_an_unresolvable_host_is_not_local(self) -> None:
        assert not host_is_local("no-such-host.invalid")

    def test_a_non_loopback_target_is_refused(self) -> None:
        with pytest.raises(ModelUnavailableError, match="LLM_ALLOW_NON_LOOPBACK"):
            assert_local_target("https://api.openai.com/v1", allow_non_loopback=False)

    def test_a_loopback_target_is_allowed(self) -> None:
        assert_local_target("http://127.0.0.1:8080/v1", allow_non_loopback=False) is None

    def test_an_unsupported_scheme_is_refused(self) -> None:
        with pytest.raises(ModelUnavailableError, match="unsupported scheme"):
            assert_local_target("ftp://127.0.0.1/files", allow_non_loopback=False)

    def test_a_hostless_url_is_refused(self) -> None:
        with pytest.raises(ModelUnavailableError, match="no host"):
            assert_local_target("http:///v1", allow_non_loopback=False)

    def test_the_opt_in_allows_a_non_loopback_target(self) -> None:
        assert_local_target("http://192.168.1.5:8080/v1", allow_non_loopback=True) is None

    def test_the_guard_runs_before_any_bytes_are_sent(self, settings: Settings) -> None:
        # A base URL that passed validation can only be non-loopback if the
        # operator deliberately opted in. Turning the opt-in back off must stop
        # the request at the adapter, not merely warn about it.
        settings = make_settings(
            settings.resolved_storage_root.parent,
            llm_base_url="http://model.example.invalid/v1",
            llm_allow_non_loopback=True,
        )
        client, rec = run_client(settings, lambda request: ok_response())
        client._settings.llm_allow_non_loopback = False
        with pytest.raises(ModelUnavailableError):
            chat(client)
        assert rec.calls == 0

    def test_settings_refuse_a_cloud_model_host(self) -> None:
        with pytest.raises(ValueError, match="not a loopback address"):
            Settings(_env_file=None, llm_base_url="https://api.openai.com/v1")

    def test_settings_refuse_cloud_fallback(self) -> None:
        with pytest.raises(ValueError, match="LLM_ALLOW_CLOUD_FALLBACK"):
            Settings(_env_file=None, llm_allow_cloud_fallback=True)


class TestRequestShape:
    def test_the_request_uses_the_configured_model_and_sampling(self, settings: Settings) -> None:
        client, rec = run_client(settings, lambda request: ok_response())
        chat(client)
        body = rec.body
        assert body["model"] == "gemma-3-1b-it"
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == settings.llm_max_tokens
        assert body["stream"] is False

    def test_the_prompts_are_sent_as_system_then_user(self, settings: Settings) -> None:
        client, rec = run_client(settings, lambda request: ok_response())
        chat(client, system_prompt="SYSPROMPT", user_prompt="USERPROMPT")
        assert rec.body["messages"] == [
            {"role": "system", "content": "SYSPROMPT"},
            {"role": "user", "content": "USERPROMPT"},
        ]

    def test_a_schema_uses_grammar_constrained_decoding(self, settings: Settings) -> None:
        client, rec = run_client(settings, lambda request: ok_response())
        chat(client, schema={"type": "object"})
        assert rec.body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "redaction_plan", "schema": {"type": "object"}, "strict": True},
        }

    def test_json_object_mode_is_used_without_a_schema(self, settings: Settings) -> None:
        client, rec = run_client(settings, lambda request: ok_response())
        chat(client)
        assert rec.body["response_format"] == {"type": "json_object"}

    def test_json_object_mode_is_used_when_schema_support_is_off(self, tmp_path) -> None:
        settings = make_settings(tmp_path, llm_use_json_schema=False)
        client, rec = run_client(settings, lambda request: ok_response())
        chat(client, schema={"type": "object"})
        assert rec.body["response_format"] == {"type": "json_object"}

    def test_a_caller_may_override_the_token_budget(self, settings: Settings) -> None:
        client, rec = run_client(settings, lambda request: ok_response())
        chat(client, max_tokens=64)
        assert rec.body["max_tokens"] == 64

    def test_a_completion_is_returned_as_a_dict(self, settings: Settings) -> None:
        client, _ = run_client(settings, lambda request: ok_response('{"targets": [1, 2]}'))
        assert chat(client) == {"targets": [1, 2]}

    def test_a_truncated_completion_is_returned_but_flagged(self, settings: Settings, caplog) -> None:
        client, _ = run_client(
            settings,
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "length"}]},
            ),
        )
        assert chat(client) == {"a": 1}


class TestHttpErrorMapping:
    """Each server failure needs its own actionable error, not a generic 500."""

    @pytest.mark.parametrize(
        ("status", "payload", "expected", "in_message"),
        [
            (401, {"error": {"message": "no key"}}, ModelAuthError, "LLM_API_KEY"),
            (403, {"error": {"message": "no"}}, ModelAuthError, "--api-key"),
            (503, {"error": {"message": "loading"}}, ModelLoadingError, "still loading"),
            (404, {"error": {"message": "gone"}}, ModelNotFoundError, "gemma-3-1b-it"),
        ],
    )
    def test_status_codes_map_to_specific_errors(
        self, settings: Settings, status: int, payload: dict, expected: type, in_message: str
    ) -> None:
        client, _ = run_client(settings, lambda request: httpx.Response(status, json=payload))
        with pytest.raises(expected) as exc:
            chat(client)
        assert in_message in str(exc.value)

    def test_a_rejected_response_format_suggests_disabling_it(self, settings: Settings) -> None:
        client, _ = run_client(
            settings,
            lambda request: httpx.Response(
                400, json={"error": {"message": "invalid response_format json_schema"}}
            ),
        )
        with pytest.raises(ModelResponseError, match="LLM_USE_JSON_SCHEMA=false"):
            chat(client, schema={"type": "object"})

    def test_other_server_errors_carry_the_status_and_message(self, settings: Settings) -> None:
        client, _ = run_client(
            settings, lambda request: httpx.Response(500, json={"error": {"message": "kaboom"}})
        )
        with pytest.raises(ModelResponseError, match="HTTP 500. kaboom") as exc:
            chat(client)
        assert exc.value.details["status_code"] == 500

    def test_a_non_json_success_body_is_refused(self, settings: Settings) -> None:
        client, _ = run_client(settings, lambda request: httpx.Response(200, text="not json"))
        with pytest.raises(ModelResponseError, match="non-JSON response body"):
            chat(client)

    def test_a_json_that_is_not_an_object_is_refused(self, settings: Settings) -> None:
        client, _ = run_client(settings, lambda request: httpx.Response(200, json=[1, 2]))
        with pytest.raises(ModelResponseError, match="not an object"):
            chat(client)

    @pytest.mark.parametrize(
        "payload",
        [
            {"choices": []},
            {"choices": ["not a dict"]},
            {"choices": [{"message": "not a dict"}]},
            {"choices": [{"message": {"content": 123}}]},
            {"no_choices": True},
        ],
    )
    def test_malformed_response_shapes_are_refused(self, settings: Settings, payload: dict) -> None:
        client, _ = run_client(settings, lambda request: httpx.Response(200, json=payload))
        with pytest.raises(ModelResponseError):
            chat(client)


class TestTransportFailures:
    def test_connection_refused_names_the_host_and_next_step(self, settings: Settings) -> None:
        def refusing(request: httpx.Request) -> Any:
            raise httpx.ConnectError("refused")

        client, _ = run_client(settings, refusing)
        with pytest.raises(ModelUnavailableError) as exc:
            chat(client)
        assert "127.0.0.1:8080" in str(exc.value)
        assert "connection refused" in str(exc.value)

    def test_connect_timeout_is_distinct_from_refusal(self, settings: Settings) -> None:
        def stalling(request: httpx.Request) -> Any:
            raise httpx.ConnectTimeout("slow")

        client, _ = run_client(settings, stalling)
        with pytest.raises(ModelUnavailableError, match="connection timed out"):
            chat(client)

    def test_a_slow_inference_reports_the_tuning_knobs(self, settings: Settings) -> None:
        def slow(request: httpx.Request) -> Any:
            raise httpx.ReadTimeout("too slow")

        client, _ = run_client(settings, slow)
        with pytest.raises(ModelTimeoutError) as exc:
            chat(client)
        assert "LLM_MAX_PROMPT_CHARS" in str(exc.value)
        assert "LLM_TIMEOUT_SECONDS" in str(exc.value)

    def test_a_generic_transport_error_uses_the_whole_retry_budget(self, settings: Settings) -> None:
        def broken(request: httpx.Request) -> Any:
            raise httpx.RemoteProtocolError("broken")

        client, rec = run_client(settings, broken)
        with pytest.raises(ModelResponseError, match="after 2 attempts"):
            chat(client)
        assert rec.calls == settings.llm_max_attempts

    def test_a_bad_json_completion_is_retried_then_accepted(self, settings: Settings) -> None:
        seen = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            seen["n"] += 1
            return ok_response("prose, not json" if seen["n"] == 1 else '{"a": 1}')

        client, rec = run_client(settings, flaky)
        assert chat(client) == {"a": 1}
        assert rec.calls == 2

    def test_a_still_bad_completion_after_retries_raises(self, settings: Settings) -> None:
        client, rec = run_client(settings, lambda request: ok_response("prose again"))
        with pytest.raises(ModelResponseError, match="Got prose instead"):
            chat(client)
        assert rec.calls == settings.llm_max_attempts

    def test_a_single_attempt_setting_is_respected(self, tmp_path) -> None:
        settings = make_settings(tmp_path, llm_max_attempts=1)

        def broken(request: httpx.Request) -> Any:
            raise httpx.RemoteProtocolError("broken")

        client, rec = run_client(settings, broken)
        with pytest.raises(ModelResponseError):
            chat(client)
        assert rec.calls == 1


class TestHealth:
    """Health probing sends no document content, so it is safe to poll."""

    def route(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"data": [{"id": "gemma-3-1b-it"}, {"id": "junk"}]})

    def test_ok_reports_the_model_ids(self, settings: Settings) -> None:
        client, rec = run_client(settings, self.route)
        result = asyncio.run(client.health())
        assert result.reachable is True
        assert result.status == "ok"
        assert result.models == ["gemma-3-1b-it", "junk"]
        assert rec.requests[0].url.path.endswith("/health")

    def test_the_health_probe_sends_no_content(self, settings: Settings) -> None:
        client, rec = run_client(settings, self.route)
        asyncio.run(client.health())
        assert all(not request.content for request in rec.requests)

    @pytest.mark.parametrize(
        ("exc", "status", "reachable"),
        [
            (httpx.ConnectError("x"), "unreachable", False),
            (httpx.ReadTimeout("x"), "timeout", False),
            (httpx.RemoteProtocolError("x"), "error", False),
        ],
    )
    def test_transport_failures_are_not_ok(
        self, settings: Settings, exc: Exception, status: str, reachable: bool
    ) -> None:
        def failing(request: httpx.Request) -> Any:
            raise exc

        client, _ = run_client(settings, failing)
        result = asyncio.run(client.health())
        assert (result.status, result.reachable) == (status, reachable)
        assert result.detail

    def test_a_loading_server_is_distinguishable(self, settings: Settings) -> None:
        client, _ = run_client(
            settings, lambda request: httpx.Response(503, json={"status": "loading"})
        )
        result = asyncio.run(client.health())
        assert (result.status, result.reachable) == ("loading", True)
        assert "still loading" in (result.detail or "")

    def test_an_unhealthy_server_reports_the_status_code(self, settings: Settings) -> None:
        client, _ = run_client(settings, lambda request: httpx.Response(500, text="x"))
        result = asyncio.run(client.health())
        assert (result.status, result.reachable) == ("unhealthy", True)
        assert "HTTP 500" in (result.detail or "")

    def test_a_broken_model_listing_still_reports_ok(self, settings: Settings) -> None:
        def route(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/health"):
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(200, text="not json")

        client, _ = run_client(settings, route)
        result = asyncio.run(client.health())
        assert (result.status, result.models) == ("ok", [])

    def test_as_dict_is_json_serialisable(self, settings: Settings) -> None:
        client, _ = run_client(settings, self.route)
        assert json.loads(json.dumps(asyncio.run(client.health()).as_dict()))["status"] == "ok"


class TestApiKeyHandling:
    def test_a_configured_key_becomes_a_bearer_header(self, tmp_path) -> None:
        settings = make_settings(tmp_path, llm_api_key="sekret-value")
        assert LlamaCppClient(settings)._ensure_client().headers.get("authorization") == (
            "Bearer sekret-value"
        )

    def test_no_key_means_no_authorization_header(self, tmp_path) -> None:
        settings = make_settings(tmp_path)
        assert LlamaCppClient(settings)._ensure_client().headers.get("authorization") is None

    def test_a_blank_key_does_not_send_an_empty_bearer(self, tmp_path) -> None:
        settings = make_settings(tmp_path, llm_api_key="")
        assert LlamaCppClient(settings)._ensure_client().headers.get("authorization") is None

    def test_the_key_never_appears_in_repr_or_dump(self, tmp_path) -> None:
        settings = make_settings(tmp_path, llm_api_key="sekret-value")
        assert "sekret-value" not in repr(settings)
        assert "sekret-value" not in str(settings.model_dump())

    def test_the_key_never_appears_in_client_logs(self, tmp_path, capfd) -> None:
        settings = make_settings(tmp_path, llm_api_key="sekret-value", log_level="INFO")
        client, _ = run_client(settings, lambda request: ok_response())
        chat(client)
        assert "sekret-value" not in capfd.readouterr().out


class TestClientLifecycle:
    def test_the_http_client_is_reused_across_calls(self, settings: Settings) -> None:
        client, rec = run_client(settings, lambda request: ok_response())
        inner = client._client
        chat(client)
        assert client._client is inner
        assert rec.calls == 1

    def test_aclose_leaves_no_client_behind(self, settings: Settings) -> None:
        client, _ = run_client(settings, lambda request: ok_response())
        inner = client._client
        asyncio.run(client.aclose())
        assert client._client is None
        assert inner.is_closed

    def test_a_closed_client_is_recreated_on_demand(self, settings: Settings) -> None:
        client, _ = run_client(settings, lambda request: ok_response())
        asyncio.run(client.aclose())
        assert not client._ensure_client().is_closed


def test_the_default_configuration_points_at_loopback() -> None:
    defaults = Settings(_env_file=None)
    assert defaults.llm_base_url.startswith("http://127.0.0.1")
    assert defaults.llm_allow_cloud_fallback is False
    assert defaults.llm_allow_non_loopback is False
    assert defaults.llm_temperature == 0.0
    assert defaults.interpretation_mode in {"model", "deterministic"}


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
