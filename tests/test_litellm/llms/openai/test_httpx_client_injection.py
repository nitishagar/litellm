"""
Tests for httpx_client injection into litellm.acompletion().

Covers:
- I1: Caller client is actually used (transport invoked)
- I2: No provider leak (httpx_client absent from optional_params)
- I3: Cache bypass + lifecycle (aclose not called)
- I4: Unsupported provider raises; custom_openai accepted
- I5: Per-call client takes precedence over aclient_session global
- I6: Omitting httpx_client is backwards compatible
"""

import json
import os
import sys
import time
import uuid

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

os.environ.setdefault("OPENAI_API_KEY", "sk-fake-key-for-tests")

import litellm
from litellm.llms.openai.openai import OpenAIChatCompletion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_completion_body(model: str = "gpt-4o-mini") -> bytes:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return json.dumps(payload).encode()


def _make_counting_transport() -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=_chat_completion_body(),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, seen


_MESSAGES = [{"role": "user", "content": "ping"}]


# ---------------------------------------------------------------------------
# I1 — Caller client is actually used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_i1_caller_transport_is_invoked():
    """The caller's mock transport must handle exactly one request."""
    client, seen = _make_counting_transport()

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=_MESSAGES,
        api_key="sk-fake-key",
        httpx_client=client,
    )

    assert len(seen) == 1, f"Expected 1 transport call, got {len(seen)}"
    assert response.choices[0].message.content == "pong"
    await client.aclose()


@pytest.mark.asyncio
async def test_i1_streaming_uses_caller_transport():
    """The caller's transport is used for streaming calls too."""
    seen: list[httpx.Request] = []
    sse_line = (
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":null}]}\n\n'
    )
    done_line = b"data: [DONE]\n\n"

    class _SseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse_line
            yield done_line

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=_SseStream(),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=_MESSAGES,
        stream=True,
        api_key="sk-fake-key",
        httpx_client=client,
    )
    [chunk async for chunk in response]
    assert len(seen) == 1, f"Expected 1 transport call for stream, got {len(seen)}"
    await client.aclose()


# ---------------------------------------------------------------------------
# I2 — No provider leak
# ---------------------------------------------------------------------------


def test_i2_httpx_client_in_all_litellm_params():
    """httpx_client must be in all_litellm_params so filter_out_litellm_params strips it."""
    from litellm.types.utils import all_litellm_params

    assert "httpx_client" in all_litellm_params


def test_i2_httpx_client_not_in_optional_params():
    """httpx_client must not appear in provider-bound optional_params."""
    from litellm.utils import get_optional_params
    from unittest.mock import MagicMock

    optional_params = get_optional_params(
        model="gpt-4o-mini",
        custom_llm_provider="openai",
        httpx_client=MagicMock(),
        temperature=0.5,
    )
    assert "httpx_client" not in optional_params
    assert optional_params.get("temperature") == 0.5


# ---------------------------------------------------------------------------
# I3 — Cache bypass + lifecycle
# ---------------------------------------------------------------------------


def test_i3_cache_bypass_when_httpx_client_supplied():
    """_get_openai_client bypasses both cache get and set when httpx_client is provided."""
    oa = OpenAIChatCompletion()
    litellm.in_memory_llm_clients_cache.flush_cache()

    mock_transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b""))
    injected = httpx.AsyncClient(transport=mock_transport)

    result = oa._get_openai_client(
        is_async=True,
        api_key="sk-fake-key",
        httpx_client=injected,
    )

    assert result is not None
    assert len(litellm.in_memory_llm_clients_cache.cache_dict) == 0, (
        "Cache must be empty after a call with httpx_client (bypass)"
    )


@pytest.mark.asyncio
async def test_i3_aclose_not_called_on_injected_client():
    """litellm must not call aclose() on the injected client."""
    client, _ = _make_counting_transport()
    aclose_calls = 0
    original_aclose = client.aclose

    async def spy_aclose(*args, **kwargs):
        nonlocal aclose_calls
        aclose_calls += 1
        return await original_aclose(*args, **kwargs)

    client.aclose = spy_aclose  # type: ignore[method-assign]

    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=_MESSAGES,
        api_key="sk-fake-key",
        httpx_client=client,
    )

    assert aclose_calls == 0, "litellm must not call aclose() on the injected client"
    await original_aclose()


def test_i3_subsequent_call_without_httpx_client_populates_cache():
    """A call without httpx_client must populate the cache; the injected-client call must not."""
    oa = OpenAIChatCompletion()
    litellm.in_memory_llm_clients_cache.flush_cache()

    mock_transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b""))
    injected = httpx.AsyncClient(transport=mock_transport)

    oa._get_openai_client(
        is_async=True,
        api_key="sk-fake-key",
        httpx_client=injected,
    )
    assert len(litellm.in_memory_llm_clients_cache.cache_dict) == 0, (
        "Injected call must leave cache empty"
    )

    oa._get_openai_client(
        is_async=True,
        api_key="sk-fake-key",
    )
    assert len(litellm.in_memory_llm_clients_cache.cache_dict) == 1, (
        "Call without httpx_client must populate the cache"
    )


# ---------------------------------------------------------------------------
# I4 — Unsupported provider raises; custom_openai does not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_i4_unsupported_provider_raises():
    """Passing httpx_client with a non-OpenAI-compatible provider raises an error naming the limitation."""
    dummy_client = httpx.AsyncClient()
    with pytest.raises(Exception, match="httpx_client is only supported for OpenAI-compatible providers"):
        await litellm.acompletion(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=_MESSAGES,
            api_key="sk-fake-key",
            httpx_client=dummy_client,
        )
    await dummy_client.aclose()


@pytest.mark.asyncio
async def test_i4_custom_openai_provider_accepted():
    """custom_openai is an OpenAI-compatible provider; the guard must not raise."""
    client, seen = _make_counting_transport()

    await litellm.acompletion(
        model="custom_openai/my-model",
        messages=_MESSAGES,
        api_base="https://my.endpoint.example",
        api_key="fake-key",
        httpx_client=client,
    )

    assert len(seen) == 1
    await client.aclose()


def test_i4_guard_fires_synchronously_via_completion_guard():
    """The guard in completion() raises ValueError for non-compatible providers."""
    from litellm.main import _is_openai_compatible_provider

    assert not _is_openai_compatible_provider("anthropic")
    assert not _is_openai_compatible_provider("bedrock")
    assert not _is_openai_compatible_provider("vertex_ai")


def test_i4_sync_completion_raises():
    """Passing httpx_client to the synchronous completion() must raise explicitly (I4 + I7)."""
    dummy_client = httpx.AsyncClient()
    with pytest.raises(Exception, match="httpx_client is only supported for acompletion()"):
        litellm.completion(
            model="gpt-4o-mini",
            messages=_MESSAGES,
            api_key="sk-fake-key",
            httpx_client=dummy_client,
        )


def test_i4_json_provider_accepted():
    """A JSON-registered provider must be accepted by _is_openai_compatible_provider."""
    from litellm.main import _is_openai_compatible_provider
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    JSONProviderRegistry.load()
    known_json_providers = JSONProviderRegistry.list_providers()
    if not known_json_providers:
        pytest.skip("No JSON providers registered in this environment")
    assert _is_openai_compatible_provider(known_json_providers[0]), (
        f"JSON-registered provider '{known_json_providers[0]}' must be accepted by _is_openai_compatible_provider"
    )


# ---------------------------------------------------------------------------
# I5 — Per-call client takes precedence over aclient_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_i5_per_call_client_overrides_aclient_session():
    """When httpx_client is supplied, it takes priority over litellm.aclient_session."""
    global_seen: list[httpx.Request] = []
    per_call_seen: list[httpx.Request] = []

    def global_handler(request: httpx.Request) -> httpx.Response:
        global_seen.append(request)
        return httpx.Response(200, headers={"content-type": "application/json"}, content=_chat_completion_body())

    def per_call_handler(request: httpx.Request) -> httpx.Response:
        per_call_seen.append(request)
        return httpx.Response(200, headers={"content-type": "application/json"}, content=_chat_completion_body())

    global_client = httpx.AsyncClient(transport=httpx.MockTransport(global_handler))
    per_call_client = httpx.AsyncClient(transport=httpx.MockTransport(per_call_handler))

    original_aclient_session = litellm.aclient_session
    litellm.aclient_session = global_client
    try:
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=_MESSAGES,
            api_key="sk-fake-key",
            httpx_client=per_call_client,
        )
    finally:
        litellm.aclient_session = original_aclient_session

    assert len(global_seen) == 0, "Global aclient_session must not be used when httpx_client is supplied"
    assert len(per_call_seen) == 1, "Per-call client must be used"

    await global_client.aclose()
    await per_call_client.aclose()


# ---------------------------------------------------------------------------
# I6 — Omitting httpx_client is identical to today (backwards compat)
# ---------------------------------------------------------------------------


def test_i6_httpx_client_param_has_none_default():
    """acompletion and completion must have httpx_client=None as default."""
    import inspect

    sig_acompletion = inspect.signature(litellm.acompletion)
    assert "httpx_client" in sig_acompletion.parameters
    assert sig_acompletion.parameters["httpx_client"].default is None

    sig_completion = inspect.signature(litellm.completion)
    assert "httpx_client" in sig_completion.parameters
    assert sig_completion.parameters["httpx_client"].default is None


# ---------------------------------------------------------------------------
# Guard predicate unit tests (_is_openai_compatible_provider)
# ---------------------------------------------------------------------------


def test_is_openai_compatible_provider_accepted():
    from litellm.main import _is_openai_compatible_provider

    assert _is_openai_compatible_provider("openai")
    assert _is_openai_compatible_provider("custom_openai")
    assert _is_openai_compatible_provider("deepinfra")
    assert _is_openai_compatible_provider("together_ai")
    assert _is_openai_compatible_provider("perplexity")


def test_is_openai_compatible_provider_rejected():
    from litellm.main import _is_openai_compatible_provider

    assert not _is_openai_compatible_provider("anthropic")
    assert not _is_openai_compatible_provider("bedrock")
    assert not _is_openai_compatible_provider("vertex_ai")
    assert not _is_openai_compatible_provider("gemini")
