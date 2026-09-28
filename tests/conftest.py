"""Shared fixtures: a stand-in for the OpenRouter client, so no test needs an
API key or network access."""

from types import SimpleNamespace

import httpx2
import openai
import pytest

import summarizer

_REQUEST = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


class FakeClient:
    """Records every completion request and replies with ``content`` or raises ``error``."""

    def __init__(self):
        self.content = "סיכום לדוגמה."
        self.error = None
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def fail_with_rate_limit(self):
        response = httpx2.Response(429, request=_REQUEST)
        self.error = openai.RateLimitError("rate limited", response=response, body=None)

    def fail_with_connection_error(self, message="Connection error."):
        self.error = openai.APIConnectionError(message=message, request=_REQUEST)


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(summarizer, "_create_client", lambda: client)
    return client
