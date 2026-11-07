"""Interchangeable inference transports. No SDK, download, or training side effects."""

from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.error
import ipaddress
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Callable
from ..types import Completion, BudgetExceeded
from ..utils.core import canonical, digest, token_upper


class ProviderError(RuntimeError):
    """Safe error categories, never remote response bodies or credentials."""


@dataclass
class RemoteBudgetLedger:
    """Shared by every provider instance in one experiment, including label pairs."""

    spent_usd: float = 0.0
    calls: int = 0


@dataclass
class HTTPSettings:
    base_url: str
    model_id: str
    timeout: float = 30.0
    max_tokens: int = 128
    temperature: float = 0.0
    retries: int = 1
    model_revision: str | None = None
    weights_sha256: str | None = None

    def __post_init__(self):
        p = urlsplit(self.base_url)
        if (
            p.scheme not in {"http", "https"}
            or not p.hostname
            or p.username
            or p.password
            or p.query
            or p.fragment
        ):
            raise ValueError(
                "invalid endpoint; credentials/query/fragment are forbidden in URLs"
            )
        if (
            self.timeout <= 0
            or not 1 <= self.max_tokens <= 8192
            or not 0 <= self.retries <= 3
        ):
            raise ValueError("invalid timeout, max_tokens, or retry count")
        if not self.model_id or not 0 <= self.temperature <= 2:
            raise ValueError("invalid model or temperature")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("HTTP redirect refused")


def http_transport(url, payload, headers, timeout):
    request = urllib.request.Request(
        url,
        data=canonical(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    # No system proxy for local connections; remote proxies must be configured outside this client.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read(2_000_001))


class LocalOpenAICompatibleProvider:
    kind = "local_llm"

    def __init__(self, settings: HTTPSettings, transport: Callable = http_transport):
        host = urlsplit(settings.base_url).hostname
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == "localhost"
        if not local:
            raise ValueError(
                "local provider accepts literal loopback or localhost only"
            )
        self.settings, self.transport = settings, transport
        self.calls = self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0

    @property
    def identity(self):
        return {
            "provider": self.kind,
            "model_id": self.settings.model_id,
            "model_revision": self.settings.model_revision,
            "weights_sha256": self.settings.weights_sha256,
            "endpoint_hash": digest(self.settings.base_url),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }

    def _headers(self):
        return {}

    def _reserve(self, payload):
        self.calls += 1

    def _settle(self, result, payload):
        usage = result.get("usage") or {}
        estimated = not ("prompt_tokens" in usage and "completion_tokens" in usage)
        incoming = usage.get("prompt_tokens", token_upper(canonical(payload)))
        outgoing = usage.get(
            "completion_tokens",
            token_upper(result["choices"][0]["message"].get("content") or ""),
        )
        if (
            type(incoming) is not int
            or type(outgoing) is not int
            or min(incoming, outgoing) < 0
        ):
            raise ProviderError("invalid usage counters")
        self.input_tokens += incoming
        self.output_tokens += outgoing
        return incoming, outgoing, estimated, 0.0

    def generate(self, messages, schemas, seed=42):
        payload = {
            "model": self.settings.model_id,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "seed": seed,
        }
        start, attempted = time.perf_counter(), 0
        for attempt in range(self.settings.retries + 1):
            self._reserve(payload)
            attempted += 1
            try:
                result = self.transport(
                    self.settings.base_url.rstrip("/") + "/chat/completions",
                    payload,
                    self._headers(),
                    self.settings.timeout,
                )
                if not isinstance(result, dict):
                    raise ProviderError("invalid response shape")
                text = result["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise ProviderError("response content must be text JSON")
                incoming, outgoing, estimated, cost = self._settle(result, payload)
                return Completion(
                    text,
                    incoming,
                    outgoing,
                    time.perf_counter() - start,
                    attempted,
                    estimated,
                    cost,
                    result.get("id"),
                    result.get("model", self.settings.model_id),
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if attempt < self.settings.retries and retryable:
                    time.sleep(min(0.1 * 2**attempt, 0.5))
                    continue
                raise ProviderError(
                    "transport failed or timed out (response body withheld)"
                ) from None
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                raise ProviderError("malformed provider response") from None


class RemoteOpenAICompatibleProvider(LocalOpenAICompatibleProvider):
    kind = "remote_llm"

    def __init__(self, *, allow_remote=False, transport=http_transport, ledger=None):
        if not allow_remote or os.environ.get("EG_REMOTE_ENABLE", "").lower() != "true":
            raise ProviderError(
                "remote execution disabled: explicit authorization and EG_REMOTE_ENABLE required"
            )

        def required(name):
            value = os.environ.get(name, "")
            if not value:
                raise ValueError(f"missing environment variable {name}")
            return value

        settings = HTTPSettings(
            required("EG_REMOTE_BASE_URL"), required("EG_REMOTE_MODEL")
        )
        if urlsplit(settings.base_url).scheme != "https":
            raise ValueError("remote endpoint must use HTTPS")
        self._api_key = required("EG_REMOTE_API_KEY")
        self.budget = float(required("EG_REMOTE_BUDGET_USD"))
        self.input_price = float(required("EG_REMOTE_INPUT_USD_PER_MILLION"))
        self.output_price = float(required("EG_REMOTE_OUTPUT_USD_PER_MILLION"))
        if not all(
            0 < x < float("inf")
            for x in (self.budget, self.input_price, self.output_price)
        ):
            raise ValueError(
                "remote budget and both token prices must be finite and positive"
            )
        self.settings, self.transport = settings, transport
        self.calls = self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0
        self._reservation = 0.0
        self.ledger = ledger if ledger is not None else RemoteBudgetLedger()

    def _headers(self):
        return {"Authorization": "Bearer " + self._api_key}

    def _reserve(self, payload):
        worst = (
            (token_upper(canonical(payload)) + 1024) * self.input_price
            + self.settings.max_tokens * self.output_price
        ) / 1_000_000
        if self.ledger.calls >= 400 or self.ledger.spent_usd + worst > self.budget:
            raise BudgetExceeded("remote cumulative request/cost cap reached")
        self._reservation = worst
        self.cost_usd += worst  # failures and absent usage keep the reservation charged
        self.ledger.spent_usd += worst
        self.ledger.calls += 1
        self.calls += 1

    def _settle(self, result, payload):
        usage = result.get("usage", {})
        if "prompt_tokens" not in usage or "completion_tokens" not in usage:
            raise ProviderError(
                "remote usage missing; reservation retained and run aborted"
            )
        incoming, outgoing, estimated, _ = super()._settle(result, payload)
        actual = (
            incoming * self.input_price + outgoing * self.output_price
        ) / 1_000_000
        self.cost_usd += actual - self._reservation
        self.ledger.spent_usd += actual - self._reservation
        if self.ledger.spent_usd > self.budget:
            raise BudgetExceeded(
                "reported usage exceeds reserved cap; execution stopped"
            )
        return incoming, outgoing, estimated, actual


class MockProvider:
    """Deterministic ENGINEERING fixture. Not an LLM and never reported as one.

    Default behavior only interprets public goal/observations, never task IDs or gold.
    A supplied script is explicitly a unit-test fixture, not an autonomous policy.
    """

    kind = "mock_engineering"

    def __init__(self, script=None):
        self.script = list(script) if script is not None else None
        self.position = self.calls = self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0

    @property
    def identity(self):
        return {
            "provider": self.kind,
            "model_id": "deterministic-public-goal-fixture-v1",
            "script_hash": digest(
                [
                    type(x).__name__ if isinstance(x, Exception) else x
                    for x in self.script
                ]
            )
            if self.script is not None
            else None,
        }

    def generate(self, messages, schemas, seed=42):
        self.calls += 1
        if self.script is not None:
            if self.position >= len(self.script):
                text = canonical({"final": "fixture exhausted"})
            else:
                value = self.script[self.position]
                self.position += 1
                if isinstance(value, Exception):
                    raise value
                text = value if isinstance(value, str) else canonical(value)
        else:
            task = next(
                json.loads(m["content"]) for m in messages if m["role"] == "user"
            )
            goal, visible = task["goal"].lower(), task["visible_state"]
            key = next(
                (
                    k
                    for k, words in {
                        "wifi": ("wifi", "wi-fi"),
                        "cellular": ("cellular",),
                        "location_service": ("location",),
                        "low_battery_mode": ("battery",),
                    }.items()
                    if any(word in goal for word in words)
                ),
                None,
            )
            from ..environments.toolsandbox import GETTERS, SETTERS

            names = {s["name"] for s in schemas}
            action = {"final": "Engineering fixture has no action for this goal"}
            if key:
                desired = not any(w in goal for w in ("disable", "turn off"))
                if visible.get(key) == desired:
                    action = {"final": "Requested setting is now satisfied"}
                elif (
                    key != "low_battery_mode"
                    and desired
                    and visible.get("low_battery_mode") is None
                    and "get_low_battery_mode_status" in names
                ):
                    action = {"tool": "get_low_battery_mode_status", "arguments": {}}
                elif (
                    key != "low_battery_mode"
                    and desired
                    and visible.get("low_battery_mode") is True
                ):
                    action = {
                        "tool": "set_low_battery_mode_status",
                        "arguments": {"on": False},
                    }
                else:
                    getter = next(n for n, k in GETTERS.items() if k == key)
                    setter = next(n for n, k in SETTERS.items() if k == key)
                    action = (
                        {"tool": getter, "arguments": {}}
                        if visible.get(key) is None and getter in names
                        else {"tool": setter, "arguments": {"on": desired}}
                    )
            text = canonical(action)
        incoming, outgoing = token_upper(canonical(messages)), token_upper(text)
        self.input_tokens += incoming
        self.output_tokens += outgoing
        return Completion(
            text,
            incoming,
            outgoing,
            0.0,
            usage_estimated=True,
            model_id=self.identity["model_id"],
        )
