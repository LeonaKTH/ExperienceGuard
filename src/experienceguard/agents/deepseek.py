"""DeepSeek non-thinking JSON transport and durable request reservations."""

from pathlib import Path
import json
import math
import os
import time
import urllib.error
from filelock import FileLock
from experienceguard.agents.providers import HTTPSettings, ProviderError, http_transport
from experienceguard.types import BudgetExceeded, Completion
from experienceguard.utils.core import canonical, digest, token_upper, write_json


class Ledger:
    """Single serialized batch. Failed/ambiguous requests retain their worst-case charge."""

    def __init__(self, path, *, budget, request_cap, binding):
        if (
            not math.isfinite(budget)
            or budget <= 0
            or type(request_cap) is not int
            or not 1 <= request_cap <= 400
        ):
            raise ValueError("finite global budget and request cap required")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget, self.request_cap, self.binding = budget, request_cap, binding

    def _state(self):
        if not self.path.exists():
            return {
                "budget": self.budget,
                "request_cap": self.request_cap,
                "binding": self.binding,
                "entries": [],
            }
        value = json.loads(self.path.read_text(encoding="utf8"))
        if (value["budget"], value["request_cap"], value["binding"]) != (
            self.budget,
            self.request_cap,
            self.binding,
        ):
            raise ValueError("ledger cannot be rebound or reset")
        return value

    def _write(self, value):
        tmp = self.path.with_suffix(".pending")
        write_json(tmp, value)
        tmp.replace(self.path)

    def reserve(self, amount, episode, episode_cap):
        if (
            not math.isfinite(amount)
            or amount <= 0
            or not math.isfinite(episode_cap)
            or not 0 < episode_cap <= self.budget
        ):
            raise ValueError("invalid reservation")
        with FileLock(str(self.path) + ".lock", timeout=2):
            state = self._state()
            spent = sum(e["charge"] for e in state["entries"])
            local = sum(
                e["charge"] for e in state["entries"] if e["episode"] == episode
            )
            if (
                len(state["entries"]) >= self.request_cap
                or spent + amount > self.budget
                or local + amount > episode_cap
            ):
                raise BudgetExceeded("persistent batch/episode cap")
            index = len(state["entries"])
            state["entries"].append(
                {
                    "index": index,
                    "episode": episode,
                    "reserved": amount,
                    "charge": amount,
                    "status": "reserved_or_ambiguous",
                }
            )
            self._write(state)
            return index

    def settle(self, index, actual):
        if not math.isfinite(actual) or actual < 0:
            raise ProviderError("invalid settlement")
        with FileLock(str(self.path) + ".lock", timeout=2):
            state = self._state()
            item = state["entries"][index]
            if item["status"] != "reserved_or_ambiguous":
                raise ProviderError("duplicate settlement")
            item.update(charge=actual, status="settled")
            self._write(state)
            if actual > item["reserved"]:
                raise BudgetExceeded(
                    "provider usage exceeded reservation; stop and reconcile invoice"
                )


class DeepSeekJSONProvider:
    kind = "remote_llm"

    def __init__(
        self, config, ledger, episode, *, allow_remote=False, transport=http_transport
    ):
        if allow_remote is not True or os.environ.get("EG_V2_REMOTE_ENABLE") != "true":
            raise ProviderError("explicit remote opt-in required")
        if (
            config["base_url"] != "https://api.deepseek.com"
            or config["model"] != "deepseek-flash"
        ):
            raise ProviderError("unreviewed provider/model")
        self.key = os.environ.get("EG_V2_API_KEY", "")
        if not self.key:
            raise ProviderError("missing key in environment")
        self.config, self.ledger, self.episode = config, ledger, episode
        self.settings = HTTPSettings(
            config["base_url"],
            config["model"],
            max_tokens=config["max_output_tokens"],
            retries=0,
            timeout=30,
            temperature=0,
        )
        self.transport = transport
        self.calls = self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0

    @property
    def identity(self):
        return {
            "provider": self.kind,
            "model_id": self.settings.model_id,
            "advertised_version": self.config["advertised_version"],
            "alias_is_mutable": True,
            "seed_supported": False,
            "cost_basis": "peak/cache-miss upper estimate, not exact invoice",
            "temperature": 0,
            "thinking": "disabled",
            "response_format": "json_object",
            "pricing_reference_sha": digest(self.config["pricing_reference"]),
            "endpoint_hash": digest(self.settings.base_url),
        }

    def guard(self, value):
        """Refuse credential echoes before the Agent persists any request or response."""
        if self.key in canonical(value):
            raise ProviderError("credential detected; content withheld")

    def generate(self, messages, schemas, seed=42):
        # No unsupported seed or native tool_calls. Host validates action schemas after JSON parsing.
        payload = {
            "model": self.settings.model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.settings.max_tokens,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        self.guard(payload)
        ip, op = (
            self.config["input_usd_per_million"],
            self.config["output_usd_per_million"],
        )
        if not all(math.isfinite(v) and v > 0 for v in [ip, op]):
            raise ValueError("invalid official price ceiling")
        reserve = (
            (token_upper(canonical(payload)) + 1024) * ip
            + self.settings.max_tokens * op
        ) / 1e6
        index = self.ledger.reserve(
            reserve, self.episode, self.config["episode_cap_usd"]
        )
        self.calls += 1
        self.cost_usd += reserve
        start = time.perf_counter()
        try:
            result = self.transport(
                self.settings.base_url + "/chat/completions",
                payload,
                {"Authorization": "Bearer " + self.key},
                self.settings.timeout,
            )
            self.guard(result)
            usage = result.get("usage") or {}
            incoming, outgoing = (
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
            text = result["choices"][0]["message"]["content"]
            if any(
                type(n) is not int or n < 0 for n in [incoming, outgoing]
            ) or not isinstance(text, str):
                raise ProviderError(
                    "missing or invalid usage/content; reservation retained"
                )
            actual = (
                incoming * ip + outgoing * op
            ) / 1e6  # peak/cache-miss ceiling, not an invoice claim
            self.cost_usd += actual - reserve
            self.input_tokens += incoming
            self.output_tokens += outgoing
            self.ledger.settle(index, actual)
            return Completion(
                text,
                incoming,
                outgoing,
                time.perf_counter() - start,
                1,
                False,
                actual,
                result.get("id"),
                result.get("model"),
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
        ):
            raise ProviderError(
                "transport/response failure; reservation retained; response body withheld"
            ) from None


def configured_provider(app, plan, *, transport=http_transport):
    """Application bridge for the V2 provider; no network during configuration."""
    from .development import AgentLimits

    if os.environ.get("EG_V2_REMOTE_ENABLE") != "true":
        raise ProviderError("remote execution disabled; EG_V2_REMOTE_ENABLE required")
    if not os.environ.get("EG_V2_API_KEY"):
        raise ProviderError("missing EG_V2_API_KEY in process environment")
    path = os.environ.get("EG_DEEPSEEK_CONFIG")
    if not path:
        raise ProviderError("missing reviewed EG_DEEPSEEK_CONFIG file")
    try:
        config = json.loads(Path(path).read_text(encoding="utf8"))
        allowed = {
            "base_url", "model", "advertised_version", "pricing_reference",
            "input_usd_per_million", "output_usd_per_million", "total_cap_usd",
            "total_request_cap", "episode_cap_usd", "max_output_tokens", "limits"
        }
        if set(config) != allowed:
            raise ValueError("configuration fields differ from the example")
        if config["base_url"] != "https://api.deepseek.com" or config["model"] != "deepseek-flash":
            raise ValueError("only the reviewed official DeepSeek endpoint/alias is accepted")
        for key in ("input_usd_per_million", "output_usd_per_million", "total_cap_usd", "episode_cap_usd"):
            if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError("positive finite prices and budgets required")
        if config["episode_cap_usd"] > config["total_cap_usd"]:
            raise ValueError("episode budget exceeds total budget")
        if not isinstance(config["pricing_reference"], str) or not config["pricing_reference"].strip() or config["pricing_reference"].startswith("REPLACE"):
            raise ValueError("check official current prices and record the reference/date")
        if not isinstance(config["advertised_version"], str) or not config["advertised_version"].strip():
            raise ValueError("record alias identity or explicitly mark it unverified")
        limits = AgentLimits(**config["limits"])
        HTTPSettings(config["base_url"], config["model"], max_tokens=config["max_output_tokens"], retries=0)
    except (OSError, KeyError, TypeError, ValueError):
        raise ProviderError("invalid/missing reviewed DeepSeek config; check the template, prices and limits") from None
    ledger = Ledger(app.data_dir / "remote-budget.json", budget=config["total_cap_usd"],
                    request_cap=config["total_request_cap"], binding=digest(config))
    ledger._state()  # Reject configuration changes before any remote request.
    return DeepSeekJSONProvider(config, ledger, plan["plan_id"], allow_remote=True,
                                transport=transport), limits
