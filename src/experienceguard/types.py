"""Public policy inputs are deliberately separate from evaluator capabilities."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
from .utils.core import digest

STATE_KEYS = ("cellular", "wifi", "location_service", "low_battery_mode")


@dataclass(frozen=True)
class PublicContext:
    goal: str
    visible_state: dict[str, bool | None]
    tools: tuple[str, ...]
    schema_hash: str
    observations: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        if set(self.visible_state) - set(STATE_KEYS):
            raise ValueError("public state contains non-whitelisted fields")
        if any(
            v is not None and type(v) is not bool for v in self.visible_state.values()
        ):
            raise ValueError("public state values must be bool or unknown")


@dataclass(frozen=True)
class Experience:
    experience_id: str
    source_task: str
    source_family: str
    goal: str
    observations: list[dict]
    tool_calls: list[dict]
    raw_trace: list[dict]
    normalized: dict
    preconditions: dict[str, bool]
    unknown_mask: dict[str, bool]
    schema_hash: str
    source_run_sha: str
    feedback: dict
    created_at: str
    provenance: str = "mock_engineering"
    reliability: float = 1.0

    def __post_init__(self):
        if set(self.preconditions) - set(STATE_KEYS) or any(
            type(v) is not bool for v in self.preconditions.values()
        ):
            raise ValueError("only public Boolean preconditions are allowed")
        if set(self.unknown_mask) != set(STATE_KEYS) or any(
            type(v) is not bool for v in self.unknown_mask.values()
        ):
            raise ValueError("complete Boolean unknown mask required")
        if any(
            self.unknown_mask[k] != (k not in self.preconditions) for k in STATE_KEYS
        ):
            raise ValueError("preconditions/unknown mask disagree")
        if not 0 <= self.reliability <= 1:
            raise ValueError("source reliability must be in [0,1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class Candidate:
    experience: Experience
    retrieval_score: float
    semantic_score: float | None = None
    raw_tokens: int = 0
    normalized_tokens: int = 0


@dataclass(frozen=True)
class Decision:
    strategy: str
    version: str
    experience_id: str | None
    reason: str
    features: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    tokens: int = 0
    evidence: list[dict] = field(default_factory=list)
    rejection: str | None = None
    representation: str = "normalized"

    def to_dict(self) -> dict:
        return asdict(self)


class UnavailablePolicy(RuntimeError):
    """A policy is missing evidence or an acceptable model; never a hidden fallback."""


class BudgetExceeded(RuntimeError):
    """The next operation would cross an explicit run limit."""


@dataclass(frozen=True)
class Action:
    tool: str | None = None
    arguments: dict = field(default_factory=dict)
    final: str | None = None

    @classmethod
    def parse(cls, content: str) -> "Action":
        import json

        def pairs(items):
            obj = {}
            for key, value in items:
                if key in obj:
                    raise ValueError("duplicate JSON key")
                obj[key] = value
            return obj

        def nonfinite(value):
            raise ValueError("non-finite JSON value")

        obj = json.loads(content, object_pairs_hook=pairs, parse_constant=nonfinite)
        if not isinstance(obj, dict):
            raise ValueError("action must be a JSON object")
        if set(obj) == {"final"} and isinstance(obj["final"], str):
            return cls(final=obj["final"])
        if (
            set(obj) == {"tool", "arguments"}
            and isinstance(obj["tool"], str)
            and isinstance(obj["arguments"], dict)
        ):
            return cls(tool=obj["tool"], arguments=obj["arguments"])
        raise ValueError("expected {tool, arguments} or {final}")


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    requests: int = 1
    usage_estimated: bool = False
    cost_usd: float = 0.0
    request_id: str | None = None
    model_id: str | None = None
