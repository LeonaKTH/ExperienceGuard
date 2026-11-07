"""Development incident records. No ToolSandbox state or evaluator capabilities."""

from dataclasses import dataclass, asdict
from ..utils.core import digest, canonical

DOMAIN = "development-ci-v1"
TOOLS = (
    "list_repository",
    "read_config",
    "parse_ci_log",
    "locate_source",
    "inspect_python_exports",
    "compare_conditions",
)
SCHEMA = digest(
    {"domain": DOMAIN, "tools": TOOLS, "conditions": ["eq", "version", "unknown"]}
)


@dataclass(frozen=True)
class DevelopmentContext:
    goal: str
    visible_state: dict
    tools: tuple = TOOLS
    schema_hash: str = SCHEMA


@dataclass(frozen=True)
class DevelopmentExperience:
    experience_id: str
    source_task: str
    source_family: str
    goal: str
    observations: list
    tool_calls: list
    raw_trace: list
    normalized: dict
    preconditions: dict
    unknown_mask: dict
    schema_hash: str
    source_run_sha: str
    feedback: dict
    created_at: str
    repository: dict
    environment: dict
    error_evidence: list
    repair: dict
    invalidations: list
    validation: dict
    provenance: str
    supersedes: str | None = None
    reliability: float = 1.0
    domain: str = DOMAIN

    def __post_init__(self):
        if self.domain != DOMAIN or self.schema_hash != SCHEMA:
            raise ValueError("not a development memory record")
        if set(self.preconditions) != set(self.unknown_mask):
            raise ValueError("condition mask mismatch")
        for key, condition in self.preconditions.items():
            if not isinstance(key, str) or condition.get("op") not in {
                "eq",
                "version",
                "unknown",
            }:
                raise ValueError("invalid development condition")
            if type(self.unknown_mask[key]) is not bool or self.unknown_mask[key] != (
                condition["op"] == "unknown"
            ):
                raise ValueError("unknown source condition must remain unknown")
        if (
            self.provenance not in {"controlled_fixture_verified", "user_attested"}
            or not 0 <= self.reliability <= 1
        ):
            raise ValueError("unverified development memory provenance")

    def to_dict(self):
        return asdict(self)

    @property
    def content_hash(self):
        return digest(self.to_dict())

    @property
    def evidence_quality(self):
        return {
            "kind": self.validation.get("kind", self.provenance),
            "independently_executed": self.provenance == "controlled_fixture_verified",
            "scope": "synthetic owned fixture"
            if self.provenance == "controlled_fixture_verified"
            else "user-attested external action; not independently verified",
            "llm_inference_is_verification": False,
        }


class DevelopmentNormalizer:
    """Keep original evidence immutable; render source-specific paths as historical only."""

    version = "development-evidence-v1"

    def render(self, experience, representation="normalized"):
        if representation == "raw":
            return canonical(
                {
                    "historical_goal": experience.goal,
                    "evidence": experience.error_evidence,
                    "repair": experience.repair,
                    "validation": experience.validation,
                }
            )
        return canonical(
            {
                "historical_goal": experience.goal,
                "conditions": experience.preconditions,
                "unknown_mask": experience.unknown_mask,
                "recipe": experience.normalized,
                "warning": "Rebind source paths and symbols from the current incident; historical success is not a guarantee.",
            }
        )
