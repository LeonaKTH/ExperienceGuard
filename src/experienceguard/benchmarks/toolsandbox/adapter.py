"""Thin adapter over UNMODIFIED Apple ToolSandbox functions and ExecutionContext.

No replacement environment is provided. Imports fail explicitly when upstream is absent.
The singleton upstream context is swapped under a lock for every operation.
"""

from __future__ import annotations
import copy
import inspect
import os
import sys
import threading
import types
import typing
from pathlib import Path
from dataclasses import dataclass
import jsonschema
from ...types import Action, PublicContext, STATE_KEYS
from ...utils.core import canonical, digest, file_hash, read_json

UPSTREAM_SHA = "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1"
TRACK = "ToolSandbox-StateReuse"
_LOCK = threading.RLock()
_VERIFIED = set()
SAFE_TOOLS = frozenset(
    {
        "get_wifi_status",
        "set_wifi_status",
        "get_cellular_service_status",
        "set_cellular_service_status",
        "get_low_battery_mode_status",
        "set_low_battery_mode_status",
        "get_location_service_status",
        "set_location_service_status",
        "search_contacts",
        "add_contact",
        "modify_contact",
        "remove_contact",
        "send_message_with_phone_number",
        "search_messages",
        "add_reminder",
        "search_reminder",
        "modify_reminder",
        "remove_reminder",
    }
)
GETTERS = {
    "get_wifi_status": "wifi",
    "get_cellular_service_status": "cellular",
    "get_low_battery_mode_status": "low_battery_mode",
    "get_location_service_status": "location_service",
}
SETTERS = {name.replace("get_", "set_", 1): key for name, key in GETTERS.items()}


def upstream():
    path = os.environ.get("EG_TOOLSANDBOX_PATH")
    if path and str(Path(path).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(path).resolve()))
    try:
        import polars as pl
        import tool_sandbox

        location = Path(tool_sandbox.__file__).resolve().parent
        with _LOCK:
            if str(location) not in _VERIFIED:
                verify_upstream(location)
                _VERIFIED.add(str(location))
        from tool_sandbox.common.execution_context import (
            ExecutionContext,
            DatabaseNamespace,
            new_context,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ToolSandbox unavailable; run the documented bootstrap, no fake fallback"
        ) from exc
    return pl, ExecutionContext, DatabaseNamespace, new_context


def verify_upstream(location):
    expected = read_json(Path(__file__).with_name("upstream_files.json"))
    actual = {
        p.relative_to(location).as_posix(): file_hash(p)
        for p in sorted(Path(location).rglob("*.py"))
    }
    if actual != expected["files"]:
        raise RuntimeError(
            "ToolSandbox source differs from the audited fixed commit; execution refused"
        )
    return expected["commit"]


def _schema_type(annotation):
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin in (typing.Union, types.UnionType):
        schemas = [
            _schema_type(a) for a in args if getattr(a, "__name__", "") != "NotGiven"
        ]
        return schemas[0] if len(schemas) == 1 else {"anyOf": schemas}
    if origin is list:
        return {"type": "array", "items": _schema_type(args[0]) if args else {}}
    primitive = {
        str: "string",
        bool: "boolean",
        int: "integer",
        float: "number",
        type(None): "null",
        dict: "object",
    }
    return {"type": primitive[annotation]} if annotation in primitive else {}


def function_schema(fn) -> dict:
    hints = typing.get_type_hints(fn)
    required, properties = [], {}
    for name, p in inspect.signature(fn).parameters.items():
        properties[name] = _schema_type(hints.get(name, p.annotation))
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    split: str
    goal: str
    tools: tuple[str, ...]
    initial_settings: dict
    expected_settings: dict
    contacts: tuple[dict, ...] = ()
    expected_message: dict | None = None
    forbidden_settings: dict | None = None
    source: str = "local_declared_fixture"

    def __post_init__(self):
        if set(self.tools) - SAFE_TOOLS:
            raise ValueError("task includes unapproved or network tools")
        if self.split not in {
            "source",
            "fit",
            "calibration",
            "validation",
            "test",
            "dev",
        }:
            raise ValueError("invalid task split")
        for mapping in (
            self.initial_settings,
            self.expected_settings,
            self.forbidden_settings or {},
        ):
            if set(mapping) - set(STATE_KEYS) or any(
                type(v) is not bool for v in mapping.values()
            ):
                raise ValueError("invalid task state constraints")

    @classmethod
    def from_dict(cls, item):
        obj = dict(item)
        obj["tools"] = tuple(obj["tools"])
        obj["contacts"] = tuple(obj.get("contacts", ()))
        return cls(**obj)


class ToolSandboxSession:
    def __init__(self, task: Task):
        pl, Context, NS, scoped = upstream()
        self._task = task  # never given to policies or model providers
        with _LOCK:
            self._context = Context(tool_allow_list=list(task.tools))
            with scoped(self._context):
                frame = self._context.get_database(NS.SETTING)
                self._context.update_database(
                    NS.SETTING,
                    frame.with_columns(
                        [pl.lit(v).alias(k) for k, v in task.initial_settings.items()]
                    ),
                )
                if task.contacts:
                    self._context.add_to_database(NS.CONTACT, list(task.contacts))
            self._tools = self._context.get_available_tools(scrambling_allowed=False)
        missing = set(task.tools) - set(self._tools)
        if missing:
            raise RuntimeError(f"upstream tools not found: {sorted(missing)}")
        self.schemas = [function_schema(self._tools[n]) for n in sorted(self._tools)]
        self.schema_hash = digest(self.schemas)
        self.visible = {k: None for k in STATE_KEYS}
        self.observations = []
        self._initial = self._snapshot()

    def public(self) -> PublicContext:
        return PublicContext(
            self._task.goal,
            copy.deepcopy(self.visible),
            tuple(sorted(self._tools)),
            self.schema_hash,
            tuple(copy.deepcopy(self.observations)),
        )

    def clone(self):
        obj = object.__new__(type(self))
        obj.__dict__ = copy.deepcopy(self.__dict__)
        return obj

    def validate(self, action: Action):
        if action.tool not in self._tools:
            raise ValueError("unknown or disallowed tool")
        schema = next(s for s in self.schemas if s["name"] == action.tool)
        jsonschema.validate(action.arguments, schema["parameters"])

    def execute(self, action: Action) -> dict:
        # Validation is before any database message or tool mutation.
        self.validate(action)
        pl, _, NS, scoped = upstream()
        with _LOCK, scoped(self._context):
            self._context.add_to_database(
                NS.SANDBOX,
                [
                    {
                        "sender": "AGENT",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": canonical(
                            {"tool": action.tool, "arguments": action.arguments}
                        ),
                    }
                ],
            )
            try:
                value = self._tools[action.tool](**action.arguments)
                result = {
                    "tool": action.tool,
                    "arguments": action.arguments,
                    "ok": True,
                    "result": value,
                }
                self._observe(action, value)
            except Exception as exc:
                # No network tools or arbitrary Python are exposed. Keep actual errors/state.
                result = {
                    "tool": action.tool,
                    "arguments": action.arguments,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            self._context.add_to_database(
                NS.SANDBOX,
                [
                    {
                        "sender": "EXECUTION_ENVIRONMENT",
                        "recipient": "AGENT",
                        "content": canonical(result),
                    }
                ],
            )
        self.observations.append(copy.deepcopy(result))
        return result

    def _observe(self, action, result):
        if action.tool in GETTERS and type(result) is bool:
            self.visible[GETTERS[action.tool]] = result
        if action.tool in SETTERS:
            key = SETTERS[action.tool]
            self.visible[key] = action.arguments["on"]
            if key == "low_battery_mode" and action.arguments["on"]:
                for dep in ("wifi", "cellular", "location_service"):
                    self.visible[dep] = (
                        False  # documented effect, not a hidden DB lookup
                    )

    def _snapshot(self):
        _, _, NS, scoped = upstream()
        with _LOCK, scoped(self._context):
            return {
                str(ns): self._context.get_database(ns).to_dicts()
                for ns in (NS.SETTING, NS.CONTACT, NS.MESSAGING, NS.REMINDER)
            }


class StateEvaluator:
    """Evaluator-only capability. No instance is passed to a policy/provider."""

    def score(self, session: ToolSandboxSession) -> dict:
        pl, _, NS, _ = upstream()
        from tool_sandbox.common.evaluation import (
            snapshot_similarity,
            column_exact_match_similarity,
        )

        final = session._snapshot()
        task = session._task
        checks = {}
        if task.expected_settings:
            frame = pl.DataFrame(final[str(NS.SETTING)])
            expected = pl.DataFrame({k: [v] for k, v in task.expected_settings.items()})
            score = float(
                snapshot_similarity(
                    snapshot=frame,
                    target_dataframe=expected,
                    column_similarities={
                        k: column_exact_match_similarity for k in task.expected_settings
                    },
                )
            )
            checks["settings"] = score == 1.0
        if task.expected_message:
            checks["message"] = any(
                all(row.get(k) == v for k, v in task.expected_message.items())
                for row in final[str(NS.MESSAGING)]
            )
        minefield = any(
            final[str(NS.SETTING)][0].get(k) == v
            for k, v in (task.forbidden_settings or {}).items()
        )
        return {
            "state_success": bool(checks) and all(checks.values()) and not minefield,
            "checks": checks,
            "minefield": minefield,
            "initial_hash": digest(session._initial),
            "final_hash": digest(final),
            "initial_state": session._initial,
            "final_state": final,
            "state_diff": {
                k: {"before": session._initial[k], "after": final[k]}
                for k in final
                if session._initial[k] != final[k]
            },
            "grader": "ToolSandbox.snapshot_similarity + declared StateReuse constraints; not official full benchmark",
        }
