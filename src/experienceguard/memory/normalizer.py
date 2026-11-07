"""Evidence-preserving normalization. Never binds parameters from an evaluator."""

from __future__ import annotations
import copy
from ..environments.toolsandbox import GETTERS, SETTERS
from ..types import Experience, STATE_KEYS
from ..utils.core import digest, utc_now


class ExperienceNormalizer:
    version = "observed-trajectory-v1"

    def normalize(self, trace: list[dict]) -> dict:
        placeholders, transitions = {}, []
        known = {k: None for k in STATE_KEYS}
        for index, event in enumerate(trace):
            if event.get("event") != "tool_result":
                continue
            obs = event["observation"]
            name, args = obs["tool"], obs["arguments"]
            normalized_args = {}
            for key, value in args.items():
                if isinstance(value, str):
                    slot = "$ARG_" + str(len(placeholders) + 1)
                    existing = next(
                        (s for s, v in placeholders.items() if v == value), None
                    )
                    slot = existing or slot
                    placeholders[slot] = value
                    normalized_args[key] = slot
                else:
                    normalized_args[key] = copy.deepcopy(value)
            transitions.append(
                {
                    "source_event_index": index,
                    "source_event_sha": digest(event),
                    "observed_before": dict(known),
                    "tool": name,
                    "arguments": normalized_args,
                    "ok": obs["ok"],
                    "error_type": obs.get("error_type"),
                    "observed_result": copy.deepcopy(obs.get("result")),
                }
            )
            if obs["ok"] and name in GETTERS and type(obs.get("result")) is bool:
                known[GETTERS[name]] = obs["result"]
            elif obs["ok"] and name in SETTERS:
                known[SETTERS[name]] = args["on"]
                if name == "set_low_battery_mode_status" and args["on"]:
                    known.update(
                        {k: False for k in ("wifi", "cellular", "location_service")}
                    )
        # Bindings retain ORIGINAL values for audit only. Rendering does not substitute them.
        return {
            "version": self.version,
            "transitions": transitions,
            "historical_bindings": placeholders,
            "raw_trace_sha": digest(trace),
            "binding_rule": "placeholders require current PUBLIC request; never reuse historical values automatically",
        }

    def from_run(
        self, run: dict, *, source_task: str, family: str, created_at=None
    ) -> Experience:
        checked = dict(run)
        checksum = checked.pop("result_sha256", None)
        if checksum != digest(checked):
            raise ValueError("source result hash mismatch")
        if (
            run["manifest"]["split"] != "source"
            or source_task != run["manifest"]["task_id"]
            or family != run["manifest"]["family"]
        ):
            raise ValueError("only independent source-partition runs may become memory")
        trace = copy.deepcopy(run["trace"])
        normalized = self.normalize(trace)
        normalized["source_model"] = copy.deepcopy(run["manifest"]["model"])
        normalized["source_prompt_hash"] = run["manifest"]["prompt_hash"]
        first_known = {}
        for step in normalized["transitions"]:
            for key, value in step["observed_before"].items():
                if value is not None and key not in first_known:
                    first_known[key] = value
            if (
                step["tool"] in GETTERS
                and step["ok"]
                and type(step.get("observed_result")) is bool
            ):
                first_known.setdefault(GETTERS[step["tool"]], step["observed_result"])
            # Later observations can be effects of our actions; they are not initial premises.
            if step["tool"] in SETTERS or step["tool"] not in GETTERS:
                break
        sha = run["execution_sha"]
        if sha != digest({"manifest": run["manifest"], "trace": trace}):
            raise ValueError("source run execution hash mismatch")
        return Experience(
            "exp-" + sha[:20],
            source_task,
            family,
            run["goal"],
            [e["observation"] for e in trace if e.get("event") == "tool_result"],
            [e["observation"] for e in trace if e.get("event") == "tool_result"],
            trace,
            normalized,
            first_known,
            {k: k not in first_known for k in STATE_KEYS},
            run["manifest"]["schema_hash"],
            sha,
            {"success": run["evaluation"]["state_success"], "stage": "source"},
            created_at or utc_now(),
            run["manifest"]["model"]["provider"],
        )

    def render(
        self, experience: Experience, representation="normalized", bindings=None
    ):
        from ..utils.core import canonical

        if representation == "raw":
            return canonical(
                {"historical_goal": experience.goal, "trace": experience.raw_trace}
            )
        value = {
            "historical_goal": experience.goal,
            "schema_hash": experience.schema_hash,
            "preconditions": experience.preconditions,
            "unknown_mask": experience.unknown_mask,
            "transitions": experience.normalized["transitions"],
            "warning": "Past actions are examples, not a current plan. Rebind placeholders from current request.",
        }
        if bindings:
            allowed = set(experience.normalized["historical_bindings"])
            if set(bindings) - allowed or any(
                not isinstance(v, str) for v in bindings.values()
            ):
                raise ValueError("invalid explicit public bindings")
            value["public_bindings"] = bindings
        return canonical(value)
