"""Read -> diagnose -> review/validate -> append memory -> independently retrieve again."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import uuid
from ..repository_tools.reader import RepositoryReader, redact
from ..ci_analysis.analyzer import analyze
from ..domain.development import (
    DevelopmentExperience,
    DevelopmentNormalizer,
    DOMAIN,
    SCHEMA,
)
from ..features.development import extract
from ..storage.experience import ExperienceStore
from ..retrieval.retriever import MemoryRetriever
from ..policies.selector import ExperienceSelector, PolicyModel
from ..types import UnavailablePolicy
from ..utils.core import digest, read_json, write_json, utc_now
from .fixtures import initialize_fixtures, validate_proposal


class DevelopmentApp:
    def __init__(
        self,
        data_dir,
        *,
        allowed_roots=(),
        policy_model=None,
        global_utilities=None,
        embedding=None,
    ):
        self.policy_model, self.global_utilities, self.embedding = (
            policy_model,
            global_utilities,
            embedding,
        )
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = [Path(p).resolve() for p in allowed_roots] + [
            self.data_dir / "workspaces"
        ]
        self.store = ExperienceStore(
            self.data_dir / "development-memory.sqlite",
            record_type=DevelopmentExperience,
        )

    def initialize(self):
        fixtures = initialize_fixtures(self.data_dir)
        write_json(
            self.data_dir / "application.json",
            {
                "version": "1.5.0",
                "domain": DOMAIN,
                "mode": "evidence_only",
                "LLM": "UNAVAILABLE / no calls authorized",
            },
        )
        return fixtures

    def _case_dir(self, case_id):
        if (
            not isinstance(case_id, str)
            or not case_id.startswith("case-")
            or len(case_id) != 37
            or any(c not in "0123456789abcdef" for c in case_id[5:])
        ):
            raise ValueError("invalid case ID")
        return self.data_dir / "incidents" / case_id

    def load_case(self, case_id):
        value = read_json(self._case_dir(case_id) / "diagnosis.json")
        check = dict(value)
        sha = check.pop("report_sha256", None)
        if sha != digest(check):
            raise ValueError("diagnostic report hash mismatch")
        return value

    def list_cases(self):
        result = []
        for p in (
            sorted((self.data_dir / "incidents").glob("case-*/diagnosis.json"))
            if (self.data_dir / "incidents").exists()
            else []
        ):
            x = self.load_case(p.parent.name)
            result.append(
                {
                    "case_id": x["case_id"],
                    "repository": x["repository"]["name"],
                    "created_at": x["created_at"],
                    "issue": x["primary_issue"]["kind"],
                    "experience_id": x["decision"].get("experience_id"),
                }
            )
        return result

    def diagnose(
        self,
        repository,
        log,
        *,
        strategy="B4",
        provenance="user_repository",
        public_url=None,
        policy_model=None,
    ):
        repo = Path(repository).resolve()
        if (self.data_dir / "fixtures.json").exists() and any(
            Path(f["repository"]).resolve() == repo
            for f in read_json(self.data_dir / "fixtures.json")
        ):
            provenance = "controlled_fixture"
        if self.data_dir.is_relative_to(repo):
            raise ValueError(
                "application data must be outside the imported repository to guarantee read-only analysis"
            )
        if (
            not isinstance(log, str)
            or not log.strip()
            or len(log.encode("utf8")) > 300000
        ):
            raise ValueError("a nonempty CI log up to 300KB is required")
        if provenance not in {
            "user_repository",
            "controlled_fixture",
            "public_repository",
        }:
            raise ValueError("invalid input provenance")
        reader = RepositoryReader(repo, allowed_roots=self.allowed_roots)
        analysis = analyze(reader, log)
        context = analysis.pop("context")
        all_memory = self.store.all()
        superseded = {e.supersedes for e in all_memory if e.supersedes}
        normalizer = DevelopmentNormalizer()
        pool = MemoryRetriever(
            [e for e in all_memory if e.experience_id not in superseded],
            embedding=self.embedding,
            renderer=normalizer,
        )
        candidates = pool.retrieve(context.goal, top_k=5, budget=24000)
        # Distinct domain schema prevents loading a phone-state model into development reuse.
        from ..features.development import FEATURE_SCHEMA

        policy_model = policy_model or self.policy_model
        model = (
            PolicyModel(policy_model, expected_schema=FEATURE_SCHEMA)
            if policy_model
            else None
        )
        selector = ExperienceSelector(
            strategy,
            model=model,
            global_utility=self.global_utilities,
            feature_extractor=extract,
            rule_eligibility=lambda c: c.experience.feedback.get("success") is True,
        )
        try:
            if strategy == "B1" and self.embedding is None:
                raise UnavailablePolicy(
                    "B1 requires a configured frozen embedding, including an empty-memory run"
                )
            decision = selector.choose(context, candidates, 24000).to_dict()
            policy_status = "AVAILABLE"
        except UnavailablePolicy as exc:
            decision = {
                "strategy": strategy,
                "version": selector.version,
                "experience_id": None,
                "reason": str(exc),
                "evidence": [],
                "scores": {},
                "rejection": str(exc),
                "representation": "normalized",
            }
            policy_status = "UNTRAINED" if strategy == "New" else "UNAVAILABLE"
        # Always expose conditions for every recalled candidate, even an unavailable policy.
        candidate_views = []
        for c in candidates:
            features, conditions = extract(context, c, 24000)
            candidate_views.append(
                {
                    "experience_id": c.experience.experience_id,
                    "retrieval_score": c.retrieval_score,
                    "semantic_score": c.semantic_score,
                    "source_repository": c.experience.repository,
                    "source_incident": c.experience.source_task,
                    "source_commit": c.experience.repository.get("commit"),
                    "historical_outcome": c.experience.feedback,
                    "provenance": c.experience.provenance,
                    "conditions": conditions,
                    "features": features,
                    "repair": c.experience.repair,
                    "content_sha256": c.experience.content_hash,
                    "evidence_quality": c.experience.evidence_quality,
                    "related_goal": c.experience.goal,
                }
            )
        reader.record(
            "compare_conditions",
            {
                "strategy": strategy,
                "candidate_ids": [c.experience.experience_id for c in candidates],
            },
            {
                "selected": decision["experience_id"],
                "policy_status": policy_status,
                "basis": "rules"
                if strategy == "B4"
                else "learned"
                if strategy == "New" and model
                else "baseline",
            },
        )
        selected = next(
            (
                c.experience
                for c in candidates
                if c.experience.experience_id == decision["experience_id"]
            ),
            None,
        )
        if selected and selected.feedback.get("success") is True:
            analysis["suggestions"].insert(
                0,
                {
                    "text": selected.repair["description"],
                    "basis": "reused historical recipe; revalidate on current evidence",
                    "experience_id": selected.experience_id,
                    "evidence_refs": [analysis["log_ref"]],
                    "historical_validation_sha": digest(selected.validation),
                    "warning": "Historical file paths/symbols are references only; use the current proposals below.",
                },
            )
        case_id = "case-" + uuid.uuid4().hex
        directory = self._case_dir(case_id)
        directory.mkdir(parents=True, exist_ok=False)
        # Original inputs stay local and unchanged; API exposes only redacted evidence.
        (directory / "raw.log").write_text(log, encoding="utf8")
        write_json(directory / "raw_repository_evidence.json", reader.raw_files)
        repository = reader.git_identity()
        if analysis.get("project_name"):
            repository["name"] = analysis.pop("project_name")
        from ..utils.core import file_hash

        source_root = Path(__file__).resolve().parents[1]
        report = {
            "version": 1,
            "domain": DOMAIN,
            "case_id": case_id,
            "created_at": utc_now(),
            "mode": "evidence_only",
            "mode_label": "无LLM · 真实工具与规则诊断",
            "llm_status": "UNAVAILABLE / NOT CALLED",
            "policy_status": policy_status,
            "policy_artifact_sha": getattr(model, "artifact_sha", None),
            "global_utility_sha": digest(self.global_utilities)
            if self.global_utilities
            else None,
            "repository": repository,
            "implementation": {
                "version": "1.5.0",
                "source_tree_sha": digest(
                    {
                        p.relative_to(source_root).as_posix(): file_hash(p)
                        for p in sorted(source_root.rglob("*.py"))
                    }
                ),
            },
            "input_provenance": provenance,
            "public_url": public_url,
            "inspected_snapshot_sha": reader.snapshot(),
            "repository_tree_sha": reader.tree_identity(analysis["files"]),
            "log_sha256": digest(log),
            "log": redact(log),
            "visible_context": asdict(context),
            "candidate_pool_sha": digest(
                [c.experience.content_hash for c in candidates]
            ),
            "source_pool_sha": digest(
                [
                    e.content_hash
                    for e in sorted(all_memory, key=lambda e: e.experience_id)
                    if e.experience_id not in superseded
                ]
            ),
            "candidates": candidate_views,
            "decision": decision,
            "evidence": reader.evidence,
            "tool_trace": reader.trace,
            **analysis,
        }
        report["report_sha256"] = digest(report)
        write_json(directory / "diagnosis.json", report)
        return report

    def verify_and_save(
        self,
        case_id,
        *,
        proposal_id=None,
        description=None,
        actor="local developer",
        manual=None,
        supersedes=None,
        agent_run_id=None,
    ):
        report = self.load_case(case_id)
        directory = self._case_dir(case_id)
        agent_execution = None
        if agent_run_id:
            from ..agents.development import load_execution

            agent_execution = load_execution(self, case_id, agent_run_id)
            if agent_execution["status"] != "completed" or manual is None:
                raise ValueError(
                    "Agent suggestions require completed execution and explicit user verification evidence"
                )
        if not actor.strip() or len(actor) > 120:
            raise ValueError("a verification actor is required")
        if manual is None:
            proposal = next(
                (p for p in report["proposals"] if p["id"] == proposal_id), None
            )
            if proposal is None:
                raise ValueError("choose a concrete current-source proposal")
            validation = validate_proposal(self.data_dir, report, proposal)
            repair_description = (
                description
                or f"历史故障通过将导入符号 {proposal['old_symbol']} 对齐到当前模块公开 API 得到修复；仅当运行时、模块和依赖前提匹配时作为参考，按当前源文件重新定位。"
            )
        else:
            if (
                manual.get("outcome") not in {"success", "failure", "inconclusive"}
                or not manual.get("verification_log", "").strip()
                or not description
                or not description.strip()
            ):
                raise ValueError(
                    "manual feedback requires actual outcome, repair description and verification evidence"
                )
            if len(manual["verification_log"]) > 100000:
                raise ValueError("verification log too large")
            validation = {
                "kind": "user_attested",
                "outcome": manual["outcome"],
                "verification_log": manual["verification_log"],
                "note": "User-attested result, not independently executed or causally attributed",
                "argv": manual.get("argv"),
                "exit_code": manual.get("exit_code"),
            }
            repair_description = description
        validation = {
            **validation,
            "case_id": case_id,
            "report_sha256": report["report_sha256"],
            "actor": actor,
            "verified_at": utc_now(),
            "agent_execution_sha": agent_execution["execution_sha"]
            if agent_execution
            else None,
            "agent_run_id": agent_run_id,
            "agent_mode": agent_execution["mode"] if agent_execution else None,
        }
        validation["receipt_sha256"] = digest(validation)
        receipt_id = "verification-" + uuid.uuid4().hex
        write_json(directory / (receipt_id + ".json"), validation)
        if validation["outcome"] == "inconclusive":
            return {
                "validation": validation,
                "receipt_id": receipt_id,
                "experience_id": None,
                "reason": "inconclusive feedback is retained but not promoted into reusable memory",
            }
        if supersedes is not None:
            self.store.get(supersedes)
        facts = report["facts"]
        issue = report["primary_issue"]
        language = facts.get("language")
        keys = ["issue_kind", "language"] + (["module"] if issue.get("module") else [])
        runtime = {"python": "python", "node": "node", "java": "java"}.get(language)
        if runtime:
            keys.append(runtime)
        if issue.get("module"):
            keys.append(
                "dependency:" + issue["module"].split(".")[0].lower().replace("_", "-")
            )
        if language == "node":
            keys.append("module_system")
        if language == "java" and facts.get("java_target"):
            keys.append("java_target")
        if issue["kind"] == "database_migration":
            keys.extend(["database", "database_version", "dependency:flyway-core"])
        conditions = {}
        for key in keys:
            value = facts.get(key)
            op = (
                "unknown"
                if value is None
                else "version"
                if key in {"python", "node", "java"} or key.startswith("dependency:")
                else "eq"
            )
            conditions[key] = {
                "op": op,
                "value": (
                    "==" + value + ".*"
                    if value
                    and key in {"python", "node", "java"}
                    and len(value.split(".")) < 3
                    else "==" + value
                )
                if op == "version"
                else value,
                "evidence_refs": [
                    e["evidence_ref"] for e in report["fact_evidence"].get(key, [])
                ],
            }
        repair = {
            "description": redact(repair_description),
            "historical_proposals": report["proposals"],
            "validation_receipt": receipt_id,
        }
        normalized = {
            "version": DevelopmentNormalizer.version,
            "recipe": redact(repair_description),
            "argument_slots": {
                "source_file": "$CURRENT_SOURCE_FILE",
                "symbol": "$CURRENT_EXPORTED_API",
            },
            "evidence_refs": [e["id"] for e in report["evidence"]],
            "original_report_sha": report["report_sha256"],
            "binding": "Current AST/evidence only; do not copy old file paths",
        }
        sha = digest(
            {
                "report": report["report_sha256"],
                "validation": validation["receipt_sha256"],
                "repair": repair,
                "conditions": conditions,
            }
        )
        exp = DevelopmentExperience(
            "devexp-" + sha[:24],
            case_id,
            issue["kind"],
            report["visible_context"]["goal"],
            report["evidence"],
            [{"tool": e["tool"]} for e in report["tool_trace"]],
            report["tool_trace"],
            normalized,
            conditions,
            {k: c["op"] == "unknown" for k, c in conditions.items()},
            SCHEMA,
            report["report_sha256"],
            {
                "success": validation["outcome"] == "success",
                "validation_kind": validation["kind"],
            },
            utc_now(),
            report["repository"],
            facts,
            report["evidence"],
            repair,
            [
                {"key": k, "invalid_when": "predicate conflicts or evidence is unknown"}
                for k in conditions
            ],
            validation,
            validation["kind"],
            supersedes,
            reliability=1.0
            if validation["kind"] == "controlled_fixture_verified"
            else 0.55,
        )
        self.store.add(exp)
        return {
            "experience_id": exp.experience_id,
            "content_sha256": exp.content_hash,
            "receipt_id": receipt_id,
            "validation": validation,
        }
