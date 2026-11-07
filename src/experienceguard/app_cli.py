"""Application CLI. Evidence mode is real file analysis, not simulated LLM planning."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import yaml
from .workflows.service import DevelopmentApp
from .utils.core import read_json, write_json

DEFAULT_DATA = Path(
    os.environ.get("EG_DATA_DIR", str(Path.home() / ".experienceguard"))
)
COMMANDS = {
    "init",
    "diagnose",
    "verify",
    "memory",
    "cases",
    "serve",
    "dev-build-labels",
    "dev-train-policy",
    "app-doctor",
    "agent-preview",
    "agent-run",
    "agent-protocol-test",
    "public-evaluate",
    "agent-pilot",
    "agent-batch",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="experienceguard",
        description="Development fault diagnosis and cross-session experience memory. Evidence Mode is offline; Agent Mode requires reviewed remote-data and cost authorization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in sorted(COMMANDS):
        p = sub.add_parser(command)
        p.add_argument("--data-dir", default=str(DEFAULT_DATA))
        if command == "diagnose":
            p.add_argument("--policy-model")
            p.add_argument("--utility-labels")
            p.add_argument("--repo")
            p.add_argument("--log-file")
            p.add_argument(
                "--fixture", choices=["source", "compatible", "conflict", "unknown"]
            )
            p.add_argument("--allow-root", action="append", default=[])
            p.add_argument(
                "--strategy",
                choices=["B0", "B1", "B2", "B3", "B4", "B5", "New"],
                default="B4",
            )
            p.add_argument("--output")
        elif command == "verify":
            p.add_argument("--case-id", required=True)
            p.add_argument("--proposal-id")
            p.add_argument("--actor", default="local developer")
            p.add_argument("--agent-run-id")
            p.add_argument(
                "--manual-result",
                help="JSON with outcome, actual verification_log and repair description",
            )
        elif command == "serve":
            p.add_argument(
                "--enable-remote",
                action="store_true",
                help="requires per-run reviewed disclosure and cost consent; not enabled by default",
            )
            p.add_argument("--policy-model")
            p.add_argument("--utility-labels")
            p.add_argument("--allow-root", action="append", default=[])
            p.add_argument("--port", type=int, default=8766)
        elif command == "agent-preview":
            p.add_argument("--case-id", required=True)
            p.add_argument("--file", action="append", default=[])
            p.add_argument("--allow-root", action="append", default=[])
        elif command in {"agent-run", "agent-protocol-test"}:
            p.add_argument("--case-id", required=True)
            p.add_argument("--plan-id", required=True)
            if command == "agent-run":
                p.add_argument("--consent-sha", required=True)
                p.add_argument("--allow-remote", action="store_true")
                p.add_argument("--acknowledge-upload", action="store_true")
                p.add_argument("--confirm-cost", action="store_true")
        elif command in {"agent-pilot", "agent-batch"}:
            p.add_argument("--config", required=True)
            p.add_argument("--allow-root", action="append", default=[])
            if command == "agent-pilot":
                p.add_argument(
                    "--protocol-test",
                    action="store_true",
                    help="scripted loopback HTTP, never real LLM",
                )
            else:
                p.add_argument("--allow-remote", action="store_true")
                p.add_argument("--acknowledge-upload", action="store_true")
                p.add_argument("--confirm-cost", action="store_true")
        elif command == "public-evaluate":
            p.add_argument("--dataset", required=True)
            p.add_argument("--output", required=True)
        elif command in {"dev-build-labels", "dev-train-policy"}:
            p.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "app-doctor":
            result = {
                "version": "1.6.0",
                "mode": "evidence_only",
                "ToolSandbox": "optional / not loaded",
                "LLM": "UNAVAILABLE / NOT CALLED",
                "New": "UNTRAINED",
                "network_requests": 0,
                "python": sys.version,
                "core_dependencies": {
                    n: importlib.metadata.version(n)
                    for n in ["numpy", "filelock", "packaging", "PyYAML", "jsonschema"]
                },
            }
        else:
            utility = None
            if getattr(args, "utility_labels", None):
                from .training.development import global_utilities

                utility = global_utilities(read_json(args.utility_labels))
            app = DevelopmentApp(
                args.data_dir,
                allowed_roots=getattr(args, "allow_root", []),
                policy_model=getattr(args, "policy_model", None),
                global_utilities=utility,
            )
            if args.command == "init":
                result = {
                    "fixtures": app.initialize(),
                    "data_dir": str(app.data_dir),
                    "mode": "evidence_only",
                }
            elif args.command == "diagnose":
                if args.fixture:
                    f = next(
                        x
                        for x in read_json(app.data_dir / "fixtures.json")
                        if x["name"] == args.fixture
                    )
                    repo = f["repository"]
                    log = Path(f["log_file"]).read_text(encoding="utf8")
                else:
                    if not args.repo or not args.log_file:
                        raise ValueError(
                            "provide --fixture, or --repo and explicitly selected --log-file"
                        )
                    repo = args.repo
                    logpath = Path(args.log_file).resolve()
                    if logpath.stat().st_size > 300000:
                        raise ValueError("log exceeds 300KB")
                    from .repository_tools.reader import allowed_name

                    if not allowed_name(logpath):
                        raise ValueError(
                            "secret/config credential files may not be imported as logs"
                        )
                    log = logpath.read_text(encoding="utf8", errors="replace")
                result = app.diagnose(repo, log, strategy=args.strategy)
                if args.output:
                    target = Path(args.output).resolve()
                    if target.is_relative_to(Path(repo).resolve()):
                        raise ValueError(
                            "report output must be outside imported repository"
                        )
                    if target.exists():
                        raise ValueError("refusing report overwrite")
                    write_json(target, result)
            elif args.command == "verify":
                manual = read_json(args.manual_result) if args.manual_result else None
                result = app.verify_and_save(
                    args.case_id,
                    proposal_id=args.proposal_id,
                    actor=args.actor,
                    manual=manual,
                    description=manual.get("description") if manual else None,
                    agent_run_id=args.agent_run_id,
                )
            elif args.command == "memory":
                result = [e.to_dict() for e in app.store.all()]
            elif args.command == "cases":
                result = app.list_cases()
            elif args.command == "serve":
                from .api.server import serve

                app.allow_remote = args.enable_remote
                serve(app, args.port)
                return 0
            elif args.command == "agent-preview":
                from .agents.development import prepare_agent

                result = prepare_agent(app, args.case_id, args.file)
            elif args.command in {"agent-run", "agent-protocol-test"}:
                from .agents.development import load_plan, run_remote

                plan = load_plan(app, args.case_id, args.plan_id)
                if args.command == "agent-protocol-test":
                    from .agents.protocol_test import protocol_test

                    result = protocol_test(app, plan)
                else:
                    result = run_remote(
                        app,
                        plan,
                        consent_sha=args.consent_sha,
                        allow_remote=args.allow_remote,
                        acknowledge_upload=args.acknowledge_upload,
                        confirm_cost=args.confirm_cost,
                    )
            elif args.command == "public-evaluate":
                from .evaluation.public_cases import evaluate_public_cases

                result = evaluate_public_cases(args.dataset, args.output)
            elif args.command in {"agent-pilot", "agent-batch"}:
                from .evaluation.agent import (
                    evaluate_agent,
                    execute_reviewed_plan_batch,
                )

                cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
                if args.command == "agent-pilot":
                    result = evaluate_agent(
                        app,
                        cfg["incidents"],
                        cfg["output"],
                        strategies=cfg.get(
                            "strategies", ["B0", "B1", "B3", "B4", "B5", "New"]
                        ),
                        protocol_only=args.protocol_test,
                    )
                else:
                    result = execute_reviewed_plan_batch(
                        app,
                        cfg["reviewed_plans"],
                        cfg["output"],
                        allow_remote=args.allow_remote,
                        acknowledge_upload=args.acknowledge_upload,
                        confirm_cost=args.confirm_cost,
                    )
            else:
                cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
                if args.command == "dev-build-labels":
                    from .training.development import build_pair

                    result = [build_pair(app, **p) for p in cfg["pairs"]]
                    if Path(cfg["output"]).exists():
                        raise ValueError("label output exists")
                    write_json(cfg["output"], result)
                else:
                    from .training.trainer import train
                    from .features.development import FEATURE_SCHEMA
                    from .domain.development import DOMAIN

                    result = train(
                        read_json(cfg["labels"]),
                        cfg["output"],
                        feature_schema=FEATURE_SCHEMA,
                        domain=DOMAIN,
                        seed=cfg.get("seed", 42),
                    )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
