"""Owned, auditable fault fixtures. Not public third-party repositories or LLM outputs."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import io
import tokenize
import uuid
from ..utils.core import digest, read_json, write_json, utc_now, file_hash

FIXTURE_NAMES = ("source", "compatible", "conflict", "unknown")
ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "examples/development").is_dir():
    ROOT = Path(sys.prefix) / "share/experienceguard"


def file_map(root):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob("*")):
        if ".git" in p.relative_to(root).parts:
            continue
        if p.is_symlink() or getattr(p, "is_junction", lambda: False)():
            raise ValueError("fixture contains links")
        if p.is_file():
            result[p.relative_to(root).as_posix()] = file_hash(p)
    return result


def minimal_env():
    return {
        k: v
        for k, v in os.environ.items()
        if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC"}
    } | {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }


def test_copy(directory):
    # Constant argv; only used on byte-verified bundled fixtures in owned copies.
    command = [
        sys.executable,
        "-I",
        "-S",
        "-X",
        "utf8",
        "-m",
        "unittest",
        "discover",
        "-s",
        ".",
        "-p",
        "test_*.py",
        "-v",
    ]
    try:
        run = subprocess.run(
            command,
            cwd=directory,
            env=minimal_env(),
            capture_output=True,
            text=True,
            encoding="utf8",
            errors="replace",
            timeout=10,
        )
        version_command = [sys.executable, "-I", "-S", "-X", "utf8", "--version"]
        runtime = subprocess.run(
            version_command,
            cwd=directory,
            env=minimal_env(),
            capture_output=True,
            text=True,
            encoding="utf8",
            timeout=5,
        )
        return {
            "argv": command,
            "cwd": str(directory),
            "exit_code": run.returncode,
            "stdout": run.stdout,
            "stderr": run.stderr,
            "log": run.stdout + run.stderr,
            "time": utc_now(),
            "runtime_probe": {
                "argv": version_command,
                "stdout": runtime.stdout,
                "stderr": runtime.stderr,
                "exit_code": runtime.returncode,
            },
        }
    except subprocess.TimeoutExpired:
        return {
            "argv": command,
            "cwd": str(directory),
            "exit_code": None,
            "stdout": "",
            "stderr": "timeout",
            "log": "Controlled fixture timeout",
            "time": utc_now(),
        }


def initialize_fixtures(data_dir):
    data_dir = Path(data_dir)
    bundled = ROOT / "examples/development"
    result = []
    for name in FIXTURE_NAMES:
        source = bundled / name
        workspace = data_dir / "workspaces" / name
        repo = workspace / "repo"
        expected = file_map(source)
        if repo.exists():
            if file_map(repo) != expected:
                raise ValueError(
                    "owned fixture differs; preserve it and choose a new data directory"
                )
            if (workspace / "initial_validation.json").exists() and (
                workspace / "failure.log"
            ).exists():
                old = read_json(workspace / "initial_validation.json")
                result.append(
                    {
                        "name": name,
                        "repository": str(repo.resolve()),
                        "log_file": str((workspace / "failure.log").resolve()),
                        "provenance": "controlled_fixture",
                        "initial_exit_code": old["exit_code"],
                    }
                )
                continue
        else:
            workspace.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, repo)
            empty = data_dir / "empty-git-template"
            empty.mkdir(exist_ok=True)
            git = shutil.which("git")
            if git:
                prefix = [
                    git,
                    "-c",
                    "init.templateDir=" + str(empty.resolve()),
                    "-c",
                    "core.hooksPath=" + str(empty.resolve()),
                    "-c",
                    "user.name=ExperienceGuard Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                ]
                for args in [
                    ["init", "-q", str(repo.resolve())],
                    ["-C", str(repo.resolve()), "add", "."],
                    [
                        "-C",
                        str(repo.resolve()),
                        "commit",
                        "-q",
                        "-m",
                        "Controlled failure fixture",
                    ],
                ]:
                    subprocess.run(
                        prefix + args,
                        env=minimal_env(),
                        capture_output=True,
                        check=True,
                        timeout=10,
                    )
        copy_dir = data_dir / "validation" / ("initial-" + uuid.uuid4().hex)
        shutil.copytree(source, copy_dir)
        receipt = test_copy(copy_dir)
        if receipt["exit_code"] in (None, 0):
            raise RuntimeError(
                "fixture must produce a real failing test before diagnosis"
            )
        write_json(workspace / "initial_validation.json", receipt)
        log = receipt["log"]
        if name != "unknown":
            log = (
                receipt["runtime_probe"]["stdout"]
                + receipt["runtime_probe"]["stderr"]
                + log
            )
        (workspace / "failure.log").write_text(log, encoding="utf8")
        write_json(
            workspace / "origin.json",
            {
                "kind": "controlled_fixture",
                "name": name,
                "repo": str(repo.resolve()),
                "bundled_tree_sha": digest(expected),
            },
        )
        result.append(
            {
                "name": name,
                "repository": str(repo.resolve()),
                "log_file": str((workspace / "failure.log").resolve()),
                "provenance": "controlled_fixture",
                "initial_exit_code": receipt["exit_code"],
            }
        )
    write_json(data_dir / "fixtures.json", result)
    return result


def validate_proposal(data_dir, report, proposal):
    data_dir = Path(data_dir)
    repo = Path(report["repository"]["path"]).resolve()
    fixtures = (
        read_json(data_dir / "fixtures.json")
        if (data_dir / "fixtures.json").exists()
        else []
    )
    registered = next(
        (x for x in fixtures if Path(x["repository"]).resolve() == repo), None
    )
    if registered is None:
        raise ValueError(
            "automatic validation is restricted to registered bundled fixtures; use manual feedback for real repositories"
        )
    bundled = ROOT / "examples/development" / registered["name"]
    expected = file_map(bundled)
    if file_map(repo) != expected:
        raise ValueError("fixture source changed; refusing execution")
    if proposal not in report["proposals"]:
        raise ValueError("proposal was not generated by this diagnostic report")
    source = (bundled / proposal["path"]).resolve()
    if not source.is_relative_to(bundled.resolve()) or proposal["path"].endswith(
        "test_worker.py"
    ):
        raise ValueError(
            "only a current application source symbol proposal may be tested"
        )
    raw = source.read_text(encoding="utf8")
    if digest(raw) != proposal["original_sha"]:
        raise ValueError("proposal is stale")
    import re

    if not all(
        re.fullmatch(r"[A-Za-z_]\w*", proposal[k]) for k in ("old_symbol", "new_symbol")
    ):
        raise ValueError("only identifier substitutions are permitted")
    workspace = data_dir / "validation" / ("verify-" + uuid.uuid4().hex)
    shutil.copytree(bundled, workspace)
    before = test_copy(workspace)
    tokens = list(tokenize.generate_tokens(io.StringIO(raw).readline))
    fixed = tokenize.untokenize(
        [
            t._replace(string=proposal["new_symbol"])
            if t.type == tokenize.NAME and t.string == proposal["old_symbol"]
            else t
            for t in tokens
        ]
    )
    (workspace / proposal["path"]).write_text(fixed, encoding="utf8")
    after = test_copy(workspace)
    unchanged = file_map(repo) == expected
    if not unchanged:
        raise RuntimeError("imported repository changed during validation")
    return {
        "kind": "controlled_fixture_verified",
        "outcome": "success"
        if before["exit_code"] not in (None, 0) and after["exit_code"] == 0
        else "failure",
        "before": before,
        "after": after,
        "repository_unchanged": unchanged,
        "fixture_tree_sha": digest(expected),
        "proposal": proposal,
        "validation_scope": "owned bundled fixture only; not general-purpose process isolation",
    }
