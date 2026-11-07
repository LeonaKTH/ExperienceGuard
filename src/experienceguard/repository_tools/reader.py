"""Bounded, read-only local file capabilities. Never imports or executes repository code."""

from __future__ import annotations
import ast
import os
import re
import time
from pathlib import Path
from ..utils.core import digest, file_hash

BLOCKED_DIRS = {
    ".git",
    ".svn",
    ".hg",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    "target",
    "dist",
    "build",
    ".ssh",
    ".aws",
    ".codex",
    ".agents",
    "models",
}
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".java",
    ".gradle",
    ".properties",
    ".txt",
    ".md",
    ".log",
    ".ini",
    ".cfg",
}
SPECIAL = {
    ".python-version",
    ".nvmrc",
    ".node-version",
    ".tool-versions",
    "requirements.txt",
    "Dockerfile",
    "pom.xml",
}
SECRET_NAME = re.compile(
    r"(^\.env($|\.)|credentials|secrets?\.|id_rsa|id_ed25519|\.pem$|\.key$|\.p12$|\.npmrc$|\.pypirc$)",
    re.I,
)


def redact(text):
    text = re.sub(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
        "[REDACTED PRIVATE KEY]",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"(?i)\b(?:gh[pousr]_[A-Za-z0-9_]{12,}|sk-[A-Za-z0-9_-]{16,})\b",
        "[REDACTED TOKEN]",
        text,
    )
    text = re.sub(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|password|secret|authorization)\b[\"\x27]?\s*[:=]\s*[\"\x27]?)([^\s\"\x27,;}]+)",
        r"\1[REDACTED]",
        text,
    )
    return re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)


def allowed_name(path):
    p = Path(path)
    return not SECRET_NAME.search(p.name) and not any(
        part in BLOCKED_DIRS for part in p.parts
    )


class RepositoryReader:
    def __init__(self, root, *, allowed_roots, max_files=300, max_bytes=160000):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir() or not any(
            self.root.is_relative_to(Path(p).resolve()) for p in allowed_roots
        ):
            raise ValueError("repository is outside explicitly allowed roots")
        self.max_files, self.max_bytes = max_files, max_bytes
        self.trace, self.evidence, self.raw_files = [], [], {}

    def record(self, tool, arguments, result, start=None, refs=()):
        self.trace.append(
            {
                "tool": tool,
                "arguments": arguments,
                "result": result,
                "evidence_refs": list(refs),
                "seconds": time.perf_counter() - start if start else 0.0,
            }
        )

    def resolve(self, relative):
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or not allowed_name(p):
            raise ValueError("file path is disallowed")
        current = self.root
        for part in p.parts:
            current = current / part
            if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
                raise ValueError("symlinks/junctions are not followed")
        target = current.resolve(strict=True)
        if not target.is_relative_to(self.root) or not target.is_file():
            raise ValueError("file escapes repository or is not regular")
        if target.suffix.lower() not in TEXT_SUFFIXES and target.name not in SPECIAL:
            raise ValueError("not an approved text file")
        if target.stat().st_size > self.max_bytes:
            raise ValueError("file exceeds read limit")
        return target

    def list_files(self):
        start = time.perf_counter()
        files = []
        truncated = False
        for directory, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in BLOCKED_DIRS
                and not (Path(directory) / d).is_symlink()
                and not getattr(Path(directory) / d, "is_junction", lambda: False)()
            )
            for name in sorted(names):
                relative = (Path(directory) / name).relative_to(self.root)
                if not allowed_name(relative):
                    continue
                try:
                    self.resolve(relative)
                except (OSError, ValueError):
                    continue
                files.append(relative.as_posix())
                if len(files) >= self.max_files:
                    truncated = True
                    break
            if truncated:
                break
        self.record(
            "list_repository", {}, {"files": files, "truncated": truncated}, start
        )
        return files

    def read(self, relative, *, tool="read_config"):
        start = time.perf_counter()
        p = self.resolve(relative)
        raw = p.read_text(encoding="utf8", errors="replace")
        name = p.relative_to(self.root).as_posix()
        self.raw_files[name] = raw
        ref = self.add_evidence(
            name, raw, 1, len(raw.splitlines()) or 1, "repository_file"
        )
        self.record(
            tool,
            {"path": name},
            {"sha256": file_hash(p), "bytes": p.stat().st_size},
            start,
            [ref],
        )
        return raw, ref

    def add_evidence(self, path, text, line_start, line_end, kind):
        value = {
            "path": str(path),
            "line_start": line_start,
            "line_end": line_end,
            "kind": kind,
            "raw_sha256": digest(text),
            "text": redact(text)[:20000],
        }
        value["id"] = "ev-" + digest(value)[:18]
        if not any(e["id"] == value["id"] for e in self.evidence):
            self.evidence.append(value)
        return value["id"]

    def python_exports(self, module, files):
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            return None
        candidates = [
            module.replace(".", "/") + ".py",
            module.replace(".", "/") + "/__init__.py",
        ]
        candidates += ["src/" + p for p in candidates]
        for path in candidates:
            if path not in files:
                continue
            raw, ref = self.read(path, tool="inspect_python_exports")
            try:
                tree = ast.parse(raw)
            except SyntaxError:
                return None
            exports = [
                n.name
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and not n.name.startswith("_")
            ]
            version = None
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "__version__"
                        for t in node.targets
                    )
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    version = node.value.value
            return {
                "path": path,
                "exports": exports,
                "version": version,
                "evidence_ref": ref,
            }
        return None

    def git_identity(self):
        # Directly inspect bounded local ref files, never run git hooks, config or network.
        metadata = self.root / ".git"
        commit = None
        if (
            metadata.is_dir()
            and not metadata.is_symlink()
            and not getattr(metadata, "is_junction", lambda: False)()
        ):
            head = metadata / "HEAD"
            if head.is_file() and not head.is_symlink() and head.stat().st_size < 512:
                value = head.read_text(encoding="ascii", errors="replace").strip()
                if value.startswith("ref: "):
                    ref = value[5:]
                    p = (metadata / ref).resolve()
                    if (
                        p.is_relative_to(metadata.resolve())
                        and p.is_file()
                        and not p.is_symlink()
                        and p.stat().st_size < 128
                    ):
                        value = p.read_text().strip()
                    else:
                        value = ""
                if re.fullmatch(r"[a-fA-F0-9]{40,64}", value):
                    commit = value
        return {
            "name": self.root.name,
            "path": str(self.root),
            "git_present": metadata.exists(),
            "commit": commit,
            "git_note": "local HEAD/ref only; worktree gitdir and packed refs remain unknown",
        }

    def snapshot(self):
        return digest({k: digest(v) for k, v in sorted(self.raw_files.items())})

    def tree_identity(self, files):
        """Byte identity of the bounded approved file list, not the entire Git worktree."""
        return digest({path: file_hash(self.resolve(path)) for path in sorted(files)})
