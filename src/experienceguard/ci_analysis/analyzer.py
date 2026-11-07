"""Evidence rules for a deliberately small set of Python/Node/Java failures; not an LLM."""

from __future__ import annotations
import ast
import configparser
import difflib
import io
import json
import re
import tokenize
import tomllib
import xml.etree.ElementTree as ET
from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version, InvalidVersion
from urllib.parse import unquote
from ..repository_tools.reader import redact
from ..domain.development import DevelopmentContext
from ..utils.core import digest


def parse_log(text):
    issues = []
    locations = []
    patterns = [
        (
            "python_native_abi",
            r"(?:_ARRAY_API not found|numpy\.core\.multiarray failed to import|module compiled against ABI version)[^\n]*",
        ),
        (
            "python_import_symbol",
            r"ImportError: cannot import name ['\"](?P<symbol>\w+)['\"] from ['\"](?P<module>[\w.]+)(?:['\"]|$)",
        ),
        (
            "python_call_signature",
            r"TypeError: (?P<symbol>[\w.]+)\(\) got an unexpected keyword argument ['\"](?P<argument>\w+)['\"]",
        ),
        (
            "python_attribute",
            r"AttributeError: (?:module ['\"](?P<module>[\w.]+)['\"]|['\"](?P<object>\w+)['\"] object) has no attribute ['\"](?P<symbol>\w+)['\"]",
        ),
        ("python_build_attribute", r"AttributeError: (?P<symbol>[A-Za-z_]\w*)\s*$"),
        (
            "python_missing_module",
            r"ModuleNotFoundError: No module named ['\"](?P<module>[\w.]+)['\"]",
        ),
        (
            "node_esm",
            r"(?:\bERR_REQUIRE_ESM\b|require is not defined in ES module scope|require\(\) of ES Module[^\n]*not supported)",
        ),
        ("node_crypto", r"(?:ERR_OSSL_EVP_UNSUPPORTED|error:0308010C:)[^\n]*"),
        ("node_engine", r"\b(?:EBADENGINE|Unsupported engine)\b"),
        ("typescript_compile", r"\berror TS(?P<code>\d+):[^\n]*"),
        (
            "java_target",
            r"(?:invalid target release|release version)[: ]+(?P<target>\d+)",
        ),
        ("java_bytecode", r"UnsupportedClassVersionError[^\n]*"),
        (
            "java_linkage",
            r"(?:java\.lang\.)?(?:NoSuchFieldError|NoSuchMethodError)[^\n]*",
        ),
        (
            "java_module_access",
            r"(?:IllegalAccessError|InaccessibleObjectException)[^\n]*",
        ),
        (
            "database_migration",
            r"(?:FlywayException|Unsupported Database:|Migration checksum mismatch)[^\n]*",
        ),
    ]
    for kind, pattern in patterns:
        for m in re.finditer(pattern, text, re.M):
            detected_kind = kind
            if kind == "python_build_attribute" and not re.search(
                r"Collecting|build|egg_info|setup\.py|get_source_files", text, re.I
            ):
                detected_kind = "python_attribute"
            if any(
                i["kind"] == detected_kind
                and i.get("module") == m.groupdict().get("module")
                for i in issues
            ):
                continue
            issues.append(
                {
                    "kind": detected_kind,
                    **{k: v for k, v in m.groupdict().items() if v is not None},
                    "line": text.count("\n", 0, m.start()) + 1,
                    "message": redact(m.group(0)[:1000]),
                }
            )
    for m in re.finditer(
        r'File ["\x27]?(?P<path>[^\n"\x27]+?\.py)["\x27]?, line (?P<line>\d+)', text
    ):
        locations.append({"path": m["path"], "line": int(m["line"])})
    for m in re.finditer(
        r"(?:file://)?(?P<path>(?:[A-Za-z]:)?[^\s:\x27\"()]+?\.(?:py|java|[cm]?js|ts))[:(]\[?(?P<line>\d+)",
        text,
        re.M,
    ):
        locations.append({"path": m["path"], "line": int(m["line"])})
    return issues, locations


def analyze(reader, log):
    files = reader.list_files()
    facts = {}
    evidence_keys = {}
    warnings = []
    project_name = None

    def fact(key, value, ref, origin):
        if value is None:
            return
        if key.startswith("dependency:") or key.startswith("locked:dependency:"):
            try:
                Version(str(value))
            except InvalidVersion:
                warnings.append("Unresolved dependency version for " + key)
                facts[key] = None
                return
        if key in facts and facts[key] != str(value):
            facts[key] = None
            warnings.append("Conflicting observations for " + key)
        else:
            facts[key] = str(value)
        evidence_keys.setdefault(key, []).append(
            {"evidence_ref": ref, "origin": origin}
        )

    config_names = {
        "pyproject.toml",
        "package.json",
        "package-lock.json",
        "pom.xml",
        "requirements.txt",
        "environment.json",
        "setup.cfg",
        ".python-version",
        ".nvmrc",
        ".node-version",
        ".tool-versions",
    }
    for path in files:
        if path.split("/")[-1] not in config_names and not (
            path.startswith(".github/workflows/") and path.endswith((".yaml", ".yml"))
        ):
            continue
        raw, ref = reader.read(path)
        try:
            if path.endswith("environment.json"):
                data = json.loads(raw)
                for key, value in data.get("runtimes", {}).items():
                    if (
                        key in {"python", "node", "java"}
                        and isinstance(value, str)
                        and len(value) < 80
                    ):
                        fact(
                            key,
                            value,
                            ref,
                            "explicit_user_report_not_independent_measurement",
                        )
                for key, value in data.get("dependencies", {}).items():
                    if re.fullmatch(r"[\w.@/-]{1,100}", key) and isinstance(value, str):
                        fact(
                            "dependency:" + key.lower().replace("_", "-"),
                            value,
                            ref,
                            "explicit_reported_dependency_not_lockfile",
                        )
                for key in ("database", "database_version"):
                    if isinstance(data.get(key), str) and len(data[key]) < 80:
                        fact(key, data[key], ref, "explicit_user_report")
            elif path.endswith("setup.cfg"):
                data = configparser.ConfigParser(interpolation=None)
                data.read_string(raw)
                if data.has_section("metadata") or data.has_section("options"):
                    fact("language", "python", ref, "build_configuration")
                    if data.has_option("options", "python_requires"):
                        fact(
                            "python_constraint",
                            data["options"]["python_requires"],
                            ref,
                            "declared_constraint_not_runtime",
                        )
            elif path.endswith(".python-version"):
                v = raw.strip()
                if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", v):
                    fact("declared_python", v, ref, "declared_runtime_not_observed")
            elif path.endswith((".nvmrc", ".node-version")):
                v = raw.strip().lstrip("v")
                if re.fullmatch(r"\d+(?:\.\d+){0,2}", v):
                    fact("declared_node", v, ref, "declared_runtime_not_observed")
            elif path.endswith("pyproject.toml"):
                data = tomllib.loads(raw).get("project", {})
                if isinstance(data.get("name"), str) and len(data["name"]) <= 100:
                    project_name = data["name"]
                fact("language", "python", ref, "build_configuration")
                if data.get("requires-python"):
                    fact(
                        "python_constraint",
                        data["requires-python"],
                        ref,
                        "declared_constraint_not_observed_runtime",
                    )
                for dep in data.get("dependencies", []):
                    try:
                        r = Requirement(dep)
                    except InvalidRequirement:
                        continue
                    specs = list(r.specifier)
                    if (
                        len(specs) == 1
                        and specs[0].operator == "=="
                        and "*" not in specs[0].version
                    ):
                        fact(
                            "dependency:" + r.name.lower().replace("_", "-"),
                            specs[0].version,
                            ref,
                            "declared_pin_not_installed_proof",
                        )
            elif path.endswith("requirements.txt"):
                for line in raw.splitlines():
                    m = re.fullmatch(r"\s*([\w.-]+)==([0-9][\w.+-]*)\s*", line)
                    if m:
                        fact(
                            "dependency:" + m[1].lower().replace("_", "-"),
                            m[2],
                            ref,
                            "declared_pin_not_installed_proof",
                        )
            elif path.endswith("package.json"):
                data = json.loads(raw)
                if isinstance(data.get("name"), str) and len(data["name"]) <= 100:
                    project_name = data["name"]
                fact("language", "node", ref, "build_configuration")
                fact(
                    "module_system",
                    data.get("type", "commonjs"),
                    ref,
                    "package_configuration",
                )
                if data.get("engines", {}).get("node"):
                    fact(
                        "node_constraint",
                        data["engines"]["node"],
                        ref,
                        "declared_constraint_not_runtime",
                    )
                for name, value in {
                    **data.get("dependencies", {}),
                    **data.get("devDependencies", {}),
                }.items():
                    if isinstance(value, str) and re.fullmatch(
                        r"\d+\.\d+\.\d+(?:-[\w.]+)?", value
                    ):
                        fact("dependency:" + name, value, ref, "declared_pin")
            elif path.endswith("package-lock.json"):
                data = json.loads(raw)
                for name, item in data.get("packages", {}).items():
                    if (
                        name.startswith("node_modules/")
                        and isinstance(item, dict)
                        and item.get("version")
                    ):
                        key = "dependency:" + name.removeprefix("node_modules/")
                        # Lock data is a separate fact, not proof that node_modules matches it.
                        fact("locked:" + key, item["version"], ref, "lockfile")
                        if key not in facts:
                            fact(key, item["version"], ref, "lockfile_resolution")
            elif path.endswith("pom.xml"):
                if "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper():
                    raise ValueError("XML DTD/entities rejected")
                doc = ET.fromstring(raw)
                fact("language", "java", ref, "build_configuration")

                def local(element):
                    return element.tag.split("}")[-1]

                props = {
                    local(e): (e.text or "").strip()
                    for e in doc.iter()
                    if local(e)
                    in {
                        "java.version",
                        "maven.compiler.release",
                        "maven.compiler.target",
                    }
                }
                for key, value in props.items():
                    fact("java_target", value, ref, key)
                for dep in doc.iter():
                    if local(dep) != "dependency":
                        continue
                    parts = {local(e): (e.text or "").strip() for e in dep}
                    if parts.get("artifactId") and re.fullmatch(
                        r"\d[\w.+-]*", parts.get("version", "")
                    ):
                        fact(
                            "dependency:" + parts["artifactId"],
                            parts["version"],
                            ref,
                            "maven_declared_version",
                        )
            else:
                for language, pattern in [
                    ("python", r"python-version:\s*[\"\x27]?(\d+\.\d+(?:\.\d+)?)"),
                    ("node", r"node-version:\s*[\"\x27]?(\d+(?:\.\d+){0,2})"),
                ]:
                    m = re.search(pattern, raw)
                    if m:
                        fact(
                            "declared_" + language,
                            m[1],
                            ref,
                            "CI_configuration_not_observed",
                        )
        except (
            ValueError,
            TypeError,
            AttributeError,
            ET.ParseError,
            configparser.Error,
        ) as exc:
            warnings.append("Could not parse " + path + ": " + type(exc).__name__)
    log_ref = reader.add_evidence(
        "ci.log", log, 1, len(log.splitlines()) or 1, "failure_log"
    )
    issues, locations = parse_log(log)
    reader.record(
        "parse_ci_log",
        {"log_sha256": digest(log)},
        {"issues": issues, "locations": locations},
        refs=[log_ref],
    )
    for key, pattern in [
        ("python", r"\bPython\s+(\d+\.\d+(?:\.\d+)?)"),
        ("node", r"\bNode(?:\.js)?(?: version:)?\s+v?(\d+\.\d+(?:\.\d+)?)"),
        ("java", r"\b(?:java|openjdk) version\s+[\"\x27]?(\d+(?:\.\d+){0,2})"),
    ]:
        m = re.search(pattern, log, re.I)
        if m:
            fact(key, m[1], log_ref, "runtime_reported_in_log")
    for key in ("python", "node"):
        declared = facts.get("declared_" + key)
        observed = facts.get(key)
        if (
            declared
            and observed
            and observed.split(".")[: len(declared.split("."))] != declared.split(".")
        ):
            warnings.append(
                "Declared "
                + key
                + " conflicts with runtime log; manual reconciliation required"
            )
            facts[key] = None
    for location in locations[-20:]:
        rawpath = unquote(location["path"]).replace("\\", "/")
        matches = [p for p in files if rawpath == p or rawpath.endswith("/" + p)]
        if not matches:
            # Java stack traces often provide a basename only; ambiguity stays unresolved.
            matches = [p for p in files if p.split("/")[-1] == rawpath.split("/")[-1]]
        if len(matches) == 1:
            raw, _ = reader.read(matches[0], tool="locate_source")
            lines = raw.splitlines()
            n = location["line"]
            if not 1 <= n <= len(lines):
                warnings.append("Source location outside current file: " + matches[0])
                continue
            reader.add_evidence(
                matches[0],
                "\n".join(lines[max(0, n - 3) : n + 2]),
                max(1, n - 2),
                min(len(lines), n + 2),
                "source_location",
            )
    primary = (
        issues[0]
        if issues
        else {
            "kind": "unclassified",
            "message": "No supported failure signature detected",
            "line": 1,
        }
    )
    fact("issue_kind", primary["kind"], log_ref, "parsed_log")
    if primary.get("module"):
        fact("module", primary["module"], log_ref, "parsed_log")
    if primary["kind"].startswith("python_"):
        facts["language"] = "python"
        evidence_keys.setdefault("language", []).append(
            {"evidence_ref": log_ref, "origin": "current_exception_signature"}
        )
    if primary["kind"].startswith("node_") or primary["kind"] == "typescript_compile":
        facts["language"] = "node"
        evidence_keys.setdefault("language", []).append(
            {"evidence_ref": log_ref, "origin": "current_exception_signature"}
        )
    if primary["kind"].startswith("java_") or primary["kind"] == "database_migration":
        facts["language"] = "java"
        evidence_keys.setdefault("language", []).append(
            {"evidence_ref": log_ref, "origin": "current_exception_signature"}
        )
    proposals = []
    suggestions = []
    exports = None
    if primary["kind"] == "python_import_symbol":
        scanned = 0
        for path in files:
            if not path.endswith(".py"):
                continue
            if scanned >= 2000000:
                warnings.append(
                    "Import-source AST search stopped at 2MB; remaining files uninspected"
                )
                break
            raw, _ = reader.read(path, tool="locate_source")
            scanned += len(raw.encode("utf8"))
            try:
                nodes = ast.walk(ast.parse(raw))
                for node in nodes:
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.module == primary["module"]
                        and any(a.name == primary["symbol"] for a in node.names)
                    ):
                        lines = raw.splitlines()
                        start = max(1, node.lineno - 2)
                        end = min(len(lines), (node.end_lineno or node.lineno) + 2)
                        reader.add_evidence(
                            path,
                            "\n".join(lines[start - 1 : end]),
                            start,
                            end,
                            "source_location",
                        )
            except SyntaxError:
                continue
    if primary.get("module"):
        exports = reader.python_exports(primary["module"], files)
        if exports and exports["version"]:
            fact(
                "dependency:"
                + primary["module"].split(".")[0].lower().replace("_", "-"),
                exports["version"],
                exports["evidence_ref"],
                "vendored_source_version",
            )
    if primary["kind"] == "python_import_symbol" and exports:
        suggestions.append(
            {
                "text": f"导入符号 {primary['symbol']} 不在当前模块 {primary['module']} 的顶层导出中；实际发现：{', '.join(exports['exports']) or '无可见顶层定义'}。请核对 API 与依赖版本。",
                "basis": "deterministic AST/log evidence",
                "evidence_refs": [log_ref, exports["evidence_ref"]],
            }
        )
        if len(exports["exports"]) == 1:
            old, new = primary["symbol"], exports["exports"][0]
            for path in files:
                if not path.endswith(".py") or path == exports["path"]:
                    continue
                raw, ref = reader.read(path, tool="locate_source")
                try:
                    tree = ast.parse(raw)
                except SyntaxError:
                    continue
                found = any(
                    isinstance(n, ast.ImportFrom)
                    and n.module == primary["module"]
                    and any(a.name == old and a.asname is None for a in n.names)
                    for n in ast.walk(tree)
                )
                if not found:
                    continue
                tokens = list(tokenize.generate_tokens(io.StringIO(raw).readline))
                fixed = tokenize.untokenize(
                    [
                        t._replace(string=new)
                        if t.type == tokenize.NAME and t.string == old
                        else t
                        for t in tokens
                    ]
                )
                proposals.append(
                    {
                        "id": "proposal-" + digest([path, raw, new])[:12],
                        "path": path,
                        "old_symbol": old,
                        "new_symbol": new,
                        "module": primary["module"],
                        "original_sha": digest(raw),
                        "evidence_refs": [log_ref, ref, exports["evidence_ref"]],
                        "diff": "\n".join(
                            difflib.unified_diff(
                                raw.splitlines(),
                                fixed.splitlines(),
                                fromfile=path,
                                tofile=path,
                                lineterm="",
                            )
                        ),
                        "manual_review_required": True,
                        "not_proven_semantically_equivalent": True,
                    }
                )
    messages = {
        "python_import_symbol": "运行时报告导入名不可用。定位调用方 import，核对提供方公开 API、实际依赖版本与模块遮蔽；本地缺少提供方源码不等于未安装。当前证据不能证明任意替代符号语义等价。",
        "python_missing_module": "当前解释器找不到该模块。核对声明依赖、实际解释器和安装环境；包名不一定等于导入名，不自动执行 pip。",
        "node_esm": "错误涉及 CommonJS/ESM 边界。核对当前 package type、调用文件扩展名和依赖导出；不要盲套旧版本 require/import 改动。",
        "node_engine": "日志报告 Node engine 不匹配。核对 CI 实际 Node 版本与 package engines；约束范围不是已观测运行时。",
        "node_crypto": "加密后端拒绝了当前哈希/算法。核对 Node/OpenSSL 版本和构建器使用的算法；区分运行时升级引起的策略变化与代码错误，检查兼容构建器，不能凭错误码断言已修复。",
        "python_native_abi": "错误指向原生扩展与 NumPy ABI 边界。核对扩展编译所用 ABI、运行时 NumPy 和平台 wheel；Python 源码相同不能证明二进制兼容。",
        "python_call_signature": "调用方传入的关键字不在当前被调用 API 的签名中。核对实际提供方/调用方版本、安装路径与是否混用 vendored 依赖，不盲目删除参数。",
        "python_attribute": "对象或模块的属性在当前运行时不可用。定位真实调用方并核对提供方 API 变化和对象类型；没有运行时对象/版本证据时不猜替代方法。",
        "python_build_attribute": "构建过程中的属性缺失。检查隔离构建环境、build-system.requires 和实际构建工具版本；运行环境锁文件不一定约束隔离构建依赖。",
        "java_linkage": "已编译调用方与当前类的字段或方法签名不一致。核对编译器内部 API、注解处理器与依赖二进制版本，单改 target 不会自动修复链接错误。",
        "java_module_access": "Java 模块访问边界阻止了反射或编译器插件。检查当前 JDK、插件支持版本和模块声明；临时开放访问不能替代兼容性验证。",
        "typescript_compile": "TypeScript 编译器报告配置或类型错误。保留 TS 错误码，核对模块输出、模块解析与 extends 配置；不能用 JavaScript 执行成功代替类型检查。",
        "java_target": "编译目标与工具链可能不兼容。核对 CI JDK、Maven/Gradle toolchain 与 target/release，勿只改版本号后宣称通过。",
        "java_bytecode": "日志包含字节码版本不兼容。核对实际运行 JDK 与构建目标，保留原错误定位。",
        "database_migration": "迁移工具报告数据库支持或校验问题。核对当前数据库、迁移工具及驱动版本，先备份并人工审阅；本应用不执行 SQL。",
        "unclassified": "现有确定性规则不能诊断该错误。补充完整首个异常、运行时和构建日志；语义 LLM 分析当前不可用。",
    }
    if primary["kind"] in messages:
        suggestions.append(
            {
                "text": messages[primary["kind"]],
                "basis": "documented deterministic rule",
                "evidence_refs": [log_ref],
            }
        )
    if not suggestions:
        suggestions.append(
            {
                "text": "确认错误符号与当前依赖公开 API 后再制定修复；当前证据不足以给出确定修复。",
                "basis": "evidence insufficiency",
                "evidence_refs": [log_ref],
            }
        )
    runtime = {"python": "python", "node": "node", "java": "java"}.get(
        facts.get("language")
    )
    if runtime and runtime not in facts:
        facts[runtime] = None
    query = " ".join(
        [
            primary["kind"],
            primary.get("module", ""),
            primary.get("symbol", ""),
            facts.get("language") or "",
        ]
    )
    return {
        "project_name": project_name,
        "context": DevelopmentContext(query, facts),
        "files": files,
        "facts": facts,
        "fact_evidence": evidence_keys,
        "warnings": warnings,
        "issues": issues,
        "primary_issue": primary,
        "proposals": proposals,
        "suggestions": suggestions,
        "log_ref": log_ref,
        "validation_suggestions": [
            {
                "argv": ["python", "-m", "unittest", "discover", "-v"]
                if facts.get("language") == "python"
                else ["npm", "test"]
                if facts.get("language") == "node"
                else ["mvn", "test"],
                "execution": "manual only in a reviewed independent copy; never auto-run on an imported repository",
            }
        ],
    }
