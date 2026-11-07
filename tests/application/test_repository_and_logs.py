import json
import shutil
import subprocess
import pytest
from experienceguard.repository_tools.reader import RepositoryReader
from experienceguard.workflows.service import DevelopmentApp
from experienceguard.ci_analysis.analyzer import analyze


def test_read_boundaries_secret_names_and_no_repository_execution(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text(
        'raise RuntimeError("MUST NOT EXECUTE")\nAPI_KEY = "CANARY_DO_NOT_UPLOAD"\n'
    )
    (repo / ".env").write_text("PASSWORD=CANARY_PRIVATE")
    outside = tmp_path / "outside.txt"
    outside.write_text("PRIVATE")
    reader = RepositoryReader(repo, allowed_roots=[repo])
    assert ".env" not in reader.list_files()
    with pytest.raises(ValueError):
        reader.read(".env")
    with pytest.raises(ValueError):
        reader.read("../outside.txt")
    with pytest.raises(ValueError):
        RepositoryReader(repo, allowed_roots=[tmp_path / "other"])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: pytest.fail("read-only analysis executed repository code"),
    )
    app = DevelopmentApp(tmp_path / "data", allowed_roots=[repo])
    before = (repo / "sample.py").read_bytes()
    result = app.diagnose(
        repo,
        "File \"sample.py\", line 1\nModuleNotFoundError: No module named 'missing_pkg'\n",
    )
    assert result["primary_issue"]["kind"] == "python_missing_module"
    assert b"MUST NOT EXECUTE" in before and (repo / "sample.py").read_bytes() == before
    assert "CANARY_DO_NOT_UPLOAD" not in json.dumps(result)
    assert (
        "CANARY_DO_NOT_UPLOAD"
        in (
            app._case_dir(result["case_id"]) / "raw_repository_evidence.json"
        ).read_text()
    )


def test_limits_and_writes_cannot_be_inside_imported_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "large.py").write_text("x" * 300)
    reader = RepositoryReader(repo, allowed_roots=[repo], max_bytes=100)
    with pytest.raises(ValueError):
        reader.read("large.py")
    app = DevelopmentApp(repo / "appdata", allowed_roots=[repo])
    with pytest.raises(ValueError):
        app.diagnose(repo, "unknown error")


def test_requires_python_is_not_runtime_and_xml_entities_not_followed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname="app"\nrequires-python=">=3.11"\n'
    )
    (repo / "pom.xml").write_text(
        '<!DOCTYPE x [<!ENTITY e SYSTEM "https://example.invalid/secret">]><project>&e;</project>'
    )
    x = analyze(
        RepositoryReader(repo, allowed_roots=[repo]),
        "ModuleNotFoundError: No module named 'x'",
    )
    assert x["facts"]["python"] is None and x["facts"]["python_constraint"] == ">=3.11"
    assert any("pom.xml" in s for s in x["warnings"])


def test_observed_runtime_configuration_conflict_stays_unknown(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".python-version").write_text("3.11")
    x = analyze(
        RepositoryReader(repo, allowed_roots=[repo]),
        "Python 3.13.2\nModuleNotFoundError: No module named 'x'",
    )
    assert x["facts"]["python"] is None and x["facts"]["declared_python"] == "3.11"
    assert any("conflicts" in s for s in x["warnings"])


def test_actual_node_esm_error_and_source_evidence(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node executable not installed; parser does not require Node")
    repo = tmp_path / "node-repo"
    repo.mkdir()
    (repo / "package.json").write_text('{"type":"module","engines":{"node":">=20"}}')
    (repo / "index.mjs").write_text(
        'const path = require("node:path");\nconsole.log(path.sep);\n'
    )
    run = subprocess.run(
        [node, "index.mjs"],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf8",
        errors="replace",
        timeout=10,
    )
    assert run.returncode != 0
    x = analyze(RepositoryReader(repo, allowed_roots=[repo]), run.stdout + run.stderr)
    assert (
        x["primary_issue"]["kind"] == "node_esm"
        and x["facts"]["module_system"] == "module"
    )
    assert x["facts"]["node"] and x["facts"]["node_constraint"] == ">=20"


def test_actual_java_compilation_target_failure(tmp_path):
    javac = shutil.which("javac")
    if not javac:
        pytest.skip("javac not installed; parser does not require a JDK")
    repo = tmp_path / "java-repo"
    repo.mkdir()
    (repo / "Main.java").write_text(
        "class Main { public static void main(String[] args) {} }"
    )
    (repo / "pom.xml").write_text(
        "<project><properties><maven.compiler.release>99</maven.compiler.release></properties></project>"
    )
    run = subprocess.run(
        [
            javac,
            "-J-Duser.language=en",
            "-J-Duser.country=US",
            "--release",
            "99",
            "Main.java",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf8",
        errors="replace",
        timeout=10,
    )
    assert run.returncode != 0
    # JDK locale may be Chinese; its numeric target is real input even if the English rule is unavailable.
    log = run.stdout + run.stderr
    x = analyze(RepositoryReader(repo, allowed_roots=[repo]), log)
    assert x["facts"]["java_target"] == "99" and x["suggestions"]
    assert x["primary_issue"]["kind"] == "java_target"
    assert not (repo / "Main.class").exists()


def test_database_migration_log_and_maven_dependency(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pom.xml").write_text(
        "<project><dependencies><dependency><artifactId>flyway-core</artifactId><version>9.0.0</version></dependency></dependencies></project>"
    )
    x = analyze(
        RepositoryReader(repo, allowed_roots=[repo]),
        "FlywayException: Unsupported Database: PostgreSQL 16.1",
    )
    assert (
        x["primary_issue"]["kind"] == "database_migration"
        and x["facts"]["dependency:flyway-core"] == "9.0.0"
    )
    assert "SQL" in x["suggestions"][0]["text"]
