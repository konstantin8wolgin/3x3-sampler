from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ".github/workflows/ci.yml"
FORBIDDEN_PAYLOADS = (
    "/home/" "konstantin",
    "execution" "/evidence",
    "handoff" ".md",
    "." "lavish",
    "." "playwright-mcp",
    "__py" "cache__",
    ".pytest" "_cache",
    "tg_3x3_" "sampler",
)
REQUIRES_DIST = "Requires-" "Dist:"
FORBIDDEN_DEPENDENCY = "taylor" "gauss"
CI_MUTATIONS = (
    ("on:\n", "true:\n"),
    ("  push:\n", ""),
    ("  pull_request:\n", ""),
    ("  test:\n", "  test:\n    name: renamed\n"),
    ('        python: ["3.11", "3.12"]', '        python: ["3.10", "3.12"]'),
    ('        python: ["3.11", "3.12"]', '        python: ["3.11", "3.13"]'),
    ("      - uses: actions/checkout@v5\n", ""),
    (
        "      - uses: actions/checkout@v5\n",
        "      - uses: actions/checkout@v4\n",
    ),
    (
        "      - uses: actions/setup-python@v6\n",
        "      - uses: actions/setup-python@v5\n",
    ),
    ("      - run: python -m pip install --upgrade pip\n", ""),
    ("      - run: python -m pip install -e '.[dev]'\n", ""),
    ("      - run: python -m pytest\n", ""),
    ("      - run: python -m build\n", ""),
    (
        "      - run: python -m pytest -q tests/test_clean_install.py "
        "tests/test_public_content.py\n",
        "",
    ),
    (
        "      - uses: actions/setup-python@v6\n"
        "        with:\n"
        "          python-version: ${{ matrix.python }}\n"
        "          cache: pip\n",
        "",
    ),
    (
        "          python-version: ${{ matrix.python }}\n",
        "          python-version: 3.12\n",
    ),
    (
        "      - uses: actions/setup-python@v6\n"
        "        with:\n"
        "          python-version: ${{ matrix.python }}\n"
        "          cache: pip\n",
        "      - uses: actions/setup-python@v6\n"
        "        with:\n"
        "          python-version: ${{ matrix.python }}\n"
        "          cache: pip\n"
        "      - uses: actions/setup-python@v6\n"
        "        with:\n"
        "          python-version: ${{ matrix.python }}\n"
        "          cache: pip\n",
    ),
)
CI_RUN_COMMANDS = (
    "python -m pip install --upgrade pip",
    "python -m pip install -e '.[dev]'",
    "python -m pytest",
    "python -m build",
    "python -m pytest -q tests/test_clean_install.py tests/test_public_content.py",
)
CACHE_PROBES = (
    ("scratch/" "__py" "cache__/sentinel.txt", "__py[c]ache__/"),
    ("scratch/" ".pytest" "_cache/state", ".pytest_[c]ache/"),
)
CACHE_RULE_MUTATIONS = (
    ("__py[c]ache__/\n", ""),
    (".pytest_[c]ache/\n", ""),
)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
    )


def _assert_allowed(path: str, payload: bytes) -> None:
    combined = path.encode("utf-8") + b"\n" + payload
    for forbidden in FORBIDDEN_PAYLOADS:
        assert forbidden.encode("utf-8") not in combined, (
            f"forbidden public content {forbidden!r} in {path}"
        )

    text = payload.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        if not line.startswith(REQUIRES_DIST):
            continue
        requirement = line.removeprefix(REQUIRES_DIST).strip()
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        assert match is not None, f"malformed dependency metadata in {path}: {line}"
        normalized = re.sub(r"[-_.]+", "-", match.group(0)).lower()
        assert normalized != FORBIDDEN_DEPENDENCY, (
            f"forbidden dependency {FORBIDDEN_DEPENDENCY!r} in {path}"
        )


def _workflow_records(payload: str) -> list[tuple[int, str]]:
    records = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        leading = raw[: len(raw) - len(raw.lstrip())]
        assert "\t" not in leading, f"tab indentation on line {line_number}"
        indent = len(leading)
        records.append((indent, raw[indent:]))
    return records


def _mapping_block(
    records: list[tuple[int, str]], key: str, indent: int
) -> list[tuple[int, str]]:
    matches = [
        index
        for index, (actual_indent, content) in enumerate(records)
        if actual_indent == indent and content == f"{key}:"
    ]
    assert len(matches) == 1, f"expected one YAML mapping key {key!r}"
    start = matches[0] + 1
    end = next(
        (
            index
            for index in range(start, len(records))
            if records[index][0] <= indent
        ),
        len(records),
    )
    return records[start:end]


def _sequence_item_blocks(
    records: list[tuple[int, str]], indent: int
) -> list[list[tuple[int, str]]]:
    starts = [
        index
        for index, (actual_indent, content) in enumerate(records)
        if actual_indent == indent and content.startswith("- ")
    ]
    return [
        records[start : starts[index + 1] if index + 1 < len(starts) else len(records)]
        for index, start in enumerate(starts)
    ]


def _assert_ci_contract(payload: str) -> None:
    """Validate the required GitHub Actions subset with YAML 1.2 key semantics."""

    records = _workflow_records(payload)
    trigger_block = _mapping_block(records, "on", 0)
    triggers = []
    for indent, content in trigger_block:
        if indent != 2:
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*):(?:\s+\{\})?", content)
        assert match is not None, f"invalid GitHub event mapping: {content}"
        triggers.append(match.group(1))
    assert {"push", "pull_request"}.issubset(triggers)

    jobs = _mapping_block(records, "jobs", 0)
    test_job = _mapping_block(jobs, "test", 2)
    assert not any(
        indent == 4 and content.startswith("name:")
        for indent, content in test_job
    ), "the test job name must remain implicit for protected matrix contexts"
    strategy = _mapping_block(test_job, "strategy", 4)
    matrix = _mapping_block(strategy, "matrix", 6)
    python_values = [
        content.removeprefix("python:").strip()
        for indent, content in matrix
        if indent == 8 and content.startswith("python:")
    ]
    assert len(python_values) == 1, "expected one Python matrix"
    try:
        python_matrix = json.loads(python_values[0])
    except json.JSONDecodeError as exc:
        raise AssertionError("Python matrix must be a YAML 1.2 flow sequence") from exc
    assert python_matrix == ["3.11", "3.12"]

    steps = _mapping_block(test_job, "steps", 4)
    step_blocks = _sequence_item_blocks(steps, 6)
    checkout_steps = [
        block
        for block in step_blocks
        if block[0] == (6, "- uses: actions/checkout@v5")
    ]
    assert len(checkout_steps) == 1, "expected one actions/checkout@v5 step"
    setup_python_steps = [
        block
        for block in step_blocks
        if block[0] == (6, "- uses: actions/setup-python@v6")
    ]
    assert len(setup_python_steps) == 1, "expected one actions/setup-python@v6 step"
    setup_python_with = _mapping_block(setup_python_steps[0], "with", 8)
    python_versions = [
        content.removeprefix("python-version:").strip()
        for indent, content in setup_python_with
        if indent == 10 and content.startswith("python-version:")
    ]
    assert python_versions == ["${{ matrix.python }}"]

    run_commands = [
        content.removeprefix("- run:").strip()
        for indent, content in steps
        if indent == 6 and content.startswith("- run:")
    ]
    assert run_commands == list(CI_RUN_COMMANDS)


def _assert_cache_ignore_contract(payload: str, workspace: Path) -> None:
    repository = workspace / "cache-ignore-repository"
    repository.mkdir()
    isolated_home = workspace / "isolated-home"
    isolated_home.mkdir()
    isolated_config = workspace / "isolated-gitconfig"
    isolated_config.write_text("", encoding="utf-8")
    isolated_template = workspace / "isolated-git-template"
    isolated_template.mkdir()
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": str(isolated_config),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TEMPLATE_DIR": str(isolated_template),
            "HOME": str(isolated_home),
            "XDG_CONFIG_HOME": str(isolated_home),
        }
    )
    (repository / ".gitignore").write_text(payload, encoding="utf-8")
    initialized = subprocess.run(
        ("git", "init", "--quiet", f"--template={isolated_template}"),
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert initialized.returncode == 0, initialized.stderr
    for path, expected_pattern in CACHE_PROBES:
        ignored = subprocess.run(
            ("git", "check-ignore", "--verbose", "--", path),
            cwd=repository,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert ignored.returncode == 0, f"cache directory rule does not ignore {path}"
        source_and_pattern, matched_path = ignored.stdout.rstrip("\n").split("\t", 1)
        source, _, matched_pattern = source_and_pattern.split(":", 2)
        assert source == ".gitignore"
        assert matched_pattern == expected_pattern
        assert matched_path == path


@pytest.mark.parametrize("forbidden", ("__py" "cache__", ".pytest" "_cache"))
@pytest.mark.parametrize("location", ("path", "payload"))
def test_every_cache_token_is_forbidden_in_paths_and_payloads(
    forbidden: str, location: str
):
    """Catches cache tokens exempted from either half of the content audit."""

    path = f"release/{forbidden}/member" if location == "path" else "release.txt"
    payload = forbidden.encode("utf-8") if location == "payload" else b"public"
    with pytest.raises(AssertionError, match="forbidden public"):
        _assert_allowed(path, payload)


def test_cache_globs_still_ignore_python_and_pytest_cache_members(tmp_path: Path):
    """Catches obfuscated forbidden-token removal that stops ignoring caches."""

    payload = (REPOSITORY / ".gitignore").read_text(encoding="utf-8")
    _assert_cache_ignore_contract(payload, tmp_path)


@pytest.mark.parametrize(("original", "replacement"), CACHE_RULE_MUTATIONS)
def test_cache_contract_rejects_directory_rule_mutations(
    tmp_path: Path, original: str, replacement: str
):
    """Catches either cache-directory rule being removed or broken."""

    payload = (REPOSITORY / ".gitignore").read_text(encoding="utf-8")
    assert original in payload
    mutated = payload.replace(original, replacement, 1)
    with pytest.raises(AssertionError):
        _assert_cache_ignore_contract(mutated, tmp_path)


def test_cache_contract_rejects_external_excludes_false_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catches user/global excludes masking both missing local cache rules."""

    external_excludes = tmp_path / "external-excludes"
    external_excludes.write_text(
        "__py" "cache__/\n" ".pytest" "_cache/\n", encoding="utf-8"
    )
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.excludesFile")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(external_excludes))

    payload = (REPOSITORY / ".gitignore").read_text(encoding="utf-8")
    for original, replacement in CACHE_RULE_MUTATIONS:
        payload = payload.replace(original, replacement, 1)
    with pytest.raises(AssertionError):
        _assert_cache_ignore_contract(payload, tmp_path)


def test_ci_workflow_has_the_release_contract():
    """Catches CI syntax or semantics that no longer run the public release gate."""

    _assert_ci_contract((REPOSITORY / CI_WORKFLOW).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("original", "replacement"), CI_MUTATIONS)
def test_ci_contract_rejects_release_semantic_mutations(
    original: str, replacement: str
):
    """Catches a checker that accepts missing triggers, runtimes, or gate steps."""

    workflow = (REPOSITORY / CI_WORKFLOW).read_text(encoding="utf-8")
    assert original in workflow
    mutated = workflow.replace(original, replacement, 1)
    with pytest.raises(AssertionError):
        _assert_ci_contract(mutated)


def test_tracked_public_content_has_the_ci_fixture_and_no_private_surface():
    """Catches omitted CI or private provenance entering a tracked release."""

    listed = _run("git", "ls-files", "-z")
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert CI_WORKFLOW in tracked, "public release gate has no tracked CI workflow"
    for relative in tracked:
        _assert_allowed(relative, (REPOSITORY / relative).read_bytes())


def test_release_metadata_freezes_numeric_runtime_and_supplies_build_tools():
    """Catches dependency resolution that changes anchors or breaks no-isolation builds."""

    metadata = tomllib.loads(
        (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["project"]["dependencies"] == [
        "numpy==2.4.4",
        "torch==2.11.0",
    ]
    assert set(metadata["build-system"]["requires"]).issubset(
        metadata["project"]["optional-dependencies"]["dev"]
    )
    assert metadata["project"]["license"] == "Apache-2.0"


def test_wheel_members_are_public_and_have_only_public_dependencies(
    tmp_path: Path,
):
    """Catches a wheel carrying repository caches, private names, or package coupling."""

    wheelhouse = tmp_path / "wheelhouse"
    built = _run(
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheelhouse),
        ".",
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = list(wheelhouse.glob("taylorgauss_3x3-*.whl"))
    assert len(wheels) == 1

    metadata_members = []
    with zipfile.ZipFile(wheels[0]) as archive:
        for member in archive.namelist():
            payload = archive.read(member)
            _assert_allowed(member, payload)
            if member.endswith(".dist-info/METADATA"):
                metadata_members.append(member)

    assert len(metadata_members) == 1


def test_built_sdist_contains_the_canonical_anchor_required_by_its_tests(
    tmp_path: Path,
):
    """Catches a source archive whose exact-observables test lacks its fixture."""

    dist_directory = tmp_path / "dist"
    built = _run(
        sys.executable,
        "-m",
        "build",
        "--sdist",
        "--no-isolation",
        "--outdir",
        str(dist_directory),
        ".",
    )
    assert built.returncode == 0, built.stdout + built.stderr
    archives = list(dist_directory.glob("taylorgauss_3x3-*.tar.gz"))
    assert len(archives) == 1

    with tarfile.open(archives[0], mode="r:gz") as archive:
        roots = {Path(name).parts[0] for name in archive.getnames() if name}
        assert len(roots) == 1
        root = roots.pop()
        assert archive.getmember(
            f"{root}/tests/test_exact_observables.py"
        ).isfile()
        fixture = archive.extractfile(
            f"{root}/tests/fixtures/canonical-anchor.json"
        )
        assert fixture is not None
        assert "mixed_linear_quadratic" in json.loads(fixture.read())
