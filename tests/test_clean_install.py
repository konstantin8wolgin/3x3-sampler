from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
RESEARCH_REPOSITORY = REPOSITORY.with_name("taylorgauss")


def _run(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=cwd, text=True, capture_output=True)


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def _environment_program(environment: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return environment / directory / f"{name}{suffix}"


def test_built_wheel_runs_the_console_and_exact_example_outside_the_checkout(
    tmp_path_factory: pytest.TempPathFactory,
):
    """Catches a wheel coupled to repository files or an undeclared local install."""

    workspace = tmp_path_factory.mktemp("standalone-wheel")
    for repository in (REPOSITORY, RESEARCH_REPOSITORY):
        assert not workspace.resolve().is_relative_to(repository.resolve())

    wheelhouse = workspace / "wheelhouse"
    built = _run(
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheelhouse),
        ".",
        cwd=REPOSITORY,
    )
    _assert_success(built)
    wheels = list(wheelhouse.glob("taylorgauss_3x3-*.whl"))
    assert len(wheels) == 1

    environment = workspace / "venv"
    created = _run(
        sys.executable,
        "-m",
        "venv",
        "--system-site-packages",
        str(environment),
        cwd=workspace,
    )
    _assert_success(created)
    python = _environment_program(environment, "python")
    installed = _run(
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        str(wheels[0]),
        cwd=workspace,
    )
    _assert_success(installed)

    imported = _run(
        str(python),
        "-c",
        "import pathlib, taylorgauss_3x3; "
        "print(pathlib.Path(taylorgauss_3x3.__file__).resolve())",
        cwd=workspace,
    )
    _assert_success(imported)
    assert Path(imported.stdout.strip()).is_relative_to(environment.resolve())

    described = _run(
        str(_environment_program(environment, "tg-3x3")),
        "describe",
        cwd=workspace,
    )
    _assert_success(described)
    description = json.loads(described.stdout)
    assert description["geometry"] == "periodic_3x3"
    assert description["euclidean_time_slices"] == 1
    assert description["exact_channel_count"] == 19_683
    assert description["method"] == "exact-enumeration"
    assert description["authority"] == "exact_reference"

    example = workspace / "exact_cpu.py"
    shutil.copy2(REPOSITORY / "examples" / "exact_cpu.py", example)
    completed = _run(str(python), str(example), cwd=workspace)
    _assert_success(completed)
    assert "authority=exact_reference channels=19683 samples=none" in completed.stdout

    summary = json.loads(
        (workspace / "runs" / "exact-example" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["exact_support"] == {"count": 19_683, "ordered": True}
    value = summary["estimate"]["value"]
    assert value["real"] == pytest.approx(1.2384040693411869, rel=2e-12, abs=2e-12)
    assert value["imag"] == pytest.approx(
        -2.0508704419898993e-15, rel=2e-12, abs=2e-12
    )
