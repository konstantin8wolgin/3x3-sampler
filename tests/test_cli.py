from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCOPE = "periodic 3x3, n_t=1"


@pytest.fixture(scope="session")
def installed_environment(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Install the distribution locally while reusing host CPU dependencies."""

    environment = tmp_path_factory.mktemp("installed") / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = environment / "bin" / "python"
    installed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(REPOSITORY),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    return environment


@pytest.fixture(scope="session")
def installed_tg_3x3(installed_environment: Path) -> str:
    return str(installed_environment / "bin" / "tg-3x3")


@pytest.fixture(scope="session")
def installed_python(installed_environment: Path) -> str:
    return str(installed_environment / "bin" / "python")


def _run(*arguments: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=cwd, text=True, capture_output=True)


def _digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_exact_cli_journey_and_immutable_offline_derivatives(
    tmp_path: Path, installed_tg_3x3: str
):
    """Catches a broken installed launcher, command route, or immutable write."""

    output = tmp_path / "exact"
    sampled = _run(
        installed_tg_3x3,
        "sample",
        "--method",
        "exact-enumeration",
        "--output",
        str(output),
    )
    assert sampled.returncode == 0, sampled.stderr
    assert (
        json.loads((output / "summary.json").read_text(encoding="utf-8"))[
            "exact_support"
        ]["count"]
        == 19_683
    )
    assert "channels=19683 samples=none" in sampled.stdout

    described = _run(installed_tg_3x3, "describe")
    assert described.returncode == 0, described.stderr
    description = json.loads(described.stdout)
    assert description["geometry"] == "periodic_3x3"
    assert description["euclidean_time_slices"] == 1
    assert description["exact_channel_count"] == 19_683

    validated = _run(installed_tg_3x3, "validate", str(output))
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["valid"] is True

    estimate_output = tmp_path / "estimated"
    estimated = _run(
        installed_tg_3x3,
        "estimate",
        str(output),
        "--output",
        str(estimate_output),
    )
    assert estimated.returncode == 0, estimated.stderr
    assert json.loads(
        (estimate_output / "summary.json").read_text(encoding="utf-8")
    )["estimate"] == json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )["estimate"]

    report_output = tmp_path / "reported"
    reported = _run(
        installed_tg_3x3,
        "report",
        str(output),
        "--output",
        str(report_output),
    )
    assert reported.returncode == 0, reported.stderr
    assert (report_output / "report.html").is_file()

    before = _digest_tree(output)
    reused = _run(
        installed_tg_3x3,
        "sample",
        "--method",
        "exact-enumeration",
        "--output",
        str(output),
    )
    assert reused.returncode == 2
    assert "immutable output already exists" in reused.stderr
    assert SCOPE in reused.stderr
    assert _digest_tree(output) == before


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("sample",),
        ("describe",),
        ("validate",),
        ("estimate",),
        ("report",),
    ],
)
def test_every_help_path_states_the_fixed_scope(installed_tg_3x3: str, arguments):
    """Catches help that presents the CLI as a general lattice sampler."""

    result = _run(installed_tg_3x3, *arguments, "--help")
    assert result.returncode == 0, result.stderr
    assert SCOPE in result.stdout


def test_module_entry_and_parser_errors_state_the_fixed_scope(
    installed_python: str, installed_tg_3x3: str
):
    """Catches divergence between module/console entry points or vague errors."""

    module_help = _run(installed_python, "-m", "taylorgauss_3x3", "--help")
    assert module_help.returncode == 0, module_help.stderr
    assert SCOPE in module_help.stdout

    invalid = _run(installed_tg_3x3, "sample", "--method", "smc")
    assert invalid.returncode == 2
    assert SCOPE in invalid.stderr
    assert "smc" in invalid.stderr


@pytest.mark.parametrize("command", ["sample", "describe"])
def test_multi_slice_request_points_to_explicit_limitation(
    tmp_path: Path, installed_tg_3x3: str, command: str
):
    """Catches silent target changes or a multi-slice rejection without guidance."""

    arguments = [installed_tg_3x3, command, "--n-t", "2"]
    if command == "sample":
        arguments.extend(["--output", str(tmp_path / "invalid")])
    result = _run(*arguments)
    assert result.returncode == 2
    assert SCOPE in result.stderr
    assert "docs/limitations.md" in result.stderr
    assert not (tmp_path / "invalid").exists()


def test_stochastic_methods_and_channel_designs_are_the_only_cli_choices(
    tmp_path: Path, installed_tg_3x3: str
):
    """Catches accidental exposure of benchmark or unsupported sampling choices."""

    described = _run(
        installed_tg_3x3,
        "describe",
        "--method",
        "exact-contour-rb",
        "--samples",
        "20",
        "--channel-design",
        "defensive_half_uniform_importance",
    )
    assert described.returncode == 0, described.stderr
    payload = json.loads(described.stdout)
    assert payload["requested_sample_count"] == 20
    assert payload["exact_channel_count"] == 19_683

    unsupported_design = _run(
        installed_tg_3x3,
        "sample",
        "--method",
        "exact-contour",
        "--channel-design",
        "adaptive",
        "--output",
        str(tmp_path / "unsupported"),
    )
    assert unsupported_design.returncode == 2
    assert SCOPE in unsupported_design.stderr


def test_public_field_docs_match_structural_stochastic_output(
    tmp_path: Path, installed_tg_3x3: str
):
    """Catches public SE/value definitions contradicting structural output."""

    output = tmp_path / "physical-log-partition"
    result = _run(
        installed_tg_3x3,
        "sample",
        "--method",
        "exact-contour-rb",
        "--observable",
        "physical_log_partition",
        "--samples",
        "2",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["estimate"] == {
        "standard_error_imag": None,
        "standard_error_real": None,
        "value": {"imag": 0.0, "real": pytest.approx(32.36358464798972)},
    }

    for relative in ("README.md", "docs/sampling-data.md", "docs/limitations.md"):
        paragraphs = (REPOSITORY / relative).read_text(encoding="utf-8").split("\n\n")
        matching = [
            paragraph.lower()
            for paragraph in paragraphs
            if "`physical_log_partition`" in paragraph and "`null`" in paragraph
        ]
        assert matching, f"{relative} omits the structural null-SE exception"
        assert "physical log partition" in matching[0]
        assert "sample" in matching[0]


@pytest.mark.parametrize(
    ("script", "output", "expected_samples"),
    [
        ("exact_cpu.py", "exact-example", None),
        ("stochastic_rb_cpu.py", "stochastic-rb-example", 20_000),
    ],
)
def test_installed_distribution_executes_cpu_examples(
    tmp_path: Path,
    installed_python: str,
    script: str,
    output: str,
    expected_samples: int | None,
):
    """Catches examples that rely on a source checkout instead of installation."""

    result = _run(
        installed_python,
        str(REPOSITORY / "examples" / script),
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    run = json.loads(
        (tmp_path / "runs" / output / "run.json").read_text(encoding="utf-8")
    )
    assert run["geometry"] == "periodic_3x3"
    assert run["euclidean_time_slices"] == 1
    if expected_samples is None:
        assert run["method"] == "exact-enumeration"
    else:
        assert run["method"] == "exact-contour-rb"
        assert run["sample_count"] == expected_samples
