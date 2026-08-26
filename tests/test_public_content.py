from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
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
    "tg_3x3_" "sampler",
)
FORBIDDEN_PATH_PARTS = ("__py" "cache__", ".pytest" "_cache")
REQUIRES_DIST = "Requires-" "Dist:"
FORBIDDEN_DEPENDENCY = "taylor" "gauss"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
    )


def _assert_allowed(path: str, payload: bytes) -> None:
    for forbidden in FORBIDDEN_PATH_PARTS:
        assert forbidden not in path, f"forbidden public member {forbidden!r}: {path}"

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


def test_tracked_public_content_has_the_ci_fixture_and_no_private_surface():
    """Catches omitted CI or private provenance entering a tracked release."""

    listed = _run("git", "ls-files", "-z")
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert CI_WORKFLOW in tracked, "public release gate has no tracked CI workflow"
    for relative in tracked:
        _assert_allowed(relative, (REPOSITORY / relative).read_bytes())


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
