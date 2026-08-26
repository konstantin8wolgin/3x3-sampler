from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from taylorgauss_3x3 import api
from taylorgauss_3x3.artifacts import derive_estimates, validate_run, write_run
from taylorgauss_3x3.config import StochasticRunConfig
from taylorgauss_3x3.core import Hubbard3x3Target
from taylorgauss_3x3.core.observables import approved_observables
from taylorgauss_3x3.reporting import render_report
from taylorgauss_3x3.sampling import allocate_channels, estimate_rao_blackwell


def _refresh_hashes(run: Path) -> None:
    import hashlib

    manifest = json.loads((run / "hashes.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        payload = (run / entry["path"]).read_bytes()
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["size_bytes"] = len(payload)
    (run / "hashes.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_offline_derivation_rejects_overflowed_sufficient_statistics(tmp_path: Path):
    """Catches JSON exponent overflow that does not affect the stored point estimate."""

    source = tmp_path / "source"
    write_run(
        StochasticRunConfig(
            method="exact-contour-rb",
            samples=3,
            seed=97,
            chunk_size=2,
        ),
        source,
    )
    copied = tmp_path / "overflowed-statistics"
    shutil.copytree(source, copied)
    rows = [
        json.loads(line)
        for line in (copied / "estimates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows[0]["real_statistics"]["positive_log_sum"] = "__OVERFLOW__"
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).replace('"__OVERFLOW__"', "1e309")
    (copied / "estimates.jsonl").write_text(payload, encoding="utf-8")
    _refresh_hashes(copied)

    with pytest.raises(ValueError, match="nonfinite|finite"):
        derive_estimates(copied)


def test_offline_derivation_rejects_nonfinite_persisted_endpoints(tmp_path: Path):
    """Catches offline use of a rehashed sample store containing a NaN endpoint."""

    source = tmp_path / "endpoint-source"
    write_run(
        StochasticRunConfig(
            method="exact-contour",
            samples=3,
            seed=101,
            chunk_size=2,
            persist_endpoints=True,
        ),
        source,
    )
    copied = tmp_path / "nonfinite-endpoint"
    shutil.copytree(source, copied)
    chunk = copied / "samples.zarr" / "endpoint" / "0.0"
    values = np.frombuffer(chunk.read_bytes(), dtype=np.complex128).copy()
    values[0] = complex(float("nan"), 0.0)
    chunk.write_bytes(values.tobytes())
    _refresh_hashes(copied)

    with pytest.raises(ValueError, match="endpoint.*finite|nonfinite"):
        derive_estimates(copied)


def test_offline_derivation_and_rendering_do_not_construct_physics(tmp_path: Path):
    """Catches offline commands that rerun the action or exact oracle."""

    run = tmp_path / "run"
    config = StochasticRunConfig(
        method="exact-contour-rb",
        samples=9,
        seed=83,
        chunk_size=4,
    )
    api.run_rao_blackwell(config, run)
    script = r'''
import json
import sys

import taylorgauss_3x3.actions as actions
import taylorgauss_3x3.core.hubbard as hubbard

def forbidden(*args, **kwargs):
    raise AssertionError("offline path constructed physics")

actions.build_action = forbidden
hubbard.Hubbard3x3Target.__init__ = forbidden
hubbard.Hubbard3x3Target.exact_indexed_oracle = forbidden

import taylorgauss_3x3.artifacts as artifacts

artifacts.build_action = forbidden
artifacts.Hubbard3x3Target = forbidden
artifacts._exact_value = forbidden
artifacts._stochastic_rows = forbidden

from taylorgauss_3x3.artifacts import derive_estimates
from taylorgauss_3x3.reporting import render_report

run = sys.argv[1]
derived = derive_estimates(run)
rendered = render_report(run)
estimate_copy = derive_estimates(run, run + "-estimate")
report_copy = render_report(run, run + "-report")
print(json.dumps({
    "derived": derived,
    "estimate_copy": str(estimate_copy),
    "rendered": str(rendered),
    "report_copy": str(report_copy),
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(run)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    target = Hubbard3x3Target(**config.parameters)
    oracle = target.exact_indexed_oracle()
    allocation = allocate_channels(
        oracle,
        sample_count=config.samples,
        seed=config.seed,
        design=config.channel_design,
    )
    expected = estimate_rao_blackwell(
        oracle,
        approved_observables()[config.observable],
        allocation,
    )

    assert result["derived"]["value"]["real"] == pytest.approx(
        expected.value.real, rel=2e-13, abs=2e-13
    )
    assert result["derived"]["value"]["imag"] == pytest.approx(
        expected.value.imag, rel=2e-13, abs=2e-13
    )
    assert result["derived"]["standard_error_real"] == pytest.approx(
        expected.standard_error_real, rel=2e-13, abs=2e-13
    )
    assert result["derived"]["standard_error_imag"] == pytest.approx(
        expected.standard_error_imag, rel=2e-13, abs=2e-13
    )
    assert result["derived"] == summary["estimate"]
    assert Path(result["rendered"]) == run / "report.html"
    source_run = json.loads((run / "run.json").read_text(encoding="utf-8"))
    source_digest = hashlib.sha256((run / "hashes.json").read_bytes()).hexdigest()
    for key, suffix, operation in (
        ("estimate_copy", "-estimate", "estimate"),
        ("report_copy", "-report", "report"),
    ):
        copied = Path(result[key])
        assert copied == Path(f"{run}{suffix}")
        assert (copied / "run.json").is_file()
        assert (copied / "hashes.json").is_file()
        assert (copied / "report.html").is_file()
        assert validate_run(copied)["valid"] is True
        copied_run = json.loads((copied / "run.json").read_text(encoding="utf-8"))
        assert copied_run["run_id"] == source_run["run_id"]
        assert copied_run["derivation"] == {
            "operation": operation,
            "source_content_sha256": source_digest,
            "source_run_id": source_run["run_id"],
        }
    assert (run / "figures" / "estimates.svg").read_text(encoding="utf-8").startswith("<svg")
    html = (run / "report.html").read_text(encoding="utf-8")
    assert "Taylor–Gauss 3×3 run report" in html
    assert "estimates.svg" in html

    for label, operation in (
        ("estimate", derive_estimates),
        ("report", render_report),
    ):
        existing = tmp_path / f"existing-{label}"
        existing.mkdir()
        with pytest.raises(FileExistsError, match="already exists"):
            operation(run, existing)
        nested = run / f"nested-{label}"
        with pytest.raises(ValueError, match="outside.*source"):
            operation(run, nested)
        assert not nested.exists()
