from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from taylorgauss_3x3 import api
from taylorgauss_3x3.artifacts import (
    ArtifactValidationError,
    derive_estimates,
    validate_run,
    write_run,
)
from taylorgauss_3x3.config import ExactRunConfig, StochasticRunConfig
from taylorgauss_3x3.core import Hubbard3x3Target
from taylorgauss_3x3.core.observables import approved_observables
from taylorgauss_3x3.sampling import allocate_channels, estimate_explicit_contour


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_run(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _refresh_hashes(run: Path) -> None:
    manifest = _read_json(run / "hashes.json")
    manifest["files"] = []
    paths = sorted(
        path.relative_to(run).as_posix()
        for path in run.rglob("*")
        if path.is_file() and path.name != "hashes.json"
    )
    for relative in paths:
        payload = (run / relative).read_bytes()
        manifest["files"].append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    (run / "hashes.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def exact_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("artifacts") / "exact"
    assert api.run_exact(ExactRunConfig(), output) == output
    return output


@pytest.fixture(scope="module")
def endpoint_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("artifacts") / "endpoint"
    config = StochasticRunConfig(
        method="exact-contour",
        samples=7,
        seed=73,
        chunk_size=3,
        persist_endpoints=True,
    )
    assert api.run_contour(config, output) == output
    return output


@pytest.fixture(scope="module")
def statistic_config() -> StochasticRunConfig:
    return StochasticRunConfig(
        method="exact-contour",
        samples=11,
        seed=79,
        chunk_size=4,
        persist_endpoints=False,
    )


@pytest.fixture(scope="module")
def statistic_run(
    tmp_path_factory: pytest.TempPathFactory,
    statistic_config: StochasticRunConfig,
) -> Path:
    output = tmp_path_factory.mktemp("artifacts") / "statistics"
    assert write_run(statistic_config, output) == output
    return output


def test_exact_run_is_a_completed_independently_validatable_artifact(exact_run: Path):
    """Catches a partial run or validator coupled to in-memory run state."""

    assert {path.relative_to(exact_run).as_posix() for path in exact_run.rglob("*") if path.is_file()} == {
        "estimates.jsonl",
        "figures/estimates.svg",
        "hashes.json",
        "report.html",
        "run.json",
        "summary.json",
    }
    run = _read_json(exact_run / "run.json")
    assert run["state"] == "completed"
    assert run["method"] == "exact-enumeration"
    assert type(run["schema_version"]) is int
    assert validate_run(exact_run)["valid"] is True
    assert api.validate(exact_run)["valid"] is True


def test_exact_artifact_matches_the_canonical_fixture(exact_run: Path):
    """Catches artifact exact values drifting with writer/validator replay together."""

    anchor = _read_json(Path(__file__).parent / "fixtures" / "canonical-anchor.json")
    estimate = _read_json(exact_run / "summary.json")["estimate"]
    expected = anchor["mixed_linear_quadratic"]

    assert estimate["value"]["real"] == pytest.approx(
        expected["real"], rel=2e-12, abs=2e-12
    )
    assert estimate["value"]["imag"] == pytest.approx(
        expected["imag"], rel=2e-12, abs=2e-12
    )
    assert estimate["standard_error_real"] is None
    assert estimate["standard_error_imag"] is None


def test_existing_output_is_refused_before_any_reuse(exact_run: Path):
    """Catches accidental overwrite or append into a finalized directory."""

    before = (exact_run / "hashes.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        write_run(ExactRunConfig(), exact_run)
    assert (exact_run / "hashes.json").read_bytes() == before


def test_hash_manifest_is_ordered_complete_and_content_addressed(exact_run: Path):
    """Catches nondeterministic manifests or unhashed completed content."""

    manifest = _read_json(exact_run / "hashes.json")
    entries = manifest["files"]
    paths = [entry["path"] for entry in entries]
    assert manifest["algorithm"] == "sha256"
    assert paths == sorted(paths)
    assert paths == [
        "estimates.jsonl",
        "figures/estimates.svg",
        "report.html",
        "run.json",
        "summary.json",
    ]
    for entry in entries:
        payload = (exact_run / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert type(entry["size_bytes"]) is int
        assert entry["size_bytes"] == len(payload)


@pytest.mark.parametrize(
    "tamper",
    [
        "wrong-count",
        "wrong-ordering",
        "missing-summary-key",
        "extra-summary-key",
        "missing-support-key",
        "extra-support-key",
    ],
)
def test_validation_rejects_rehashed_summary_schema_and_support_tampering(
    exact_run: Path, tmp_path: Path, tamper: str
):
    """Catches exact-support claims remaining outside semantic validation."""

    copied = _copy_run(exact_run, tmp_path / tamper)
    summary = _read_json(copied / "summary.json")
    support = summary["exact_support"]
    if tamper == "wrong-count":
        support["count"] = 1
    elif tamper == "wrong-ordering":
        support["ordered"] = False
    elif tamper == "missing-summary-key":
        del summary["exact_support"]
    elif tamper == "extra-summary-key":
        summary["unexpected"] = True
    elif tamper == "missing-support-key":
        del support["ordered"]
    else:
        support["unexpected"] = True
    (copied / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_hashes(copied)

    with pytest.raises(
        ArtifactValidationError,
        match="summary.*schema|exact_support",
    ):
        validate_run(copied)


def test_completed_layout_rejects_an_unexpected_empty_directory(
    exact_run: Path, tmp_path: Path
):
    """Catches unhashed empty directories outside the exact completed layout."""

    copied = _copy_run(exact_run, tmp_path / "unexpected-directory")
    (copied / "unexpected").mkdir()
    with pytest.raises(ValueError, match="content|layout|unexpected"):
        validate_run(copied)


@pytest.mark.parametrize("placement", ["zarr-root", "array-directory"])
def test_validation_rejects_special_zarr_nodes_before_hash_or_science_reads(
    endpoint_run: Path, tmp_path: Path, placement: str
):
    """Catches unhashed special nodes being ignored by completed-run traversal."""

    if not hasattr(os, "mkfifo"):
        pytest.skip("platform does not provide FIFO creation")
    copied = _copy_run(endpoint_run, tmp_path / f"special-node-{placement}")
    parent = copied / "samples.zarr"
    if placement == "array-directory":
        parent /= "channel_id"
    special = parent / "unhashed-special-entry"
    try:
        os.mkfifo(special)
    except NotImplementedError:
        pytest.skip("platform does not implement FIFO creation")
    except OSError as exc:
        unsupported = {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
        if exc.errno in unsupported:
            pytest.skip("test filesystem does not support FIFOs")
        raise

    with pytest.raises(
        ArtifactValidationError,
        match="unsupported special filesystem content",
    ):
        validate_run(copied)


def test_strict_json_rejects_nonfinite_and_preserves_integer_semantics(
    exact_run: Path, tmp_path: Path
):
    """Catches NaN acceptance and integer fields silently becoming floats."""

    copied = tmp_path / "nonfinite"
    copied.mkdir()
    for source in exact_run.rglob("*"):
        relative = source.relative_to(exact_run)
        target = copied / relative
        if source.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.write_bytes(source.read_bytes())
    (copied / "run.json").write_text(
        (copied / "run.json").read_text(encoding="utf-8").replace(
            '"schema_version":1', '"schema_version":NaN'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nonfinite|JSON"):
        validate_run(copied)

    copied = tmp_path / "float-integer"
    copied.mkdir()
    for source in exact_run.rglob("*"):
        relative = source.relative_to(exact_run)
        target = copied / relative
        if source.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.write_bytes(source.read_bytes())
    (copied / "run.json").write_text(
        (copied / "run.json").read_text(encoding="utf-8").replace(
            '"schema_version":1', '"schema_version":1.0'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version.*integer"):
        validate_run(copied)


def test_validation_translates_excessively_nested_hostile_json(
    exact_run: Path, tmp_path: Path
):
    """Catches hostile nesting escaping the public validator as RecursionError."""

    copied = _copy_run(exact_run, tmp_path / "excessively-nested-json")
    depth = sys.getrecursionlimit() + 100
    (copied / "run.json").write_text(
        '{"hostile":' + "[" * depth + "null" + "]" * depth + "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="nested|JSON"):
        validate_run(copied)


def test_estimate_rows_preserve_strict_json_integer_semantics(
    exact_run: Path, tmp_path: Path
):
    """Catches an exact chunk index silently changing from an integer to a float."""

    copied = _copy_run(exact_run, tmp_path / "float-row-index")
    estimates = copied / "estimates.jsonl"
    estimates.write_text(
        estimates.read_text(encoding="utf-8").replace(
            '"chunk_index":0', '"chunk_index":0.0'
        ),
        encoding="utf-8",
    )
    _refresh_hashes(copied)
    with pytest.raises(ValueError, match="chunk_index.*integer"):
        validate_run(copied)


def test_zarr_metadata_preserves_strict_json_integer_semantics(
    endpoint_run: Path, tmp_path: Path
):
    """Catches the Zarr format integer silently changing to an equal-valued float."""

    copied = _copy_run(endpoint_run, tmp_path / "float-zarr-format")
    group = copied / "samples.zarr" / ".zgroup"
    group.write_text('{"zarr_format":2.0}\n', encoding="utf-8")
    _refresh_hashes(copied)
    with pytest.raises(ValueError, match="Zarr|zarr_format|integer"):
        validate_run(copied)


def test_run_id_is_bound_to_the_stored_configuration(exact_run: Path, tmp_path: Path):
    """Catches a well-formed but unrelated identifier accepted after rehashing."""

    copied = _copy_run(exact_run, tmp_path / "unbound-run-id")
    run = _read_json(copied / "run.json")
    run["run_id"] = "0" * 64
    (copied / "run.json").write_text(
        json.dumps(run, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_hashes(copied)
    with pytest.raises(ValueError, match="run_id.*configuration"):
        validate_run(copied)


@pytest.mark.parametrize("tamper", ["missing", "truncated", "reordered"])
def test_validation_rejects_missing_truncated_or_reordered_zarr_chunks(
    endpoint_run: Path, tmp_path: Path, tamper: str
):
    """Catches incomplete or reordered endpoint streams being accepted."""

    copied = tmp_path / tamper
    copied.mkdir()
    for source in endpoint_run.rglob("*"):
        relative = source.relative_to(endpoint_run)
        target = copied / relative
        if source.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.write_bytes(source.read_bytes())
    chunks = sorted(
        path
        for path in (copied / "samples.zarr").rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    assert len(chunks) >= 2
    if tamper == "missing":
        chunks[0].unlink()
    elif tamper == "truncated":
        chunks[0].write_bytes(chunks[0].read_bytes()[:-1])
    else:
        first = chunks[0].read_bytes()
        second = chunks[1].read_bytes()
        chunks[0].write_bytes(second)
        chunks[1].write_bytes(first)
    _refresh_hashes(copied)
    expected_error = {
        "missing": "missing Zarr chunk",
        "truncated": "truncated Zarr chunk",
        "reordered": "channel sequence.*exact seeded allocation",
    }[tamper]
    with pytest.raises(ArtifactValidationError, match=expected_error):
        validate_run(copied)


def test_endpoint_disabled_run_matches_an_independent_in_memory_estimator(
    statistic_run: Path, statistic_config: StochasticRunConfig
):
    """Catches artifact replay and summary agreeing on the wrong stochastic result."""

    assert not (statistic_run / "samples.zarr").exists()
    records = [
        json.loads(line)
        for line in (statistic_run / "estimates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(record["count"] for record in records) == 11
    assert [record["chunk_index"] for record in records] == [0, 1, 2]
    assert all("real_statistics" in record and "imag_statistics" in record for record in records)
    derived = derive_estimates(statistic_run)
    summary = _read_json(statistic_run / "summary.json")
    target = Hubbard3x3Target(**statistic_config.parameters)
    oracle = target.exact_indexed_oracle()
    allocation = allocate_channels(
        oracle,
        sample_count=statistic_config.samples,
        seed=statistic_config.seed,
        design=statistic_config.channel_design,
    )
    expected = estimate_explicit_contour(
        oracle,
        approved_observables()[statistic_config.observable],
        allocation,
    )

    assert derived["value"]["real"] == pytest.approx(
        expected.value.real, rel=2e-15, abs=2e-15
    )
    assert derived["value"]["imag"] == pytest.approx(
        expected.value.imag, rel=2e-15, abs=2e-15
    )
    assert derived["standard_error_real"] == pytest.approx(
        expected.standard_error_real, rel=2e-15, abs=2e-15
    )
    assert derived["standard_error_imag"] == pytest.approx(
        expected.standard_error_imag, rel=2e-15, abs=2e-15
    )
    assert derived == summary["estimate"]
    assert api.estimate(statistic_run) == derived
    assert api.report(statistic_run) == statistic_run / "report.html"
    assert validate_run(statistic_run)["valid"] is True


def test_explicit_structural_run_honors_endpoint_persistence(tmp_path: Path):
    """Catches an approved structural observable skipping its requested sample stream."""

    output = tmp_path / "structural-endpoints"
    config = StochasticRunConfig(
        method="exact-contour",
        observable="physical_log_partition",
        samples=3,
        seed=89,
        chunk_size=2,
        persist_endpoints=True,
    )
    assert write_run(config, output) == output
    assert (output / "samples.zarr").is_dir()
    assert validate_run(output)["valid"] is True


def test_validation_rejects_persisted_sources_outside_the_seeded_stream(
    tmp_path: Path,
):
    """Catches self-consistent endpoints whose source is not the declared seeded draw."""

    output = tmp_path / "seeded-structural"
    write_run(
        StochasticRunConfig(
            method="exact-contour",
            observable="physical_log_partition",
            samples=3,
            seed=103,
            chunk_size=2,
            persist_endpoints=True,
        ),
        output,
    )
    source_chunk = output / "samples.zarr" / "source" / "0.0"
    endpoint_chunk = output / "samples.zarr" / "endpoint" / "0.0"
    source = np.frombuffer(source_chunk.read_bytes(), dtype=np.float64).copy()
    endpoint = np.frombuffer(endpoint_chunk.read_bytes(), dtype=np.complex128).copy()
    source[0] += 1.0
    endpoint[0] += 1.0
    source_chunk.write_bytes(source.tobytes())
    endpoint_chunk.write_bytes(endpoint.tobytes())
    _refresh_hashes(output)

    with pytest.raises(ValueError, match="seeded.*source|source.*stream"):
        validate_run(output)


def test_validation_rejects_incomplete_state_even_with_structural_files(
    exact_run: Path, tmp_path: Path
):
    """Catches structurally plausible artifacts that were never finalized."""

    copied = tmp_path / "incomplete"
    copied.mkdir()
    for source in exact_run.rglob("*"):
        relative = source.relative_to(exact_run)
        target = copied / relative
        if source.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.write_bytes(source.read_bytes())
    payload = _read_json(copied / "run.json")
    payload["state"] = "running"
    (copied / "run.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed"):
        validate_run(copied)
