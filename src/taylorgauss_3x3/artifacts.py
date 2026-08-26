"""Immutable run construction, offline derivation, and independent validation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable

import numpy as np
import torch

from .actions import build_action
from .config import (
    EXACT_METHOD,
    EXPLICIT_STOCHASTIC_METHOD,
    RB_STOCHASTIC_METHOD,
    ExactRunConfig,
    StochasticRunConfig,
)
from .core import ExactIndexedContourOracle, Hubbard3x3Target
from .core.observables import approved_observables, exact_enumeration, physical_log_partition
from .reporting import _render_html, _render_svg
from .sampling import (
    ChannelAllocation,
    LogComponentStatistics,
    _counter_torch_seed,
    _estimate_from_log_statistics,
    _log_component_statistics,
    allocate_channels,
)
from .storage import (
    ZarrStreamWriter,
    ensure_json_value,
    iter_zarr_chunks,
    json_bytes,
    read_json,
    write_json,
    zarr_metadata,
)


SCHEMA_VERSION = 1
EXACT_SUPPORT_COUNT = 3**9


class ArtifactValidationError(ValueError):
    """A completed artifact failed a structural or scientific invariant."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complex(value: complex) -> dict[str, float]:
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError("artifact estimate must be finite")
    return {"imag": float(value.imag), "real": float(value.real)}


def _statistics(value: LogComponentStatistics) -> dict[str, float | None]:
    result = asdict(value)
    for item in result.values():
        if item is not None and not math.isfinite(item):
            raise FloatingPointError("sufficient statistics must be finite")
    return result


def _as_statistics(value: Any, location: str) -> LogComponentStatistics:
    fields = {
        "positive_log_sum", "negative_log_sum", "log_sum_squares", "log_scale",
        "scaled_sum", "scaled_sum_squared_deviations",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ArtifactValidationError(f"{location} has an invalid sufficient-statistic schema")
    for key in ("positive_log_sum", "negative_log_sum", "log_sum_squares", "log_scale"):
        if value[key] is not None and type(value[key]) is not float:
            raise ArtifactValidationError(f"{location}.{key} must be a float or null")
        if value[key] is not None and not math.isfinite(value[key]):
            raise ArtifactValidationError(f"{location}.{key} must be finite")
    for key in ("scaled_sum", "scaled_sum_squared_deviations"):
        if type(value[key]) is not float:
            raise ArtifactValidationError(f"{location}.{key} must be a float")
        if not math.isfinite(value[key]):
            raise ArtifactValidationError(f"{location}.{key} must be finite")
    if value["scaled_sum_squared_deviations"] < 0.0:
        raise ArtifactValidationError(
            f"{location}.scaled_sum_squared_deviations must be nonnegative"
        )
    return LogComponentStatistics(**value)


def _logaddexp(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    scale = max(finite)
    return scale + math.log(math.fsum(math.exp(value - scale) for value in finite))


def _merge_statistics(
    pieces: list[tuple[int, LogComponentStatistics]],
) -> LogComponentStatistics:
    if not pieces:
        raise ArtifactValidationError("no sufficient-statistic chunks were stored")
    scale_values = [value.log_scale for _, value in pieces if value.log_scale is not None]
    global_scale = max(scale_values) if scale_values else None
    count = 0
    mean = 0.0
    m2 = 0.0
    for piece_count, value in pieces:
        factor = 0.0 if value.log_scale is None else math.exp(value.log_scale - global_scale)  # type: ignore[operator]
        piece_mean = value.scaled_sum * factor / piece_count
        piece_m2 = value.scaled_sum_squared_deviations * factor * factor
        if count == 0:
            count = piece_count
            mean = piece_mean
            m2 = piece_m2
            continue
        total = count + piece_count
        delta = piece_mean - mean
        m2 += piece_m2 + delta * delta * count * piece_count / total
        mean += delta * piece_count / total
        count = total
    return LogComponentStatistics(
        positive_log_sum=_logaddexp(value.positive_log_sum for _, value in pieces),
        negative_log_sum=_logaddexp(value.negative_log_sum for _, value in pieces),
        log_sum_squares=_logaddexp(value.log_sum_squares for _, value in pieces),
        log_scale=global_scale,
        scaled_sum=mean * count,
        scaled_sum_squared_deviations=max(0.0, m2),
    )


def _strict_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError("missing estimates.jsonl") from exc
    if not payload or not payload.endswith(b"\n"):
        raise ArtifactValidationError("estimates.jsonl is empty or truncated")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {token}")
                ),
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ) as exc:
            raise ArtifactValidationError(
                f"invalid JSON in estimates.jsonl line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ArtifactValidationError("each estimates.jsonl row must be an object")
        try:
            ensure_json_value(value, f"estimates.jsonl[{line_number}]")
        except ValueError as exc:
            raise ArtifactValidationError(str(exc)) from exc
        rows.append(value)
    return rows


def _derive_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) == 1 and rows[0].get("kind") in {"exact", "structural"}:
        row = rows[0]
        if set(row) != {"chunk_index", "count", "kind", "value"}:
            raise ArtifactValidationError("stored exact estimate has an invalid row schema")
        if type(row.get("chunk_index")) is not int:
            raise ArtifactValidationError("estimate chunk_index must be an integer")
        if row["chunk_index"] != 0:
            raise ArtifactValidationError("stored exact estimate must be chunk zero")
        if type(row.get("count")) is not int or row["count"] < 1:
            raise ArtifactValidationError("estimate chunk count must be a positive integer")
        value = row.get("value")
        if (
            not isinstance(value, dict)
            or set(value) != {"imag", "real"}
            or any(type(value[key]) is not float for key in value)
            or any(not math.isfinite(value[key]) for key in value)
        ):
            raise ArtifactValidationError("stored exact estimate has an invalid value")
        return {
            "standard_error_imag": None,
            "standard_error_real": None,
            "value": value,
        }
    real: list[tuple[int, LogComponentStatistics]] = []
    imag: list[tuple[int, LogComponentStatistics]] = []
    for index, row in enumerate(rows):
        if set(row) != {
            "chunk_index",
            "count",
            "imag_statistics",
            "kind",
            "real_statistics",
        }:
            raise ArtifactValidationError("stored stochastic estimate has an invalid row schema")
        if row.get("kind") != "stochastic_chunk":
            raise ArtifactValidationError("mixed or invalid estimate row kinds")
        if type(row.get("chunk_index")) is not int or row["chunk_index"] != index:
            raise ArtifactValidationError("estimate chunks are missing or reordered")
        count = row.get("count")
        if type(count) is not int or count < 1:
            raise ArtifactValidationError("estimate chunk count must be a positive integer")
        real.append((count, _as_statistics(row.get("real_statistics"), "real_statistics")))
        imag.append((count, _as_statistics(row.get("imag_statistics"), "imag_statistics")))
    total = sum(count for count, _ in real)
    if total < 2:
        raise ArtifactValidationError("stochastic estimates require at least two records")
    value, se_real, se_imag = _estimate_from_log_statistics(
        real=_merge_statistics(real), imag=_merge_statistics(imag), count=total
    )
    return {
        "standard_error_imag": float(se_imag),
        "standard_error_real": float(se_real),
        "value": _complex(value),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(json_bytes(row))


def _validated_content_entries(directory: Path) -> list[tuple[Path, int]]:
    """Return run entries only after rejecting symlinks and special nodes."""

    entries: list[tuple[Path, int]] = []
    try:
        for path in directory.rglob("*"):
            mode = path.lstat().st_mode
            relative = path.relative_to(directory).as_posix()
            if stat.S_ISLNK(mode):
                raise ArtifactValidationError(
                    f"completed run content must not contain symlinks: {relative}"
                )
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ArtifactValidationError(
                    "completed run contains unsupported special filesystem "
                    f"content: {relative}"
                )
            entries.append((path, mode))
    except ArtifactValidationError:
        raise
    except OSError as exc:
        raise ArtifactValidationError(
            "completed run content could not be inspected"
        ) from exc
    return entries


def _manifest_files(run_directory: Path) -> list[dict[str, Any]]:
    paths = sorted(
        path.relative_to(run_directory).as_posix()
        for path, mode in _validated_content_entries(run_directory)
        if stat.S_ISREG(mode) and path.name != "hashes.json"
    )
    return [
        {
            "path": relative,
            "sha256": _sha256(run_directory / relative),
            "size_bytes": (run_directory / relative).stat().st_size,
        }
        for relative in paths
    ]


def _write_completed(
    output: Path,
    run: dict[str, Any],
    rows: list[dict[str, Any]],
    estimate: dict[str, Any],
) -> None:
    run = {**run, "schema_version": SCHEMA_VERSION, "state": "completed"}
    summary = {
        "estimate": estimate,
        "schema_version": SCHEMA_VERSION,
        "state": "completed",
        "valid": True,
    }
    write_json(output / "run.json", run)
    _write_jsonl(output / "estimates.jsonl", rows)
    write_json(output / "summary.json", summary)
    (output / "figures").mkdir()
    (output / "figures" / "estimates.svg").write_text(
        _render_svg(run, estimate), encoding="utf-8"
    )
    (output / "report.html").write_text(
        _render_html(run, estimate), encoding="utf-8"
    )
    write_json(
        output / "hashes.json",
        {"algorithm": "sha256", "files": _manifest_files(output), "schema_version": SCHEMA_VERSION},
    )


def _base_run(config: ExactRunConfig | StochasticRunConfig) -> dict[str, Any]:
    action = build_action(config)
    metadata = action.metadata
    result: dict[str, Any] = {
        "authority": "exact_reference" if isinstance(config, ExactRunConfig) else config.authority,
        "euclidean_time_slices": metadata.euclidean_time_slices,
        "field_shape": list(metadata.field_shape),
        "geometry": metadata.geometry,
        "method": config.method,
        "model": metadata.model_id,
        "observable": config.observable,
        "parameters": dict(metadata.parameters),
        "weight_character": metadata.weight_character,
    }
    if isinstance(config, StochasticRunConfig):
        result.update(
            {
                "channel_design": config.channel_design,
                "chunk_size": config.chunk_size,
                "endpoint_persistence": bool(config.persist_endpoints),
                "sample_count": config.samples,
                "seed": config.seed,
            }
        )
    return result


def _run_id_for(run: dict[str, Any]) -> str:
    scientific_configuration = {
        key: value
        for key, value in run.items()
        if key not in {"derivation", "run_id", "schema_version", "state"}
    }
    payload = json.dumps(
        scientific_configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _explicit_sample_chunk(
    oracle: ExactIndexedContourOracle,
    allocation: ChannelAllocation,
    *,
    seed: int,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[torch.Tensor] = []
    endpoints: list[torch.Tensor] = []
    for record_index in range(start, stop):
        generator = torch.Generator(device=oracle.modes.device).manual_seed(
            _counter_torch_seed(
                seed=seed,
                domain="explicit_contour.gaussian_source",
                counter=record_index,
            )
        )
        sample = oracle.sample_for_channels(
            allocation.channel[record_index : record_index + 1],
            generator=generator,
        )
        sources.append(sample.source)
        endpoints.append(sample.endpoint)
    return torch.cat(sources, dim=0), torch.cat(endpoints, dim=0)


def _finish_sample_writer(writer: ZarrStreamWriter, config: StochasticRunConfig) -> None:
    writer.finish(
        {
            "chunk_size": config.chunk_size,
            "format": "zarr-v2",
            "sample_count": config.samples,
            "schema_version": SCHEMA_VERSION,
        }
    )


def _exact_value(config: ExactRunConfig) -> complex:
    target = Hubbard3x3Target(**config.parameters)
    oracle = target.exact_indexed_oracle()
    if config.observable == "physical_log_partition":
        return complex(physical_log_partition(target, oracle), 0.0)
    return exact_enumeration(oracle, approved_observables()[config.observable])


def _stochastic_rows(
    config: StochasticRunConfig,
    output: Path,
    *,
    write_samples: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = Hubbard3x3Target(**config.parameters)
    oracle = target.exact_indexed_oracle()
    writer = None
    if (
        write_samples
        and config.method == EXPLICIT_STOCHASTIC_METHOD
        and config.persist_endpoints
    ):
        writer = ZarrStreamWriter(
            output / "samples.zarr",
            arrays={"channel_id": ("int64", ()), "endpoint": ("complex128", (9,)), "source": ("float64", (9,))},
            row_count=config.samples,
            chunk_size=config.chunk_size,
        )
    if config.observable == "physical_log_partition":
        value = complex(physical_log_partition(target, oracle), 0.0)
        row = {
            "chunk_index": 0,
            "count": config.samples,
            "kind": "structural",
            "value": _complex(value),
        }
        if writer is not None:
            allocation = allocate_channels(
                oracle,
                sample_count=config.samples,
                seed=config.seed,
                design=config.channel_design,
            )
            for start in range(0, config.samples, config.chunk_size):
                stop = min(config.samples, start + config.chunk_size)
                source, endpoint = _explicit_sample_chunk(
                    oracle,
                    allocation,
                    seed=config.seed,
                    start=start,
                    stop=stop,
                )
                writer.write(
                    {
                        "channel_id": allocation.channel[start:stop],
                        "endpoint": endpoint,
                        "source": source,
                    }
                )
            _finish_sample_writer(writer, config)
        return [row], _derive_from_rows([row])
    observable = approved_observables()[config.observable]
    allocation = allocate_channels(
        oracle,
        sample_count=config.samples,
        seed=config.seed,
        design=config.channel_design,
    )
    rows: list[dict[str, Any]] = []
    conditional_means = (
        None
        if config.method == EXPLICIT_STOCHASTIC_METHOD
        else observable.conditional_moments(oracle.means, oracle.precision)[0]
    )
    for chunk_index, start in enumerate(range(0, config.samples, config.chunk_size)):
        stop = min(config.samples, start + config.chunk_size)
        channel = allocation.channel[start:stop]
        log_weight = allocation.log_design_weight[start:stop]
        source: torch.Tensor | None = None
        endpoint: torch.Tensor | None = None
        if config.method == EXPLICIT_STOCHASTIC_METHOD:
            source, endpoint = _explicit_sample_chunk(
                oracle,
                allocation,
                seed=config.seed,
                start=start,
                stop=stop,
            )
            values = observable.evaluate(endpoint)
        else:
            assert conditional_means is not None
            values = conditional_means[channel]
        phased = values * oracle.phase[channel]
        row = {
            "chunk_index": chunk_index,
            "count": stop - start,
            "imag_statistics": _statistics(_log_component_statistics(log_weight, phased.imag)),
            "kind": "stochastic_chunk",
            "real_statistics": _statistics(_log_component_statistics(log_weight, phased.real)),
        }
        rows.append(row)
        if writer is not None:
            assert source is not None and endpoint is not None
            writer.write({"channel_id": channel, "endpoint": endpoint, "source": source})
    if writer is not None:
        _finish_sample_writer(writer, config)
    return rows, _derive_from_rows(rows)


def write_run(
    config: ExactRunConfig | StochasticRunConfig, output: str | Path
) -> Path:
    """Create one finalized run, refusing every existing output path."""

    if type(config) not in {ExactRunConfig, StochasticRunConfig}:
        raise TypeError("config must be an ExactRunConfig or StochasticRunConfig")
    output_path = Path(output)
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable output already exists: {output_path}") from exc
    run = _base_run(config)
    if isinstance(config, ExactRunConfig):
        value = _exact_value(config)
        rows = [{"chunk_index": 0, "count": EXACT_SUPPORT_COUNT, "kind": "exact", "value": _complex(value)}]
        estimate = _derive_from_rows(rows)
    else:
        rows, estimate = _stochastic_rows(config, output_path)
    run["run_id"] = _run_id_for(run)
    _write_completed(output_path, run, rows, estimate)
    validate_run(output_path)
    return output_path


def _validate_manifest(run_directory: Path) -> None:
    manifest = read_json(run_directory / "hashes.json")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"algorithm", "files", "schema_version"}
        or manifest["algorithm"] != "sha256"
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
        or not isinstance(manifest["files"], list)
    ):
        raise ArtifactValidationError("invalid hash manifest")
    expected_paths = sorted(
        path.relative_to(run_directory).as_posix()
        for path, mode in _validated_content_entries(run_directory)
        if stat.S_ISREG(mode) and path.name != "hashes.json"
    )
    if any(not isinstance(entry, dict) for entry in manifest["files"]):
        raise ArtifactValidationError("invalid hash manifest entry")
    paths = [entry.get("path") for entry in manifest["files"]]
    if paths != expected_paths or paths != sorted(paths):
        raise ArtifactValidationError("hash manifest has missing or unordered content")
    for entry in manifest["files"]:
        if (
            set(entry) != {"path", "sha256", "size_bytes"}
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
        ):
            raise ArtifactValidationError("invalid hash manifest entry")
        path = run_directory / entry["path"]
        if path.stat().st_size != entry["size_bytes"]:
            raise ArtifactValidationError(f"artifact size mismatch: {entry['path']}")
        if _sha256(path) != entry["sha256"]:
            raise ArtifactValidationError(f"artifact hash mismatch: {entry['path']}")


def _validate_layout(directory: Path) -> None:
    try:
        root_mode = directory.lstat().st_mode
    except OSError as exc:
        raise ArtifactValidationError("completed run must be a directory") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ArtifactValidationError("completed run must be a directory")
    _validated_content_entries(directory)
    required_root = {
        "estimates.jsonl",
        "figures",
        "hashes.json",
        "report.html",
        "run.json",
        "summary.json",
    }
    present_root = {path.name for path in directory.iterdir()}
    if (
        present_root != required_root
        and present_root != required_root | {"samples.zarr"}
    ):
        raise ArtifactValidationError(
            "completed run has missing or unexpected top-level content"
        )
    required_files = required_root - {"figures"}
    if any(not (directory / name).is_file() for name in required_files):
        raise ArtifactValidationError("completed run layout requires regular files")
    figures = directory / "figures"
    if (
        not figures.is_dir()
        or {path.name for path in figures.iterdir()} != {"estimates.svg"}
        or not (figures / "estimates.svg").is_file()
    ):
        raise ArtifactValidationError("completed run has an invalid figures layout")
    samples = directory / "samples.zarr"
    if samples.exists() and not samples.is_dir():
        raise ArtifactValidationError("samples.zarr must be a directory")


def _load_completed(run_directory: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    directory = Path(run_directory)
    _validate_layout(directory)
    run = read_json(directory / "run.json")
    if not isinstance(run, dict):
        raise ArtifactValidationError("run.json must be an object")
    if type(run.get("schema_version")) is not int:
        raise ArtifactValidationError("run schema_version must be an integer")
    if run["schema_version"] != SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported run schema_version")
    if run.get("state") != "completed":
        raise ArtifactValidationError("run state must be completed")
    summary = read_json(directory / "summary.json")
    if not isinstance(summary, dict) or summary.get("state") != "completed":
        raise ArtifactValidationError("summary state must be completed")
    if type(summary.get("schema_version")) is not int or summary["schema_version"] != SCHEMA_VERSION:
        raise ArtifactValidationError("summary schema_version must be an integer supported version")
    rows = _strict_json_lines(directory / "estimates.jsonl")
    derived = _derive_from_rows(rows)
    if summary.get("estimate") != derived:
        raise ArtifactValidationError("stored sufficient statistics do not reproduce summary estimate")
    if summary.get("valid") is not True:
        raise ArtifactValidationError("completed summary must be valid")
    _validate_manifest(directory)
    _validate_stored_contract(directory, run, rows)
    if (directory / "figures" / "estimates.svg").read_text(encoding="utf-8") != _render_svg(run, derived):
        raise ArtifactValidationError("stored figure does not render from completed content")
    if (directory / "report.html").read_text(encoding="utf-8") != _render_html(run, derived):
        raise ArtifactValidationError("stored report does not render from completed content")
    return directory, run, summary, rows


def _validate_scope(run: dict[str, Any]) -> None:
    common_fields = {
        "authority",
        "euclidean_time_slices",
        "field_shape",
        "geometry",
        "method",
        "model",
        "observable",
        "parameters",
        "run_id",
        "schema_version",
        "state",
        "weight_character",
    }
    expected_fields = set(common_fields)
    if run.get("method") in {EXPLICIT_STOCHASTIC_METHOD, RB_STOCHASTIC_METHOD}:
        expected_fields.update(
            {
                "channel_design",
                "chunk_size",
                "endpoint_persistence",
                "sample_count",
                "seed",
            }
        )
    if "derivation" in run:
        expected_fields.add("derivation")
    if set(run) != expected_fields:
        raise ArtifactValidationError("run.json has an invalid public artifact schema")
    if run.get("model") != "hubbard_auxiliary_field_exponential":
        raise ArtifactValidationError("run model is outside the public exact authority")
    if run.get("geometry") != "periodic_3x3" or run.get("field_shape") != [1, 3, 3]:
        raise ArtifactValidationError("run geometry or field shape is outside fixed scope")
    if type(run.get("euclidean_time_slices")) is not int or run["euclidean_time_slices"] != 1:
        raise ArtifactValidationError("run Euclidean slice count must be integer n_t=1")
    parameters = run.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != {"U", "beta", "kappa", "mu_chem"}:
        raise ArtifactValidationError("run parameters have an invalid schema")
    if any(type(value) is not float or not math.isfinite(value) for value in parameters.values()):
        raise ArtifactValidationError("run parameters must be finite floats")
    expected_weight = "positive" if parameters["mu_chem"] == 0.0 else "complex"
    if run.get("weight_character") != expected_weight:
        raise ArtifactValidationError("run weight character disagrees with its parameters")
    run_id = run.get("run_id")
    if (
        not isinstance(run_id, str)
        or len(run_id) != 64
        or run_id != _run_id_for(run)
    ):
        raise ArtifactValidationError("run_id does not bind the stored configuration")
    derivation = run.get("derivation")
    if derivation is not None:
        if (
            not isinstance(derivation, dict)
            or set(derivation)
            != {"operation", "source_content_sha256", "source_run_id"}
            or derivation.get("operation") not in {"estimate", "report"}
            or not isinstance(derivation.get("source_content_sha256"), str)
            or len(derivation["source_content_sha256"]) != 64
            or not isinstance(derivation.get("source_run_id"), str)
            or derivation["source_run_id"] != run_id
        ):
            raise ArtifactValidationError("run derivation metadata is invalid")


def _config_from_run(run: dict[str, Any]) -> ExactRunConfig | StochasticRunConfig:
    common = {**run["parameters"], "geometry": run["geometry"], "n_t": run["euclidean_time_slices"], "observable": run["observable"]}
    if run["method"] == EXACT_METHOD:
        return ExactRunConfig(**common)
    if run["method"] not in {EXPLICIT_STOCHASTIC_METHOD, RB_STOCHASTIC_METHOD}:
        raise ArtifactValidationError("unsupported public artifact method")
    return StochasticRunConfig(
        **common,
        method=run["method"],
        samples=run["sample_count"],
        seed=run["seed"],
        channel_design=run["channel_design"],
        chunk_size=run["chunk_size"],
        persist_endpoints=run["endpoint_persistence"],
    )


def _validate_stored_contract(
    directory: Path, run: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Validate completed stored semantics without constructing physics."""

    _validate_scope(run)
    try:
        config = _config_from_run(run)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"invalid stored run configuration: {exc}") from exc
    if isinstance(config, ExactRunConfig):
        if run.get("authority") != "exact_reference":
            raise ArtifactValidationError("exact run authority must be exact_reference")
        if (
            len(rows) != 1
            or rows[0].get("kind") != "exact"
            or rows[0].get("count") != EXACT_SUPPORT_COUNT
        ):
            raise ArtifactValidationError(
                "exact row must declare the complete 19,683-channel support"
            )
        if (directory / "samples.zarr").exists():
            raise ArtifactValidationError("exact runs must not contain samples.zarr")
        return
    if run.get("authority") != config.authority:
        raise ArtifactValidationError("stochastic authority disagrees with its method")
    if config.observable == "physical_log_partition":
        if (
            len(rows) != 1
            or rows[0].get("kind") != "structural"
            or rows[0].get("count") != config.samples
        ):
            raise ArtifactValidationError("structural estimate row is invalid")
    else:
        expected_counts = [
            min(config.chunk_size, config.samples - start)
            for start in range(0, config.samples, config.chunk_size)
        ]
        if [row.get("count") for row in rows] != expected_counts:
            raise ArtifactValidationError(
                "estimate chunks are missing, reordered, or truncated"
            )
    should_persist = (
        config.method == EXPLICIT_STOCHASTIC_METHOD
        and bool(config.persist_endpoints)
    )
    samples = directory / "samples.zarr"
    if samples.is_dir() is not should_persist:
        raise ArtifactValidationError(
            "samples.zarr presence disagrees with endpoint persistence"
        )
    if not should_persist:
        return
    attrs, arrays = zarr_metadata(samples)
    if set(attrs) != {"chunk_size", "format", "sample_count", "schema_version"}:
        raise ArtifactValidationError("samples.zarr attributes have an invalid schema")
    if (
        attrs.get("format") != "zarr-v2"
        or type(attrs.get("schema_version")) is not int
        or attrs["schema_version"] != SCHEMA_VERSION
        or type(attrs.get("sample_count")) is not int
        or attrs["sample_count"] != config.samples
        or type(attrs.get("chunk_size")) is not int
        or attrs["chunk_size"] != config.chunk_size
    ):
        raise ArtifactValidationError("samples.zarr attributes disagree with run.json")
    expected_arrays = {
        "channel_id": (
            np.dtype("int64"),
            [config.samples],
            [min(config.chunk_size, config.samples)],
        ),
        "endpoint": (
            np.dtype("complex128"),
            [config.samples, 9],
            [min(config.chunk_size, config.samples), 9],
        ),
        "source": (
            np.dtype("float64"),
            [config.samples, 9],
            [min(config.chunk_size, config.samples), 9],
        ),
    }
    if set(arrays) != set(expected_arrays):
        raise ArtifactValidationError("samples.zarr arrays do not match endpoint schema")
    for name, (dtype, shape, chunks) in expected_arrays.items():
        metadata = arrays[name]
        if (
            np.dtype(metadata["dtype"]) != dtype
            or metadata["shape"] != shape
            or metadata["chunks"] != chunks
        ):
            raise ArtifactValidationError(f"samples.zarr schema mismatch: {name}")
    position = 0
    channel_chunks = iter_zarr_chunks(samples, "channel_id")
    source_chunks = iter_zarr_chunks(samples, "source")
    endpoint_chunks = iter_zarr_chunks(samples, "endpoint")
    for channel, source, endpoint in zip(
        channel_chunks, source_chunks, endpoint_chunks, strict=True
    ):
        if not (len(channel) == len(source) == len(endpoint)):
            raise ArtifactValidationError("samples.zarr arrays have misaligned chunks")
        if bool(((channel < 0) | (channel >= EXACT_SUPPORT_COUNT)).any()):
            raise ArtifactValidationError("stored endpoint channel is outside exact support")
        if not np.isfinite(source).all():
            raise ArtifactValidationError("stored endpoint source must be finite")
        if not np.isfinite(endpoint.real).all() or not np.isfinite(endpoint.imag).all():
            raise ArtifactValidationError("stored endpoint content must be finite")
        position += len(channel)
    if position != config.samples:
        raise ArtifactValidationError("stored endpoint chunks are truncated")


def _validate_science(directory: Path, run: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    config = _config_from_run(run)
    if isinstance(config, ExactRunConfig):
        expected = _complex(_exact_value(config))
        if _derive_from_rows(rows)["value"] != expected:
            raise ArtifactValidationError("exact estimate does not match internal exact-law authority")
        return
    should_persist = (
        config.method == EXPLICIT_STOCHASTIC_METHOD
        and bool(config.persist_endpoints)
    )
    if not should_persist:
        expected_rows, _ = _stochastic_rows(
            config, directory, write_samples=False
        )
        if rows != expected_rows:
            raise ArtifactValidationError(
                "stored sufficient statistics do not match the exact seeded stochastic law"
            )
        return
    target = Hubbard3x3Target(**config.parameters)
    oracle = target.exact_indexed_oracle()
    allocation = allocate_channels(
        oracle, sample_count=config.samples, seed=config.seed, design=config.channel_design
    )
    position = 0
    channel_chunks = iter_zarr_chunks(directory / "samples.zarr", "channel_id")
    source_chunks = iter_zarr_chunks(directory / "samples.zarr", "source")
    endpoint_chunks = iter_zarr_chunks(directory / "samples.zarr", "endpoint")
    for channel, source, endpoint in zip(channel_chunks, source_chunks, endpoint_chunks, strict=True):
        count = len(channel)
        expected_channel = allocation.channel[position : position + count].cpu().numpy()
        if not np.array_equal(channel, expected_channel):
            raise ArtifactValidationError("stored endpoint channel sequence is not the exact seeded allocation")
        if not np.isfinite(source).all() or not np.isfinite(endpoint.real).all() or not np.isfinite(endpoint.imag).all():
            raise ArtifactValidationError("stored endpoint content must be finite")
        expected_source, expected_endpoint = _explicit_sample_chunk(
            oracle,
            allocation,
            seed=config.seed,
            start=position,
            stop=position + count,
        )
        if not np.array_equal(source, expected_source.cpu().numpy()):
            raise ArtifactValidationError(
                "stored source does not match the exact seeded source stream"
            )
        if not np.array_equal(endpoint, expected_endpoint.cpu().numpy()):
            raise ArtifactValidationError(
                "stored endpoint does not match the exact seeded contour stream"
            )
        position += count
    if position != config.samples:
        raise ArtifactValidationError("stored endpoint chunks are truncated")
    expected_rows, _ = _stochastic_rows(config, directory, write_samples=False)
    if rows != expected_rows:
        raise ArtifactValidationError(
            "stored sufficient statistics do not match persisted exact-law endpoints"
        )


def validate_run(run_directory: str | Path) -> dict[str, Any]:
    """Independently validate structure, hashes, and public exact-law identities."""

    try:
        directory, run, _, rows = _load_completed(run_directory)
        _validate_science(directory, run, rows)
    except (
        ArtifactValidationError,
        FileNotFoundError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, ArtifactValidationError):
            raise
        raise ArtifactValidationError(str(exc)) from exc
    return {"path": str(directory), "state": "completed", "valid": True}


def _copy_derivative(source: Path, output: Path, operation: str) -> Path:
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("offline output must be outside the immutable source run")
    _, run, summary, rows = _load_completed(source)
    output.mkdir(parents=True, exist_ok=False)
    if (source / "samples.zarr").is_dir():
        shutil.copytree(source / "samples.zarr", output / "samples.zarr")
    source_digest = hashlib.sha256((source / "hashes.json").read_bytes()).hexdigest()
    derived_run = {
        **run,
        "derivation": {"operation": operation, "source_content_sha256": source_digest, "source_run_id": run["run_id"]},
    }
    _write_completed(output, derived_run, rows, summary["estimate"])
    _load_completed(output)
    return output


def derive_estimates(source: str | Path, output: str | Path | None = None) -> dict[str, Any] | Path:
    """Re-derive estimates from stored statistics, optionally as a new artifact."""

    source_path, _, summary, _ = _load_completed(source)
    if output is None:
        return summary["estimate"]
    return _copy_derivative(source_path, Path(output), "estimate")


__all__ = [
    "ArtifactValidationError",
    "SCHEMA_VERSION",
    "derive_estimates",
    "validate_run",
    "write_run",
]
