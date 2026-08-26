"""Public API façade for the fixed 3x3 sampler contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .actions import build_action
from .artifacts import derive_estimates, validate_run, write_run
from .config import (
    EXPLICIT_STOCHASTIC_METHOD,
    RB_STOCHASTIC_METHOD,
    ExactRunConfig,
    StochasticRunConfig,
)
from .reporting import render_report


_EXACT_CHANNEL_COUNT = 3**9


def describe(
    config: ExactRunConfig | StochasticRunConfig | None = None,
) -> dict[str, object]:
    """Describe the exact support and requested stochastic work separately."""

    config = ExactRunConfig() if config is None else config
    action = build_action(config)
    metadata = action.metadata
    authority = (
        "exact_reference"
        if isinstance(config, ExactRunConfig)
        else config.authority
    )
    description: dict[str, object] = {
        "model": metadata.model_id,
        "geometry": metadata.geometry,
        "field_shape": list(metadata.field_shape),
        "euclidean_time_slices": metadata.euclidean_time_slices,
        "parameters": dict(metadata.parameters),
        "weight_character": metadata.weight_character,
        "observable": config.observable,
        "supported_observables": list(action.supported_observables()),
        "method": config.method,
        "capabilities": list(action.capability_names),
        "authority": authority,
        "authority_boundary": metadata.authority_boundary,
        "exact_channel_count": _EXACT_CHANNEL_COUNT,
    }
    if isinstance(config, StochasticRunConfig):
        endpoint_persistence = (
            config.method == "exact-contour" and config.persist_endpoints
        )
        description.update(
            {
                "requested_sample_count": config.samples,
                "seed": config.seed,
                "channel_design": config.channel_design,
                "persist_endpoints": endpoint_persistence,
                "chunk_size": config.chunk_size,
                "source_representation": (
                    "persisted_real_gaussian"
                    if endpoint_persistence
                    else "not_persisted_sufficient_statistics"
                    if config.method == "exact-contour"
                    else "analytically_integrated"
                ),
                "endpoint_representation": (
                    "persisted_complex_contour"
                    if endpoint_persistence
                    else "not_persisted_sufficient_statistics"
                    if config.method == "exact-contour"
                    else "analytically_integrated"
                ),
            }
        )
    return description


def estimate(
    source: str | Path, output: str | Path | None = None
) -> dict[str, Any] | Path:
    """Re-derive an estimate from completed stored content."""

    return derive_estimates(source, output)


def report(source: str | Path, output: str | Path | None = None) -> Path:
    """Access or immutably reproduce a completed offline report."""

    return render_report(source, output)


def run_contour(config: StochasticRunConfig, output: str | Path) -> Path:
    """Run the explicit exact-contour estimator into an immutable directory."""

    if (
        type(config) is not StochasticRunConfig
        or config.method != EXPLICIT_STOCHASTIC_METHOD
    ):
        raise ValueError("run_contour requires an exact-contour configuration")
    return write_run(config, output)


def run_exact(config: ExactRunConfig, output: str | Path) -> Path:
    """Run exact enumeration into an immutable directory."""

    if type(config) is not ExactRunConfig:
        raise ValueError("run_exact requires an ExactRunConfig")
    return write_run(config, output)


def run_rao_blackwell(config: StochasticRunConfig, output: str | Path) -> Path:
    """Run the Rao--Blackwell exact-contour estimator immutably."""

    if type(config) is not StochasticRunConfig or config.method != RB_STOCHASTIC_METHOD:
        raise ValueError("run_rao_blackwell requires an exact-contour-rb configuration")
    return write_run(config, output)


def validate(run_directory: str | Path) -> dict[str, Any]:
    """Delegate to the artifact-owned independent validator."""

    return validate_run(run_directory)
