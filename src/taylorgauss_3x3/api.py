"""Public API façade for the fixed 3x3 sampler contracts."""

from __future__ import annotations

from .actions import build_action
from .config import ExactRunConfig, StochasticRunConfig


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


def _not_implemented(*args, **kwargs):
    raise NotImplementedError("This public API is implemented in a later task.")


def estimate(*args, **kwargs):
    return _not_implemented(*args, **kwargs)


def report(*args, **kwargs):
    return _not_implemented(*args, **kwargs)


def run_contour(*args, **kwargs):
    return _not_implemented(*args, **kwargs)


def run_exact(*args, **kwargs):
    return _not_implemented(*args, **kwargs)


def run_rao_blackwell(*args, **kwargs):
    return _not_implemented(*args, **kwargs)


def validate(*args, **kwargs):
    return _not_implemented(*args, **kwargs)
