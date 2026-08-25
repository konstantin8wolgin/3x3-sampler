"""Fixed-core action adapter used by the public configuration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .config import APPROVED_OBSERVABLES, ExactRunConfig, StochasticRunConfig
from .core import ExactIndexedContourOracle, Hubbard3x3Target


@dataclass(frozen=True, slots=True)
class ActionMetadata:
    model_id: str
    geometry: str
    field_shape: tuple[int, int, int]
    euclidean_time_slices: int
    parameters: dict[str, float]
    weight_character: str
    authority_boundary: str


@runtime_checkable
class FermionAction(Protocol):
    """The constrained action boundary required by the fixed sampler."""

    @property
    def metadata(self) -> ActionMetadata: ...

    @property
    def capability_names(self) -> tuple[str, ...]: ...

    def evaluate(self, real_field: torch.Tensor) -> torch.Tensor: ...

    def evaluate_holomorphic(self, field: torch.Tensor) -> torch.Tensor: ...

    def exact_indexed_oracle(self) -> ExactIndexedContourOracle: ...

    def supported_observables(self) -> tuple[str, ...]: ...


class HubbardActionAdapter:
    """Adapter over the standalone package's exact fixed 3x3 mathematics."""

    def __init__(self, config: ExactRunConfig | StochasticRunConfig):
        self._target = Hubbard3x3Target(**config.parameters)
        self._metadata = ActionMetadata(
            model_id="hubbard_auxiliary_field_exponential",
            geometry=config.geometry,
            field_shape=(1, 3, 3),
            euclidean_time_slices=config.n_t,
            parameters=config.parameters,
            weight_character="positive" if config.mu_chem == 0.0 else "complex",
            authority_boundary=(
                "complete exact indexed contour authority for periodic 3x3, n_t=1, "
                "and approved entire polynomials plus physical log partition"
            ),
        )

    @property
    def metadata(self) -> ActionMetadata:
        return self._metadata

    @property
    def capability_names(self) -> tuple[str, ...]:
        return ("exact_fourier_channels", "holomorphic_evaluation")

    def evaluate(self, real_field: torch.Tensor) -> torch.Tensor:
        return self._target.evaluate(real_field)

    def evaluate_holomorphic(self, field: torch.Tensor) -> torch.Tensor:
        return self._target.evaluate_holomorphic(field)

    def exact_indexed_oracle(self) -> ExactIndexedContourOracle:
        return self._target.exact_indexed_oracle()

    def supported_observables(self) -> tuple[str, ...]:
        return APPROVED_OBSERVABLES


def build_action(config: ExactRunConfig | StochasticRunConfig) -> FermionAction:
    return HubbardActionAdapter(config)
