"""Validated configuration for the fixed periodic 3x3 sampler."""

from __future__ import annotations

from dataclasses import dataclass
import math


SUPPORTED_GEOMETRY = "periodic_3x3"
SUPPORTED_TIME_SLICES = 1
EXACT_METHOD = "exact-enumeration"
EXPLICIT_STOCHASTIC_METHOD = "exact-contour"
RB_STOCHASTIC_METHOD = "exact-contour-rb"
STOCHASTIC_METHODS = (EXPLICIT_STOCHASTIC_METHOD, RB_STOCHASTIC_METHOD)
IID_CHANNEL_DESIGN = "iid_exact_categorical"
WEIGHTED_CHANNEL_DESIGN = "defensive_half_uniform_importance"
CHANNEL_DESIGNS = (IID_CHANNEL_DESIGN, WEIGHTED_CHANNEL_DESIGN)
APPROVED_OBSERVABLES = (
    "quadratic_mean_field",
    "mixed_linear_quadratic",
    "odd_linear",
    "physical_log_partition",
)


def _finite_number(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _validate_fixed_scope(geometry: str, n_t: int) -> None:
    if geometry != SUPPORTED_GEOMETRY:
        raise ValueError("supported geometry is exactly periodic 3x3")
    if type(n_t) is not int or n_t != SUPPORTED_TIME_SLICES:
        raise ValueError("supported Euclidean slice count is exactly n_t=1")


@dataclass(frozen=True, slots=True)
class ExactRunConfig:
    """Configuration for analytic integration of every fixed-support channel."""

    U: float = 2.0
    beta: float = 1.5
    kappa: float = 1.0
    mu_chem: float = 0.75
    geometry: str = SUPPORTED_GEOMETRY
    n_t: int = SUPPORTED_TIME_SLICES
    method: str = EXACT_METHOD
    observable: str = "mixed_linear_quadratic"
    samples: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        _validate_fixed_scope(self.geometry, self.n_t)
        if self.method != EXACT_METHOD:
            raise ValueError("exact configuration supports only exact-enumeration")
        if self.observable not in APPROVED_OBSERVABLES:
            raise ValueError(f"observable is not approved: {self.observable}")
        if self.samples is not None or self.seed is not None:
            raise ValueError(
                "exact-enumeration integrates all channels and Gaussian sources; "
                "samples and seed are invalid"
            )
        object.__setattr__(self, "U", _finite_number("U", self.U, positive=True))
        object.__setattr__(self, "beta", _finite_number("beta", self.beta, positive=True))
        object.__setattr__(self, "kappa", _finite_number("kappa", self.kappa))
        object.__setattr__(self, "mu_chem", _finite_number("mu_chem", self.mu_chem))

    @property
    def parameters(self) -> dict[str, float]:
        return {
            "U": self.U,
            "beta": self.beta,
            "kappa": self.kappa,
            "mu_chem": self.mu_chem,
        }


@dataclass(frozen=True, slots=True)
class StochasticRunConfig:
    """Configuration for one of the two exact-law stochastic estimators."""

    U: float = 2.0
    beta: float = 1.5
    kappa: float = 1.0
    mu_chem: float = 0.75
    geometry: str = SUPPORTED_GEOMETRY
    n_t: int = SUPPORTED_TIME_SLICES
    method: str = RB_STOCHASTIC_METHOD
    observable: str = "mixed_linear_quadratic"
    samples: int = 512
    seed: int = 202609010001
    channel_design: str = IID_CHANNEL_DESIGN
    persist_endpoints: bool | None = None
    chunk_size: int = 256

    def __post_init__(self) -> None:
        _validate_fixed_scope(self.geometry, self.n_t)
        if self.method not in STOCHASTIC_METHODS:
            raise ValueError(f"unsupported exact-law stochastic method: {self.method}")
        if self.observable not in APPROVED_OBSERVABLES:
            raise ValueError(f"observable is not approved: {self.observable}")
        if type(self.samples) is not int or self.samples < 2:
            raise ValueError("stochastic uncertainty requires at least two samples")
        if type(self.seed) is not int or not -(2**63) <= self.seed < 2**63:
            raise ValueError("seed must be a signed 64-bit integer")
        if self.channel_design not in CHANNEL_DESIGNS:
            raise ValueError(f"unsupported channel design: {self.channel_design}")
        if self.persist_endpoints is None:
            object.__setattr__(
                self,
                "persist_endpoints",
                self.method == EXPLICIT_STOCHASTIC_METHOD,
            )
        if type(self.persist_endpoints) is not bool:
            raise ValueError("persist_endpoints must be boolean")
        if self.method == RB_STOCHASTIC_METHOD and self.persist_endpoints:
            raise ValueError(
                "persist_endpoints is not applicable to exact-contour-rb; "
                "sources and endpoints are analytically integrated"
            )
        if type(self.chunk_size) is not int or not 1 <= self.chunk_size <= 4096:
            raise ValueError("chunk_size must be an integer from 1 through 4096")
        object.__setattr__(self, "U", _finite_number("U", self.U, positive=True))
        object.__setattr__(self, "beta", _finite_number("beta", self.beta, positive=True))
        object.__setattr__(self, "kappa", _finite_number("kappa", self.kappa))
        object.__setattr__(self, "mu_chem", _finite_number("mu_chem", self.mu_chem))

    @property
    def parameters(self) -> dict[str, float]:
        return {
            "U": self.U,
            "beta": self.beta,
            "kappa": self.kappa,
            "mu_chem": self.mu_chem,
        }

    @property
    def authority(self) -> str:
        return (
            "exact_law_stochastic"
            if self.method == EXPLICIT_STOCHASTIC_METHOD
            else "exact_law_stochastic_rb"
        )
