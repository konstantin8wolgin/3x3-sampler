"""Exact-law stochastic estimators for the fixed indexed contour."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch

from .config import IID_CHANNEL_DESIGN, WEIGHTED_CHANNEL_DESIGN
from .core import CDTYPE, RDTYPE, ExactIndexedContourOracle
from .core.observables import EntirePolynomial


_RNG_SCIENCE_ALGORITHM_VERSION = "tg3x3-record-indexed-sha256-v1"


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _counter_digest(*, seed: int, domain: str, counter: int) -> bytes:
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError("counter RNG seed must be a signed 64-bit integer")
    if not isinstance(domain, str) or not domain or "\x00" in domain:
        raise ValueError("counter RNG domain must be a nonempty text label")
    if type(counter) is not int or not 0 <= counter < 2**64:
        raise ValueError("counter RNG record index must be an unsigned 64-bit integer")
    digest = hashlib.sha256()
    digest.update(_RNG_SCIENCE_ALGORITHM_VERSION.encode("ascii"))
    digest.update(b"\x00")
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(seed.to_bytes(8, "big", signed=True))
    digest.update(counter.to_bytes(8, "big", signed=False))
    return digest.digest()


def _counter_uniform_float64(*, seed: int, domain: str, counter: int) -> float:
    mantissa = int.from_bytes(
        _counter_digest(seed=seed, domain=domain, counter=counter)[:8], "big"
    ) >> 11
    return (mantissa + 0.5) * (2.0**-53)


def _counter_torch_seed(*, seed: int, domain: str, counter: int) -> int:
    return int.from_bytes(
        _counter_digest(seed=seed, domain=domain, counter=counter)[:8], "big"
    )


def _proposal_for_design(
    oracle: ExactIndexedContourOracle, design: str
) -> torch.Tensor:
    probabilities = oracle.channel_probabilities
    if design == IID_CHANNEL_DESIGN:
        if bool((probabilities == 0.0).any()):
            raise ValueError(
                "some finite-log channel probabilities are below float64 resolution; "
                "select defensive_half_uniform_importance for complete support"
            )
        return probabilities
    if design == WEIGHTED_CHANNEL_DESIGN:
        represented = probabilities / probabilities.sum()
        return 0.5 * represented + 0.5 / oracle.mode_count
    raise ValueError(f"unsupported channel design: {design}")


@dataclass(frozen=True, slots=True)
class ChannelAllocation:
    """An ordered channel draw and its exact proposal correction."""

    channel: torch.Tensor
    proposal_probabilities: torch.Tensor
    log_design_weight: torch.Tensor
    design: str
    seed: int
    proposal_has_complete_support: bool
    iid_categorical_used: bool
    underflowed_iid_categorical_refused: bool
    unrepresentable_channel_count: int
    unrepresentable_log_probability_mass: float | None

    @property
    def selected_proposal_probability(self) -> torch.Tensor:
        return self.proposal_probabilities[self.channel]

    @property
    def allocation_sha256(self) -> str:
        return _tensor_sha256(self.channel)

    @property
    def channel_sha256(self) -> str:
        return self.allocation_sha256


def allocate_channels(
    oracle: ExactIndexedContourOracle,
    *,
    sample_count: int,
    seed: int,
    design: str,
) -> ChannelAllocation:
    """Draw ordered exact-law channels, retaining ``log(p / q)``."""

    if type(oracle) is not ExactIndexedContourOracle:
        raise TypeError("oracle must be an ExactIndexedContourOracle")
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("at least two samples are required for uncertainty diagnostics")
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError("seed must be a signed 64-bit integer")

    probabilities = oracle.channel_probabilities
    underflow_mask = probabilities == 0.0
    underflowed = bool(underflow_mask.any())
    proposal = _proposal_for_design(oracle, design)
    domain = (
        "iid_exact_categorical.channel"
        if design == IID_CHANNEL_DESIGN
        else "defensive_half_uniform_importance.channel"
    )
    uniforms = torch.tensor(
        [
            _counter_uniform_float64(seed=seed, domain=domain, counter=index)
            for index in range(sample_count)
        ],
        dtype=RDTYPE,
        device=oracle.modes.device,
    )
    cumulative = torch.cumsum(proposal, dim=0)
    cumulative[-1] = 1.0
    channel = torch.searchsorted(cumulative, uniforms, right=False).to(torch.int64)
    if design == IID_CHANNEL_DESIGN:
        log_weight = torch.zeros(
            sample_count, dtype=RDTYPE, device=oracle.modes.device
        )
        iid_used = True
    else:
        log_weight = oracle.channel_log_probabilities[channel] - torch.log(
            proposal[channel]
        )
        iid_used = False
    if not bool(torch.isfinite(log_weight).all()):
        raise FloatingPointError(
            "channel design must have finite complete-support log weights"
        )

    return ChannelAllocation(
        channel=channel,
        proposal_probabilities=proposal,
        log_design_weight=log_weight,
        design=design,
        seed=seed,
        proposal_has_complete_support=True,
        iid_categorical_used=iid_used,
        underflowed_iid_categorical_refused=underflowed and not iid_used,
        unrepresentable_channel_count=int(underflow_mask.sum()),
        unrepresentable_log_probability_mass=(
            float(
                torch.logsumexp(
                    oracle.channel_log_probabilities[underflow_mask], dim=0
                )
            )
            if underflowed
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class LogComponentStatistics:
    """Stable signed first- and second-moment sufficient statistics."""

    positive_log_sum: float | None
    negative_log_sum: float | None
    log_sum_squares: float | None


@dataclass(frozen=True, slots=True)
class StochasticEstimate:
    """One weighted complex estimate with observable-component uncertainty."""

    value: complex
    standard_error_real: float
    standard_error_imag: float
    allocation: ChannelAllocation
    source: torch.Tensor | None
    endpoint: torch.Tensor | None
    count: int
    real_statistics: LogComponentStatistics
    imag_statistics: LogComponentStatistics

    @property
    def channel(self) -> torch.Tensor:
        return self.allocation.channel


def _optional_logsumexp(values: torch.Tensor) -> float | None:
    return float(torch.logsumexp(values, dim=0)) if values.numel() else None


def _log_component_statistics(
    log_weight: torch.Tensor, component: torch.Tensor
) -> LogComponentStatistics:
    if not bool(torch.isfinite(component).all()):
        raise FloatingPointError("stochastic observable components must be finite")
    positive = component > 0.0
    negative = component < 0.0
    nonzero = component != 0.0
    return LogComponentStatistics(
        positive_log_sum=_optional_logsumexp(
            log_weight[positive] + torch.log(component[positive])
        ),
        negative_log_sum=_optional_logsumexp(
            log_weight[negative] + torch.log(-component[negative])
        ),
        log_sum_squares=_optional_logsumexp(
            2.0 * (log_weight[nonzero] + torch.log(component[nonzero].abs()))
        ),
    )


def _signed_log_sum(
    statistics: LogComponentStatistics,
) -> tuple[float, float | None]:
    positive = statistics.positive_log_sum
    negative = statistics.negative_log_sum
    if positive is None and negative is None:
        return 0.0, None
    scale = max(value for value in (positive, negative) if value is not None)
    scaled = (
        (0.0 if positive is None else math.exp(positive - scale))
        - (0.0 if negative is None else math.exp(negative - scale))
    )
    if scaled == 0.0:
        return 0.0, None
    return math.copysign(1.0, scaled), scale + math.log(abs(scaled))


def _estimate_from_log_statistics(
    *,
    real: LogComponentStatistics,
    imag: LogComponentStatistics,
    count: int,
) -> tuple[complex, float, float]:
    if type(count) is not int or count < 2:
        raise ValueError("log sufficient statistics require at least two samples")
    signed_log_sums = (_signed_log_sum(real), _signed_log_sum(imag))
    estimates: list[float] = []
    errors: list[float] = []
    log_count = math.log(count)
    for (sign, log_sum), statistics in zip(
        signed_log_sums, (real, imag), strict=True
    ):
        estimate = 0.0 if log_sum is None else sign * math.exp(log_sum - log_count)
        estimates.append(estimate)
        if statistics.log_sum_squares is None:
            errors.append(0.0)
            continue
        centered_term = None if log_sum is None else 2.0 * log_sum - log_count
        scale = max(
            value
            for value in (statistics.log_sum_squares, centered_term)
            if value is not None
        )
        scaled_variance_numerator = math.exp(
            statistics.log_sum_squares - scale
        ) - (0.0 if centered_term is None else math.exp(centered_term - scale))
        if scaled_variance_numerator <= 0.0:
            errors.append(0.0)
            continue
        log_variance_of_mean = (
            scale
            + math.log(scaled_variance_numerator)
            - math.log(count * (count - 1))
        )
        errors.append(math.exp(0.5 * log_variance_of_mean))
    if not all(math.isfinite(value) for value in (*estimates, *errors)):
        raise FloatingPointError("log-domain estimate and uncertainty must be finite")
    return complex(*estimates), errors[0], errors[1]


def _validate_estimator_inputs(
    oracle: ExactIndexedContourOracle,
    observable: EntirePolynomial,
    allocation: ChannelAllocation,
) -> None:
    if type(oracle) is not ExactIndexedContourOracle:
        raise TypeError("oracle must be an ExactIndexedContourOracle")
    if type(observable) is not EntirePolynomial:
        raise TypeError("observable must be an EntirePolynomial")
    if type(allocation) is not ChannelAllocation:
        raise TypeError("allocation must be a ChannelAllocation")
    if allocation.channel.ndim != 1 or allocation.channel.numel() < 2:
        raise ValueError("allocation must contain at least two ordered channels")
    if bool((allocation.channel < 0).any()) or bool(
        (allocation.channel >= oracle.mode_count).any()
    ):
        raise ValueError("allocation channels must belong to the oracle")
    expected_proposal = _proposal_for_design(oracle, allocation.design)
    if not torch.equal(allocation.proposal_probabilities, expected_proposal):
        raise ValueError("allocation proposal does not belong to this oracle")
    if allocation.log_design_weight.shape != allocation.channel.shape:
        raise ValueError("allocation log weights must match its ordered channels")
    expected_log_weight = (
        torch.zeros_like(allocation.log_design_weight)
        if allocation.design == IID_CHANNEL_DESIGN
        else oracle.channel_log_probabilities[allocation.channel]
        - torch.log(expected_proposal[allocation.channel])
    )
    if not torch.equal(allocation.log_design_weight, expected_log_weight):
        raise ValueError("allocation correction does not belong to this oracle")
    if not bool(torch.equal(oracle.phase, torch.ones_like(oracle.phase))):
        raise ValueError(
            "supported exact-law stochastic estimation requires the positive "
            "channel-phase contract"
        )


def _estimate(
    oracle: ExactIndexedContourOracle,
    observable: EntirePolynomial,
    allocation: ChannelAllocation,
    *,
    explicit: bool,
) -> StochasticEstimate:
    _validate_estimator_inputs(oracle, observable, allocation)
    source: torch.Tensor | None = None
    endpoint: torch.Tensor | None = None
    if explicit:
        sources: list[torch.Tensor] = []
        endpoints: list[torch.Tensor] = []
        for index in range(int(allocation.channel.numel())):
            generator = torch.Generator(device=oracle.modes.device).manual_seed(
                _counter_torch_seed(
                    seed=allocation.seed,
                    domain="explicit_contour.gaussian_source",
                    counter=index,
                )
            )
            sample = oracle.sample_for_channels(
                allocation.channel[index : index + 1], generator=generator
            )
            sources.append(sample.source)
            endpoints.append(sample.endpoint)
        source = torch.cat(sources, dim=0)
        endpoint = torch.cat(endpoints, dim=0)
        values = observable.evaluate(endpoint)
    else:
        conditional_means, _ = observable.conditional_moments(
            oracle.means, oracle.precision
        )
        values = conditional_means[allocation.channel]

    phased_values = values * oracle.phase[allocation.channel]
    real_statistics = _log_component_statistics(
        allocation.log_design_weight, phased_values.real
    )
    imag_statistics = _log_component_statistics(
        allocation.log_design_weight, phased_values.imag
    )
    count = int(phased_values.numel())
    value, standard_error_real, standard_error_imag = _estimate_from_log_statistics(
        real=real_statistics,
        imag=imag_statistics,
        count=count,
    )
    return StochasticEstimate(
        value=value,
        standard_error_real=standard_error_real,
        standard_error_imag=standard_error_imag,
        allocation=allocation,
        source=source,
        endpoint=endpoint,
        count=count,
        real_statistics=real_statistics,
        imag_statistics=imag_statistics,
    )


def estimate_explicit_contour(
    oracle: ExactIndexedContourOracle,
    observable: EntirePolynomial,
    allocation: ChannelAllocation,
) -> StochasticEstimate:
    """Draw real Gaussian sources and estimate on translated endpoints."""

    return _estimate(oracle, observable, allocation, explicit=True)


def estimate_rao_blackwell(
    oracle: ExactIndexedContourOracle,
    observable: EntirePolynomial,
    allocation: ChannelAllocation,
) -> StochasticEstimate:
    """Integrate the Gaussian source analytically for the same allocation."""

    return _estimate(oracle, observable, allocation, explicit=False)


__all__ = [
    "ChannelAllocation",
    "LogComponentStatistics",
    "StochasticEstimate",
    "allocate_channels",
    "estimate_explicit_contour",
    "estimate_rao_blackwell",
]
