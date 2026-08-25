"""Exact-law stochastic estimators for the fixed indexed contour."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import math

import torch

from .config import IID_CHANNEL_DESIGN, WEIGHTED_CHANNEL_DESIGN
from .core import CDTYPE, RDTYPE, ExactIndexedContourOracle
from .core.observables import EntirePolynomial


_RNG_SCIENCE_ALGORITHM_VERSION = "tg3x3-record-indexed-exact-rational-v2"


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _counter_digest(
    *, seed: int, domain: str, counter: int, attempt: int = 0, block: int = 0
) -> bytes:
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError("counter RNG seed must be a signed 64-bit integer")
    if not isinstance(domain, str) or not domain or "\x00" in domain:
        raise ValueError("counter RNG domain must be a nonempty text label")
    if type(counter) is not int or not 0 <= counter < 2**64:
        raise ValueError("counter RNG record index must be an unsigned 64-bit integer")
    if type(attempt) is not int or not 0 <= attempt < 2**64:
        raise ValueError("counter RNG rejection attempt must be an unsigned 64-bit integer")
    if type(block) is not int or not 0 <= block < 2**32:
        raise ValueError("counter RNG block index must be an unsigned 32-bit integer")
    digest = hashlib.sha256()
    digest.update(_RNG_SCIENCE_ALGORITHM_VERSION.encode("ascii"))
    digest.update(b"\x00")
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(seed.to_bytes(8, "big", signed=True))
    digest.update(counter.to_bytes(8, "big", signed=False))
    digest.update(attempt.to_bytes(8, "big", signed=False))
    digest.update(block.to_bytes(4, "big", signed=False))
    return digest.digest()


def _counter_uniform_below(
    *, seed: int, domain: str, counter: int, upper: int
) -> int:
    """Draw an exact uniform integer in ``range(upper)`` by rejection."""

    if type(upper) is not int or upper < 1:
        raise ValueError("counter RNG integer upper bound must be positive")
    bit_count = upper.bit_length()
    block_count = (bit_count + 255) // 256
    mask = (1 << bit_count) - 1
    attempt = 0
    while True:
        candidate = 0
        for block in range(block_count):
            candidate = (candidate << 256) | int.from_bytes(
                _counter_digest(
                    seed=seed,
                    domain=domain,
                    counter=counter,
                    attempt=attempt,
                    block=block,
                ),
                "big",
            )
        candidate &= mask
        if candidate < upper:
            return candidate
        attempt += 1
        if attempt >= 2**64:
            raise RuntimeError("counter RNG exhausted its rejection-attempt space")


def _counter_torch_seed(*, seed: int, domain: str, counter: int) -> int:
    return int.from_bytes(
        _counter_digest(seed=seed, domain=domain, counter=counter)[:8], "big"
    )


def _represented_integer_masses(probabilities: torch.Tensor) -> tuple[tuple[int, ...], int]:
    ratios = [float(value).as_integer_ratio() for value in probabilities]
    exponents = [denominator.bit_length() - 1 for _, denominator in ratios]
    common_exponent = max(exponents)
    masses = tuple(
        numerator << (common_exponent - exponent)
        for (numerator, _), exponent in zip(ratios, exponents, strict=True)
    )
    total = sum(masses)
    if total <= 0:
        raise ValueError("channel probabilities must have positive represented mass")
    return masses, total


def _proposal_for_design(
    oracle: ExactIndexedContourOracle, design: str
) -> tuple[tuple[int, ...], int, torch.Tensor, torch.Tensor]:
    probabilities = oracle.channel_probabilities
    if design == IID_CHANNEL_DESIGN:
        if bool((probabilities == 0.0).any()):
            raise ValueError(
                "some finite-log channel probabilities are below float64 resolution; "
                "select defensive_half_uniform_importance for complete support"
            )
    elif design != WEIGHTED_CHANNEL_DESIGN:
        raise ValueError(f"unsupported channel design: {design}")

    represented_masses, represented_total = _represented_integer_masses(
        probabilities
    )
    if design == IID_CHANNEL_DESIGN:
        numerators = represented_masses
        denominator = represented_total
    else:
        mode_count = oracle.mode_count
        numerators = tuple(
            mode_count * mass + represented_total for mass in represented_masses
        )
        denominator = 2 * mode_count * represented_total
    if sum(numerators) != denominator:
        raise RuntimeError("exact rational proposal masses must sum to their denominator")
    if any(numerator <= 0 for numerator in numerators):
        raise ValueError(
            "the represented categorical proposal does not have complete support"
        )
    proposal = torch.tensor(
        [numerator / denominator for numerator in numerators],
        dtype=RDTYPE,
        device=oracle.modes.device,
    )
    log_proposal = torch.log(proposal)
    return numerators, denominator, proposal, log_proposal


def _channel_from_uniform_integer(
    cumulative_masses: tuple[int, ...], draw: int
) -> int:
    """Map one exact uniform integer to its categorical mass interval."""

    if not cumulative_masses or cumulative_masses[0] <= 0:
        raise ValueError("categorical cumulative masses must be positive")
    if type(draw) is not int or not 0 <= draw < cumulative_masses[-1]:
        raise ValueError("categorical integer draw must belong to the proposal range")
    return bisect_right(cumulative_masses, draw)


@dataclass(frozen=True, slots=True)
class ChannelAllocation:
    """An ordered channel draw and its exact proposal correction."""

    channel: torch.Tensor
    proposal_probabilities: torch.Tensor
    proposal_log_probabilities: torch.Tensor
    proposal_numerators: tuple[int, ...]
    proposal_denominator: int
    log_design_weight: torch.Tensor
    design: str
    seed: int
    proposal_has_complete_support: bool
    iid_categorical_used: bool
    underflowed_iid_categorical_refused: bool
    unrepresentable_channel_count: int
    unrepresentable_log_probability_mass: float | None
    rng_algorithm_version: str
    oracle_probability_sha256: str

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
    (
        proposal_numerators,
        proposal_denominator,
        proposal,
        proposal_log_probabilities,
    ) = _proposal_for_design(oracle, design)
    domain = (
        "iid_exact_categorical.channel"
        if design == IID_CHANNEL_DESIGN
        else "defensive_half_uniform_importance.channel"
    )
    cumulative_values: list[int] = []
    cumulative = 0
    for numerator in proposal_numerators:
        cumulative += numerator
        cumulative_values.append(cumulative)
    cumulative_masses = tuple(cumulative_values)
    channel = torch.tensor(
        [
            _channel_from_uniform_integer(
                cumulative_masses,
                _counter_uniform_below(
                    seed=seed,
                    domain=domain,
                    counter=index,
                    upper=proposal_denominator,
                ),
            )
            for index in range(sample_count)
        ],
        dtype=torch.int64,
        device=oracle.modes.device,
    )
    log_weight = (
        oracle.channel_log_probabilities[channel]
        - proposal_log_probabilities[channel]
    )
    iid_used = design == IID_CHANNEL_DESIGN
    if not bool(torch.isfinite(log_weight).all()):
        raise FloatingPointError(
            "channel design must have finite complete-support log weights"
        )

    return ChannelAllocation(
        channel=channel,
        proposal_probabilities=proposal,
        proposal_log_probabilities=proposal_log_probabilities,
        proposal_numerators=proposal_numerators,
        proposal_denominator=proposal_denominator,
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
        rng_algorithm_version=_RNG_SCIENCE_ALGORITHM_VERSION,
        oracle_probability_sha256=_tensor_sha256(oracle.channel_probabilities),
    )


@dataclass(frozen=True, slots=True)
class LogComponentStatistics:
    """Stable signed first- and second-moment sufficient statistics."""

    positive_log_sum: float | None
    negative_log_sum: float | None
    log_sum_squares: float | None
    log_scale: float | None
    scaled_sum: float
    scaled_sum_squared_deviations: float


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
    log_magnitude = log_weight[nonzero] + torch.log(component[nonzero].abs())
    if log_magnitude.numel():
        log_scale = float(log_magnitude.max())
        scaled = torch.zeros_like(component)
        scaled[nonzero] = torch.sign(component[nonzero]) * torch.exp(
            log_magnitude - log_scale
        )
        scaled_values = scaled.tolist()
        scaled_sum = math.fsum(scaled_values)
        scaled_mean = scaled_sum / int(component.numel())
        scaled_sum_squared_deviations = math.fsum(
            (value - scaled_mean) ** 2 for value in scaled_values
        )
    else:
        log_scale = None
        scaled_sum = 0.0
        scaled_sum_squared_deviations = 0.0
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
        log_scale=log_scale,
        scaled_sum=scaled_sum,
        scaled_sum_squared_deviations=scaled_sum_squared_deviations,
    )


def _rescale_signed(value: float, log_scale: float | None) -> float:
    if value == 0.0 or log_scale is None:
        return 0.0
    return math.copysign(
        math.exp(log_scale + math.log(abs(value))),
        value,
    )


def _estimate_from_log_statistics(
    *,
    real: LogComponentStatistics,
    imag: LogComponentStatistics,
    count: int,
) -> tuple[complex, float, float]:
    if type(count) is not int or count < 2:
        raise ValueError("log sufficient statistics require at least two samples")
    estimates: list[float] = []
    errors: list[float] = []
    for statistics in (real, imag):
        estimates.append(
            _rescale_signed(
                statistics.scaled_sum / count,
                statistics.log_scale,
            )
        )
        scaled_standard_error = math.sqrt(
            statistics.scaled_sum_squared_deviations / (count * (count - 1))
        )
        errors.append(
            _rescale_signed(scaled_standard_error, statistics.log_scale)
        )
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
    if allocation.oracle_probability_sha256 != _tensor_sha256(
        oracle.channel_probabilities
    ):
        raise ValueError("allocation proposal does not belong to this oracle")
    if (
        len(allocation.proposal_numerators) != oracle.mode_count
        or allocation.proposal_probabilities.shape != (oracle.mode_count,)
        or allocation.proposal_log_probabilities.shape != (oracle.mode_count,)
        or allocation.proposal_denominator <= 0
        or sum(allocation.proposal_numerators) != allocation.proposal_denominator
        or any(numerator <= 0 for numerator in allocation.proposal_numerators)
    ):
        raise ValueError("allocation rational proposal is invalid")
    if allocation.rng_algorithm_version != _RNG_SCIENCE_ALGORITHM_VERSION:
        raise ValueError("allocation RNG algorithm does not belong to this estimator")
    if allocation.log_design_weight.shape != allocation.channel.shape:
        raise ValueError("allocation log weights must match its ordered channels")
    expected_log_weight = (
        oracle.channel_log_probabilities[allocation.channel]
        - allocation.proposal_log_probabilities[allocation.channel]
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
