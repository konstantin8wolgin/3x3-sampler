"""Approved entire-polynomial observables for the exact 3x3 contour."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .atoms import CDTYPE, RDTYPE
from .hubbard import Hubbard3x3Target
from .indexed import ExactIndexedContourOracle


@dataclass(frozen=True, slots=True)
class EntirePolynomial:
    """An approved entire observable ``c + l^T z + z^T Q z``."""

    constant: complex = 0.0j
    linear: torch.Tensor | None = None
    quadratic: torch.Tensor | None = None

    def _coefficients(
        self, d: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _polynomial_coefficients(self, d, device)

    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the polynomial directly at real or complex points."""

        points = torch.atleast_2d(torch.as_tensor(z)).to(dtype=CDTYPE)
        if points.ndim != 2:
            raise ValueError("polynomial points must have shape (samples, dimension)")
        if not torch.isfinite(points.real).all() or not torch.isfinite(
            points.imag
        ).all():
            raise ValueError("polynomial points must be finite")
        constant, linear, quadratic = self._coefficients(
            points.shape[1], points.device
        )
        return (
            constant
            + points @ linear
            + torch.einsum("ni,ij,nj->n", points, quadratic, points)
        )

    def conditional_moments(
        self,
        means: torch.Tensor,
        precision: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian conditional means and complex variances by channel."""

        return _polynomial_conditional_moments(self, means, precision)


def _polynomial_coefficients(
    observable: EntirePolynomial,
    d: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and validate stored coefficients without virtual dispatch."""

    constant = torch.as_tensor(observable.constant, dtype=CDTYPE, device=device)
    linear = (
        torch.zeros(d, dtype=CDTYPE, device=device)
        if observable.linear is None
        else torch.as_tensor(observable.linear, dtype=CDTYPE, device=device)
    )
    quadratic = (
        torch.zeros(d, d, dtype=CDTYPE, device=device)
        if observable.quadratic is None
        else torch.as_tensor(observable.quadratic, dtype=CDTYPE, device=device)
    )
    if constant.ndim != 0:
        raise ValueError("constant coefficient must be scalar")
    if linear.shape != (d,):
        raise ValueError("linear coefficient must match the oracle dimension")
    if quadratic.shape != (d, d):
        raise ValueError(
            "quadratic coefficient must be a square oracle-dimensional matrix"
        )
    for name, value in (
        ("constant", constant),
        ("linear", linear),
        ("quadratic", quadratic),
    ):
        if not torch.isfinite(value.real).all() or not torch.isfinite(
            value.imag
        ).all():
            raise ValueError(f"{name} coefficients must be finite")
    return constant, linear, quadratic


def _polynomial_conditional_moments(
    observable: EntirePolynomial,
    means: torch.Tensor,
    precision: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    channel_means = torch.atleast_2d(torch.as_tensor(means)).to(dtype=CDTYPE)
    if channel_means.ndim != 2:
        raise ValueError("channel means must have shape (channels, dimension)")
    if not torch.isfinite(channel_means.real).all() or not torch.isfinite(
        channel_means.imag
    ).all():
        raise ValueError("channel means must be finite")
    try:
        precision = float(precision)
    except (TypeError, ValueError) as exc:
        raise ValueError("precision must be finite and positive") from exc
    if not math.isfinite(precision) or precision <= 0.0:
        raise ValueError("precision must be finite and positive")

    constant, linear, quadratic = _polynomial_coefficients(
        observable, channel_means.shape[1], channel_means.device
    )
    symmetric = 0.5 * (quadratic + quadratic.transpose(-1, -2))
    variance = torch.as_tensor(
        1.0 / precision, dtype=RDTYPE, device=channel_means.device
    )
    conditional_mean = (
        constant
        + channel_means @ linear
        + torch.einsum(
            "ni,ij,nj->n", channel_means, symmetric, channel_means
        )
        + variance.to(CDTYPE) * torch.trace(symmetric)
    )
    effective_linear = linear.unsqueeze(0) + channel_means @ (2.0 * symmetric)
    conditional_variance = (
        variance * effective_linear.abs().square().sum(dim=1)
        + 2.0 * variance.square() * symmetric.abs().square().sum()
    )
    return conditional_mean, conditional_variance.to(RDTYPE)


def approved_observables(dimension: int = 9) -> dict[str, EntirePolynomial]:
    """Return the frozen three-polynomial suite for a positive dimension."""

    if type(dimension) is not int or dimension <= 0:
        raise ValueError("polynomial dimension must be a positive integer")
    quadratic = torch.eye(dimension, dtype=RDTYPE) / dimension
    return {
        "quadratic_mean_field": EntirePolynomial(quadratic=quadratic),
        "mixed_linear_quadratic": EntirePolynomial(
            linear=torch.linspace(-0.4, 0.4, dimension, dtype=RDTYPE),
            quadratic=quadratic.clone(),
        ),
        "odd_linear": EntirePolynomial(
            linear=torch.ones(dimension, dtype=RDTYPE) / dimension
        ),
    }


def exact_enumeration(
    oracle: ExactIndexedContourOracle,
    observable: EntirePolynomial,
) -> complex:
    """Enumerate all channels and integrate each real Gaussian analytically."""

    if type(observable) is not EntirePolynomial:
        raise TypeError("observable must be an approved EntirePolynomial")
    probabilities = torch.softmax(oracle.channel_log_magnitudes, dim=0)
    channel_mean, _ = observable.conditional_moments(
        oracle.means, oracle.precision
    )
    weighted_phase = probabilities.to(CDTYPE) * oracle.phase
    mean_phase = weighted_phase.sum()
    if float(mean_phase.abs()) == 0.0:
        raise ZeroDivisionError("the exact phase average is zero")
    return complex(((weighted_phase * channel_mean).sum() / mean_phase).item())


def physical_log_partition(
    target: Hubbard3x3Target,
    oracle: ExactIndexedContourOracle,
) -> float:
    """Return the physical log partition with the chemical-potential factor."""

    if oracle.d != target.d:
        raise ValueError("target and oracle dimensions must agree")
    return float(
        torch.logsumexp(oracle.channel_log_magnitudes, dim=0)
        + target.beta * target.mu_chem * target.n_x
    )


__all__ = [
    "EntirePolynomial",
    "approved_observables",
    "exact_enumeration",
    "physical_log_partition",
]
