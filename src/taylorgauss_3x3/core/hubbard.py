"""Exact periodic 3x3, one-slice auxiliary-field Hubbard target."""

from __future__ import annotations

import math

import torch

from .atoms import CDTYPE, RDTYPE
from .indexed import ExactIndexedContourOracle


def square_3x3_hopping() -> torch.Tensor:
    """Return the real-symmetric nearest-neighbor periodic 3x3 hopping matrix."""

    width = height = 3

    def site(row: int, col: int) -> int:
        return row * width + col

    edges: set[tuple[int, int]] = set()
    for row in range(height):
        for col in range(width):
            for neighbor in ((row, (col + 1) % width), ((row + 1) % height, col)):
                i = site(row, col)
                j = site(*neighbor)
                edges.add(tuple(sorted((i, j))))

    hopping = torch.zeros(width * height, width * height, dtype=RDTYPE)
    for i, j in edges:
        hopping[i, j] = 1.0
        hopping[j, i] = 1.0
    if not torch.equal(hopping, hopping.T):
        raise RuntimeError("periodic 3x3 hopping must be real symmetric")
    return hopping


def _poly_mul(
    p: dict[tuple[int, ...], complex], q: dict[tuple[int, ...], complex]
) -> dict[tuple[int, ...], complex]:
    out: dict[tuple[int, ...], complex] = {}
    for np_, cp in p.items():
        for nq, cq in q.items():
            key = tuple(a + b for a, b in zip(np_, nq))
            out[key] = out.get(key, 0j) + cp * cq
    return out


def _principal_minor_poly(
    expk: torch.Tensor, sign: int, d: int
) -> dict[tuple[int, ...], complex]:
    """Polynomial for ``det(I + expk * diag(z_i**sign))``."""

    expk_cpu = expk.detach().cpu()
    n = expk_cpu.shape[0]
    poly: dict[tuple[int, ...], complex] = {}
    for mask in range(1 << n):
        idx = [i for i in range(n) if mask & (1 << i)]
        powers = [0] * d
        for i in idx:
            powers[i] = sign
        if idx:
            sub = expk_cpu[idx][:, idx]
            coeff = complex(torch.linalg.det(sub))
        else:
            coeff = 1.0 + 0j
        poly[tuple(powers)] = coeff
    return poly


def _finite_float(name: str, value: float, *, positive: bool = False) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}") from exc
    if not math.isfinite(value) or positive and value <= 0.0:
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return value


class Hubbard3x3Target:
    """The fixed periodic 3x3 Hubbard target with one Euclidean-time slice."""

    def __init__(self, U: float, beta: float, kappa: float, mu_chem: float):
        self.U = _finite_float("U", U, positive=True)
        self.beta = _finite_float("beta", beta, positive=True)
        self.kappa = _finite_float("kappa", kappa)
        self.mu_chem = _finite_float("mu_chem", mu_chem)
        self.hop = square_3x3_hopping()
        self.n_x = 9
        self.n_t = 1
        self.d = 9
        self.dt = self.beta
        self.precision = 1.0 / (self.U * self.beta)
        self.prior_prec = self.precision

        mu_beta = self.mu_chem * self.beta
        if abs(mu_beta) > math.log(torch.finfo(RDTYPE).max):
            raise ValueError("mu_chem branch factor is not representable in float64")
        self.expk_p = (
            torch.matrix_exp(self.kappa * self.beta * self.hop)
            * math.exp(mu_beta)
        )
        self.expk_m = (
            torch.matrix_exp(-self.kappa * self.beta * self.hop)
            * math.exp(-mu_beta)
        )
        if not torch.isfinite(self.expk_p).all() or not torch.isfinite(
            self.expk_m
        ).all():
            raise ValueError("parameters must produce finite hopping exponentials")

    def _real_field(self, phi: torch.Tensor) -> torch.Tensor:
        raw = torch.as_tensor(phi)
        if raw.is_complex():
            raise ValueError("the real-axis evaluator requires a real field")
        field = torch.atleast_2d(raw).to(dtype=RDTYPE)
        if field.ndim != 2 or field.shape[1] != self.d:
            raise ValueError("field must have shape (samples, 9)")
        if not torch.isfinite(field).all():
            raise ValueError("field values must be finite")
        return field

    def _complex_field(self, phi: torch.Tensor) -> torch.Tensor:
        field = torch.atleast_2d(torch.as_tensor(phi)).to(dtype=CDTYPE)
        if field.ndim != 2 or field.shape[1] != self.d:
            raise ValueError("field must have shape (samples, 9)")
        if not torch.isfinite(field.real).all() or not torch.isfinite(field.imag).all():
            raise ValueError("field values must be finite")
        return field

    def _dets(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(self.n_x, dtype=CDTYPE)
        z = torch.exp(1j * phi.to(CDTYPE))
        product_p = self.expk_p.to(CDTYPE).unsqueeze(0) * z.unsqueeze(1)
        product_m = self.expk_m.to(CDTYPE).unsqueeze(0) * z.conj().unsqueeze(1)
        return torch.linalg.det(eye + product_p), torch.linalg.det(eye + product_m)

    def _holomorphic_dets(
        self, phi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(self.n_x, dtype=CDTYPE)
        z_p = torch.exp(1j * phi)
        z_m = torch.exp(-1j * phi)
        product_p = self.expk_p.to(CDTYPE).unsqueeze(0) * z_p.unsqueeze(1)
        product_m = self.expk_m.to(CDTYPE).unsqueeze(0) * z_m.unsqueeze(1)
        return torch.linalg.det(eye + product_p), torch.linalg.det(eye + product_m)

    def evaluate(self, phi: torch.Tensor) -> torch.Tensor:
        """Evaluate the original real-axis Boltzmann weight."""

        field = self._real_field(phi)
        det_p, det_m = self._dets(field)
        prior = torch.exp(-0.5 * self.precision * field.square().sum(-1)).to(
            CDTYPE
        )
        return prior * det_p * det_m

    def evaluate_holomorphic(self, phi: torch.Tensor) -> torch.Tensor:
        """Evaluate the holomorphic continuation at complex contour points."""

        field = self._complex_field(phi)
        det_p, det_m = self._holomorphic_dets(field)
        prior = torch.exp(-0.5 * self.precision * (field * field).sum(-1))
        return prior * det_p * det_m

    def exact_indexed_oracle(self) -> ExactIndexedContourOracle:
        """Return all ``3**9`` ordered positive one-slice contour channels."""

        if not torch.equal(self.hop, self.hop.T):
            raise RuntimeError("the indexed contour oracle requires real-symmetric hopping")
        try:
            torch.linalg.cholesky(self.expk_p)
            torch.linalg.cholesky(self.expk_m)
        except RuntimeError as exc:
            raise RuntimeError(
                "the indexed contour oracle requires SPD hopping exponentials"
            ) from exc

        poly = _poly_mul(
            _principal_minor_poly(self.expk_p, +1, self.d),
            _principal_minor_poly(self.expk_m, -1, self.d),
        )
        ordered_modes = sorted(poly)
        modes = torch.tensor(ordered_modes, dtype=torch.int64)
        coefficients = torch.tensor(
            [poly[mode].real for mode in ordered_modes], dtype=RDTYPE
        )
        if len(ordered_modes) != 3**9:
            raise RuntimeError("exactly 3**9 indexed labels must be collected")
        if not bool((coefficients > 0.0).all()):
            raise RuntimeError("every indexed coefficient must be strictly positive")

        log_mass = torch.log(coefficients) - modes.square().sum(1) / (
            2.0 * self.precision
        )
        return ExactIndexedContourOracle(
            modes=modes,
            channel_masses=torch.exp(log_mass).to(CDTYPE),
            precision=self.precision,
            channel_log_magnitudes=log_mass,
            channel_phase=torch.ones_like(log_mass, dtype=CDTYPE),
        )


__all__ = ["Hubbard3x3Target", "square_3x3_hopping"]
