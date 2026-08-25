"""The complete indexed complex-Gaussian contour representation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .atoms import CDTYPE, RDTYPE


@dataclass(frozen=True, slots=True)
class IndexedContourSample:
    """Samples from explicit channels of the exact translated contour."""

    channel: torch.Tensor
    source: torch.Tensor
    endpoint: torch.Tensor
    channel_log_probability: torch.Tensor
    source_log_density: torch.Tensor
    joint_log_proposal: torch.Tensor
    phase: torch.Tensor
    jacobian: torch.Tensor


@dataclass(frozen=True, slots=True)
class ExactIndexedContourOracle:
    """Complete one-slice Fourier/Gaussian expansion with explicit modes."""

    modes: torch.Tensor
    channel_masses: torch.Tensor
    precision: float
    channel_log_magnitudes: torch.Tensor | None = None
    channel_phase: torch.Tensor | None = None

    def __post_init__(self) -> None:
        raw_modes = torch.atleast_2d(self.modes)
        real_modes = raw_modes.real if raw_modes.is_complex() else raw_modes
        if (
            not torch.isfinite(real_modes).all()
            or raw_modes.is_complex()
            and not torch.equal(raw_modes.imag, torch.zeros_like(raw_modes.imag))
        ):
            raise ValueError("modes must be finite real integers")
        rounded_modes = torch.round(real_modes)
        if not torch.equal(real_modes, rounded_modes):
            raise ValueError("modes must be integral")
        modes = rounded_modes.to(dtype=torch.int64)
        masses = torch.atleast_1d(self.channel_masses).to(
            device=modes.device, dtype=CDTYPE
        )
        if modes.shape[0] != masses.numel():
            raise ValueError("mode count must match channel mass count")
        if modes.numel() == 0:
            raise ValueError("an exact indexed oracle requires at least one mode")
        if not torch.isfinite(masses.real).all() or not torch.isfinite(masses.imag).all():
            raise ValueError("channel masses must be finite")
        if torch.unique(modes, dim=0).shape[0] != modes.shape[0]:
            raise ValueError("modes must be unique")
        try:
            precision = float(self.precision)
        except (TypeError, ValueError) as exc:
            raise ValueError("shared precision must be finite and positive") from exc
        if not math.isfinite(precision) or precision <= 0.0:
            raise ValueError("shared precision must be finite and positive")

        if self.channel_log_magnitudes is None:
            log_magnitudes = torch.log(masses.abs())
        else:
            log_magnitudes = torch.atleast_1d(self.channel_log_magnitudes).to(
                device=modes.device, dtype=RDTYPE
            )
            if log_magnitudes.shape != masses.shape or not torch.isfinite(
                log_magnitudes
            ).all():
                raise ValueError(
                    "channel log magnitudes must be finite and match channel masses"
                )

        if self.channel_phase is None:
            magnitudes = masses.abs()
            phase = torch.where(
                magnitudes > 0.0, masses / magnitudes, torch.ones_like(masses)
            )
        else:
            phase = torch.atleast_1d(self.channel_phase).to(
                device=modes.device, dtype=CDTYPE
            )
            if (
                phase.shape != masses.shape
                or not torch.isfinite(phase.real).all()
                or not torch.isfinite(phase.imag).all()
            ):
                raise ValueError("channel phases must be finite and match channel masses")
            if not torch.allclose(
                phase.abs(), torch.ones_like(phase.abs()), rtol=1e-12, atol=1e-12
            ):
                raise ValueError("channel phases must have unit magnitude")

        expected_masses = torch.exp(log_magnitudes).to(CDTYPE) * phase
        representable = expected_masses.abs() > 0.0
        if not torch.allclose(
            masses[representable],
            expected_masses[representable],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "representable channel masses must be consistent with log magnitudes and phases"
            )
        if not torch.equal(
            masses[~representable], torch.zeros_like(masses[~representable])
        ):
            raise ValueError("underflowed channel masses must be stored as zero")

        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "channel_masses", masses)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "channel_log_magnitudes", log_magnitudes)
        object.__setattr__(self, "channel_phase", phase)

    @property
    def mode_count(self) -> int:
        return int(self.modes.shape[0])

    @property
    def d(self) -> int:
        return int(self.modes.shape[1])

    @property
    def means(self) -> torch.Tensor:
        """Analytic translations ``m_n = i A0^-1 n``."""

        return 1j * self.modes.to(RDTYPE) / self.precision

    @property
    def phase(self) -> torch.Tensor:
        return self.channel_phase

    @property
    def channel_log_masses(self) -> torch.Tensor:
        """Exact log magnitudes, retained even when linear masses underflow."""

        return self.channel_log_magnitudes

    @property
    def channel_log_probabilities(self) -> torch.Tensor:
        return self.channel_log_masses - torch.logsumexp(
            self.channel_log_masses, dim=0
        )

    @property
    def channel_probabilities(self) -> torch.Tensor:
        return torch.exp(self.channel_log_probabilities).real

    def evaluate(self, phi: torch.Tensor) -> torch.Tensor:
        """Evaluate the complete mixture on real or complex field points."""

        phi = torch.atleast_2d(phi).to(device=self.modes.device, dtype=CDTYPE)
        if phi.ndim != 2 or phi.shape[1] != self.d:
            raise ValueError("field dimension must match mode dimension")
        diff = phi[:, None, :] - self.means[None, :, :]
        log_atoms = -0.5 * self.precision * (diff * diff).sum(dim=-1)
        log_terms = log_atoms + self.channel_log_masses[None, :]
        shift = log_terms.real.max(dim=1, keepdim=True).values
        return torch.exp(shift[:, 0]) * (
            torch.exp(log_terms - shift) * self.phase[None, :]
        ).sum(dim=1)

    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> IndexedContourSample:
        """Draw IID exact-law channels and real Gaussian sources."""

        if type(n) is not int or n < 0:
            raise ValueError("sample count must be a non-negative integer")
        probabilities = self.channel_probabilities
        if bool((probabilities == 0.0).any()):
            raise ValueError(
                "some finite-log channel probabilities are below float64 resolution; "
                "use sample_for_channels with explicit channel ids"
            )
        channel = torch.multinomial(
            probabilities, n, replacement=True, generator=generator
        )
        return self.sample_for_channels(channel, generator=generator)

    def sample_for_channels(
        self,
        channel: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> IndexedContourSample:
        """Transport explicit channel ids, including float64-rare channels."""

        raw_channel = torch.atleast_1d(torch.as_tensor(channel))
        real_channel = raw_channel.real if raw_channel.is_complex() else raw_channel
        if (
            not torch.isfinite(real_channel).all()
            or raw_channel.is_complex()
            and not torch.equal(
                raw_channel.imag, torch.zeros_like(raw_channel.imag)
            )
            or not torch.equal(real_channel, torch.round(real_channel))
        ):
            raise ValueError("channel ids must be finite integers")
        channel = real_channel.to(device=self.modes.device, dtype=torch.int64)
        if bool((channel < 0).any()) or bool((channel >= self.mode_count).any()):
            raise ValueError("channel ids must belong to the oracle")
        n = int(channel.numel())
        source = torch.randn(
            n,
            self.d,
            dtype=RDTYPE,
            device=self.modes.device,
            generator=generator,
        ) / self.precision**0.5
        source_log_density = (
            0.5
            * self.d
            * torch.log(
                torch.tensor(
                    self.precision, dtype=RDTYPE, device=self.modes.device
                )
            )
            - 0.5
            * self.d
            * torch.log(
                torch.tensor(2.0 * torch.pi, dtype=RDTYPE, device=self.modes.device)
            )
            - 0.5 * self.precision * source.square().sum(dim=1)
        )
        channel_log_probability = self.channel_log_probabilities[channel]
        return IndexedContourSample(
            channel=channel,
            source=source,
            endpoint=source.to(CDTYPE) + self.means[channel],
            channel_log_probability=channel_log_probability,
            source_log_density=source_log_density,
            joint_log_proposal=channel_log_probability + source_log_density,
            phase=self.phase[channel],
            jacobian=torch.ones(n, dtype=CDTYPE, device=self.modes.device),
        )

    def recover_source(
        self, endpoint: torch.Tensor, channel: torch.Tensor
    ) -> torch.Tensor:
        """Invert the unit-Jacobian translation using its retained channel."""

        endpoint = torch.atleast_2d(endpoint).to(
            device=self.modes.device, dtype=CDTYPE
        )
        channel = torch.atleast_1d(channel).to(
            device=self.modes.device, dtype=torch.int64
        )
        if endpoint.shape[0] != channel.numel() or endpoint.shape[1] != self.d:
            raise ValueError("endpoint and channel shapes must match the oracle")
        if bool((channel < 0).any()) or bool((channel >= self.mode_count).any()):
            raise ValueError("channel ids must belong to the oracle")
        return (endpoint - self.means[channel]).real


__all__ = ["ExactIndexedContourOracle", "IndexedContourSample"]
