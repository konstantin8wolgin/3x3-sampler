import math
from itertools import product

import pytest
import torch

from taylorgauss_3x3.core import (
    CDTYPE,
    RDTYPE,
    Hubbard3x3Target,
    square_3x3_hopping,
)


def _target() -> Hubbard3x3Target:
    return Hubbard3x3Target(U=2.0, beta=1.5, kappa=1.0, mu_chem=0.75)


@pytest.fixture(scope="module")
def anchor_oracle():
    return _target().exact_indexed_oracle()


def test_square_hopping_is_the_fixed_periodic_3x3_geometry():
    hop = square_3x3_hopping()
    expected = torch.tensor(
        [
            [0, 1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 1, 0],
        ],
        dtype=RDTYPE,
    )

    assert torch.equal(hop, expected)


def test_exact_oracle_has_complete_positive_ternary_support(anchor_oracle):
    target = _target()
    oracle = anchor_oracle
    expected_modes = torch.tensor(
        list(product((-1, 0, 1), repeat=9)), dtype=torch.int64
    )

    assert oracle.mode_count == 3**9 == 19_683
    assert torch.equal(oracle.modes, expected_modes)
    assert bool((oracle.channel_masses.real > 0.0).all())
    assert torch.equal(
        oracle.channel_masses.imag, torch.zeros_like(oracle.channel_masses.imag)
    )
    assert torch.equal(oracle.phase, torch.ones_like(oracle.phase))
    assert bool((oracle.channel_log_magnitudes.isfinite()).all())
    torch.testing.assert_close(
        oracle.means,
        1j * target.U * target.beta * oracle.modes.to(RDTYPE),
    )


def test_contour_translation_round_trips_with_unit_jacobian():
    target = _target()
    oracle = target.exact_indexed_oracle()
    channel = torch.tensor([0, 100, 19_682])

    sample = oracle.sample_for_channels(
        channel, generator=torch.Generator().manual_seed(9)
    )

    torch.testing.assert_close(
        sample.endpoint - oracle.means[channel], sample.source.to(CDTYPE)
    )
    torch.testing.assert_close(
        oracle.recover_source(sample.endpoint, channel), sample.source
    )
    assert torch.equal(sample.jacobian, torch.ones(3, dtype=CDTYPE))


@pytest.mark.parametrize("name", ["U", "beta"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_target_rejects_nonpositive_scale_parameters(name, value):
    parameters = {"U": 2.0, "beta": 1.5, "kappa": 1.0, "mu_chem": 0.75}
    parameters[name] = value

    with pytest.raises(ValueError, match=name):
        Hubbard3x3Target(**parameters)


@pytest.mark.parametrize("name", ["U", "beta", "kappa", "mu_chem"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_target_rejects_nonfinite_parameters(name, value):
    parameters = {"U": 2.0, "beta": 1.5, "kappa": 1.0, "mu_chem": 0.75}
    parameters[name] = value

    with pytest.raises(ValueError, match=name):
        Hubbard3x3Target(**parameters)


@pytest.mark.parametrize(
    ("U", "beta"),
    [
        (1e308, 2.0),
        (1e308, 1.0),
        (1e-308, 1.0),
        (1e-308, 1e-308),
    ],
    ids=[
        "scale-overflows",
        "precision-underflows",
        "scale-underflows",
        "scale-becomes-zero",
    ],
)
def test_target_rejects_unrepresentable_derived_precision(U, beta):
    with pytest.raises(ValueError, match="precision"):
        Hubbard3x3Target(U=U, beta=beta, kappa=0.0, mu_chem=0.0)


@pytest.mark.parametrize(
    "unsupported", [{"n_t": 1}, {"geometry": "periodic_3x3"}, {"periodic": True}]
)
def test_target_refuses_geometry_and_slice_arguments(unsupported):
    with pytest.raises(TypeError):
        Hubbard3x3Target(
            U=2.0,
            beta=1.5,
            kappa=1.0,
            mu_chem=0.75,
            **unsupported,
        )


def test_real_target_and_holomorphic_continuation_agree_on_real_fields():
    target = _target()
    phi = torch.tensor(
        [
            [0.0, 0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.4, -0.3],
            [-0.5, 0.25, 0.125, -0.375, 0.5, -0.25, 0.75, -0.125, 0.625],
        ],
        dtype=RDTYPE,
    )

    torch.testing.assert_close(
        target.evaluate(phi),
        target.evaluate_holomorphic(phi.to(CDTYPE)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_exact_mixture_matches_target_at_deterministic_identity_points():
    target = _target()
    oracle = target.exact_indexed_oracle()
    real_phi = torch.tensor(
        [
            [0.0, 0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.4, -0.3],
            [-0.5, 0.25, 0.125, -0.375, 0.5, -0.25, 0.75, -0.125, 0.625],
        ],
        dtype=RDTYPE,
    )
    complex_phi = real_phi.to(CDTYPE) + 0.25j * torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.0],
            [-0.5] * 9,
        ],
        dtype=RDTYPE,
    )

    torch.testing.assert_close(
        oracle.evaluate(real_phi), target.evaluate(real_phi), rtol=2e-9, atol=2e-9
    )
    torch.testing.assert_close(
        oracle.evaluate(complex_phi),
        target.evaluate_holomorphic(complex_phi),
        rtol=3e-9,
        atol=3e-9,
    )


@pytest.mark.parametrize(
    "phi",
    [
        torch.tensor([[math.nan] + [0.0] * 8], dtype=RDTYPE),
        torch.tensor([[math.inf] + [0.0] * 8], dtype=RDTYPE),
        torch.tensor([[complex(0.0, math.nan)] + [0.0j] * 8], dtype=CDTYPE),
        torch.tensor([[complex(0.0, math.inf)] + [0.0j] * 8], dtype=CDTYPE),
    ],
    ids=["real-nan", "real-inf", "imaginary-nan", "imaginary-inf"],
)
def test_oracle_evaluate_rejects_nonfinite_fields(anchor_oracle, phi):
    with pytest.raises(ValueError, match="finite"):
        anchor_oracle.evaluate(phi)


@pytest.mark.parametrize(
    "endpoint",
    [
        torch.tensor([[math.nan] + [0.0] * 8], dtype=RDTYPE),
        torch.tensor([[math.inf] + [0.0] * 8], dtype=RDTYPE),
        torch.tensor([[complex(0.0, math.nan)] + [0.0j] * 8], dtype=CDTYPE),
        torch.tensor([[complex(0.0, math.inf)] + [0.0j] * 8], dtype=CDTYPE),
    ],
    ids=["real-nan", "real-inf", "imaginary-nan", "imaginary-inf"],
)
def test_oracle_recover_rejects_nonfinite_endpoints(anchor_oracle, endpoint):
    with pytest.raises(ValueError, match="finite"):
        anchor_oracle.recover_source(endpoint, torch.tensor([0]))


@pytest.mark.parametrize(
    "channel",
    [
        torch.tensor([0.5]),
        torch.tensor([True]),
        torch.tensor([math.nan]),
        torch.tensor([math.inf]),
    ],
    ids=["fractional", "boolean", "nan", "inf"],
)
def test_oracle_sampling_rejects_nonintegral_channel_ids(anchor_oracle, channel):
    with pytest.raises(ValueError, match="finite integers"):
        anchor_oracle.sample_for_channels(
            channel, generator=torch.Generator().manual_seed(11)
        )


@pytest.mark.parametrize(
    "channel",
    [
        torch.tensor([0.5]),
        torch.tensor([True]),
        torch.tensor([math.nan]),
        torch.tensor([math.inf]),
    ],
    ids=["fractional", "boolean", "nan", "inf"],
)
def test_oracle_recover_rejects_nonintegral_channel_ids(anchor_oracle, channel):
    endpoint = torch.zeros(1, 9, dtype=CDTYPE)

    with pytest.raises(ValueError, match="finite integers"):
        anchor_oracle.recover_source(endpoint, channel)
