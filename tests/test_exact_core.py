import math

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


def test_square_hopping_is_the_fixed_periodic_3x3_geometry():
    hop = square_3x3_hopping()

    assert hop.dtype == RDTYPE
    assert hop.shape == (9, 9)
    assert torch.equal(hop, hop.T)
    assert torch.equal(torch.diag(hop), torch.zeros(9, dtype=RDTYPE))
    assert torch.equal(hop.sum(dim=1), torch.full((9,), 4.0, dtype=RDTYPE))


def test_exact_oracle_has_complete_positive_ternary_support():
    target = _target()
    oracle = target.exact_indexed_oracle()

    assert oracle.mode_count == 3**9 == 19_683
    assert torch.unique(oracle.modes, dim=0).shape[0] == 19_683
    assert oracle.modes.tolist() == sorted(oracle.modes.tolist())
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
