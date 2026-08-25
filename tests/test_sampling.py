from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import pytest
import torch

from taylorgauss_3x3.config import (
    IID_CHANNEL_DESIGN,
    WEIGHTED_CHANNEL_DESIGN,
)
from taylorgauss_3x3.core import CDTYPE, RDTYPE, ExactIndexedContourOracle
from taylorgauss_3x3.core.hubbard import Hubbard3x3Target
from taylorgauss_3x3.core.observables import (
    EntirePolynomial,
    approved_observables,
    exact_enumeration,
)
from taylorgauss_3x3.sampling import (
    allocate_channels,
    estimate_explicit_contour,
    estimate_rao_blackwell,
)


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def canonical():
    target = Hubbard3x3Target(U=2.0, beta=1.5, kappa=1.0, mu_chem=0.75)
    return target, target.exact_indexed_oracle(), approved_observables()


def test_fixed_seed_freezes_ordered_channel_allocation_hash(canonical):
    """Catches nondeterministic allocation or drift from the frozen counter stream."""

    _, oracle, _ = canonical
    first = allocate_channels(
        oracle,
        sample_count=512,
        seed=202609010001,
        design=IID_CHANNEL_DESIGN,
    )
    repeated = allocate_channels(
        oracle,
        sample_count=512,
        seed=202609010001,
        design=IID_CHANNEL_DESIGN,
    )
    different = allocate_channels(
        oracle,
        sample_count=512,
        seed=202609010003,
        design=IID_CHANNEL_DESIGN,
    )

    assert first.channel[:8].tolist() == [
        10110,
        9850,
        9869,
        10570,
        15943,
        17140,
        13000,
        9931,
    ]
    assert (
        first.allocation_sha256
        == "f786741d13a5f662a972374116082a18f2a70d2c139b3156a6ef378645151a8e"
    )
    assert repeated.allocation_sha256 == first.allocation_sha256
    assert (
        different.allocation_sha256
        == "dff19297687a3d9a309b4ea72e77b24790ce8878f121a6dd160f90d04ad79fa6"
    )
    assert different.allocation_sha256 != first.allocation_sha256


def test_paired_estimators_share_channels_and_only_explicit_draws_sources(canonical):
    """Catches lane-local reallocation or fabricated Rao--Blackwell records."""

    target, oracle, observables = canonical
    allocation = allocate_channels(
        oracle,
        sample_count=64,
        seed=202609010001,
        design=WEIGHTED_CHANNEL_DESIGN,
    )
    explicit = estimate_explicit_contour(
        oracle, observables["mixed_linear_quadratic"], allocation
    )
    rb = estimate_rao_blackwell(
        oracle, observables["mixed_linear_quadratic"], allocation
    )

    assert explicit.allocation is allocation
    assert rb.allocation is allocation
    assert torch.equal(explicit.channel, rb.channel)
    assert explicit.source is not None
    assert explicit.endpoint is not None
    assert rb.source is None
    assert rb.endpoint is None
    assert not hasattr(allocation, "source")
    assert not hasattr(allocation, "endpoint")
    expected_endpoint = explicit.source.to(CDTYPE) + (
        1j
        * target.U
        * target.beta
        * oracle.modes[allocation.channel].to(CDTYPE)
    )
    torch.testing.assert_close(
        explicit.endpoint, expected_endpoint, rtol=0.0, atol=0.0
    )
    assert (
        _tensor_sha256(explicit.source)
        == "e6175f280d950c325bf94697facd3c297469e691696dcd58bf624c17aeea0fee"
    )
    assert (
        _tensor_sha256(explicit.endpoint)
        == "4c2b54434938b429d0a6bc05db7aede2f686adbe45d2fa98dc93fa14e78eaa90"
    )


@pytest.mark.parametrize(
    "estimator",
    [estimate_explicit_contour, estimate_rao_blackwell],
)
def test_component_standard_errors_are_observable_level_finite_and_nonnegative(
    canonical, estimator
):
    """Catches amplitude/ESS proxies substituted for observable component SEs."""

    _, oracle, _ = canonical
    allocation = allocate_channels(
        oracle,
        sample_count=512,
        seed=202609010002,
        design=WEIGHTED_CHANNEL_DESIGN,
    )
    base = estimator(oracle, EntirePolynomial(constant=2.0), allocation)
    scaled = estimator(oracle, EntirePolynomial(constant=-7.0), allocation)

    for estimate in (base, scaled):
        assert math.isfinite(estimate.standard_error_real)
        assert math.isfinite(estimate.standard_error_imag)
        assert estimate.standard_error_real >= 0.0
        assert estimate.standard_error_imag >= 0.0
    assert base.standard_error_real > 0.0
    assert scaled.standard_error_real == pytest.approx(
        3.5 * base.standard_error_real, rel=2e-13, abs=2e-13
    )
    assert base.standard_error_imag == 0.0
    assert scaled.standard_error_imag == 0.0


def _underflow_oracle() -> ExactIndexedContourOracle:
    log_magnitudes = torch.tensor([-1000.0, 0.0, math.log(3.0)], dtype=RDTYPE)
    return ExactIndexedContourOracle(
        modes=torch.tensor([[-1], [0], [1]], dtype=torch.int64),
        channel_masses=torch.exp(log_magnitudes).to(CDTYPE),
        channel_log_magnitudes=log_magnitudes,
        channel_phase=torch.ones(3, dtype=CDTYPE),
        precision=1.0,
    )


def test_iid_categorical_fails_closed_when_finite_log_probability_underflows():
    """Catches silently truncated IID support being reported as exact sampling."""

    oracle = _underflow_oracle()
    assert bool((oracle.channel_probabilities == 0.0).any())

    with pytest.raises(ValueError, match="below float64 resolution"):
        allocate_channels(
            oracle,
            sample_count=64,
            seed=91,
            design=IID_CHANNEL_DESIGN,
        )


def test_defensive_proposal_has_complete_support_and_exact_log_correction():
    """Catches missing uniform defense or a linear-space p/q correction."""

    oracle = _underflow_oracle()
    allocation = allocate_channels(
        oracle,
        sample_count=20_000,
        seed=91,
        design=WEIGHTED_CHANNEL_DESIGN,
    )
    represented = oracle.channel_probabilities / oracle.channel_probabilities.sum()
    expected_proposal = 0.5 * represented + 0.5 / oracle.mode_count
    expected_log_weight = oracle.channel_log_probabilities[allocation.channel] - torch.log(
        expected_proposal[allocation.channel]
    )

    assert allocation.proposal_has_complete_support is True
    assert bool((allocation.proposal_probabilities > 0.0).all())
    torch.testing.assert_close(
        allocation.proposal_probabilities,
        expected_proposal,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        allocation.log_design_weight, expected_log_weight, rtol=0.0, atol=0.0
    )
    assert bool(torch.isfinite(allocation.log_design_weight).all())
    assert allocation.iid_categorical_used is False
    assert allocation.underflowed_iid_categorical_refused is True
    assert allocation.unrepresentable_channel_count == 1
    assert allocation.unrepresentable_log_probability_mass == pytest.approx(
        float(oracle.channel_log_probabilities[0])
    )

    estimate = estimate_rao_blackwell(
        oracle,
        EntirePolynomial(linear=torch.tensor([1.0], dtype=RDTYPE)),
        allocation,
    )
    assert estimate.value.real == pytest.approx(0.0, abs=1e-15)
    assert estimate.value.imag == pytest.approx(0.75, abs=3.0 * estimate.standard_error_imag)


_RB_WEIGHTED_FIXTURE = {
    202609010011: {
        "quadratic_mean_field": (1.2079351304918835 + 0.0j, 0.030981723133504054, 0.0),
        "mixed_linear_quadratic": (
            1.2079351304918835 - 0.014990778807533817j,
            0.030981723133504054,
            0.02041548934434753,
        ),
        "odd_linear": (0.0 + 0.3110697357505275j, 0.0, 0.007954840161786709),
    },
    202609010012: {
        "quadratic_mean_field": (1.1971793003757627 + 0.0j, 0.030403823830372583, 0.0),
        "mixed_linear_quadratic": (
            1.1971793003757627 - 0.0001301846854827749j,
            0.030403823830372583,
            0.020338926471866536,
        ),
        "odd_linear": (0.0 + 0.31539474309533744j, 0.0, 0.007873929708595467),
    },
    202609010013: {
        "quadratic_mean_field": (1.238244464428187 + 0.0j, 0.0313998300461717, 0.0),
        "mixed_linear_quadratic": (
            1.238244464428187 + 0.008210189441205449j,
            0.0313998300461717,
            0.019985081540120544,
        ),
        "odd_linear": (0.0 + 0.3136508548069849j, 0.0, 0.007834471243276523),
    },
    202609010014: {
        "quadratic_mean_field": (1.2107721042222304 + 0.0j, 0.03069501339918047, 0.0),
        "mixed_linear_quadratic": (
            1.2107721042222304 + 0.014377010452360346j,
            0.03069501339918047,
            0.02073958718288596,
        ),
        "odd_linear": (0.0 + 0.3273101096486488j, 0.0, 0.0079932894458357),
    },
    202609010015: {
        "quadratic_mean_field": (1.1867452537449004 + 0.0j, 0.03052981016682849, 0.0),
        "mixed_linear_quadratic": (
            1.1867452537449004 - 0.0012929386384264926j,
            0.03052981016682849,
            0.020163381359623478,
        ),
        "odd_linear": (0.0 + 0.30845381371974334j, 0.0, 0.007657912588337734),
    },
    202609010016: {
        "quadratic_mean_field": (1.2472098394062732 + 0.0j, 0.03118672596903868, 0.0),
        "mixed_linear_quadratic": (
            1.2472098394062732 + 0.014142067665739163j,
            0.03118672596903868,
            0.020327350107856853,
        ),
        "odd_linear": (0.0 + 0.31451898179740373j, 0.0, 0.007959671668233158),
    },
    202609010017: {
        "quadratic_mean_field": (1.1914527835489306 + 0.0j, 0.0304836032980873, 0.0),
        "mixed_linear_quadratic": (
            1.1914527835489306 + 0.01206413004742729j,
            0.0304836032980873,
            0.020158391087716908,
        ),
        "odd_linear": (0.0 + 0.3158957181282722j, 0.0, 0.007911709495330037),
    },
}


def test_frozen_weighted_rb_replicates_match_exact_authority_within_intervals(
    canonical,
):
    """Catches biased p/q weighting or miscalibrated observable component SEs."""

    _, oracle, observables = canonical
    exact = {
        name: exact_enumeration(oracle, observable)
        for name, observable in observables.items()
    }
    for seed, frozen in _RB_WEIGHTED_FIXTURE.items():
        allocation = allocate_channels(
            oracle,
            sample_count=4096,
            seed=seed,
            design=WEIGHTED_CHANNEL_DESIGN,
        )
        for name, expected in frozen.items():
            estimate = estimate_rao_blackwell(oracle, observables[name], allocation)
            assert estimate.value == pytest.approx(expected[0], rel=2e-13, abs=2e-13)
            assert estimate.standard_error_real == pytest.approx(
                expected[1], rel=2e-13, abs=2e-13
            )
            assert estimate.standard_error_imag == pytest.approx(
                expected[2], rel=2e-13, abs=2e-13
            )
            for component, standard_error in (
                ("real", estimate.standard_error_real),
                ("imag", estimate.standard_error_imag),
            ):
                if standard_error > 0.0:
                    error = abs(getattr(estimate.value - exact[name], component))
                    assert error <= 1.96 * standard_error


def test_estimators_reject_an_allocation_from_a_different_oracle(canonical):
    """Catches accidental reuse of p/q corrections against a different exact law."""

    _, oracle, observables = canonical
    allocation = allocate_channels(
        oracle,
        sample_count=16,
        seed=1,
        design=IID_CHANNEL_DESIGN,
    )
    other = replace(
        oracle,
        channel_log_magnitudes=oracle.channel_log_magnitudes + torch.linspace(
            0.0, 0.1, oracle.mode_count, dtype=RDTYPE
        ),
        channel_masses=torch.exp(
            oracle.channel_log_magnitudes
            + torch.linspace(0.0, 0.1, oracle.mode_count, dtype=RDTYPE)
        ).to(CDTYPE),
    )

    with pytest.raises(ValueError, match="allocation.*oracle"):
        estimate_rao_blackwell(
            other, observables["mixed_linear_quadratic"], allocation
        )
