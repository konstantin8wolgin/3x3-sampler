from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
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
        9923,
        12352,
        9842,
        9844,
        9436,
        11977,
        9841,
        9841,
    ]
    assert (
        first.allocation_sha256
        == "95b65be5dd5b47f5507d6f8601aca7bf83d394cdfa8efc01948aeb08b312f262"
    )
    assert repeated.allocation_sha256 == first.allocation_sha256
    assert first.rng_algorithm_version == "tg3x3-record-indexed-exact-rational-v2"
    assert bool((first.log_design_weight != 0.0).all())
    assert (
        different.allocation_sha256
        == "031f847802d7d316e170507c21642cbb0f206e422bf51d414e40f5ad792bf89b"
    )
    assert different.allocation_sha256 != first.allocation_sha256


@pytest.mark.parametrize("design", [IID_CHANNEL_DESIGN, WEIGHTED_CHANNEL_DESIGN])
def test_stored_rational_proposal_is_the_exact_attainable_law(canonical, design):
    """Catches a finite RNG lattice whose bin counts differ from stored q."""

    _, oracle, _ = canonical
    allocation = allocate_channels(
        oracle,
        sample_count=32,
        seed=202609010004,
        design=design,
    )
    represented = [
        Fraction.from_float(float(value)) for value in oracle.channel_probabilities
    ]
    represented_total = sum(represented)
    represented = [value / represented_total for value in represented]
    if design == IID_CHANNEL_DESIGN:
        expected = represented
    else:
        uniform = Fraction(1, oracle.mode_count)
        expected = [(value + uniform) / 2 for value in represented]
    actual = [
        Fraction(numerator, allocation.proposal_denominator)
        for numerator in allocation.proposal_numerators
    ]

    assert actual == expected
    assert sum(allocation.proposal_numerators) == allocation.proposal_denominator
    assert all(numerator > 0 for numerator in allocation.proposal_numerators)
    assert allocation.proposal_has_complete_support is True


def test_integer_categorical_partition_has_exact_bin_counts():
    """Catches boundary comparisons inconsistent with the declared integer masses."""

    from taylorgauss_3x3.sampling import _channel_from_uniform_integer

    cumulative = (1, 4, 8)
    selected = [
        _channel_from_uniform_integer(cumulative, draw) for draw in range(8)
    ]

    assert [selected.count(channel) for channel in range(3)] == [1, 3, 4]


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
        == "bcf6dce4b444cee4b64b17c7f561b36e6e0a693607a8483a331174a61eb04a7c"
    )
    assert (
        _tensor_sha256(explicit.endpoint)
        == "dec236821045ab60fce96fc8b68b120f751179d44a97b03805cd3d6c831ad457"
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


def _two_channel_oracle() -> ExactIndexedContourOracle:
    log_magnitudes = torch.tensor([0.0, 0.0], dtype=RDTYPE)
    return ExactIndexedContourOracle(
        modes=torch.tensor([[0], [1]], dtype=torch.int64),
        channel_masses=torch.ones(2, dtype=CDTYPE),
        channel_log_magnitudes=log_magnitudes,
        channel_phase=torch.ones(2, dtype=CDTYPE),
        precision=1.0,
    )


def _ordered_two_channel_allocation(oracle: ExactIndexedContourOracle):
    allocation = allocate_channels(
        oracle,
        sample_count=2,
        seed=7,
        design=IID_CHANNEL_DESIGN,
    )
    channel = torch.tensor([0, 1], dtype=torch.int64)
    return replace(
        allocation,
        channel=channel,
        log_design_weight=(
            oracle.channel_log_probabilities[channel]
            - allocation.proposal_log_probabilities[channel]
        ),
    )


@pytest.mark.parametrize("epsilon", [1e-8, 1e-9])
def test_nearly_constant_nonzero_variance_has_reference_standard_error(epsilon):
    """Catches cancellation of tiny variance around a large nonzero mean."""

    oracle = _two_channel_oracle()
    allocation = _ordered_two_channel_allocation(oracle)
    observable = EntirePolynomial(
        constant=1.0,
        quadratic=torch.tensor([[epsilon]], dtype=RDTYPE),
    )
    estimate = estimate_rao_blackwell(oracle, observable, allocation)
    high = float(1.0 + epsilon)
    expected_mean = math.fsum([high, 1.0]) / 2.0
    expected_standard_error = abs(high - 1.0) / 2.0

    assert estimate.value.real == pytest.approx(expected_mean, abs=1e-16)
    assert estimate.standard_error_real == pytest.approx(
        expected_standard_error, rel=2e-8, abs=1e-18
    )
    assert estimate.standard_error_real > 0.0


def test_near_cancelling_signed_mean_matches_compensated_reference():
    """Catches loss of a small signed mean when large components cancel."""

    oracle = _two_channel_oracle()
    allocation = _ordered_two_channel_allocation(oracle)
    epsilon = 1e-12
    low = float(-1.0 + epsilon)
    observable = EntirePolynomial(
        constant=low,
        quadratic=torch.tensor([[1.0 - low]], dtype=RDTYPE),
    )
    estimate = estimate_rao_blackwell(oracle, observable, allocation)
    expected_mean = math.fsum([1.0, low]) / 2.0
    expected_standard_error = abs(1.0 - low) / 2.0

    assert estimate.value.real == pytest.approx(expected_mean, abs=1e-16)
    assert estimate.standard_error_real == pytest.approx(
        expected_standard_error, rel=2e-15, abs=2e-15
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
    expected_log_weight = (
        oracle.channel_log_probabilities[allocation.channel]
        - torch.log(allocation.proposal_probabilities[allocation.channel])
    )

    assert allocation.proposal_has_complete_support is True
    assert bool((allocation.proposal_probabilities > 0.0).all())
    torch.testing.assert_close(
        allocation.proposal_probabilities,
        expected_proposal,
        rtol=3e-16,
        atol=0.0,
    )
    torch.testing.assert_close(
        allocation.log_design_weight, expected_log_weight, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        allocation.proposal_log_probabilities,
        torch.log(allocation.proposal_probabilities),
        rtol=0.0,
        atol=0.0,
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


@pytest.mark.parametrize(
    "mutation",
    [
        "rational_numerators",
        "rational_denominator",
        "float_proposal",
        "log_proposal_and_selected_weights",
        "declared_design",
    ],
)
def test_estimator_rejects_tampered_allocation_proposal_state(mutation):
    """Catches public allocations whose mutually consistent-looking q fields drift."""

    oracle = _underflow_oracle()
    allocation = allocate_channels(
        oracle,
        sample_count=16,
        seed=92,
        design=WEIGHTED_CHANNEL_DESIGN,
    )
    if mutation == "rational_numerators":
        numerators = list(allocation.proposal_numerators)
        numerators[0] += 1
        numerators[-1] -= 1
        tampered = replace(allocation, proposal_numerators=tuple(numerators))
    elif mutation == "rational_denominator":
        tampered = replace(
            allocation,
            proposal_numerators=tuple(
                2 * numerator for numerator in allocation.proposal_numerators
            ),
            proposal_denominator=2 * allocation.proposal_denominator,
        )
    elif mutation == "float_proposal":
        proposal = allocation.proposal_probabilities.clone()
        proposal[0] = torch.nextafter(proposal[0], torch.ones_like(proposal[0]))
        tampered = replace(allocation, proposal_probabilities=proposal)
    elif mutation == "log_proposal_and_selected_weights":
        log_proposal = allocation.proposal_log_probabilities + math.log(2.0)
        tampered = replace(
            allocation,
            proposal_log_probabilities=log_proposal,
            log_design_weight=(
                oracle.channel_log_probabilities[allocation.channel]
                - log_proposal[allocation.channel]
            ),
        )
    else:
        tampered = replace(allocation, design=IID_CHANNEL_DESIGN)

    with pytest.raises(ValueError):
        estimate_rao_blackwell(
            oracle,
            EntirePolynomial(constant=1.0),
            tampered,
        )


_RB_WEIGHTED_FIXTURE = {
    202609010011: {
        "quadratic_mean_field": (1.2217453687292403 + 0.0j, 0.03100371044410752, 0.0),
        "mixed_linear_quadratic": (
            1.2217453687292403 - 0.009518478597796266j,
            0.03100371044410752,
            0.020122927641520648,
        ),
        "odd_linear": (0.0 + 0.30663851999496095j, 0.0, 0.007714149645346787),
    },
    202609010012: {
        "quadratic_mean_field": (1.2548734231996856 + 0.0j, 0.031149377470451933, 0.0),
        "mixed_linear_quadratic": (
            1.2548734231996856 + 0.009545793119380848j,
            0.031149377470451933,
            0.01995790357137028,
        ),
        "odd_linear": (0.0 + 0.321236521717924j, 0.0, 0.007867374161635708),
    },
    202609010013: {
        "quadratic_mean_field": (1.255780231201275 + 0.0j, 0.03137229293684074, 0.0),
        "mixed_linear_quadratic": (
            1.255780231201275 + 0.0031260748409568436j,
            0.03137229293684074,
            0.020028607870254785,
        ),
        "odd_linear": (0.0 + 0.3224600152344152j, 0.0, 0.007840288563794482),
    },
    202609010014: {
        "quadratic_mean_field": (1.2067114423327814 + 0.0j, 0.03099089419844108, 0.0),
        "mixed_linear_quadratic": (
            1.2067114423327814 + 0.014954299214067865j,
            0.03099089419844108,
            0.02003621988371334,
        ),
        "odd_linear": (0.0 + 0.3249133282965423j, 0.0, 0.007894190724252204),
    },
    202609010015: {
        "quadratic_mean_field": (1.265752008979593 + 0.0j, 0.031131673175891536, 0.0),
        "mixed_linear_quadratic": (
            1.265752008979593 + 0.008001007966639767j,
            0.031131673175891536,
            0.020635027494584998,
        ),
        "odd_linear": (0.0 + 0.3176082882452296j, 0.0, 0.00801588385509753),
    },
    202609010016: {
        "quadratic_mean_field": (1.264344259981549 + 0.0j, 0.03143607470720236, 0.0),
        "mixed_linear_quadratic": (
            1.264344259981549 + 0.015617859938299328j,
            0.03143607470720236,
            0.02026447549531093,
        ),
        "odd_linear": (0.0 + 0.31938984410888305j, 0.0, 0.007822018868681942),
    },
    202609010017: {
        "quadratic_mean_field": (1.1590669989428668 + 0.0j, 0.03034988779152547, 0.0),
        "mixed_linear_quadratic": (
            1.1590669989428668 - 0.012311776192460214j,
            0.03034988779152547,
            0.020046115771196527,
        ),
        "odd_linear": (0.0 + 0.32832379149401353j, 0.0, 0.00800435228395894),
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
    standardized_errors: list[float] = []
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
                    standardized_errors.append(error / standard_error)

    assert sum(error <= 1.96 for error in standardized_errors) >= math.ceil(
        0.85 * len(standardized_errors)
    )
    assert math.sqrt(
        math.fsum(error**2 for error in standardized_errors)
        / len(standardized_errors)
    ) < 1.5
    assert max(standardized_errors) < 3.0


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
