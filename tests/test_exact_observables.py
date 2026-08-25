import json
from pathlib import Path

import pytest
import torch

from taylorgauss_3x3.core import CDTYPE, RDTYPE, Hubbard3x3Target
from taylorgauss_3x3.core.observables import (
    EntirePolynomial,
    approved_observables,
    exact_enumeration,
    physical_log_partition,
)


@pytest.fixture(scope="module")
def canonical_anchor():
    path = Path(__file__).parent / "fixtures" / "canonical-anchor.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def anchor_target_and_oracle(canonical_anchor):
    target = Hubbard3x3Target(**canonical_anchor["parameters"])
    return target, target.exact_indexed_oracle()


def test_approved_observables_have_the_frozen_coefficients():
    observables = approved_observables()
    quadratic = torch.eye(9, dtype=RDTYPE) / 9
    linear = torch.linspace(-0.4, 0.4, 9, dtype=RDTYPE)

    assert tuple(observables) == (
        "quadratic_mean_field",
        "mixed_linear_quadratic",
        "odd_linear",
    )
    assert torch.equal(observables["quadratic_mean_field"].quadratic, quadratic)
    assert torch.equal(observables["mixed_linear_quadratic"].linear, linear)
    assert torch.equal(observables["mixed_linear_quadratic"].quadratic, quadratic)
    assert torch.equal(
        observables["odd_linear"].linear, torch.ones(9, dtype=RDTYPE) / 9
    )


@pytest.mark.parametrize("dimension", [0, -1, 2.5, True])
def test_approved_observables_reject_invalid_dimensions(dimension):
    with pytest.raises(ValueError, match="positive integer"):
        approved_observables(dimension)


def test_entire_polynomial_evaluates_constant_linear_and_quadratic_terms():
    observable = EntirePolynomial(
        constant=1.0 - 0.5j,
        linear=torch.tensor([2.0, -1.0], dtype=RDTYPE),
        quadratic=torch.tensor([[0.5, 0.25], [0.25, -1.0]], dtype=RDTYPE),
    )
    points = torch.tensor([[1.0 + 1.0j, 2.0 - 0.5j]], dtype=CDTYPE)

    expected = torch.tensor([-1.5 + 5.75j], dtype=CDTYPE)
    torch.testing.assert_close(observable.evaluate(points), expected)


@pytest.mark.parametrize(
    "observable, message",
    [
        (EntirePolynomial(constant=[1.0]), "constant"),
        (EntirePolynomial(linear=torch.ones(3)), "linear"),
        (EntirePolynomial(quadratic=torch.eye(3)), "quadratic"),
        (EntirePolynomial(linear=torch.tensor([float("nan"), 0.0])), "linear"),
    ],
)
def test_entire_polynomial_rejects_invalid_coefficients(observable, message):
    with pytest.raises(ValueError, match=message):
        observable.evaluate(torch.zeros(1, 2, dtype=RDTYPE))


def test_conditional_gaussian_moments_match_a_hand_derived_example():
    observable = EntirePolynomial(
        constant=0.5,
        linear=torch.tensor([1.0, -2.0], dtype=RDTYPE),
        quadratic=torch.tensor([[2.0, 1.0], [3.0, -1.0]], dtype=RDTYPE),
    )
    means = torch.tensor([[1.0j, -0.5j]], dtype=CDTYPE)

    conditional_mean, conditional_variance = observable.conditional_moments(
        means, precision=2.0
    )

    torch.testing.assert_close(
        conditional_mean, torch.tensor([1.25 + 2.0j], dtype=CDTYPE)
    )
    torch.testing.assert_close(
        conditional_variance, torch.tensor([23.5], dtype=RDTYPE)
    )


def test_canonical_fixture_matches_exact_observable_authority(
    canonical_anchor, anchor_target_and_oracle
):
    target, oracle = anchor_target_and_oracle
    observables = approved_observables(target.d)

    assert oracle.mode_count == canonical_anchor["mode_count"]
    for name in (
        "mixed_linear_quadratic",
        "odd_linear",
        "quadratic_mean_field",
    ):
        actual = exact_enumeration(oracle, observables[name])
        expected = complex(
            canonical_anchor[name]["real"], canonical_anchor[name]["imag"]
        )
        assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)

    assert physical_log_partition(target, oracle) == pytest.approx(
        canonical_anchor["physical_log_partition"], rel=2e-12, abs=2e-12
    )
