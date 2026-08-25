from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from taylorgauss_3x3.actions import FermionAction, build_action
from taylorgauss_3x3.api import describe
from taylorgauss_3x3.config import ExactRunConfig, StochasticRunConfig
from taylorgauss_3x3.core import RDTYPE


def test_describe_distinguishes_support_from_samples():
    row = describe(StochasticRunConfig(samples=20_000, seed=202609010001))

    assert row["exact_channel_count"] == 19_683
    assert row["requested_sample_count"] == 20_000
    assert row["geometry"] == "periodic_3x3"
    assert row["euclidean_time_slices"] == 1
    assert row["authority"] == "exact_law_stochastic_rb"


def test_configurations_are_frozen_and_stochastic_defaults_are_explicit():
    exact = ExactRunConfig()
    stochastic = StochasticRunConfig()

    with pytest.raises(FrozenInstanceError):
        exact.U = 3.0

    assert stochastic.method == "exact-contour-rb"
    assert stochastic.channel_design == "iid_exact_categorical"
    assert stochastic.persist_endpoints is False
    assert stochastic.authority == "exact_law_stochastic_rb"


@pytest.mark.parametrize(
    ("factory", "changes", "message"),
    [
        (ExactRunConfig, {"U": True}, "U must be a finite number"),
        (StochasticRunConfig, {"samples": True}, "at least two samples"),
        (StochasticRunConfig, {"seed": True}, "signed 64-bit"),
        (ExactRunConfig, {"n_t": True}, "n_t=1"),
        (StochasticRunConfig, {"n_t": 2}, "n_t=1"),
        (ExactRunConfig, {"geometry": "open_3x3"}, "periodic 3x3"),
        (StochasticRunConfig, {"geometry": "periodic_4x4"}, "periodic 3x3"),
        (ExactRunConfig, {"U": float("nan")}, "U must be finite"),
        (StochasticRunConfig, {"beta": float("inf")}, "beta must be finite"),
        (StochasticRunConfig, {"samples": 1}, "at least two samples"),
        (StochasticRunConfig, {"seed": None}, "signed 64-bit"),
        (ExactRunConfig, {"observable": "green_function"}, "not approved"),
        (StochasticRunConfig, {"method": "frozen-m9-smc"}, "unsupported"),
        (ExactRunConfig, {"method": "exact-contour"}, "exact-enumeration"),
        (ExactRunConfig, {"samples": 2}, "samples and seed are invalid"),
        (ExactRunConfig, {"seed": 1}, "samples and seed are invalid"),
        (
            StochasticRunConfig,
            {"method": "exact-contour-rb", "persist_endpoints": True},
            "not applicable",
        ),
        (StochasticRunConfig, {"channel_design": "smc"}, "channel design"),
    ],
)
def test_configurations_reject_scope_drift_and_invalid_values(factory, changes, message):
    with pytest.raises(ValueError, match=message):
        factory(**changes)


def test_action_uses_only_the_fixed_core_target():
    action = build_action(ExactRunConfig())
    field = torch.zeros((1, 9), dtype=RDTYPE)

    assert isinstance(action, FermionAction)
    assert action.metadata.field_shape == (1, 3, 3)
    assert action.metadata.euclidean_time_slices == 1
    assert action.exact_indexed_oracle().mode_count == 19_683
    assert action.evaluate(field).shape == (1,)
    assert action.evaluate_holomorphic(field).shape == (1,)
