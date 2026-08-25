import taylorgauss_3x3 as tg


def test_public_surface_is_small_and_versioned():
    assert tg.__version__ == "0.1.0"
    assert tg.__all__ == [
        "ExactRunConfig",
        "StochasticRunConfig",
        "describe",
        "estimate",
        "report",
        "run_contour",
        "run_exact",
        "run_rao_blackwell",
        "validate",
    ]
