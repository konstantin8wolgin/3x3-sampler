"""Run a Rao--Blackwell exact-law periodic 3x3, n_t=1 estimate on CPU."""

from pathlib import Path

from taylorgauss_3x3 import StochasticRunConfig, run_rao_blackwell


samples = 20_000
output = run_rao_blackwell(
    StochasticRunConfig(
        method="exact-contour-rb",
        samples=samples,
        seed=202609010001,
    ),
    Path("runs/stochastic-rb-example"),
)
print(
    f"completed={output} authority=exact_law_stochastic_rb "
    f"channels=19683 samples={samples}"
)
