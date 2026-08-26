"""Run the deterministic exact periodic 3x3, n_t=1 calculation on CPU."""

from pathlib import Path

from taylorgauss_3x3 import ExactRunConfig, run_exact


output = run_exact(ExactRunConfig(), Path("runs/exact-example"))
print(f"completed={output} authority=exact_reference channels=19683 samples=none")
