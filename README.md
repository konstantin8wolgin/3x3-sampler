# Taylor–Gauss 3x3 Sampler

Exact and exact-law sampling for the periodic `3x3` Hubbard auxiliary-field target with exactly one Euclidean-time slice (`n_t=1`).

## Install from a clean clone

```bash
git clone https://github.com/konstantin8wolgin/3x3-sampler.git
cd 3x3-sampler
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

The distribution name is `taylorgauss-3x3`, the Python import is
`taylorgauss_3x3`, and the installed command is `tg-3x3`.

## Exact quick start

This deterministic calculation analytically integrates the Gaussian sources
and enumerates every ordered channel:

```bash
tg-3x3 sample \
  --method exact-enumeration \
  --output runs/exact
tg-3x3 validate runs/exact
```

The same calculation through Python is:

```python
from taylorgauss_3x3 import ExactRunConfig, run_exact

run_exact(ExactRunConfig(), "runs/exact-python")
```

## Stochastic Rao–Blackwell quick start

The Rao–Blackwell estimator samples channels but analytically integrates each
selected channel's Gaussian source:

```bash
tg-3x3 sample \
  --method exact-contour-rb \
  --samples 20000 \
  --seed 202609010001 \
  --channel-design iid_exact_categorical \
  --output runs/stochastic-rb
```

See [`examples/stochastic_rb_cpu.py`](examples/stochastic_rb_cpu.py) for the
corresponding Python program.

## Explicit contour example

`exact-contour` draws both a channel and its real Gaussian source. Endpoints
are stored by default, so this mode has a larger artifact:

```bash
tg-3x3 sample \
  --method exact-contour \
  --samples 20000 \
  --seed 202609010001 \
  --channel-design defensive_half_uniform_importance \
  --output runs/explicit-contour
```

The only channel designs are `iid_exact_categorical` and
`defensive_half_uniform_importance`.

## Exact channels and stochastic samples are different

The fixed target has exactly `3^9 = 19,683` lexicographically ordered
channels. That count is the complete discrete support and does not change with
the requested workload. A run with `--samples 20000` makes 20,000 stochastic
draws from that 19,683-channel support; draws may repeat channels and need not
visit every channel. It is therefore incorrect to describe 20,000 samples as
20,000 channels or to interpret 19,683 channels as the stochastic sample
count.

## Returned fields

Each run's `summary.json` contains:

- `exact_support.count`: `19683`, the number of ordered channels;
- `exact_support.ordered`: `true`, recording that channel identities use the
  fixed lexicographic order;
- `estimate.value.real` and `estimate.value.imag`: the two components of the
  observable estimate;
- `estimate.standard_error_real` and
  `estimate.standard_error_imag`: componentwise Monte Carlo standard errors
  for stochastic polynomial estimates, or `null` when the value is
  analytically determined;
- `state`, `valid`, and `schema_version`: completion, validation, and schema
  metadata.

`run.json` records the physical parameters, observable, method, authority,
channel design and stochastic sample count where applicable. The remaining
artifact files and offline commands are documented in
[`docs/sampling-data.md`](docs/sampling-data.md).

## Exact observable accuracy and uncertainty

For `exact-enumeration`, all 19,683 ordered channels and the approved
observable's Gaussian moments are integrated; the standard-error fields are
`null` because there is no Monte Carlo uncertainty. Correct physical accuracy
means reproducing the fixed mathematical contract for the requested
parameters and approved observable, including both real and imaginary
components. The canonical default-parameter values and the float64/complex128
comparison tolerance are recorded in
[`docs/mathematical-contract.md`](docs/mathematical-contract.md).

For either stochastic method and one of the three polynomial observables, the
value fields are estimates and the two standard-error fields quantify
finite-sample uncertainty separately for the real and imaginary components. A
single estimate is not required to equal the exact value, and one standard
error is not a deterministic accuracy bound. Physical validation should
compare repeated seeded runs with the exact-enumeration reference and check
componentwise uncertainty calibration; it should not judge accuracy from
channel coverage or sample count alone.

`physical_log_partition` is the structural exception for stochastic runs:
`value.real` is the analytically determined physical log partition,
`value.imag` is zero, and both standard-error fields are `null`. Its reported
value does not depend on sampled channels, Gaussian sources, or sample count;
an explicit-contour run may still persist its requested endpoint samples.

## Limitations and citation

The supported boundary is intentionally fixed: periodic `3x3`, `n_t=1`, the
three methods shown above, exactly 19,683 ordered channels, and the approved
observables. Requests outside that boundary fail instead of changing the
target. Read [`docs/limitations.md`](docs/limitations.md) before interpreting
results.

To cite version `0.1.0`, use [`CITATION.cff`](CITATION.cff). The software is
licensed under the [Apache License 2.0](LICENSE).
