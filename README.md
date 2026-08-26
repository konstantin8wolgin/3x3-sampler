# 3x3 Hubbard Sampler

Small, self-contained Python code for the periodic `3x3` Hubbard
auxiliary-field target with one Euclidean-time slice.

It provides:

- exact enumeration of all `3^9 = 19,683` Fourier channels;
- exact contour sampling;
- Rao–Blackwellized contour sampling for lower-variance polynomial estimates;
- reproducible JSON summaries and optional contour samples.

This is a one-slice reference implementation: `n_t=1` means Euclidean
imaginary time, not real-time evolution. It is not a multi-slice Hubbard
simulation.

## Install

```bash
git clone https://github.com/konstantin8wolgin/3x3-sampler.git
cd 3x3-sampler
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

The command is `tg-3x3`.

## Run an exact calculation

```bash
tg-3x3 sample \
  --method exact-enumeration \
  --observable mixed_linear_quadratic \
  --output runs/exact

tg-3x3 validate runs/exact
```

Exact enumeration does not draw random samples. It integrates all channels and
the Gaussian source analytically.

## Draw contour samples

For a practical stochastic estimate, use the Rao–Blackwellized method:

```bash
tg-3x3 sample \
  --method exact-contour-rb \
  --samples 20000 \
  --seed 202609010001 \
  --channel-design iid_exact_categorical \
  --output runs/stochastic-rb
```

To store the Gaussian sources and complex contour endpoints, use
`--method exact-contour`. The default channel design is the exact categorical
law. `defensive_half_uniform_importance` is also available when complete
support is needed for very rare channels.

## Parameters and observables

The defaults are `U=2`, `beta=1.5`, `kappa=1`, and `mu-chem=0.75`. Override
them with the corresponding command-line options. The supported observables
are:

- `quadratic_mean_field`;
- `mixed_linear_quadratic`;
- `odd_linear`;
- `physical_log_partition`.

The first three are entire polynomials of the auxiliary field. The
`physical_log_partition` observable (the physical log partition) is computed
analytically, so its reported standard error is `null` even when a sample
method is selected.

## Output

Each run creates a new output directory containing `run.json`, `summary.json`,
`estimates.jsonl`, a small HTML report, and hashes. Explicit contour runs also
store `samples.zarr` unless endpoint persistence is disabled.

`summary.json` contains the complex estimate and separate standard errors for
its real and imaginary components. `exact_support.count` is always 19,683:
it is the complete exact support, not a sample count and not the number of
stochastic draws.

Run validation with:

```bash
tg-3x3 validate runs/stochastic-rb
```

## Mathematical scope

For `A=1/(U beta)`, the target is expanded as positive Gaussian channels with
centers

```text
z = x + i U beta n,   n in {-1, 0, 1}^9.
```

The positivity statement concerns the channel masses in this one-slice
representation. Complex contour endpoints are not ordinary real-axis samples.
Rao–Blackwellization analytically integrates the Gaussian source. The
one-slice discretization is not the exact finite-temperature
Hubbard model; increasing the number of Euclidean-time slices requires a
different implementation. This package makes no multi-slice, 4x4, 3D,
Green-function, or production-SMC claim.

More details are in [`docs/mathematical-contract.md`](docs/mathematical-contract.md),
[`docs/sampling-data.md`](docs/sampling-data.md), and
[`docs/limitations.md`](docs/limitations.md).

## Development

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest
python3 -m build
```

The project uses Python 3.11 or 3.12, NumPy, and PyTorch.
