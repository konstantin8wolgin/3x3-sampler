# Sampling data

Every command and artifact in this document refers only to the periodic `3x3`,
`n_t=1` Hubbard auxiliary-field target with exactly 19,683 ordered channels.

## Immutable run directory

`tg-3x3 sample --output PATH` requires `PATH` not to exist. A completed path is
never overwritten or appended to. It contains:

- `run.json`: parameters, method, observable, authority, and run identity;
- `summary.json`: exact-support metadata and the returned estimate;
- `estimates.jsonl`: exact values or chunked stochastic sufficient statistics;
- `hashes.json`: an ordered SHA-256 manifest for completed content;
- `report.html` and `figures/estimates.svg`: deterministic stored reporting;
- `samples.zarr`: present only when explicit `exact-contour` endpoints are
  persisted.

The validator checks the fixed scientific scope, layout, hashes, stored
statistics, and exact-law identities. Run it with:

```bash
tg-3x3 validate PATH
```

## Estimate fields

`summary.json` has an `exact_support` object whose `count` is `19683` and whose
`ordered` flag is `true`. Its `estimate` object contains:

```json
{
  "standard_error_imag": null,
  "standard_error_real": null,
  "value": {"imag": 0.0, "real": 1.0}
}
```

`value.real` and `value.imag` are the observable's real and imaginary
components. The example's numbers are illustrative. Exact enumeration returns
`null` standard errors because all channels and Gaussian moments are
integrated. Stochastic runs of the polynomial observables return finite
nonnegative componentwise Monte Carlo standard errors.

`physical_log_partition` is a structural observable: `value.real` is the
analytically determined physical log partition, `value.imag` is zero, and both
standard-error fields are `null`, including under a stochastic method. That
value is independent of sampled channels, Gaussian sources, and sample count;
requested explicit-contour endpoint samples may still be persisted.

## Channel allocation and endpoint storage

Both stochastic methods allocate the requested number of independent channel
draws using either `iid_exact_categorical` or
`defensive_half_uniform_importance`. The stored `sample_count` is the number of
draws, not the number of channels. `exact-contour-rb` integrates sources and
endpoints analytically. `exact-contour` draws them explicitly and persists
them unless `--no-persist-endpoints` is supplied.

## Offline operations

The offline commands first validate a completed source. They do not sample:

```bash
tg-3x3 estimate runs/source --output runs/estimate-copy
tg-3x3 report runs/source --output runs/report-copy
```

Each output must be a new path outside the source directory. The derivative is
itself an immutable validated artifact with lineage metadata binding it to the
source run.
