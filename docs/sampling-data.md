# Sampling data

Each completed run is a new directory containing:

- `run.json`: parameters, method, observable, and seed;
- `summary.json`: the estimate and standard errors;
- `estimates.jsonl`: exact values or stochastic sufficient statistics;
- `report.html` and `figures/estimates.svg`;
- `hashes.json`;
- `samples.zarr` for explicit contour endpoints when persistence is enabled.

Use the CLI to validate a run:

```bash
tg-3x3 validate runs/source
```

The summary always declares the exact channel support separately:

```json
{"exact_support": {"count": 19683, "ordered": true}}
```

That count is not the stochastic sample count. The stochastic count is stored
in `run.json` as `sample_count`.

The exact method reports analytic standard errors as `null`. Stochastic
polynomial estimates report separate real and imaginary component errors.
`physical_log_partition` (the physical log partition) is analytic and
therefore reports `null` errors even when a sample method is selected.

`estimate` and `report` can reproduce derived artifacts from a validated source
run without sampling again:

```bash
tg-3x3 estimate runs/source --output runs/estimate-copy
tg-3x3 report runs/source --output runs/report-copy
```
