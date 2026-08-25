# Mathematical contract

## Frozen scope

This package supports one model only: the periodic real-symmetric `3x3`
nearest-neighbor Hubbard lattice with one Euclidean-time slice (`n_t = 1`).
Its auxiliary field has dimension nine. The exact indexed representation has
exactly `3^9 = 19,683` lexicographically ordered modes
`n in {-1, 0, 1}^9`; these are ordered channels, not symmetry-reduced classes.

For `U > 0` and `beta > 0`, the shared Gaussian precision is

```text
A0 = 1 / (U beta).
```

Each mode uses the imaginary translation

```text
z = x + i U beta n,
```

where `x` is a real centered Gaussian with precision `A0`. This is a
translation, so its Jacobian is exactly one.

The positivity claim is deliberately narrow. It applies to the fixed
one-slice construction when the hopping matrix is real symmetric and the two
hopping exponentials are symmetric positive definite. Their principal minors
are then positive, giving strictly positive coefficients and unit channel
phase. The package does not claim this positivity for nonsymmetric hopping,
multiple time slices, other geometries, or arbitrary matrix products.

## Approved observables

Contour evaluation is authorized only for the following entire polynomials of
degree at most two:

```text
O(z) = c + l^T z + z^T Q z.
```

At dimension `d`, the frozen suite is:

- `quadratic_mean_field`: `c = 0`, `l = 0`, `Q = I / d`;
- `mixed_linear_quadratic`: `c = 0`,
  `l = linspace(-0.4, 0.4, d)`, `Q = I / d`;
- `odd_linear`: `c = 0`, `l = 1 / d`, `Q = 0`.

No callback, inverse-matrix, meromorphic, or otherwise unclassified observable
is approved. Such an observable requires a separate holomorphy and contour-tail
argument before it can be added.

For these polynomials, each channel's real Gaussian is integrated
analytically. Deterministic exact enumeration then averages all 19,683 channel
moments with their log-space normalized masses and phases. The physical
normalization reported by this package is

```text
log Z_physical = logsumexp(channel_log_magnitudes)
                 + beta * mu_chem * 9.
```

## Canonical numerical anchor

For `U = 2.0`, `beta = 1.5`, `kappa = 1.0`, and `mu_chem = 0.75`, the frozen
float64/complex128 values are:

| Quantity | Real | Imaginary |
|---|---:|---:|
| `quadratic_mean_field` | `1.2384040693411869` | `0.0` |
| `mixed_linear_quadratic` | `1.2384040693411869` | `-2.0508704419898993e-15` |
| `odd_linear` | `0.0` | `0.3228805203648307` |
| `log Z_physical` | `32.36358464798972` | — |

The machine-readable authority is
`tests/fixtures/canonical-anchor.json`. Public tests require parity with every
component at `rtol = 2e-12` and `atol = 2e-12`.

## Private provenance

The extraction source is private commit
`b7f285d047b2f0c46403106d94ce7f121a406f83`. Its canonical ancestor is the
merge base with `codex/canonical-baseline`,
`013bb6293654180d949e75ac85a7c6e1fb419c86` (`docs: design fermion sampling
products`).

The three frozen private mathematical source blobs have these SHA-256 values.
The values are computed from `git show <source-commit>:<path>`, so they identify
the committed blobs rather than a mutable checkout:

| Private source path | SHA-256 |
|---|---|
| `ccc/taylorgauss/atoms.py` | `5371105f2a60f7a0e5da3a67b00fed310a77599b88ae5f93480ef6502616c1f0` |
| `ccc/taylorgauss/targets.py` | `b488bf959642d709f7403de5aeff7a288a5b685afc9032ab269e48853c52e9fd` |
| `ccc/taylorgauss/indexed_estimators.py` | `d6259fb89c91cb5f4ed2cb07006833c5d3057bb1cdff9202f4db2360fb1e190d` |

The approved coefficient tensors are declared locally in this public package;
the runtime and tests never import the private checkout, campaign modules, or
validation modules.
