# Mathematical contract

## Target

The package implements one fixed target: a periodic real-symmetric `3x3`
nearest-neighbor Hubbard lattice with one Euclidean-time slice. The auxiliary
field has dimension nine.

Let `H` be the 9-by-9 hopping matrix and define

```text
A  = 1 / (U beta)
B+ = exp(kappa beta H) exp(mu_chem beta)
B- = exp(-kappa beta H) exp(-mu_chem beta)
```

The real-axis weight is

```text
W(phi) = exp(-A ||phi||^2 / 2)
         det(I + B+ diag(exp(i phi)))
         det(I + B- diag(exp(-i phi)))
```

For a complex field, `exp(-i phi)` is used literally; no complex conjugation
is applied.

## Exact channel expansion

For any matrix `B`,

```text
det(I + B diag(z)) = sum_S det(B[S,S]) prod(i in S, z_i)
```

The two determinants therefore produce exactly one channel for every

```text
n in {-1, 0, 1}^9
```

For `U > 0` and `beta > 0`, `B+` and `B-` are symmetric positive definite.
Their principal minors are positive, so every channel coefficient is positive.
Completing the square gives

```text
exp(-A ||phi||^2 / 2) exp(i n dot phi)
  = exp(-||n||^2 / (2 A))
    exp(-A ||phi - i n/A||^2 / 2)
```

so the contour center is `i U beta n` and the translation has unit Jacobian.

## Observables

The approved observables have the form

```text
O(z) = c + l^T z + z^T Q z
```

Their Gaussian conditional moments are integrated analytically. This is why
the Rao–Blackwellized method only samples channels.

Arbitrary callbacks, inverse matrices, Green functions, and meromorphic
observables are not part of this contract.

## Normalization

The reported physical log partition is

```text
log Z = logsumexp(channel_log_masses) + beta * mu_chem * 9
```

This uses the normalized Gaussian convention and the particle–hole chemical
potential convention of the target. It is a one-slice, exponential-discretized
quantity, not the exact continuum-time Hubbard partition function.

For the default parameters, the exact reference values are:

| Quantity | Real | Imaginary |
|---|---:|---:|
| `quadratic_mean_field` | `1.2384040693411869` | `0.0` |
| `mixed_linear_quadratic` | `1.2384040693411869` | `-2.0508704419898993e-15` |
| `odd_linear` | `0.0` | `0.3228805203648307` |
| `log Z` | `32.36358464798972` | — |
