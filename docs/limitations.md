# Limitations

Taylor–Gauss 3x3 Sampler version 0.1.0 has one deliberately fixed scientific
boundary: the periodic `3x3` Hubbard auxiliary-field target with `n_t=1` and a
field shape of `(1, 3, 3)`. The Euclidean-time slice count cannot be changed;
any `--n-t` value other than `1` is rejected.

The exact discrete support is always the same 19,683 lexicographically ordered
channels. The only sampling methods are `exact-enumeration`, `exact-contour`,
and `exact-contour-rb`. The only channel designs for the two stochastic
methods are `iid_exact_categorical` and
`defensive_half_uniform_importance`.

Observables are limited to `quadratic_mean_field`,
`mixed_linear_quadratic`, `odd_linear`, and `physical_log_partition`. The
polynomial observables satisfy the holomorphy and contour-tail contract stated
in [`mathematical-contract.md`](mathematical-contract.md); arbitrary callbacks
are not accepted.

An exact-enumeration result has analytic uncertainty fields set to `null`.
Stochastic standard errors cover finite-sample variability of the reported
observable components. They do not cover model mismatch, altered physical
parameters, or an observable outside the approved contract.

Completed output directories are immutable. Sampling, estimate reproduction,
and report reproduction all refuse an existing output path. Use `validate`
before consuming or copying a run.
