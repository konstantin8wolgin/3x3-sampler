# Limitations

This package deliberately supports only:

- periodic `3x3` geometry;
- one Euclidean-time slice (`n_t=1`), where `n_t=1` is Euclidean imaginary
  time, not real-time evolution;
- the three exact/contour methods exposed by the CLI;
- the four named observables in the README.

The 19,683 channels are the complete exact support, not a sample count. A
stochastic run with `--samples 20000` still contains 20,000 draws, and channel
identities may repeat.

The explicit contour method samples complex contour endpoints; they are not
ordinary real-axis samples. Rao–Blackwellization analytically integrates the
Gaussian source; the Rao–Blackwellized method therefore samples channels only.

The `physical_log_partition` observable (the physical log partition) is a
structural analytic result even when requested with a sample method, so its
standard error is `null`.
Polynomial estimates have componentwise Monte Carlo standard errors; those
errors do not cover model mismatch or Trotter error.

This package makes no multi-slice, 4x4, 3D, Green-function, or production-SMC
claim. It also does not implement real-time dynamics or arbitrary observables.
