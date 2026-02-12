# hIRT Benchmark Log

Performance tracking across code versions and datasets.

---

## 2026-02-12: SQUAREM bug fix + prior restore

**Commit**: (this commit)
**Baseline**: cb20354 (4 EM optimizations)

### Changes

1. **Restore prior default**: `prior_sigma_beta = Inf` → `5` (Gaussian, σ=5). Prevents Heywood cases on high-NA/high-J data without the −1/b Jacobian singularity of the old lognormal prior.
2. **Constrained SQUAREM**: `squarem_active = TRUE` → `FALSE`. SQUAREM now operates on the constrained EM map directly, eliminating the unconstrained→constrained fixed-point mismatch that caused polish-phase oscillation (300+ evals on economic ideology).
3. **LL-based polish exit**: Added `ll_change < 1e-6` early exit to polish phase as safety net.

### Results (init="irt", acceleration="squarem", compute_se=FALSE)

| Dataset | N | J | NA% | Old (cb20354) | New | Speedup |
|---|---|---|---|---|---|---|
| nes_econ2008 | 2,268 | 10 | 18% | ~1.4s | 0.28s | 5.0x |
| Nationalism | 17,839 | 14 | 68% | ~8s | 1.54s | 5.2x |
| Economic Ideology | 149,616 | 84 | 95% | 282s | 14.0s | 20.2x |

Economic ideology, SQUAREM vs plain EM accuracy:
- beta RMSE: 0.007, correlation: 0.999994
- LL difference: 0.075 (negligible)

### Notes

- The 20x speedup on economic ideology is mostly from eliminating polish oscillation (333+ → ~1 eval) and the prior preventing Heywood-case divergence.
- Smaller datasets see ~5x from the cumulative effect of the earlier C++ E/M-step, K=20, and SQUAREM optimizations (cb20354) combined with the polish fix here.

---

## cb20354: Four EM performance optimizations

**Commit**: cb20354
**Baseline**: e7c793f (profiling improvements)

### Changes

1. C++ E-step for `hltm2` (was R-only)
2. Default quadrature K: 25 → 20
3. SQUAREM acceleration (S3 variant)
4. Pattern collapsing for duplicate response vectors

### Results

These optimizations reduced total wall time substantially but introduced the SQUAREM polish-phase oscillation bug on high-J data (fixed in the entry above).

---

## e7c793f: Profiling and SE bypass

**Commit**: e7c793f

### Changes

- Added `profile = TRUE` control option for EM timing breakdown
- Added `compute_se = FALSE` option to skip inference phase

---

## a506cf6: M-step lognormal prior

**Commit**: a506cf6

### Changes

- Added optional lognormal prior on beta to prevent Heywood cases
- E-step now returns log-likelihood for convergence monitoring

---

## d02c438: C++ EM core

**Commit**: d02c438

### Changes

- Rewrote E-step and M-step in C++ via Rcpp
- Sparse CSR representation for response matrices
- Per-item Newton-Raphson in M-step
