# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Load package from source (compiles C++)
Rscript -e "pkgload::load_all('.')"

# Run all tests
Rscript -e "pkgload::load_all('.'); testthat::test_dir('tests/testthat')"

# Regenerate C++ exports after modifying src/*.cpp
Rscript -e "Rcpp::compileAttributes('.')"

# R CMD check
R CMD build . && R CMD check hIRT_*.tar.gz

# Benchmark on production nationalism data (if bench_data_nationalism.rds exists)
Rscript bench_compare.R
```

## Architecture

hIRT implements hierarchical IRT models where latent trait mean and variance depend on covariates. The EM algorithm has three accelerated phases in C++.

### Model Family

| Model | Data | Item Params | Use Case |
|-------|------|-------------|----------|
| `hltm` | Binary | Estimated | Primary 2PL model |
| `hgrm` | Ordinal | Estimated | Graded response model |
| `hltm2` / `hgrm2` | Binary/Ordinal | Fixed (user-supplied) | Two-step: train items on one sample, apply to another |
| `hgrmDIF` | Ordinal | With covariate effects | Differential item functioning |

### EM Flow (hltm.R)

```
Init → em_step_fn closure × SQUAREM/plain → Final E-step → Inference (C++)
```

The full EM cycle (E-step → M-step → VarReg → Constraints) is wrapped in `em_step_fn`, a closure over packed parameters `θ = c(alpha, beta, gamma, lambda)`. Two acceleration modes:

- **`acceleration = "squarem"`** (default): SQUAREM S3 (Varadhan & Roland 2008). Each cycle: 3 F-calls (two consecutive EM steps + one stabilization). Step size grows adaptively on success, shrinks on failure. Monotonicity enforced via LL check with fallback to plain EM step. Reduces iteration count 2-4x on slow-converging datasets.
- **`acceleration = "none"`**: Plain EM via same `em_step_fn`. `max_iter` counts F-calls in both modes.

- **E-step** (`src/estep_ltm.cpp: compute_estep_ltm_cpp`): Sparse CSR log-likelihood with precomputed J×K utility tables, log-sum-exp normalization. Returns posterior weights w(K×N), EAP, VAP.
- **M-step** (`src/estep_ltm.cpp: compute_mstep_ltm_cpp`): Sparse sufficient statistics → per-item 2×2 Newton-Raphson. Supports Gaussian (`prior_type=1`) or lognormal (`prior_type=0`) prior on beta.
- **Inference** (`src/inference_ltm.cpp: compute_inference_ltm_cpp`): Streaming OPG information matrix via rank-1 updates. O(J*K + d²) memory — no N-scaled intermediates. Eigen LLT with LU fallback.

### Sparse Y Representation

Response matrices (often 60-80% NA) are stored in CSR format (`build_sparse_y` in `R/utils_ltm.R`). All C++ functions take `(row_ptr, col_idx, values)` triplets. Only observed entries contribute to computation.

### Identification Constraints

Two constraint modes applied at end of each EM iteration:
- `"latent_scale"` (default): mean(theta)=0, geom_mean(prior_var)=1
- `"items"`: mean(alpha/beta)=0, geom_mean(|beta|)=1 — preserves natural scale for cross-dimension comparison

### Key Dependencies

- `Rcpp` + `RcppEigen`: C++ acceleration (LinkingTo, requires C++17)
- `ltm`: Provides `ltm()` and `grm()` for `init="irt"` starting values
- `pryr`: `compose`/`partial` used in hgrm/hltm2/hgrm2 inference blocks (not in hltm — already C++)
- `Matrix::bdiag`: Block diagonal matrices for items-constraint reparameterization

### Numerical Safeguards

- `softplus_()`: Piecewise log(1+exp(x)) — 4 branches for full double range
- `sigmoid_()`: Overflow-safe logistic via exp(min/max) trick
- Gaussian prior on beta (default σ=5): prevents Heywood cases without the -1/b Jacobian singularity of lognormal
- NR step capped at ±1.0 in M-step to prevent overshooting

## Tests

All tests in `tests/testthat/test-correctness.R`. Key test groups:
- Full pipeline validation on `nes_econ2008` (N=2268, J=10)
- C++ E-step/M-step numerical equivalence against R reference implementations
- Missing data handling, `compute_se=FALSE` path, profiling timing structure
- Prior shrinkage behavior (both Gaussian and lognormal)
