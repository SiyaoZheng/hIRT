#!/usr/bin/env Rscript
# Benchmark the current hltm() implementation path with staged timings.

`%||%` <- function(a, b) if (!is.null(a)) a else b

# Keep benchmark runs safe by default.
thread_vars <- c(
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
  "BLAS_NUM_THREADS"
)
for (var in thread_vars) {
  if (!nzchar(Sys.getenv(var))) Sys.setenv(setNames("1", var))
}

loaded_source <- FALSE
if (requireNamespace("pkgload", quietly = TRUE) && file.exists("DESCRIPTION")) {
  pkgload::load_all(".", quiet = TRUE)
  loaded_source <- TRUE
} else {
  library(hIRT)
}

if (!"package:hIRT" %in% search()) library(hIRT)

cat("Loaded hIRT version:", as.character(utils::packageVersion("hIRT")), "\n")
cat("Loaded from source:", loaded_source, "\n")
cat("Thread caps:",
    paste(sprintf("%s=%s", thread_vars, Sys.getenv(thread_vars)), collapse = ", "),
    "\n\n")

data(nes_econ2008, package = "hIRT")
y0 <- nes_econ2008[, -(1:3)]
x0 <- model.matrix(~ party * educ, nes_econ2008)
z0 <- model.matrix(~ party, nes_econ2008)

dichotomize <- function(v) findInterval(v, c(mean(v, na.rm = TRUE)))
y0[] <- lapply(y0, dichotomize)

run_case <- function(mult = 1L, init = "naive", compute_se = TRUE,
                     max_iter = 150, eps = 1e-3) {
  y <- do.call(rbind, rep(list(y0), mult))
  x <- do.call(rbind, rep(list(x0), mult))
  z <- do.call(rbind, rep(list(z0), mult))

  elapsed <- system.time({
    model <- hltm(
      y, x, z, init = init, compute_se = compute_se,
      control = list(max_iter = max_iter, eps = eps, profile = TRUE)
    )
  })[["elapsed"]]

  em <- model$timing$em %||% list()
  inf <- model$timing$inference %||% list()
  cat(sprintf("N=%d J=%d init=%s compute_se=%s\n",
              nrow(y), ncol(y), init, compute_se))
  cat(sprintf("  elapsed(total): %.3f sec\n", elapsed))
  cat(sprintf("  init: %.3f sec\n", model$timing$init %||% NA_real_))
  cat(sprintf("  em_total: %.3f sec\n", model$timing$em_total %||% NA_real_))
  cat(sprintf("    estep: %.3f | mstep: %.3f | varreg: %.3f | constr: %.3f\n",
              em$estep %||% NA_real_, em$mstep %||% NA_real_,
              em$varreg %||% NA_real_, em$constr %||% NA_real_))
  cat(sprintf("  inference: loglik %.3f | gradients %.3f | information %.3f | reparam %.3f\n",
              inf$loglik %||% NA_real_, inf$gradients %||% NA_real_,
              inf$information %||% NA_real_, inf$reparam %||% NA_real_))
  cat(sprintf("  logLik: %.3f\n\n", model$log_Lik))

  invisible(model)
}

cat("=== hltm Current-Path Benchmark ===\n\n")
run_case(mult = 1, init = "naive", compute_se = TRUE)
run_case(mult = 2, init = "naive", compute_se = TRUE)
run_case(mult = 4, init = "naive", compute_se = TRUE)
run_case(mult = 4, init = "naive", compute_se = FALSE)
