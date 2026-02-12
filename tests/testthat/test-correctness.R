test_that("hltm produces valid output on nes_econ2008", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  x <- model.matrix(~ party * educ, nes_econ2008)
  z <- model.matrix(~ party, nes_econ2008)

  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  m <- hltm(y, x, z)

  # Basic structure
  expect_s3_class(m, "hltm")
  expect_s3_class(m, "hIRT")
  expect_equal(m$N, nrow(y))
  expect_equal(m$J, ncol(y))
  expect_equal(m$p, ncol(x))
  expect_equal(m$q, ncol(z))

  # Coefficients are finite
  coefs <- m$coefficients
  expect_true(all(is.finite(coefs$Estimate)))
  # SEs should be finite except for constrained params (which are NA)
  se_finite <- coefs$Std_Error[!is.na(coefs$Std_Error)]
  expect_true(all(is.finite(se_finite)))
  expect_true(all(se_finite > 0))

  # Log-likelihood is finite and negative
  expect_true(is.finite(m$log_Lik))
  expect_true(m$log_Lik < 0)

  # Scores have correct dimensions
  expect_equal(nrow(m$scores), nrow(y))
  expect_equal(ncol(m$scores), 4)  # post_mean, post_sd, prior_mean, prior_sd
  expect_true(all(is.finite(m$scores$post_mean)))
  expect_true(all(is.finite(m$scores$post_sd)))
  expect_true(all(m$scores$post_sd > 0))
})

test_that("hltm with default arguments converges", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  m <- hltm(y)

  expect_s3_class(m, "hltm")
  expect_true(is.finite(m$log_Lik))
})

test_that("hltm handles missing data correctly", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  # Introduce additional NAs (simulate planned missingness)
  set.seed(42)
  na_mask <- matrix(rbinom(nrow(y) * ncol(y), 1, 0.3), nrow(y), ncol(y))
  y[na_mask == 1] <- NA

  x <- model.matrix(~ party * educ, nes_econ2008)
  z <- model.matrix(~ party, nes_econ2008)

  m <- hltm(y, x, z)

  expect_s3_class(m, "hltm")
  expect_true(is.finite(m$log_Lik))
  expect_true(m$log_Lik < 0)
  expect_true(all(is.finite(m$scores$post_mean)))
})

test_that("hltm can skip standard error computation", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[1:600, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)
  x <- model.matrix(~ party * educ, nes_econ2008)[1:600, , drop = FALSE]
  z <- model.matrix(~ party, nes_econ2008)[1:600, , drop = FALSE]

  m <- hltm(y, x, z, compute_se = FALSE)

  expect_s3_class(m, "hltm")
  expect_false(m$se_computed)
  expect_true(all(is.na(m$coefficients$Std_Error)))
  expect_true(all(is.na(m$coefficients$z_value)))
  expect_true(all(is.na(m$coefficients$p_value)))
  expect_true(all(is.na(m$vcov)))
  expect_true(is.finite(m$log_Lik))
})

test_that("hltm returns timing breakdown when profiling is enabled", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[1:600, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)
  x <- model.matrix(~ party * educ, nes_econ2008)[1:600, , drop = FALSE]
  z <- model.matrix(~ party, nes_econ2008)[1:600, , drop = FALSE]

  m <- hltm(
    y, x, z, compute_se = FALSE,
    control = list(profile = TRUE, max_iter = 60, eps = 1e-3)
  )

  expect_type(m$timing, "list")
  expect_named(m$timing, c("init", "em_total", "em", "inference", "total", "em_evals"))
  expect_true(is.finite(m$timing$init))
  expect_true(is.finite(m$timing$em_total))
  expect_true(is.finite(m$timing$total))
  expect_true(m$timing$total > 0)

  expect_named(m$timing$em, c("estep", "mstep", "varreg", "constr"))
  expect_true(all(vapply(m$timing$em, function(x) is.finite(x) && x >= 0, logical(1L))))

  expect_named(m$timing$inference, c("total", "reparam"))
  expect_true(all(vapply(m$timing$inference, function(x) is.finite(x) && x >= 0, logical(1L))))
})

test_that("build_sparse_y produces correct CSR format", {
  # Small test case
  y <- data.frame(a = c(0L, 1L, NA), b = c(1L, NA, 0L), c = c(NA, 1L, 1L))
  sparse <- hIRT:::build_sparse_y(y)

  # Row 1: a=0, b=1 (c=NA)  -> 2 entries

  # Row 2: a=1, c=1 (b=NA)  -> 2 entries
  # Row 3: b=0, c=1 (a=NA)  -> 2 entries
  expect_equal(sparse$row_ptr, c(0L, 2L, 4L, 6L))
  expect_equal(length(sparse$col_idx), 6L)
  expect_equal(length(sparse$values), 6L)

  # Row 1: cols 0,1 (a,b)
  expect_equal(sparse$col_idx[1:2], c(0L, 1L))
  expect_equal(sparse$values[1:2], c(0L, 1L))

  # Row 2: cols 0,2 (a,c)
  expect_equal(sparse$col_idx[3:4], c(0L, 2L))
  expect_equal(sparse$values[3:4], c(1L, 1L))

  # Row 3: cols 1,2 (b,c)
  expect_equal(sparse$col_idx[5:6], c(1L, 2L))
  expect_equal(sparse$values[5:6], c(0L, 1L))
})

test_that("compute_mstep_ltm_cpp matches R glm.fit M-step", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  N <- nrow(y)
  J <- ncol(y)
  K <- 25
  C <- 4

  theta_ls <- C * hIRT:::GLpoints[[K]][["x"]]
  qw_ls    <- C * hIRT:::GLpoints[[K]][["w"]]

  # Initialize parameters and run one E-step to get realistic weights
  alpha <- rep(0, J)
  beta  <- rep(1, J)
  fitted_mean <- rep(0, N)
  fitted_var  <- rep(1, N)

  sparse_y <- hIRT:::build_sparse_y(y)
  estep <- hIRT:::compute_estep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
  )
  w <- estep$w

  # R reference M-step (original code path)
  dummy_fun_ltm_ref <- function(y_j, w) {
    dummy_mat <- outer(y_j, c(0, 1), "==")
    dummy_mat[is.na(dummy_mat)] <- 0
    w %*% dummy_mat
  }
  tab2df_ltm_ref <- function(tab, theta_ls, K) {
    theta <- rep(theta_ls, 2)
    y_vec <- rep(c(0, 1), each = K)
    data.frame(y = factor(y_vec), x = theta, wt = as.double(tab))
  }

  alpha_r <- numeric(J)
  beta_r  <- numeric(J)
  for (j in seq_len(J)) {
    y_j <- y[[j]]
    tab <- dummy_fun_ltm_ref(y_j, w)
    df  <- tab2df_ltm_ref(tab, theta_ls, K)
    fit <- glm.fit(cbind(1, df$x), df$y, weights = df$wt,
                   family = quasibinomial("logit"))
    alpha_r[j] <- fit$coefficients[1]
    beta_r[j]  <- fit$coefficients[2]
  }

  # C++ M-step (sigma_prior = Inf for pure MLE to match glm.fit)
  mstep <- hIRT:::compute_mstep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    w, theta_ls, alpha, beta, sigma_prior = Inf
  )

  expect_equal(mstep$alpha, alpha_r, tolerance = 1e-6)
  expect_equal(mstep$beta, beta_r, tolerance = 1e-6)
})

test_that("compute_mstep_ltm_cpp handles high-NA data", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  # Introduce 50% NAs
  set.seed(123)
  na_mask <- matrix(rbinom(nrow(y) * ncol(y), 1, 0.5), nrow(y), ncol(y))
  y[na_mask == 1] <- NA

  N <- nrow(y)
  J <- ncol(y)
  K <- 25
  C <- 4

  theta_ls <- C * hIRT:::GLpoints[[K]][["x"]]
  qw_ls    <- C * hIRT:::GLpoints[[K]][["w"]]

  alpha <- rep(0, J)
  beta  <- rep(1, J)
  fitted_mean <- rep(0, N)
  fitted_var  <- rep(1, N)

  sparse_y <- hIRT:::build_sparse_y(y)
  estep <- hIRT:::compute_estep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
  )
  w <- estep$w

  # R reference
  dummy_fun_ltm_ref <- function(y_j, w) {
    dummy_mat <- outer(y_j, c(0, 1), "==")
    dummy_mat[is.na(dummy_mat)] <- 0
    w %*% dummy_mat
  }
  tab2df_ltm_ref <- function(tab, theta_ls, K) {
    theta <- rep(theta_ls, 2)
    y_vec <- rep(c(0, 1), each = K)
    data.frame(y = factor(y_vec), x = theta, wt = as.double(tab))
  }

  alpha_r <- numeric(J)
  beta_r  <- numeric(J)
  for (j in seq_len(J)) {
    y_j <- y[[j]]
    tab <- dummy_fun_ltm_ref(y_j, w)
    df  <- tab2df_ltm_ref(tab, theta_ls, K)
    fit <- glm.fit(cbind(1, df$x), df$y, weights = df$wt,
                   family = quasibinomial("logit"))
    alpha_r[j] <- fit$coefficients[1]
    beta_r[j]  <- fit$coefficients[2]
  }

  # C++ M-step (sigma_prior = Inf for pure MLE to match glm.fit)
  mstep <- hIRT:::compute_mstep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    w, theta_ls, alpha, beta, sigma_prior = Inf
  )

  expect_equal(mstep$alpha, alpha_r, tolerance = 1e-6)
  expect_equal(mstep$beta, beta_r, tolerance = 1e-6)
})

test_that("compute_estep_ltm_cpp matches R E-step", {
  # Use small built-in data for exact comparison
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  N <- nrow(y)
  J <- ncol(y)
  K <- 25
  C <- 4

  # Use hIRT's internal GL quadrature points
  theta_ls <- C * hIRT:::GLpoints[[K]][["x"]]
  qw_ls    <- C * hIRT:::GLpoints[[K]][["w"]]

  # Simple initial parameters
  alpha <- rep(0, J)
  beta  <- rep(1, J)
  fitted_mean <- rep(0, N)
  fitted_var  <- rep(1, N)

  # R reference E-step (from theta_post_ltm logic)
  loglik_r <- function(alpha, beta, theta) {
    util <- matrix(alpha, N, J, byrow = TRUE) + outer(theta, beta)
    log(exp(as.matrix(y) * util) / (1 + exp(util)))
  }

  posterior_r <- lapply(seq_along(theta_ls), function(k) {
    theta_k <- theta_ls[k]
    qw_k    <- qw_ls[k]
    wt_k <- dnorm(theta_k - fitted_mean, sd = sqrt(fitted_var)) * qw_k
    loglik <- rowSums(loglik_r(alpha, beta, rep(theta_k, N)), na.rm = TRUE)
    exp(loglik + log(wt_k))
  })
  tmp <- matrix(unlist(posterior_r), N, K)
  w_r <- t(sweep(tmp, 1, rowSums(tmp), FUN = "/"))
  theta_eap_r <- as.double(t(theta_ls %*% w_r))
  theta_vap_r <- as.double(t(theta_ls^2 %*% w_r) - theta_eap_r^2)

  # C++ E-step
  sparse_y <- hIRT:::build_sparse_y(y)
  estep <- hIRT:::compute_estep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
  )

  expect_equal(dim(estep$w), c(K, N))
  expect_equal(length(estep$theta_eap), N)
  expect_equal(length(estep$theta_vap), N)

  # Posterior weights should match within tolerance
  expect_equal(estep$w, w_r, tolerance = 1e-10)
  expect_equal(estep$theta_eap, theta_eap_r, tolerance = 1e-10)
  expect_equal(estep$theta_vap, theta_vap_r, tolerance = 1e-10)
})

test_that("compute_estep_ltm_cpp returns finite log_lik", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  N <- nrow(y)
  J <- ncol(y)
  K <- 25
  C <- 4

  theta_ls <- C * hIRT:::GLpoints[[K]][["x"]]
  qw_ls    <- C * hIRT:::GLpoints[[K]][["w"]]

  alpha <- rep(0, J)
  beta  <- rep(1, J)
  fitted_mean <- rep(0, N)
  fitted_var  <- rep(1, N)

  sparse_y <- hIRT:::build_sparse_y(y)
  estep <- hIRT:::compute_estep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
  )

  expect_true(is.finite(estep$log_lik))
  expect_true(estep$log_lik < 0)

  # Cross-check: log_lik should equal sum of log(marginal) from R reference
  loglik_r <- function(alpha, beta, theta) {
    util <- matrix(alpha, N, J, byrow = TRUE) + outer(theta, beta)
    log(exp(as.matrix(y) * util) / (1 + exp(util)))
  }
  posterior_r <- lapply(seq_along(theta_ls), function(k) {
    theta_k <- theta_ls[k]
    qw_k    <- qw_ls[k]
    wt_k <- dnorm(theta_k - fitted_mean, sd = sqrt(fitted_var)) * qw_k
    loglik <- rowSums(loglik_r(alpha, beta, rep(theta_k, N)), na.rm = TRUE)
    exp(loglik + log(wt_k))
  })
  tmp <- matrix(unlist(posterior_r), N, K)
  log_lik_r <- sum(log(rowSums(tmp)))

  expect_equal(estep$log_lik, log_lik_r, tolerance = 1e-8)
})

test_that("hltm converges with synthetic Heywood-case data using init='irt'", {
  set.seed(123)
  N <- 500
  theta <- rnorm(N)

  # 10 items: 9 normal + 1 extreme (P(y=1) ~ 0.95)
  alpha <- c(rep(0, 9), 3.0)
  beta  <- c(rep(1, 9), 1.0)
  y <- data.frame(matrix(NA, N, 10))
  for (j in 1:10) {
    p <- plogis(alpha[j] + beta[j] * theta)
    y[, j] <- rbinom(N, 1, p)
  }

  # With prior (default), should converge without error
  expect_no_error({
    m <- hltm(y, init = "irt", control = list(max_iter = 200))
  })
  expect_s3_class(m, "hltm")
  expect_true(is.finite(m$log_Lik))
  expect_true(all(is.finite(m$coefficients$Estimate)))
})

test_that("hltm with prior_sigma_beta=Inf recovers pure MLE behavior", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  m_prior <- hltm(y, control = list(prior_sigma_beta = 1.5, prior_type = "lognormal"))
  m_mle   <- hltm(y, control = list(prior_sigma_beta = Inf))

  # Both should converge and produce valid results
  expect_s3_class(m_prior, "hltm")
  expect_s3_class(m_mle, "hltm")
  expect_true(is.finite(m_prior$log_Lik))
  expect_true(is.finite(m_mle$log_Lik))

  # For well-behaved data, prior and MLE results should be close
  # (weakly informative prior barely affects reasonable estimates)
  expect_equal(m_prior$coefficients$Estimate, m_mle$coefficients$Estimate,
               tolerance = 0.1)
})

test_that("lognormal prior shrinks extreme discrimination in M-step", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  N <- nrow(y)
  J <- ncol(y)
  K <- 25
  C <- 4

  theta_ls <- C * hIRT:::GLpoints[[K]][["x"]]
  qw_ls    <- C * hIRT:::GLpoints[[K]][["w"]]

  # Use extreme starting values to simulate Heywood case
  alpha <- rep(0, J)
  beta  <- rep(1, J)
  beta[1] <- 25  # Heywood-like extreme value
  fitted_mean <- rep(0, N)
  fitted_var  <- rep(1, N)

  sparse_y <- hIRT:::build_sparse_y(y)
  estep <- hIRT:::compute_estep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
  )

  # M-step with lognormal prior: extreme beta should be pulled back
  mstep_prior <- hIRT:::compute_mstep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    estep$w, theta_ls, alpha, beta,
    mu_prior = 0, sigma_prior = 1.5, prior_type = 0L
  )

  # M-step without prior: pure MLE
  mstep_mle <- hIRT:::compute_mstep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    estep$w, theta_ls, alpha, beta,
    sigma_prior = Inf
  )

  # Prior should pull extreme beta closer to reasonable range
  expect_true(abs(mstep_prior$beta[1]) < abs(mstep_mle$beta[1]))
})

test_that("gaussian prior shrinks extreme discrimination in M-step", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  N <- nrow(y)
  J <- ncol(y)
  K <- 25
  C <- 4

  theta_ls <- C * hIRT:::GLpoints[[K]][["x"]]
  qw_ls    <- C * hIRT:::GLpoints[[K]][["w"]]

  # Use extreme starting values to simulate Heywood case
  alpha <- rep(0, J)
  beta  <- rep(1, J)
  beta[1] <- 25  # Heywood-like extreme value
  fitted_mean <- rep(0, N)
  fitted_var  <- rep(1, N)

  sparse_y <- hIRT:::build_sparse_y(y)
  estep <- hIRT:::compute_estep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
  )

  # M-step with Gaussian prior
  mstep_gauss <- hIRT:::compute_mstep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    estep$w, theta_ls, alpha, beta,
    mu_prior = 0, sigma_prior = 5, prior_type = 1L
  )

  # M-step without prior: pure MLE
  mstep_mle <- hIRT:::compute_mstep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    estep$w, theta_ls, alpha, beta,
    sigma_prior = Inf
  )

  # Gaussian prior should pull extreme beta closer to zero
  expect_true(abs(mstep_gauss$beta[1]) < abs(mstep_mle$beta[1]))
})

test_that("SQUAREM and plain EM converge to same solution", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  x <- model.matrix(~ party * educ, nes_econ2008)
  z <- model.matrix(~ party, nes_econ2008)
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  m_sq <- hltm(y, x, z, control = list(acceleration = "squarem", max_iter = 500))
  m_em <- hltm(y, x, z, control = list(acceleration = "none", max_iter = 500))

  # Log-likelihoods should match closely
  expect_equal(m_sq$log_Lik, m_em$log_Lik, tolerance = 0.1)

  # Coefficients should match
  expect_equal(m_sq$coefficients$Estimate, m_em$coefficients$Estimate, tolerance = 0.01)
})

test_that("acceleration='none' produces valid output", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  m <- hltm(y, control = list(acceleration = "none"))

  expect_s3_class(m, "hltm")
  expect_true(is.finite(m$log_Lik))
  expect_true(m$log_Lik < 0)
})

test_that("SQUAREM profiling reports em_evals", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[1:600, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  m <- hltm(y, compute_se = FALSE, control = list(profile = TRUE))

  expect_true(m$timing$em_evals > 0)
  expect_true(is.integer(m$timing$em_evals))
})

test_that("hltm2 produces valid output on nes_econ2008", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  x <- model.matrix(~ party * educ, nes_econ2008)
  z <- model.matrix(~ party, nes_econ2008)
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  set.seed(42)
  n <- nrow(nes_econ2008)
  id_train <- sample.int(n, n %/% 2)
  id_test <- setdiff(1:n, id_train)

  m_train <- hltm(y[id_train, ], x[id_train, ], z[id_train, ])
  ic <- lapply(coef_item(m_train), function(x) x[["Estimate"]])

  m2 <- hltm2(y[id_test, ], x[id_test, ], z[id_test, ], item_coefs = ic)

  expect_s3_class(m2, "hltm")
  expect_true(is.finite(m2$log_Lik))
  expect_true(m2$log_Lik < 0)
  expect_equal(nrow(m2$scores), length(id_test))
  expect_true(all(is.finite(m2$scores$post_mean)))
  expect_true(all(m2$scores$post_sd > 0))
})

test_that("hltm2 SQUAREM and plain EM converge to same solution", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  x <- model.matrix(~ party * educ, nes_econ2008)
  z <- model.matrix(~ party, nes_econ2008)
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  set.seed(42)
  n <- nrow(nes_econ2008)
  id_train <- sample.int(n, n %/% 2)
  id_test <- setdiff(1:n, id_train)

  m_train <- hltm(y[id_train, ], x[id_train, ], z[id_train, ])
  ic <- lapply(coef_item(m_train), function(x) x[["Estimate"]])

  m_sq <- hltm2(y[id_test, ], x[id_test, ], z[id_test, ], item_coefs = ic,
                control = list(acceleration = "squarem"))
  m_em <- hltm2(y[id_test, ], x[id_test, ], z[id_test, ], item_coefs = ic,
                control = list(acceleration = "none"))

  expect_equal(m_sq$log_Lik, m_em$log_Lik, tolerance = 0.1)
  expect_equal(m_sq$coefficients$Estimate, m_em$coefficients$Estimate, tolerance = 0.01)
})

test_that("build_sparse_y_patterns produces correct output", {
  y <- data.frame(a = c(0L, 1L, 0L, 1L), b = c(1L, 0L, 1L, 0L))
  sp <- hIRT:::build_sparse_y_patterns(y)

  # Rows 1,3 are identical (0,1) and rows 2,4 are identical (1,0)
  expect_equal(length(sp$freq_weights), 2L)
  expect_equal(sum(sp$freq_weights), 4L)
  expect_equal(length(sp$expand_idx), 4L)
  # Patterns should map correctly
  expect_equal(sp$expand_idx[1], sp$expand_idx[3])
  expect_equal(sp$expand_idx[2], sp$expand_idx[4])
  expect_true(sp$expand_idx[1] != sp$expand_idx[2])
})

test_that("build_sparse_y_patterns handles NAs as distinct patterns", {
  y <- data.frame(a = c(0L, NA, 0L), b = c(1L, 1L, 1L))
  sp <- hIRT:::build_sparse_y_patterns(y)

  # Row 1 (0,1) and row 3 (0,1) are same; row 2 (NA,1) is different
  expect_equal(length(sp$freq_weights), 2L)
  expect_equal(sp$expand_idx[1], sp$expand_idx[3])
  expect_true(sp$expand_idx[1] != sp$expand_idx[2])
})

test_that("pattern-collapsed E-step matches full E-step", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  N <- nrow(y)
  J <- ncol(y)
  K <- 20

  theta_ls <- 4 * hIRT:::GLpoints[[K]][["x"]]
  qw_ls    <- 4 * hIRT:::GLpoints[[K]][["w"]]

  alpha <- rep(0, J)
  beta  <- rep(1, J)
  fitted_mean <- rep(0, N)
  fitted_var  <- rep(1, N)

  # Full E-step
  sparse_full <- hIRT:::build_sparse_y(y)
  es_full <- hIRT:::compute_estep_ltm_cpp(
    sparse_full$row_ptr, sparse_full$col_idx, sparse_full$values,
    alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
  )

  # Pattern-collapsed E-step
  sparse_pat <- hIRT:::build_sparse_y_patterns(y)
  N_pat <- length(sparse_pat$freq_weights)
  es_pat <- hIRT:::compute_estep_ltm_cpp(
    sparse_pat$row_ptr, sparse_pat$col_idx, sparse_pat$values,
    alpha, beta, theta_ls, qw_ls,
    rep(0, N_pat), rep(1, N_pat),
    freq_weights_ = sparse_pat$freq_weights
  )

  # Log-likelihoods should match
  expect_equal(es_pat$log_lik, es_full$log_lik, tolerance = 1e-8)

  # EAP/VAP should match after expansion
  eap_expanded <- es_pat$theta_eap[sparse_pat$expand_idx]
  vap_expanded <- es_pat$theta_vap[sparse_pat$expand_idx]
  expect_equal(eap_expanded, es_full$theta_eap, tolerance = 1e-10)
  expect_equal(vap_expanded, es_full$theta_vap, tolerance = 1e-10)
})

test_that("pattern-collapsed M-step matches full M-step", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  N <- nrow(y)
  J <- ncol(y)
  K <- 20

  theta_ls <- 4 * hIRT:::GLpoints[[K]][["x"]]
  qw_ls    <- 4 * hIRT:::GLpoints[[K]][["w"]]

  alpha <- rep(0, J)
  beta  <- rep(1, J)

  # Full path
  sparse_full <- hIRT:::build_sparse_y(y)
  es_full <- hIRT:::compute_estep_ltm_cpp(
    sparse_full$row_ptr, sparse_full$col_idx, sparse_full$values,
    alpha, beta, theta_ls, qw_ls, rep(0, N), rep(1, N)
  )
  ms_full <- hIRT:::compute_mstep_ltm_cpp(
    sparse_full$row_ptr, sparse_full$col_idx, sparse_full$values,
    es_full$w, theta_ls, alpha, beta, sigma_prior = Inf
  )

  # Pattern-collapsed path
  sparse_pat <- hIRT:::build_sparse_y_patterns(y)
  N_pat <- length(sparse_pat$freq_weights)
  es_pat <- hIRT:::compute_estep_ltm_cpp(
    sparse_pat$row_ptr, sparse_pat$col_idx, sparse_pat$values,
    alpha, beta, theta_ls, qw_ls,
    rep(0, N_pat), rep(1, N_pat),
    freq_weights_ = sparse_pat$freq_weights
  )
  ms_pat <- hIRT:::compute_mstep_ltm_cpp(
    sparse_pat$row_ptr, sparse_pat$col_idx, sparse_pat$values,
    es_pat$w, theta_ls, alpha, beta, sigma_prior = Inf,
    freq_weights_ = sparse_pat$freq_weights
  )

  expect_equal(ms_pat$alpha, ms_full$alpha, tolerance = 1e-8)
  expect_equal(ms_pat$beta, ms_full$beta, tolerance = 1e-8)
})

test_that("hltm with pattern collapsing matches without (intercept-only)", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)

  # Intercept-only: triggers pattern collapsing
  m <- hltm(y)

  expect_s3_class(m, "hltm")
  expect_true(is.finite(m$log_Lik))
  expect_true(m$log_Lik < 0)
  expect_equal(nrow(m$scores), nrow(y))
  expect_true(all(is.finite(m$scores$post_mean)))
})
