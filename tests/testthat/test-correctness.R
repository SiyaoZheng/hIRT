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

  # C++ M-step
  mstep <- hIRT:::compute_mstep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    w, theta_ls, alpha, beta
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

  # C++ M-step
  mstep <- hIRT:::compute_mstep_ltm_cpp(
    sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
    w, theta_ls, alpha, beta
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
