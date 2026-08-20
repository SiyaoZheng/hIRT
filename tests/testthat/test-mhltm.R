# ---- Simulation helper ----
# Simulate binary responses from a between-item multidimensional 2PL model.
simulate_mhltm_data <- function(N, J_per_dim, D, R_true, alpha_true, beta_true,
                                 seed = 123) {
  set.seed(seed)
  J <- sum(J_per_dim)
  item_dim <- rep(seq_len(D), times = J_per_dim)

  # Correlated latent traits via Cholesky
  L <- chol(R_true)  # upper triangular: t(L) %*% L = R_true
  theta <- matrix(rnorm(N * D), N, D) %*% L

  # Generate binary responses
  y <- matrix(NA_integer_, N, J)
  for (j in seq_len(J)) {
    d <- item_dim[j]
    prob <- plogis(alpha_true[j] + beta_true[j] * theta[, d])
    y[, j] <- rbinom(N, 1, prob)
  }
  colnames(y) <- paste0("item", seq_len(J))
  list(y = as.data.frame(y), theta = theta, item_dim = item_dim)
}

# ---- Tests ----
test_that("mhltm converges on simulated 2D data", {
  D <- 2L
  J_per_dim <- c(5L, 5L)
  J <- sum(J_per_dim)
  N <- 1500L
  R_true <- matrix(c(1, 0.5, 0.5, 1), 2, 2)
  alpha_true <- rnorm(J, 0, 0.5)
  beta_true <- runif(J, 0.5, 2.0)
  sim <- simulate_mhltm_data(N, J_per_dim, D, R_true, alpha_true, beta_true)

  m <- mhltm(sim$y, item_dim = sim$item_dim, compute_se = FALSE,
             control = list(max_iter = 150))

  # Basic structure
  expect_s3_class(m, "mhltm")
  expect_s3_class(m, "hIRT")
  expect_equal(m$N, N)
  expect_equal(m$J, J)
  expect_equal(m$D, D)
  expect_equal(m$item_dim, sim$item_dim)

  # Log-likelihood is finite and negative
  expect_true(is.finite(m$log_Lik))
  expect_true(m$log_Lik < 0)

  # Coefficients are finite
  expect_true(all(is.finite(m$coefficients$Estimate)))

  # Scores have correct structure: 4 columns per dimension
  expect_equal(nrow(m$scores), N)
  expect_equal(ncol(m$scores), 4 * D)
  expect_true(all(is.finite(m$scores$post_mean_1)))
  expect_true(all(is.finite(m$scores$post_mean_2)))
  expect_true(all(m$scores$post_sd_1 > 0))
  expect_true(all(m$scores$post_sd_2 > 0))

  # Correlation matrix is D x D, symmetric, PD
  expect_equal(dim(m$R), c(D, D))
  expect_equal(m$R, t(m$R))
  expect_true(all(diag(m$R) == 1))
  expect_true(all(eigen(m$R, symmetric = TRUE)$values > 0))
})

test_that("mhltm recovers item parameters reasonably", {
  D <- 2L
  J_per_dim <- c(6L, 6L)
  J <- sum(J_per_dim)
  N <- 3000L
  R_true <- matrix(c(1, 0.4, 0.4, 1), 2, 2)
  set.seed(42)
  alpha_true <- rnorm(J, 0, 0.5)
  beta_true <- runif(J, 0.8, 2.0)
  sim <- simulate_mhltm_data(N, J_per_dim, D, R_true, alpha_true, beta_true,
                               seed = 42)

  m <- mhltm(sim$y, item_dim = sim$item_dim, compute_se = FALSE,
             control = list(max_iter = 200))

  # Discrimination estimates should correlate well with true values
  # (up to a per-dimension scale transformation due to identification)
  est_beta <- m$coefficients$Estimate[seq(2, 2 * J, by = 2)]
  cor_beta <- cor(abs(est_beta), beta_true)
  expect_gt(cor_beta, 0.8)

  # Correlation matrix recovery: estimated R[1,2] should be close to 0.4
  # Allow generous tolerance since Gibbs-EM is approximate
  expect_lt(abs(m$R[1, 2] - R_true[1, 2]), 0.2)
})

test_that("mhltm with D=1 produces valid output", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[, -(1:3)]
  dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
  y[] <- lapply(y, dichotomize)
  J <- ncol(y)

  m <- mhltm(y, item_dim = rep(1L, J), compute_se = FALSE)

  expect_s3_class(m, "mhltm")
  expect_equal(m$D, 1L)
  expect_true(is.finite(m$log_Lik))
  expect_true(m$log_Lik < 0)
  expect_equal(dim(m$R), c(1, 1))
  expect_equal(m$R[1, 1], 1)
})

test_that("mhltm extractors work correctly", {
  D <- 2L
  J_per_dim <- c(4L, 4L)
  J <- sum(J_per_dim)
  N <- 800L
  R_true <- matrix(c(1, 0.3, 0.3, 1), 2, 2)
  alpha_true <- rnorm(J, 0, 0.3)
  beta_true <- runif(J, 0.5, 1.5)
  sim <- simulate_mhltm_data(N, J_per_dim, D, R_true, alpha_true, beta_true,
                               seed = 99)

  m <- mhltm(sim$y, item_dim = sim$item_dim, compute_se = FALSE,
             control = list(max_iter = 100))

  # coef_item
  ci <- coef_item(m, by_item = TRUE)
  expect_length(ci, J)
  expect_equal(nrow(ci[[1]]), 2)  # Diff, Dscrmn

  ci_flat <- coef_item(m, by_item = FALSE)
  expect_equal(nrow(ci_flat), 2 * J)

  # coef_mean: should have p * D rows (p=1 intercept-only, D=2)
  cm <- coef_mean(m)
  expect_equal(nrow(cm), m$p * D)

  # coef_var: should have q * D rows
  cv <- coef_var(m)
  expect_equal(nrow(cv), m$q * D)

  # coef_corr
  cc <- coef_corr(m)
  expect_equal(dim(cc), c(D, D))

  # print should not error
  expect_output(print(m))

  # latent_scores
  ls <- latent_scores(m)
  expect_equal(nrow(ls), N)
  expect_equal(ncol(ls), 4 * D)
})

test_that("mhltm handles missing data", {
  D <- 2L
  J_per_dim <- c(5L, 5L)
  J <- sum(J_per_dim)
  N <- 1000L
  R_true <- diag(D)  # independent dimensions
  alpha_true <- rep(0, J)
  beta_true <- rep(1, J)
  sim <- simulate_mhltm_data(N, J_per_dim, D, R_true, alpha_true, beta_true)

  # Introduce 30% NAs
  set.seed(777)
  na_mask <- matrix(rbinom(N * J, 1, 0.3), N, J)
  y_na <- sim$y
  y_na[na_mask == 1] <- NA

  m <- mhltm(y_na, item_dim = sim$item_dim, compute_se = FALSE,
             control = list(max_iter = 100))

  expect_s3_class(m, "mhltm")
  expect_true(is.finite(m$log_Lik))
  expect_true(all(is.finite(m$scores$post_mean_1)))
  expect_true(all(is.finite(m$scores$post_mean_2)))
})

test_that("mhltm with SQUAREM acceleration converges", {
  D <- 2L
  J_per_dim <- c(5L, 5L)
  J <- sum(J_per_dim)
  N <- 1000L
  R_true <- matrix(c(1, 0.3, 0.3, 1), 2, 2)
  alpha_true <- rnorm(J, 0, 0.3)
  beta_true <- runif(J, 0.5, 1.5)
  sim <- simulate_mhltm_data(N, J_per_dim, D, R_true, alpha_true, beta_true,
                               seed = 55)

  m <- mhltm(sim$y, item_dim = sim$item_dim, compute_se = FALSE,
             control = list(acceleration = "squarem", max_iter = 200))

  expect_s3_class(m, "mhltm")
  expect_true(is.finite(m$log_Lik))
  expect_true(m$log_Lik < 0)
})

test_that("mhltm input validation catches errors", {
  y <- data.frame(a = c(0, 1, 0), b = c(1, 0, 1), c = c(1, 1, 0))

  # Missing item_dim
  expect_error(mhltm(y, compute_se = FALSE), "item_dim")

  # Wrong length
  expect_error(mhltm(y, item_dim = c(1, 2), compute_se = FALSE), "length J")

  # Non-contiguous dimensions
  expect_error(mhltm(y, item_dim = c(1, 1, 3), compute_se = FALSE), "contiguous")

  # Dimension with < 2 items
  expect_error(mhltm(y, item_dim = c(1, 1, 2), compute_se = FALSE),
               "at least 2 items")

  # compute_se = TRUE (use valid item_dim so compute_se check is reached)
  y4 <- data.frame(a = c(0,1,0), b = c(1,0,1), c = c(1,1,0), d = c(0,0,1))
  expect_error(mhltm(y4, item_dim = c(1, 1, 2, 2), compute_se = TRUE),
               "compute_se must be FALSE")
})

test_that("mhltm with 3 dimensions converges", {
  D <- 3L
  J_per_dim <- c(4L, 4L, 4L)
  J <- sum(J_per_dim)
  N <- 2000L
  R_true <- matrix(0.3, D, D)
  diag(R_true) <- 1
  alpha_true <- rnorm(J, 0, 0.3)
  beta_true <- runif(J, 0.5, 1.5)
  sim <- simulate_mhltm_data(N, J_per_dim, D, R_true, alpha_true, beta_true,
                               seed = 77)

  m <- mhltm(sim$y, item_dim = sim$item_dim, compute_se = FALSE,
             control = list(max_iter = 150))

  expect_s3_class(m, "mhltm")
  expect_equal(m$D, 3L)
  expect_true(is.finite(m$log_Lik))
  expect_equal(dim(m$R), c(3, 3))
  expect_true(all(eigen(m$R, symmetric = TRUE)$values > 0))
})
