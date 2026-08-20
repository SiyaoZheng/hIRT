# Multidimensional hIRT helpers (mhltm)

# Build per-dimension sparse Y representations.
# Returns a list of CSR structures, one per dimension.
build_sparse_y_by_dim <- function(y, items_by_dim) {
  lapply(items_by_dim, function(jj) {
    build_sparse_y(y[, jj, drop = FALSE])
  })
}

# Compute conditional MVN parameters for dimension d given all others.
# Given D x D correlation matrix R, returns:
#   beta_d: (D-1)-vector of regression coefficients
#   c_d:    scalar Schur complement (conditional variance ratio)
# When D=1, returns trivial values (no conditioning needed).
conditional_mvn_params <- function(R, d) {
  D <- nrow(R)
  if (D == 1L) {
    return(list(beta_d = numeric(0), c_d = 1.0))
  }
  idx <- setdiff(seq_len(D), d)
  R_mm <- R[idx, idx, drop = FALSE]
  R_md <- R[idx, d]
  beta_d <- as.double(solve(R_mm, R_md))
  c_d <- max(1.0 - sum(R_md * beta_d), 1e-6)
  list(beta_d = beta_d, c_d = c_d)
}

# Compute per-person conditional prior moments for dimension d.
#
# The model assumes theta_i ~ MVN(mu_i, Sigma_i) where
# Sigma_i = diag(sigma_i) %*% R %*% diag(sigma_i).
#
# The conditional distribution theta_id | theta_i,-d is:
#   cond_mean_i = mu_id + sigma_id * sum(beta_d * e_i,-d)
#   cond_var_i  = sigma2_id * c_d
# where e_i,-d = (theta_i,-d - mu_i,-d) / sigma_i,-d are standardized residuals.
#
# @param mu       N x D matrix of marginal prior means
# @param sigma2   N x D matrix of marginal prior variances
# @param theta_other N x D matrix of current theta estimates (used for conditioning)
# @param beta_d   (D-1)-vector from conditional_mvn_params
# @param c_d      scalar from conditional_mvn_params
# @param d        dimension index
# @return list(cond_mean, cond_var) each N-vectors
conditional_prior_moments <- function(mu, sigma2, theta_other, beta_d, c_d, d) {
  D <- ncol(mu)
  if (D == 1L) {
    return(list(cond_mean = mu[, 1L], cond_var = sigma2[, 1L]))
  }
  idx <- setdiff(seq_len(D), d)
  sigma_d <- sqrt(sigma2[, d])

  # Standardized residuals for other dimensions
  e_other <- (theta_other[, idx, drop = FALSE] - mu[, idx, drop = FALSE]) /
    sqrt(sigma2[, idx, drop = FALSE])

  # Conditional mean and variance
  cond_mean <- mu[, d] + sigma_d * as.double(e_other %*% beta_d)
  cond_var <- sigma2[, d] * c_d

  list(cond_mean = cond_mean, cond_var = cond_var)
}

# M-step for the correlation matrix R.
# Computes the sample correlation of standardized residuals with
# posterior variance correction on the diagonal, then normalizes
# to a proper correlation matrix and ensures positive-definiteness.
#
# @param theta_eap N x D EAP estimates
# @param theta_vap N x D VAP estimates
# @param mu        N x D marginal prior means
# @param sigma2    N x D marginal prior variances
# @return D x D correlation matrix
update_correlation_matrix <- function(theta_eap, theta_vap, mu, sigma2) {
  N <- nrow(theta_eap)
  D <- ncol(theta_eap)

  if (D == 1L) return(matrix(1, 1, 1))

  # Standardized residuals
  e_std <- (theta_eap - mu) / sqrt(sigma2)

  # Sample covariance + posterior variance correction
  S <- crossprod(e_std) / N
  for (d in seq_len(D)) {
    S[d, d] <- S[d, d] + mean(theta_vap[, d] / sigma2[, d])
  }

  # Normalize to correlation
  d_diag <- sqrt(pmax(diag(S), 1e-10))
  R <- S / outer(d_diag, d_diag)

  ensure_pd(R)
}

# Eigenvalue floor to guarantee positive-definiteness.
ensure_pd <- function(R, eps = 1e-6) {
  eig <- eigen(R, symmetric = TRUE)
  if (any(eig$values < eps)) {
    eig$values <- pmax(eig$values, eps)
    R <- eig$vectors %*% diag(eig$values, nrow = length(eig$values)) %*% t(eig$vectors)
    d <- sqrt(diag(R))
    R <- R / outer(d, d)
  }
  R
}

# Pack lower triangle of correlation matrix to vector (row-major).
corr_to_vec <- function(R) {
  R[lower.tri(R)]
}

# Unpack vector to symmetric correlation matrix.
vec_to_corr <- function(v, D) {
  R <- diag(D)
  R[lower.tri(R)] <- v
  R <- R + t(R) - diag(D)
  R
}

# Apply per-dimension identification constraints (latent_scale).
#
# For each dimension d:
#   Location: mean(x %*% gamma_d) = 0
#   Scale:    mean(z %*% lambda_d) = 0 (i.e., geom_mean(prior_var) = 1)
#   Direction: sign(beta[beta_set_d]) matches sign_set_d
#
# Sign flips also negate the corresponding row/column of R.
apply_constraints_md <- function(params, J, p, q, D, x, z,
                                  items_by_dim, beta_set, sign_set) {
  n_corr <- (D * (D - 1L)) %/% 2L

  alpha <- params[1:J]
  beta_vec <- params[(J + 1):(2 * J)]
  gamma_all <- matrix(params[(2 * J + 1):(2 * J + p * D)], nrow = p, ncol = D)
  lambda_all <- matrix(params[(2 * J + p * D + 1):(2 * J + p * D + q * D)],
                        nrow = q, ncol = D)
  if (n_corr > 0L) {
    R <- vec_to_corr(params[(2 * J + p * D + q * D + 1):
                             (2 * J + p * D + q * D + n_corr)], D)
  } else {
    R <- matrix(1, 1, 1)
  }

  for (d in seq_len(D)) {
    jj <- items_by_dim[[d]]
    g <- gamma_all[, d]
    l <- lambda_all[, d]
    a <- alpha[jj]
    b <- beta_vec[jj]
    bs <- beta_set[d]
    ss <- sign_set[d]

    # Location
    loc <- mean(x %*% g)
    a <- a + loc * b
    g[1L] <- g[1L] - loc

    # Scale
    sc <- mean(z %*% l)
    g <- g / exp(sc / 2)
    b <- b * exp(sc / 2)
    l[1L] <- l[1L] - sc

    # Direction
    if (ss == (b[bs] < 0)) {
      g <- -g
      b <- -b
      if (D > 1L) {
        R[d, ] <- -R[d, ]
        R[, d] <- -R[, d]
        R[d, d] <- 1
      }
    }

    alpha[jj] <- a
    beta_vec[jj] <- b
    gamma_all[, d] <- g
    lambda_all[, d] <- l
  }

  if (n_corr > 0L) {
    R_vec <- corr_to_vec(R)
  } else {
    R_vec <- numeric(0)
  }
  c(alpha, beta_vec, as.double(gamma_all), as.double(lambda_all), R_vec)
}
