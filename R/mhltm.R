#' Fitting Multidimensional Between-Item Hierarchical Latent Trait Models.
#'
#' \code{mhltm} fits a multidimensional between-item hierarchical latent trait
#' model where each binary item loads on exactly one dimension. The mean and
#' variance of each latent dimension may depend on person-specific covariates,
#' and dimensions are correlated through a population-level MVN prior.
#'
#' The model uses a Gibbs-EM algorithm: the joint E-step over D dimensions
#' is approximated by cycling through conditional 1D E-steps, reusing the
#' existing C++ \code{compute_estep_ltm_cpp} and \code{compute_mstep_ltm_cpp}
#' kernels. Dimensions couple only through the prior correlation matrix R.
#'
#' @param y A data frame or matrix of binary responses (N x J).
#' @param x Covariate matrix for the mean equation (N x p), including intercept.
#' @param z Covariate matrix for the variance equation (N x q), including intercept.
#' @param item_dim Integer vector of length J assigning each item to a dimension
#'   (values in 1, ..., D where D = max(item_dim)).
#' @param constr Identification constraint: \code{"latent_scale"} (default) or
#'   \code{"items"}.
#' @param beta_set Integer vector (length D, recycled) specifying the reference
#'   item index within each dimension for sign identification.
#' @param sign_set Logical vector (length D, recycled) specifying sign direction.
#' @param init Initialization method: \code{"glm"} (default), \code{"tetrachoric"},
#'   or \code{"irt"}.
#' @param control List of control parameters. See details for \code{n_gibbs}.
#' @param compute_se Logical. Must be \code{FALSE} (SE not yet implemented).
#'
#' @return An object of class \code{c("mhltm", "hIRT")}.
#'  \item{coefficients}{Data frame of parameter estimates.}
#'  \item{scores}{Data frame of per-dimension EAP/prior estimates.}
#'  \item{R}{D x D estimated correlation matrix.}
#'  \item{item_dim}{Integer vector of dimension assignments.}
#'  \item{D}{Number of dimensions.}
#'  \item{log_Lik}{Sum of per-dimension conditional log-likelihoods at convergence.}
#'  \item{N}{Number of units.}
#'  \item{J}{Number of items.}
#'  \item{H}{Vector of response categories per item.}
#'  \item{ylevels}{List of factor levels per item.}
#'  \item{p}{Number of mean-equation predictors.}
#'  \item{q}{Number of variance-equation predictors.}
#'  \item{control}{List of control values.}
#'  \item{se_computed}{Logical, always FALSE for now.}
#'  \item{call}{The matched call.}
#'
#' @importFrom ltm ltm
#' @import stats
#' @export

mhltm <- function(y, x = NULL, z = NULL,
                  item_dim,
                  constr = c("latent_scale", "items"),
                  beta_set = 1L,
                  sign_set = TRUE,
                  init = c("glm", "tetrachoric", "irt"),
                  control = list(),
                  compute_se = FALSE) {

  cl <- match.call()

  # ---- Input validation ----
  if (missing(y)) stop("`y` must be provided.")

  if ((!is.data.frame(y) && !is.matrix(y)) || ncol(y) == 1L)
    stop("'y' must be either a data.frame or a matrix with at least two columns.")
  if (is.matrix(y)) y <- as.data.frame(y)

  N <- nrow(y)
  J <- ncol(y)

  # item_dim
  if (missing(item_dim)) stop("`item_dim` must be provided.")
  item_dim <- as.integer(item_dim)
  if (length(item_dim) != J)
    stop("`item_dim` must have length J (", J, ").")
  if (any(is.na(item_dim))) stop("`item_dim` must not contain NAs.")
  D <- max(item_dim)
  if (min(item_dim) < 1L) stop("`item_dim` values must be >= 1.")
  if (!all(seq_len(D) %in% item_dim))
    stop("item_dim must use contiguous dimensions 1:", D, ".")

  items_by_dim <- split(seq_len(J), item_dim)
  J_per_dim <- vapply(items_by_dim, length, integer(1L))
  if (any(J_per_dim < 2L))
    stop("Each dimension must have at least 2 items. Dimensions with < 2: ",
         paste(which(J_per_dim < 2L), collapse = ", "))

  # Convert y
  y[] <- lapply(y, factor, exclude = c(NA, NaN))
  ylevels <- lapply(y, levels)
  y[] <- lapply(y, function(x) as.integer(x) - 1)
  if (!is.na(invalid <- match(TRUE, vapply(y, invalid_ltm, logical(1L)))))
    stop(paste(names(y)[invalid], "is not a dichotomous variable"))
  H <- vapply(y, max, double(1L), na.rm = TRUE) + 1

  # Covariates
  x <- x %||% as.matrix(rep(1, N))
  z <- z %||% as.matrix(rep(1, N))
  if (!is.matrix(x)) stop("`x` must be a matrix.")
  if (!is.matrix(z)) stop("`z` must be a matrix.")
  if (nrow(x) != N || nrow(z) != N)
    stop("both 'x' and 'z' must have the same number of rows as 'y'")
  p <- ncol(x)
  q <- ncol(z)
  colnames(x) <- colnames(x) %||% paste0("x", 1:p)
  colnames(z) <- colnames(z) %||% paste0("z", 1:q)

  # beta_set / sign_set: recycle to length D
  beta_set <- rep_len(as.integer(beta_set), D)
  sign_set <- rep_len(as.logical(sign_set), D)
  for (d in seq_len(D)) {
    if (beta_set[d] < 1L || beta_set[d] > J_per_dim[d])
      stop("beta_set[", d, "] = ", beta_set[d],
           " is out of range for dimension ", d, " (", J_per_dim[d], " items).")
  }

  if (!identical(compute_se, FALSE))
    stop("compute_se must be FALSE for mhltm (SE not yet implemented).")

  constr <- match.arg(constr)
  init <- match.arg(init)

  # ---- Control parameters ----
  con <- list(max_iter = 150, max_iter2 = 15, eps = 1e-03, eps2 = 1e-03,
              K = 25, C = 4,
              prior_mu_beta = 0, prior_sigma_beta = Inf,
              prior_type = "lognormal", prior_warmup = 0L,
              acceleration = "none", n_gibbs = 1L,
              profile = FALSE, verbose = FALSE, lazy_varreg = 0)
  con[names(control)] <- control

  if (identical(con[["prior_warmup"]], "auto")) {
    con[["prior_warmup"]] <- if (con[["prior_type"]] == "lognormal" &&
                                  init != "irt") 20L else 0L
  }
  con[["prior_warmup"]] <- as.integer(con[["prior_warmup"]])
  verbose <- isTRUE(con[["verbose"]])
  n_gibbs <- as.integer(con[["n_gibbs"]])
  profile <- isTRUE(con[["profile"]])
  timing <- NULL
  if (profile) {
    t_total_start <- proc.time()[["elapsed"]]
    t_init_start <- t_total_start
    timing <- list(init = 0, em_total = 0, total = 0)
  }

  # ---- Quadrature ----
  K <- con[["K"]]
  theta_ls <- con[["C"]] * GLpoints[[K]][["x"]]
  qw_ls <- con[["C"]] * GLpoints[[K]][["w"]]

  # ---- Imputation ----
  y_imp <- y
  if (anyNA(y)) y_imp[] <- lapply(y, impute)

  # ---- Per-dimension sparse Y ----
  sparse_y_dim <- build_sparse_y_by_dim(y, items_by_dim)
  n_corr <- (D * (D - 1L)) %/% 2L

  # ---- Initialization ----
  alpha <- numeric(J)
  beta <- numeric(J)
  item_names <- names(y)

  # PCA per dimension for initial theta
  theta_init <- matrix(0, N, D)
  for (d in seq_len(D)) {
    jj <- items_by_dim[[d]]
    y_d <- y_imp[, jj, drop = FALSE]
    if (ncol(y_d) >= 2L) {
      tmp <- princomp(y_d, cor = TRUE)$scores[, 1]
      theta_init[, d] <- (tmp - mean(tmp)) / sd(tmp)
    } else {
      theta_init[, d] <- as.double(scale(as.numeric(y_d[[1]]))[, 1])
    }
  }

  if (init == "glm") {
    for (d in seq_len(D)) {
      jj <- items_by_dim[[d]]
      pseudo_logit <- lapply(y_imp[, jj, drop = FALSE], function(yj) {
        glm.fit(cbind(1, theta_init[, d]), yj,
                family = binomial("logit"))[["coefficients"]]
      })
      alpha[jj] <- vapply(pseudo_logit, function(x) x[1L], double(1L))
      beta[jj] <- vapply(pseudo_logit, function(x) x[2L], double(1L))
    }
  } else if (init == "tetrachoric") {
    for (d in seq_len(D)) {
      jj <- items_by_dim[[d]]
      y_mat_d <- as.matrix(y[, jj, drop = FALSE])
      obs <- (!is.na(y_mat_d)) * 1.0
      y_1 <- y_mat_d; y_1[is.na(y_1)] <- 0
      y_0 <- (1 - y_mat_d) * obs
      ct_00 <- crossprod(y_0)
      ct_01 <- crossprod(y_0, y_1)
      ct_10 <- crossprod(y_1, y_0)
      ct_11 <- crossprod(y_1)
      ad <- ct_00 * ct_11; bc <- ct_01 * ct_10
      sqrt_ad <- sqrt(pmax(ad, 0)); sqrt_bc <- sqrt(pmax(bc, 0))
      denom <- sqrt_ad + sqrt_bc
      ratio <- sqrt_bc / denom
      ratio[!is.finite(ratio)] <- 0.5
      r_tet <- cos(pi * ratio); diag(r_tet) <- 1
      eig <- eigen(r_tet, symmetric = TRUE)
      eig$values <- pmax(eig$values, 0)
      f <- eig$vectors[, 1] * sqrt(eig$values[1])
      if (mean(f) < 0) f <- -f
      f <- pmin(pmax(f, -0.999), 0.999)
      D_const <- 1.702
      p_j <- colMeans(y_mat_d, na.rm = TRUE)
      scale_j <- 1 / sqrt(1 - f^2)
      beta[jj] <- D_const * f * scale_j
      alpha[jj] <- D_const * qnorm(p_j) * scale_j
    }
  } else {
    for (d in seq_len(D)) {
      jj <- items_by_dim[[d]]
      ltm_coefs <- ltm(y[, jj, drop = FALSE] ~ z1)[["coefficients"]]
      beta[jj] <- ltm_coefs[, 2, drop = TRUE]
      alpha[jj] <- ltm_coefs[, 1, drop = TRUE]
    }
  }

  # Initial gamma, lambda per dimension
  lm_opr <- tcrossprod(solve(crossprod(x)), x)
  gamma_all <- matrix(0, p, D)
  lambda_all <- matrix(0, q, D)
  for (d in seq_len(D)) {
    gamma_all[, d] <- lm_opr %*% theta_init[, d]
    lambda_all[, d] <- rep(0, q)
  }

  # Initial correlation: identity
  R_init <- diag(D)

  if (profile) timing$init <- proc.time()[["elapsed"]] - t_init_start

  # ---- EM setup ----
  prior_type_int <- match(con[["prior_type"]],
                          c("lognormal", "gaussian"), nomatch = 2L) - 1L

  # Pack initial parameters
  params <- c(alpha, beta, as.double(gamma_all), as.double(lambda_all),
              corr_to_vec(R_init))
  n_eval <- 0L
  squarem_active <- FALSE

  apply_constraints_packed <- function(params) {
    apply_constraints_md(params, J, p, q, D, x, z,
                         items_by_dim, beta_set, sign_set)
  }

  # ---- EM step function (one Gibbs-EM cycle) ----
  em_step_fn <- function(params) {
    # Unpack
    a <- params[1:J]
    b <- params[(J + 1):(2 * J)]
    g_all <- matrix(params[(2 * J + 1):(2 * J + p * D)], nrow = p, ncol = D)
    l_all <- matrix(params[(2 * J + p * D + 1):(2 * J + p * D + q * D)],
                     nrow = q, ncol = D)
    if (n_corr > 0L) {
      R_cur <- vec_to_corr(
        params[(2 * J + p * D + q * D + 1):(2 * J + p * D + q * D + n_corr)], D)
    } else {
      R_cur <- matrix(1, 1, 1)
    }

    # Marginal prior
    mu_all <- x %*% g_all
    sigma2_all <- matrix(0, N, D)
    for (dd in seq_len(D)) {
      sigma2_all[, dd] <- exp(as.double(z %*% l_all[, dd]))
    }

    # Initialize Gibbs conditioning from marginal means
    theta_gibbs <- mu_all

    # Storage
    w_mats <- vector("list", D)
    theta_eap_new <- matrix(0, N, D)
    theta_vap_new <- matrix(0, N, D)
    total_ll <- 0

    # Gibbs-EM sweep(s)
    for (gibbs_iter in seq_len(n_gibbs)) {
      total_ll <- 0
      for (dd in seq_len(D)) {
        cond_p <- conditional_mvn_params(R_cur, dd)
        cond_m <- conditional_prior_moments(
          mu_all, sigma2_all, theta_gibbs, cond_p$beta_d, cond_p$c_d, dd)

        jj <- items_by_dim[[dd]]

        es <- compute_estep_ltm_cpp(
          sparse_y_dim[[dd]]$row_ptr,
          sparse_y_dim[[dd]]$col_idx,
          sparse_y_dim[[dd]]$values,
          a[jj], b[jj],
          theta_ls, qw_ls,
          cond_m$cond_mean, cond_m$cond_var)

        w_mats[[dd]] <- es$w
        theta_eap_new[, dd] <- es$theta_eap
        theta_vap_new[, dd] <- es$theta_vap
        total_ll <- total_ll + es$log_lik

        theta_gibbs[, dd] <- es$theta_eap
      }
    }

    # M-step: items per dimension
    warmup_n <- con[["prior_warmup"]]
    sigma_eff <- if (warmup_n > 0L && n_eval < warmup_n) Inf else
      con[["prior_sigma_beta"]]

    for (dd in seq_len(D)) {
      jj <- items_by_dim[[dd]]
      ms <- compute_mstep_ltm_cpp(
        sparse_y_dim[[dd]]$row_ptr,
        sparse_y_dim[[dd]]$col_idx,
        sparse_y_dim[[dd]]$values,
        w_mats[[dd]], theta_ls,
        a[jj], b[jj],
        mu_prior = con[["prior_mu_beta"]],
        sigma_prior = sigma_eff,
        prior_type = prior_type_int)
      a[jj] <- ms$alpha
      b[jj] <- ms$beta
    }

    # M-step: variance regression per dimension
    for (dd in seq_len(D)) {
      g_d <- lm_opr %*% theta_eap_new[, dd]
      r2_d <- (theta_eap_new[, dd] - x %*% g_d)^2 + theta_vap_new[, dd]

      if (ncol(z) == 1L) {
        l_d <- log(mean(r2_d))
      } else {
        skip_full_vr <- as.integer(con[["lazy_varreg"]]) > 0L &&
          n_eval < as.integer(con[["lazy_varreg"]])
        if (skip_full_vr) {
          l_d <- log(mean(r2_d))
          l_d <- rep(l_d / ncol(z), ncol(z))
        } else {
          s2_d <- glm.fit(x = z, y = r2_d, intercept = FALSE,
                          family = Gamma(link = "log"))[["fitted.values"]]
          loglik_vr <- -0.5 * (log(s2_d) + r2_d / s2_d)
          LL0 <- sum(loglik_vr)
          for (m_vr in seq_len(con[["max_iter2"]])) {
            g_d <- lm.wfit(x, theta_eap_new[, dd], w = 1 / s2_d)[["coefficients"]]
            r2_d <- (theta_eap_new[, dd] - x %*% g_d)^2 + theta_vap_new[, dd]
            var_reg <- glm.fit(x = z, y = r2_d, intercept = FALSE,
                               family = Gamma(link = "log"))
            s2_d <- var_reg[["fitted.values"]]
            loglik_vr <- -0.5 * (log(s2_d) + r2_d / s2_d)
            LL_temp <- sum(loglik_vr)
            if (LL_temp - LL0 < con[["eps2"]]) break
            LL0 <- LL_temp
          }
          l_d <- var_reg[["coefficients"]]
        }
      }
      g_all[, dd] <- g_d
      l_all[, dd] <- l_d
    }

    # M-step: correlation matrix
    mu_new <- x %*% g_all
    sigma2_new <- matrix(0, N, D)
    for (dd in seq_len(D)) {
      sigma2_new[, dd] <- exp(as.double(z %*% l_all[, dd]))
    }
    R_cur <- update_correlation_matrix(theta_eap_new, theta_vap_new,
                                        mu_new, sigma2_new)

    # Constraints
    if (!squarem_active) {
      params_tmp <- c(a, b, as.double(g_all), as.double(l_all),
                       corr_to_vec(R_cur))
      params_tmp <- apply_constraints_packed(params_tmp)
    } else {
      params_tmp <- c(a, b, as.double(g_all), as.double(l_all),
                       corr_to_vec(R_cur))
    }

    list(params = params_tmp, log_lik = total_ll)
  }

  # ---- EM algorithm ----
  if (profile) t_em_start <- proc.time()[["elapsed"]]
  trace_log <- if (verbose) list() else NULL

  if (verbose) {
    cat(sprintf("\n=== mhltm Gibbs-EM (N=%d, J=%d, D=%d, K=%d) ===\n",
                N, J, D, K))
    cat(sprintf("acceleration=%s, n_gibbs=%d, max_iter=%d, eps=%.1e\n",
                con$acceleration, n_gibbs, con$max_iter, con$eps))
  }

  # Convergence: for D>1 Gibbs-EM, beta_rmse has a noise floor from cycling.
  # Use LL-based convergence: range of LL over rolling window < tol.
  ll_window_size <- 10L
  ll_history <- numeric(0)
  use_ll_convergence <- D > 1L

  if (con[["acceleration"]] == "squarem") {
    # SQUAREM S3 (Varadhan & Roland 2008)
    step_max <- 1
    squarem_active <- FALSE
    converged <- FALSE
    cycle <- 0L

    while (n_eval < con[["max_iter"]]) {
      cycle <- cycle + 1L

      # Step 1: F(theta_0)
      res1 <- em_step_fn(params)
      n_eval <- n_eval + 1L
      if (!verbose) cat(".")

      theta_1 <- res1$params
      ll_0 <- res1$log_lik

      # Convergence check
      params_c <- apply_constraints_packed(params)
      theta_1_c <- apply_constraints_packed(theta_1)
      beta_prev <- params_c[(J + 1):(2 * J)]
      beta_curr <- theta_1_c[(J + 1):(2 * J)]
      beta_rmse <- sqrt(mean((beta_curr - beta_prev)^2))

      if (use_ll_convergence) {
        ll_history <- c(ll_history, ll_0)
        if (length(ll_history) > ll_window_size)
          ll_history <- ll_history[(length(ll_history) - ll_window_size + 1L):
                                    length(ll_history)]
        ll_range <- if (length(ll_history) >= ll_window_size)
          diff(range(ll_history)) else Inf
        ll_converged <- ll_range < 10
      } else {
        ll_converged <- FALSE
        ll_range <- NA
      }

      if (beta_rmse < con[["eps"]] || ll_converged) {
        params <- theta_1
        converged <- TRUE
        if (verbose) {
          cat(sprintf("[Cycle %d] CONVERGED: beta_rmse=%.2e, ll_range=%.2e\n",
                      cycle, beta_rmse,
                      if (is.na(ll_range)) 0 else ll_range))
        } else {
          cat("\n converged at evaluation", n_eval, "\n")
        }
        break
      }

      if (n_eval >= con[["max_iter"]]) { params <- theta_1; break }

      # Step 2: F(theta_1)
      res2 <- em_step_fn(theta_1)
      n_eval <- n_eval + 1L
      if (!verbose) cat(".")

      theta_2 <- res2$params
      ll_1 <- res2$log_lik

      if (n_eval >= con[["max_iter"]]) { params <- theta_2; break }

      # SQUAREM extrapolation
      r <- theta_1 - params
      v <- (theta_2 - theta_1) - r
      sr2 <- sum(r^2)
      sv2 <- sum(v^2)

      if (sv2 < 1e-30) {
        params <- theta_2
        next
      }

      alpha_raw <- sqrt(sr2 / sv2)
      alpha_sq <- min(step_max, max(1, alpha_raw))

      theta_prop <- params + 2 * alpha_sq * r + alpha_sq^2 * v
      if (any(!is.finite(theta_prop))) {
        params <- theta_2
        step_max <- max(1, step_max / 2)
        next
      }

      # Step 3: stabilize
      res3 <- em_step_fn(theta_prop)
      n_eval <- n_eval + 1L
      if (!verbose) cat(".")
      ll_prop <- res3$log_lik

      if (!is.finite(ll_prop)) {
        params <- theta_2
        step_max <- max(1, step_max / 2)
      } else if (ll_prop >= ll_0 - 1e-4) {
        params <- res3$params
        if (alpha_sq >= step_max - 0.01) step_max <- 2 * alpha_sq
      } else {
        params <- theta_2
        step_max <- max(1, step_max / 2)
      }

      if (verbose) {
        cat(sprintf("[Cycle %3d | eval %3d] LL=%.4f->%.4f  beta_rmse=%.2e  alpha=%.2f\n",
                    cycle, n_eval, ll_0, ll_1, beta_rmse, alpha_sq))
      }
    }

    if (!converged && n_eval >= con[["max_iter"]]) {
      stop("algorithm did not converge; try increasing max_iter.")
    }

    # Polish phase
    params <- apply_constraints_packed(params)
    ll_polish <- -Inf
    for (polish_iter in seq_len(con[["max_iter"]] - n_eval)) {
      beta_prev_p <- params[(J + 1):(2 * J)]
      res_p <- em_step_fn(params)
      n_eval <- n_eval + 1L
      params <- res_p$params
      if (!verbose) cat(".")
      beta_rmse_p <- sqrt(mean((params[(J + 1):(2 * J)] - beta_prev_p)^2))
      ll_change <- abs(res_p$log_lik - ll_polish)
      ll_polish <- res_p$log_lik
      if (verbose) {
        cat(sprintf("[Polish %d | eval %d] LL=%.4f  beta_rmse=%.2e\n",
                    polish_iter, n_eval, res_p$log_lik, beta_rmse_p))
      }
      if (beta_rmse_p < con[["eps"]] || ll_change < 1e-6) break
    }

  } else {
    # Plain EM
    converged <- FALSE
    for (iter in seq_len(con[["max_iter"]])) {
      beta_prev <- params[(J + 1):(2 * J)]
      res <- em_step_fn(params)
      n_eval <- n_eval + 1L
      params <- res$params
      if (!verbose) cat(".")

      beta_curr <- params[(J + 1):(2 * J)]
      beta_rmse <- sqrt(mean((beta_curr - beta_prev)^2))

      if (use_ll_convergence) {
        ll_history <- c(ll_history, res$log_lik)
        if (length(ll_history) > ll_window_size)
          ll_history <- ll_history[(length(ll_history) - ll_window_size + 1L):
                                    length(ll_history)]
        ll_range <- if (length(ll_history) >= ll_window_size)
          diff(range(ll_history)) else Inf
        ll_converged <- ll_range < 10
      } else {
        ll_converged <- FALSE
        ll_range <- NA
      }

      if (verbose) {
        cat(sprintf("[Iter %3d | eval %3d] LL=%.4f  beta_rmse=%.2e  ll_range=%.2e\n",
                    iter, n_eval, res$log_lik, beta_rmse,
                    if (is.na(ll_range)) 0 else ll_range))
      }

      if (beta_rmse < con[["eps"]] || ll_converged) {
        converged <- TRUE
        if (verbose) {
          cat(sprintf("CONVERGED at iter %d: beta_rmse=%.2e\n", iter, beta_rmse))
        } else {
          cat("\n converged at iteration", iter, "\n")
        }
        break
      }
    }
    if (!converged) {
      stop("algorithm did not converge; try increasing max_iter.")
    }
  }

  if (profile) timing$em_total <- proc.time()[["elapsed"]] - t_em_start

  # ---- Unpack final parameters ----
  alpha  <- setNames(params[1:J], item_names)
  beta   <- setNames(params[(J + 1):(2 * J)], item_names)
  gamma_all <- matrix(params[(2 * J + 1):(2 * J + p * D)], nrow = p, ncol = D)
  lambda_all <- matrix(params[(2 * J + p * D + 1):(2 * J + p * D + q * D)],
                        nrow = q, ncol = D)
  if (n_corr > 0L) {
    R_final <- vec_to_corr(
      params[(2 * J + p * D + q * D + 1):(2 * J + p * D + q * D + n_corr)], D)
  } else {
    R_final <- matrix(1, 1, 1)
  }

  mu_all <- x %*% gamma_all
  sigma2_all <- matrix(0, N, D)
  for (d in seq_len(D)) {
    sigma2_all[, d] <- exp(as.double(z %*% lambda_all[, d]))
  }

  # ---- Final E-step (Gibbs sweep for posterior estimates) ----
  theta_gibbs <- mu_all
  theta_eap <- matrix(0, N, D)
  theta_vap <- matrix(0, N, D)
  log_Lik <- 0

  for (d in seq_len(D)) {
    cond_p <- conditional_mvn_params(R_final, d)
    cond_m <- conditional_prior_moments(
      mu_all, sigma2_all, theta_gibbs, cond_p$beta_d, cond_p$c_d, d)

    jj <- items_by_dim[[d]]
    final_es <- compute_estep_ltm_cpp(
      sparse_y_dim[[d]]$row_ptr,
      sparse_y_dim[[d]]$col_idx,
      sparse_y_dim[[d]]$values,
      alpha[jj], beta[jj],
      theta_ls, qw_ls,
      cond_m$cond_mean, cond_m$cond_var)

    theta_eap[, d] <- final_es$theta_eap
    theta_vap[, d] <- final_es$theta_vap
    theta_gibbs[, d] <- final_es$theta_eap
    log_Lik <- log_Lik + final_es$log_lik
  }

  # ---- Name parameters ----
  gamma_names <- character(p * D)
  lambda_names <- character(q * D)
  for (d in seq_len(D)) {
    gamma_names[((d - 1) * p + 1):(d * p)] <- paste0("d", d, " x", colnames(x))
    lambda_names[((d - 1) * q + 1):(d * q)] <- paste0("d", d, " z", colnames(z))
  }
  gamma_vec <- setNames(as.double(gamma_all), gamma_names)
  lambda_vec <- setNames(as.double(lambda_all), lambda_names)

  # ---- Build coefficients data frame ----
  names_ab <- paste(rep(item_names, each = 2), c("Diff", "Dscrmn"))
  coefs_item_vec <- c(unlist(Map(function(a, b) c(Diff = a, Dscrmn = b),
                                  alpha, beta)))
  coef_all <- c(coefs_item_vec, gamma_vec, lambda_vec)
  coef_names <- c(names_ab, gamma_names, lambda_names)
  se_all <- rep(NA_real_, length(coef_all))
  names(se_all) <- coef_names

  coefs <- data.frame(Estimate = coef_all, Std_Error = se_all,
                       z_value = coef_all / se_all,
                       p_value = 2 * (1 - pnorm(abs(coef_all / se_all))))
  rownames(coefs) <- coef_names

  # ---- Items constraint reparameterization (post-hoc) ----
  if (constr == "items") {
    gamma0_prev <- gamma_all[1L, ]

    for (d in seq_len(D)) {
      jj <- items_by_dim[[d]]
      a_d <- alpha[jj]; b_d <- beta[jj]

      # Location: sum(alpha_d) = 0
      c1 <- sum(a_d) / sum(b_d)
      gamma_all[1L, d] <- gamma_all[1L, d] + c1
      alpha[jj] <- a_d - c1 * b_d

      # Scale: geom_mean(|beta_d|) = 1
      c2 <- 2 * mean(log(abs(b_d)))
      gamma_all[, d] <- gamma_all[, d] * exp(c2 / 2)
      lambda_all[1L, d] <- lambda_all[1L, d] + c2
      beta[jj] <- b_d / exp(c2 / 2)

      # Theta reparameterization
      theta_eap[, d] <- (theta_eap[, d] - gamma0_prev[d]) * exp(c2 / 2) +
        gamma_all[1L, d]
      theta_vap[, d] <- theta_vap[, d] * exp(c2)
    }

    # Recompute fitted values
    mu_all <- x %*% gamma_all
    sigma2_all <- matrix(0, N, D)
    for (d in seq_len(D)) {
      sigma2_all[, d] <- exp(as.double(z %*% lambda_all[, d]))
    }

    # Rebuild coefficient vectors
    gamma_vec <- setNames(as.double(gamma_all), gamma_names)
    lambda_vec <- setNames(as.double(lambda_all), lambda_names)
    coefs_item_vec <- c(unlist(Map(function(a, b) c(Diff = a, Dscrmn = b),
                                    alpha, beta)))
    coef_all <- c(coefs_item_vec, gamma_vec, lambda_vec)
    se_all <- rep(NA_real_, length(coef_all))
    names(se_all) <- coef_names

    coefs <- data.frame(Estimate = coef_all, Std_Error = se_all,
                         z_value = coef_all / se_all,
                         p_value = 2 * (1 - pnorm(abs(coef_all / se_all))))
    rownames(coefs) <- coef_names
  }

  # ---- Scores data frame ----
  scores <- data.frame(row.names = seq_len(N))
  for (d in seq_len(D)) {
    scores[[paste0("post_mean_", d)]] <- theta_eap[, d]
    scores[[paste0("post_sd_", d)]] <- sqrt(theta_vap[, d])
    scores[[paste0("prior_mean_", d)]] <- mu_all[, d]
    scores[[paste0("prior_sd_", d)]] <- sqrt(sigma2_all[, d])
  }

  if (profile) timing$total <- proc.time()[["elapsed"]] - t_total_start

  # ---- Output ----
  out <- list(
    coefficients = coefs,
    scores = scores,
    R = R_final,
    item_dim = item_dim,
    D = D,
    log_Lik = log_Lik,
    constr = constr,
    N = N, J = J, H = H, ylevels = ylevels,
    p = p, q = q,
    control = con,
    se_computed = FALSE,
    timing = timing,
    call = cl
  )
  class(out) <- c("mhltm", "hIRT")
  out
}
