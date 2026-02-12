#' Fitting Hierarchical Graded Response Models (for Ordinal Responses)
#'
#' \code{hgrm} fits a hierarchical graded response model in which both
#' the mean and the variance of the latent preference (ability parameter)
#' may depend on person-specific covariates (\code{x} and \code{z}).
#' Specifically, the mean is specified as a linear combination of \code{x}
#' and the log of the variance is specified as a linear combination of
#' \code{z}. Nonresponses are treated as missing at random.
#'
#' @param y A data frame or matrix of item responses.
#' @param x An optional model matrix, including the intercept term, that predicts the
#'   mean of the latent preference. If not supplied, only the intercept term is included.
#' @param z An optional model matrix, including the intercept term, that predicts the
#'   variance of the latent preference. If not supplied, only the intercept term is included.
#' @param constr The type of constraints used to identify the model: "latent_scale",
#'   or "items". The default, "latent_scale" constrains the mean of latent preferences
#'   to zero and the geometric mean of prior variance to one; "items" places constraints
#'   on item parameters instead and sets the mean of item difficulty parameters to zero
#'   and the geometric mean of the discrimination parameters to one.
#' @param beta_set The index of the item for which the discrimination parameter is
#'   restricted to be positive (or negative). It may take any integer value from
#'   1 to \code{ncol(y)}.
#' @param sign_set Logical. Should the discrimination parameter of
#'   the corresponding item (indexed by \code{beta_set}) be positive
#'   (if \code{TRUE}) or negative (if \code{FALSE})?
#' @param init A character string indicating how item parameters are initialized. It can be
#'   "glm" or "irt".
#' @param control A list of control values
#' \describe{
#'  \item{max_iter}{The maximum number of EM F-evaluations.
#'    The default is 300.}
#'  \item{eps}{Tolerance parameter used to determine convergence of the
#'   EM algorithm. Specifically, iterations continue until the RMSE of
#'   beta changes falls under \code{eps}.
#'   \code{eps}=1e-3 by default.}
#'  \item{max_iter2}{The maximum number of iterations of the conditional
#'    maximization procedures for updating \eqn{\gamma} and \eqn{\lambda}.
#'    The default is 15.}
#'  \item{eps2}{Tolerance parameter used to determine convergence of the
#'    conditional maximization procedures for updating \eqn{\gamma} and
#'    \eqn{\lambda}. \code{eps2}=1e-3 by default.}
#'  \item{K}{Number of Gauss-Legendre quadrature points for the E-step. The default is 25.}
#'  \item{C}{[-C, C] sets the range of integral in the E-step. \code{C}=4 by default.}
#'  \item{acceleration}{"squarem" (default) or "none".}
#'  \item{prior_sigma_beta}{Prior SD for discrimination parameters. Default 1.0.}
#'  \item{prior_mu_beta}{Prior mean for discrimination (log scale if lognormal). Default 0.}
#'  \item{prior_type}{"lognormal" (default) or "gaussian".}
#'  \item{prior_warmup}{Number of F-evals before activating prior. "auto" or integer.}
#' }
#' @param compute_se Logical. If \code{TRUE} (default), compute standard errors.
#'
#' @return An object of class \code{hgrm}.
#'  \item{coefficients}{A data frame of parameter estimates, standard errors,
#'   z values and p values.}
#'  \item{scores}{A data frame of EAP estimates of latent preferences and
#'   their approximate standard errors.}
#'  \item{vcov}{Variance-covariance matrix of parameter estimates.}
#'  \item{log_Lik}{The log-likelihood value at convergence.}
#'  \item{N}{Number of units.}
#'  \item{J}{Number of items.}
#'  \item{H}{A vector denoting the number of response categories for each item.}
#'  \item{ylevels}{A list showing the levels of the factorized response categories.}
#'  \item{p}{The number of predictors for the mean equation.}
#'  \item{q}{The number of predictors for the variance equation.}
#'  \item{control}{List of control values.}
#'  \item{se_computed}{Logical, whether standard errors were computed.}
#'  \item{call}{The matched call.}
#' @references Zhou, Xiang. 2019. "\href{https://doi.org/10.1017/pan.2018.63}{Hierarchical Item Response Models for Analyzing Public Opinion.}" Political Analysis.
#' @importFrom rms lrm.fit
#' @importFrom pryr compose
#' @importFrom pryr partial
#' @importFrom ltm grm
#' @importFrom ltm ltm
#' @import stats
#' @export
#' @examples
#' y <- nes_econ2008[, -(1:3)]
#' x <- model.matrix( ~ party * educ, nes_econ2008)
#' z <- model.matrix( ~ party, nes_econ2008)
#' nes_m1 <- hgrm(y, x, z)
#' nes_m1

hgrm <- function(y, x = NULL, z = NULL, constr = c("latent_scale", "items"),
                 beta_set = 1L, sign_set = TRUE, init = c("glm", "irt"),
                 control = list(), compute_se = TRUE) {

  # match call
  cl <- match.call()

  # check y and convert y into data.frame if needed
  if(missing(y)) stop("`y` must be provided.")
  if ((!is.data.frame(y) && !is.matrix(y)) || ncol(y) == 1L)
    stop("'y' must be either a data.frame or a matrix with at least two columns.")
  if(is.matrix(y)) y <- as.data.frame(y)

  # number of units and items
  N <- nrow(y)
  J <- ncol(y)

  # convert each y_j into an integer vector
  y[] <- lapply(y, factor, exclude = c(NA, NaN))
  ylevels <- lapply(y, levels)
  y[] <- lapply(y, as.integer)
  if (!is.na(invalid <- match(TRUE, vapply(y, invalid_grm, logical(1L)))))
    stop(paste(names(y)[invalid], "does not have at least two valid responses"))
  H <- vapply(y, max, integer(1L), na.rm = TRUE)

  # check x and z (x and z should contain an intercept column)
  x <- x %||% as.matrix(rep(1, N))
  z <- z %||% as.matrix(rep(1, N))
  if (!is.matrix(x)) stop("`x` must be a matrix.")
  if (!is.matrix(z)) stop("`z` must be a matrix.")
  if (nrow(x) != N || nrow(z) != N) stop("both 'x' and 'z' must have the same number of rows as 'y'")
  p <- ncol(x)
  q <- ncol(z)
  colnames(x) <- colnames(x) %||% paste0("x", 1:p)
  colnames(z) <- colnames(z) %||% paste0("x", 1:q)

  # check beta_set and sign_set
  stopifnot(beta_set %in% 1:J, is.logical(sign_set))
  if (!is.logical(compute_se) || length(compute_se) != 1L || is.na(compute_se))
    stop("`compute_se` must be TRUE or FALSE.")

  # check constraint
  constr <- match.arg(constr)
  init <- match.arg(init)

  # control parameters
  con <- list(max_iter = 300, max_iter2 = 15, eps = 1e-03, eps2 = 1e-03, K = 25, C = 4,
              prior_mu_beta = 0, prior_sigma_beta = 1.0, prior_type = "lognormal",
              prior_warmup = "auto",
              acceleration = "squarem", verbose = FALSE)
  con[names(control)] <- control

  # Auto-detect prior warmup
  if (identical(con[["prior_warmup"]], "auto")) {
    con[["prior_warmup"]] <- if (con[["prior_type"]] == "lognormal" && init != "irt") 20L else 0L
  }
  con[["prior_warmup"]] <- as.integer(con[["prior_warmup"]])

  # set environments for utility functions (needed for inference)
  environment(loglik_grm) <- environment(theta_post_grm) <- environment(dummy_fun_grm) <- environment(tab2df_grm) <- environment()

  # GL points
  K <- con[["K"]]
  theta_ls <- con[["C"]] * GLpoints[[K]][["x"]]
  qw_ls <- con[["C"]] * GLpoints[[K]][["w"]]

  # imputation
  y_imp <- y
  if(anyNA(y)) y_imp[] <- lapply(y, impute)

  # pca
  theta_eap <- {
    tmp <- princomp(y_imp, cor = TRUE)$scores[, 1]
    (tmp - mean(tmp, na.rm = TRUE))/sd(tmp, na.rm = TRUE)
  }

  # initialization of alpha and beta parameters
  if (init == "glm"){

    pseudo_lrm <- lapply(y_imp, function(y) lrm.fit(theta_eap, y)[["coefficients"]])
    beta <- vapply(pseudo_lrm, function(x) x[[length(x)]], double(1L))
    alpha <- lapply(pseudo_lrm, function(x) c(Inf, x[-length(x)], -Inf))

  } else {

    grm_coefs <- grm(y)[["coefficients"]]
    beta <- vapply(grm_coefs, function(x) x[[length(x)]], double(1L))
    alpha <- lapply(grm_coefs, function(x) c(Inf, rev(x[-length(x)]), -Inf))

  }

  # initial values of gamma and lambda
  lm_opr <- tcrossprod(solve(crossprod(x)), x)
  gamma <- lm_opr %*% theta_eap
  lambda <- rep(0, q)
  fitted_mean <- as.double(x %*% gamma)
  fitted_var <- rep(1, N)

  # Pre-compute sparse representation and flat alpha for C++
  sparse_y <- build_sparse_y(y)
  af_obj <- flatten_alpha_grm(alpha, H)
  alpha_offsets <- af_obj$alpha_offsets
  H_int <- as.integer(H)
  n_alpha_flat <- sum(H - 1L)

  # Prior type integer
  prior_type_int <- match(con[["prior_type"]], c("lognormal", "gaussian"), nomatch = 2L) - 1L
  item_names <- names(y)

  verbose <- isTRUE(con[["verbose"]])

  # Flag for SQUAREM constraint skipping
  squarem_active <- FALSE

  # Standalone constraint application for packed params
  apply_constraints_packed <- function(params) {
    af <- params[1:n_alpha_flat]
    b  <- params[(n_alpha_flat + 1):(n_alpha_flat + J)]
    g  <- params[(n_alpha_flat + J + 1):(n_alpha_flat + J + p)]
    l  <- params[(n_alpha_flat + J + p + 1):(n_alpha_flat + J + p + q)]
    # location
    loc <- mean(x %*% g)
    shifts <- rep(b * loc, times = H - 1L)
    af <- af + shifts
    g[1L] <- g[1L] - loc
    # scale
    sc <- mean(z %*% l)
    g <- g / exp(sc / 2)
    b <- b * exp(sc / 2)
    l[1L] <- l[1L] - sc
    # direction
    if (sign_set == (b[beta_set] < 0)) {
      g <- -g
      b <- -b
    }
    c(af, b, as.double(g), as.double(l))
  }

  # EM fixed-point mapping
  em_step_fn <- function(params) {
    # Unpack
    af <- params[1:n_alpha_flat]
    b  <- params[(n_alpha_flat + 1):(n_alpha_flat + J)]
    g  <- params[(n_alpha_flat + J + 1):(n_alpha_flat + J + p)]
    l  <- params[(n_alpha_flat + J + p + 1):(n_alpha_flat + J + p + q)]
    fm <- as.double(x %*% g)
    fv <- exp(as.double(z %*% l))

    # E-step (C++)
    es <- compute_estep_grm_cpp(
        sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
        af, alpha_offsets, H_int, b, theta_ls, qw_ls, fm, fv
    )
    ll <- es$log_lik
    w_mat <- es$w
    t_eap <- es$theta_eap
    t_vap <- es$theta_vap

    # M-step (C++)
    warmup_n <- con[["prior_warmup"]]
    sigma_eff <- if (warmup_n > 0L && n_eval < warmup_n) Inf else con[["prior_sigma_beta"]]
    ms <- compute_mstep_grm_cpp(
        sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
        w_mat, theta_ls, af, alpha_offsets, H_int, b,
        mu_prior = con[["prior_mu_beta"]],
        sigma_prior = sigma_eff,
        prior_type = prior_type_int
    )
    af <- ms$alpha_flat
    b  <- ms$beta

    # Variance regression
    g <- lm_opr %*% t_eap
    r2 <- (t_eap - x %*% g)^2 + t_vap
    if (ncol(z) == 1) {
      l <- log(mean(r2))
    } else {
      s2 <- glm.fit(x = z, y = r2, intercept = FALSE, family = Gamma(link = "log"))[["fitted.values"]]
      loglik_vr <- -0.5 * (log(s2) + r2 / s2)
      LL0 <- sum(loglik_vr)
      for (m_vr in seq(1, con[["max_iter2"]])) {
        g <- lm.wfit(x, t_eap, w = 1 / s2)[["coefficients"]]
        r2 <- (t_eap - x %*% g)^2 + t_vap
        var_reg <- glm.fit(x = z, y = r2, intercept = FALSE, family = Gamma(link = "log"))
        s2 <- var_reg[["fitted.values"]]
        loglik_vr <- -0.5 * (log(s2) + r2 / s2)
        LL_temp <- sum(loglik_vr)
        if (LL_temp - LL0 < con[["eps2"]]) break
        LL0 <- LL_temp
      }
      l <- var_reg[["coefficients"]]
    }

    # Constraints (skipped during SQUAREM)
    if (!squarem_active) {
      # location
      loc <- mean(x %*% g)
      shifts <- rep(b * loc, times = H - 1L)
      af <- af + shifts
      g[1L] <- g[1L] - loc
      # scale
      sc <- mean(z %*% l)
      g <- g / exp(sc / 2)
      b <- b * exp(sc / 2)
      l[1L] <- l[1L] - sc
      # direction
      if (sign_set == (b[beta_set] < 0)) {
        g <- -g
        b <- -b
      }
    }

    list(params = c(af, b, as.double(g), as.double(l)), log_lik = ll)
  }

  # Pack initial parameters
  params <- c(af_obj$alpha_flat, beta, as.double(gamma), as.double(lambda))
  n_eval <- 0L

  if (con[["acceleration"]] == "squarem") {
    # SQUAREM S3 (Varadhan & Roland 2008)
    step_max <- 1
    squarem_active <- FALSE
    converged <- FALSE
    cycle <- 0L

    while (n_eval < con[["max_iter"]]) {
      cycle <- cycle + 1L

      # Step 1: F(theta_0) -> theta_1
      res1 <- em_step_fn(params)
      n_eval <- n_eval + 1L
      cat(".")

      theta_1 <- res1$params
      ll_0 <- res1$log_lik

      # Convergence check on beta RMSE
      params_c  <- apply_constraints_packed(params)
      theta_1_c <- apply_constraints_packed(theta_1)
      beta_prev <- params_c[(n_alpha_flat + 1):(n_alpha_flat + J)]
      beta_curr <- theta_1_c[(n_alpha_flat + 1):(n_alpha_flat + J)]
      beta_rmse <- sqrt(mean((beta_curr - beta_prev)^2))
      if (beta_rmse < con[["eps"]]) {
        params <- theta_1
        converged <- TRUE
        cat("\n converged at evaluation", n_eval, "\n")
        break
      }

      if (n_eval >= con[["max_iter"]]) { params <- theta_1; break }

      # Step 2: F(theta_1) -> theta_2
      res2 <- em_step_fn(theta_1)
      n_eval <- n_eval + 1L
      cat(".")

      theta_2 <- res2$params

      if (n_eval >= con[["max_iter"]]) { params <- theta_2; break }

      # SQUAREM extrapolation
      r <- theta_1 - params
      v <- (theta_2 - theta_1) - r
      sr2 <- sum(r^2)
      sv2 <- sum(v^2)

      if (sv2 < 1e-30) { params <- theta_2; next }

      alpha_raw <- sqrt(sr2 / sv2)
      alpha_sq <- min(step_max, max(1, alpha_raw))
      theta_prop <- params + 2 * alpha_sq * r + alpha_sq^2 * v

      if (any(!is.finite(theta_prop))) {
        params <- theta_2
        step_max <- max(1, step_max / 2)
        next
      }

      # Step 3: Stabilize with F(theta_prop)
      res3 <- em_step_fn(theta_prop)
      n_eval <- n_eval + 1L
      cat(".")

      ll_prop <- res3$log_lik

      # Monotonicity check
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
    }

    if (!converged && n_eval >= con[["max_iter"]]) {
      stop("algorithm did not converge; try increasing max_iter.")
    }

    # Polish: constrained EM steps
    params <- apply_constraints_packed(params)
    ll_polish <- -Inf
    for (polish_iter in seq_len(con[["max_iter"]] - n_eval)) {
      beta_prev_p <- params[(n_alpha_flat + 1):(n_alpha_flat + J)]
      res_p <- em_step_fn(params)
      n_eval <- n_eval + 1L
      params <- res_p$params
      cat(".")
      beta_rmse_p <- sqrt(mean((params[(n_alpha_flat + 1):(n_alpha_flat + J)] - beta_prev_p)^2))
      ll_change <- abs(res_p$log_lik - ll_polish)
      ll_polish <- res_p$log_lik
      if (beta_rmse_p < con[["eps"]] || ll_change < 1e-6) break
    }

  } else {
    # Plain EM
    converged <- FALSE
    for (iter in seq(1, con[["max_iter"]])) {
      beta_prev <- params[(n_alpha_flat + 1):(n_alpha_flat + J)]
      res <- em_step_fn(params)
      n_eval <- n_eval + 1L
      params <- res$params
      cat(".")

      beta_curr <- params[(n_alpha_flat + 1):(n_alpha_flat + J)]
      beta_rmse <- sqrt(mean((beta_curr - beta_prev)^2))

      if (beta_rmse < con[["eps"]]) {
        converged <- TRUE
        cat("\n converged at iteration", iter, "\n")
        break
      }
    }
    if (!converged) {
      stop("algorithm did not converge; try increasing `max_iter` or decreasing `eps`")
    }
  }

  # Unpack final parameters
  alpha_flat_final <- params[1:n_alpha_flat]
  beta   <- setNames(params[(n_alpha_flat + 1):(n_alpha_flat + J)], item_names)
  gamma  <- params[(n_alpha_flat + J + 1):(n_alpha_flat + J + p)]
  lambda <- params[(n_alpha_flat + J + p + 1):(n_alpha_flat + J + p + q)]
  alpha  <- unflatten_alpha_grm(alpha_flat_final, alpha_offsets, H)
  names(alpha) <- item_names

  fitted_mean <- as.double(x %*% gamma)
  fitted_var  <- exp(as.double(z %*% lambda))

  # Final E-step for theta_eap/theta_vap
  final_es <- compute_estep_grm_cpp(
      sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
      alpha_flat_final, alpha_offsets, H_int, beta, theta_ls, qw_ls,
      fitted_mean, fitted_var
  )
  theta_eap <- final_es$theta_eap
  theta_vap <- final_es$theta_vap

  gamma <- setNames(as.double(gamma), paste("x", colnames(x), sep = ""))
  lambda <- setNames(as.double(lambda), paste("z", colnames(z), sep = ""))

  # inference
  pik <- matrix(unlist(Map(partial(dnorm, x = theta_ls), mean = fitted_mean, sd = sqrt(fitted_var))),
                N, K, byrow = TRUE) * matrix(qw_ls, N, K, byrow = TRUE)
  Lijk <- lapply(theta_ls, function(theta_k) exp(loglik_grm(alpha = alpha, beta = beta, rep(theta_k, N))))  # K-list
  Lik <- vapply(Lijk, compose(exp, partial(rowSums, na.rm = TRUE), log), double(N))
  Li <- rowSums(Lik * pik)

  # log likelihood
  log_Lik <- sum(log(Li))

  # Name structures
  # Add threshold names to alpha list elements
  for (jj in seq_along(alpha)) {
    tmp <- alpha[[jj]]
    nms <- c(paste0("y>=", seq(2, length(tmp) - 1)), "Dscrmn")
    # Only name the interior thresholds + Dscrmn for naming consistency
  }

  names_ab <- unlist(lapply(names(alpha), function(x) {
    tmp <- alpha[[x]]
    paste(x, c(names(tmp)[-c(1L, length(tmp))], "Dscrmn"))
  }))
  # If thresholds aren't named, use y>=h format
  if (is.null(names(alpha[[1]])) || all(is.na(names(alpha[[1]])))) {
    names_ab <- unlist(lapply(seq_along(alpha), function(jj) {
      nm <- item_names[jj]
      H_j <- H[jj]
      paste(nm, c(paste0("y>=", seq(2, H_j)), "Dscrmn"))
    }))
  }

  coef_names <- c(names_ab, names(gamma), names(lambda))

  if (compute_se) {
    # outer product of gradients
    environment(sj_ab_grm) <- environment(si_gamma) <- environment(si_lambda) <- environment()
    s_ab <- unname(Reduce(rbind, lapply(1:J, sj_ab_grm)))
    s_gamma <- vapply(1:N, si_gamma, double(p))
    s_lambda <- vapply(1:N, si_lambda, double(q))

    # covariance matrix and standard errors
    s_all <- rbind(s_ab[-c(1L, nrow(s_ab)), , drop = FALSE], s_gamma, s_lambda)
    s_all[is.na(s_all)] <- 0
    covmat <- tryCatch(solve(tcrossprod(s_all)),
                       error = function(e) {warning("The information matrix is singular; SE calculation failed.");
                         matrix(NA, nrow(s_all), nrow(s_all))})
    se_all <- sqrt(diag(covmat))

    # reorganize se_all
    sH <- sum(H)
    gamma_indices <- (sH - 1):(sH + p - 2)
    lambda_indices <- (sH + p - 1):(sH + p + q - 2)
    se_all <- c(NA, se_all[1:(sH-2)], NA, se_all[gamma_indices], se_all[lambda_indices])
    names(se_all) <- coef_names

    free_coef_names <- coef_names[-c(1L, sH)]
    rownames(covmat) <- colnames(covmat) <- free_coef_names
  } else {
    sH <- sum(H)
    free_coef_names <- coef_names[-c(1L, sH)]
    covmat <- matrix(NA_real_, length(free_coef_names), length(free_coef_names))
    rownames(covmat) <- colnames(covmat) <- free_coef_names
    se_all <- rep(NA_real_, length(coef_names))
    names(se_all) <- coef_names
  }

  # item coefficients
  coef_item <- Map(function(a, b) c(a[-c(1L, length(a))], Dscrmn = b), alpha, beta)

  # all coefficients
  coef_all <- c(unlist(coef_item), gamma, lambda)
  coefs <- data.frame(Estimate = coef_all, Std_Error = se_all, z_value = coef_all/se_all,
                      p_value = 2 * (1 - pnorm(abs(coef_all/se_all))))
  rownames(coefs) <- names(se_all)

  # item constraints
  if (constr == "items"){

    gamma0_prev <- gamma[[1L]]

    # location constraint
    alpha_sum <- sum(vapply(alpha, function(x) sum(x[-c(1L, length(x))]), double(1L)))
    beta_sum <- sum((H-1) * beta)
    c1 <- alpha_sum/beta_sum
    gamma[[1L]] <- gamma[[1L]] + c1
    alpha <- Map(function(x, y) x - c1 * y, alpha, beta)

    # scale constraint
    c2 <- 2 * mean(log(abs(beta)))
    gamma <- gamma * exp(c2/2)
    lambda[[1L]] <- lambda[[1L]] + c2
    beta <- beta / exp(c2/2)

    # fitted means and variances
    fitted_mean <- as.double(x %*% gamma)
    fitted_var <- exp(as.double(z %*% lambda))

    # theta_eap and theta_vap
    theta_eap <- (theta_eap - gamma0_prev) * exp(c2/2) + gamma[[1L]]
    theta_vap <- theta_vap * (exp(c2/2))^2

    # covmat for new parameterization
    if (compute_se) {
      tmp_fun <- function(d) {
        mat <- diag(d)
        mat[d, d] <- exp(-c2/2)
        mat[1:(d-1), d] <- rep(-c1, d-1)
        mat
      }
      A <- Reduce(Matrix::bdiag, lapply(H, tmp_fun))
      A2 <- A[seq(2, nrow(A)-1), seq(2, ncol(A)-1)]
      B <- Matrix::bdiag(exp(c2/2) * diag(p), diag(q))
      C <- Matrix::bdiag(A2, B)
      covmat <- C %*% Matrix::tcrossprod(covmat, C)

      se_all <- sqrt(Matrix::diag(covmat))
    }

    # reorganize se_all
    sH <- sum(H)
    gamma_indices <- (sH - 1):(sH + p - 2)
    lambda_indices <- (sH + p - 1):(sH + p + q - 2)
    if (compute_se) {
      se_all <- c(NA, se_all[1:(sH-2)], NA, se_all[gamma_indices], se_all[lambda_indices])
    } else {
      se_all <- rep(NA_real_, sH + p + q)
    }

    # name se_all and covmat
    names_ab <- unlist(lapply(names(alpha), function(x) {
      tmp <- alpha[[x]]
      paste(x, c(names(tmp)[-c(1L, length(tmp))], "Dscrmn"))
    }))
    if (is.null(names(alpha[[1]])) || all(is.na(names(alpha[[1]])))) {
      names_ab <- unlist(lapply(seq_along(alpha), function(jj) {
        nm <- item_names[jj]
        H_j <- H[jj]
        paste(nm, c(paste0("y>=", seq(2, H_j)), "Dscrmn"))
      }))
    }
    names(se_all) <- c(names_ab, names(gamma), names(lambda))
    if (compute_se) {
      rownames(covmat) <- colnames(covmat) <- c(names_ab[-c(1L, length(names_ab))], names(gamma), names(lambda))
    }

    # item coefficients
    coef_item <- Map(function(a, b) c(a[-c(1L, length(a))], Dscrmn = b), alpha, beta)

    # all coefficients
    coef_all <- c(unlist(coef_item), gamma, lambda)
    coefs <- data.frame(Estimate = coef_all, Std_Error = se_all, z_value = coef_all/se_all,
                        p_value = 2 * (1 - pnorm(abs(coef_all/se_all))))
    rownames(coefs) <- names(se_all)
  }

  # ability parameter estimates
  theta <- data.frame(post_mean = theta_eap, post_sd = sqrt(theta_vap),
                      prior_mean = fitted_mean, prior_sd = sqrt(fitted_var))

  # output
  out <- list(coefficients = coefs, scores = theta, vcov = covmat, log_Lik = log_Lik, constr = constr,
              N = N, J = J, H = H, ylevels = ylevels, p = p, q = q,
              control = con, se_computed = compute_se, call = cl)
  class(out) <- c("hgrm", "hIRT")
  out
}
