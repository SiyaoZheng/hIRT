#' Fitting Hierarchical Latent Trait Models (for Binary Responses).
#'
#' \code{hltm} fits a hierarchical latent trait model in which both
#' the mean and the variance of the latent preference (ability parameter)
#' may depend on person-specific covariates (\code{x} and \code{z}).
#' Specifically, the mean is specified as a linear combination of \code{x}
#' and the log of the variance is specified as a linear combination of
#' \code{z}.
#'
#' @inheritParams hgrm
#'
#' @return An object of class \code{hltm}.
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
#'  \item{timing}{Named list of timing diagnostics when \code{control$profile = TRUE}.}
#'  \item{call}{The matched call.}
#' @param compute_se Logical. If \code{TRUE} (default), compute standard
#'   errors from the observed information matrix. If \code{FALSE}, skip this
#'   step and return \code{NA} for standard-error related outputs.
#' @references Zhou, Xiang. 2019. "\href{https://doi.org/10.1017/pan.2018.63}{Hierarchical Item Response Models for Analyzing Public Opinion.}" Political Analysis.
#' @importFrom rms lrm.fit
#' @importFrom pryr compose
#' @importFrom pryr partial
#' @import stats
#' @export
#' @examples
#' y <- nes_econ2008[, -(1:3)]
#' x <- model.matrix( ~ party * educ, nes_econ2008)
#' z <- model.matrix( ~ party, nes_econ2008)
#'
#' dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
#' y[] <- lapply(y, dichotomize)
#' nes_m1 <- hltm(y, x, z)
#' nes_m1

hltm <- function(y, x = NULL, z = NULL, constr = c("latent_scale", "items"),
                 beta_set = 1L, sign_set = TRUE, init = c("naive", "glm", "tetrachoric", "irt"),
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
  y[] <- lapply(y, function(x) as.integer(x) - 1)
  if (!is.na(invalid <- match(TRUE, vapply(y, invalid_ltm, logical(1L)))))
    stop(paste(names(y)[invalid], "is not a dichotomous variable"))
  H <- vapply(y, max, double(1L), na.rm = TRUE) + 1

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
  con <- list(max_iter = 150, max_iter2 = 15, eps = 1e-03, eps2 = 1e-03, K = 25, C = 4,
              prior_mu_beta = 0, prior_sigma_beta = Inf, prior_type = "lognormal",
              prior_warmup = 0L,
              acceleration = "none", profile = FALSE, verbose = FALSE,
              lazy_varreg = 0)
  con[names(control)] <- control

  # Auto-detect prior warmup: lognormal prior with glm init needs warmup
  # because glm init gives small betas where lognormal gradient explodes
  if (identical(con[["prior_warmup"]], "auto")) {
    con[["prior_warmup"]] <- if (con[["prior_type"]] == "lognormal" && init != "irt") 20L else 0L
  }
  con[["prior_warmup"]] <- as.integer(con[["prior_warmup"]])

  profile <- isTRUE(con[["profile"]])
  timing <- NULL
  if (profile) {
    t_total_start <- proc.time()[["elapsed"]]
    t_init_start <- t_total_start
    timing <- list(
      init = 0,
      em_total = 0,
      em = list(estep = 0, mstep = 0, varreg = 0, constr = 0),
      inference = list(total = 0, reparam = 0),
      total = 0
    )
  }

  # set environments for utility functions
  environment(loglik_ltm) <- environment(theta_post_ltm) <- environment(dummy_fun_ltm) <- environment(tab2df_ltm) <- environment()

  # GL points
  K <- con[["K"]]
  theta_ls <- con[["C"]] * GLpoints[[K]][["x"]]
  qw_ls <- con[["C"]] * GLpoints[[K]][["w"]]

  # imputation
  y_imp <- y
  if(anyNA(y)) y_imp[] <- lapply(y, impute)

  # pca for initial values of theta_eap
  theta_eap <- {
    tmp <- princomp(y_imp, cor = TRUE)$scores[, 1]
    (tmp - mean(tmp, na.rm = TRUE))/sd(tmp, na.rm = TRUE)
  }

  # initialization of alpha and beta parameters
  if (init == "naive"){
    alpha <- rep(0, J)
    beta <- vapply(y, function(y) cov(y, theta_eap, use = "complete.obs")/var(theta_eap), double(1L))
  } else if (init == "glm"){
    pseudo_logit <- lapply(y_imp, function(y) glm.fit(cbind(1, theta_eap), y, family = binomial("logit"))[["coefficients"]])
    beta <- vapply(pseudo_logit, function(x) x[2L], double(1L))
    alpha <- vapply(pseudo_logit, function(x) x[1L], double(1L))
  } else if (init == "tetrachoric") {
    # mirt-style: tetrachoric correlations + factor analysis + algebraic IRT conversion
    # Digby (1983) approximation for tetrachoric correlations
    y_mat <- as.matrix(y)
    obs <- (!is.na(y_mat)) * 1.0
    y_1 <- y_mat; y_1[is.na(y_1)] <- 0
    y_0 <- (1 - y_mat) * obs
    # Pairwise 2x2 counts via BLAS (J x J matrices)
    ct_00 <- crossprod(y_0)
    ct_01 <- crossprod(y_0, y_1)
    ct_10 <- crossprod(y_1, y_0)
    ct_11 <- crossprod(y_1)
    ad <- ct_00 * ct_11
    bc <- ct_01 * ct_10
    sqrt_ad <- sqrt(pmax(ad, 0))
    sqrt_bc <- sqrt(pmax(bc, 0))
    denom <- sqrt_ad + sqrt_bc
    ratio <- sqrt_bc / denom
    ratio[!is.finite(ratio)] <- 0.5  # no information -> r_tet = 0
    r_tet <- cos(pi * ratio)
    diag(r_tet) <- 1
    # Ensure positive semi-definite and extract first factor
    eig <- eigen(r_tet, symmetric = TRUE)
    eig$values <- pmax(eig$values, 0)
    f <- eig$vectors[, 1] * sqrt(eig$values[1])
    if (mean(f) < 0) f <- -f
    f <- pmin(pmax(f, -0.999), 0.999)
    # Convert normal ogive loadings to logistic 2PL parameters
    D <- 1.702
    p_j <- colMeans(y_mat, na.rm = TRUE)
    scale_j <- 1 / sqrt(1 - f^2)
    beta <- D * f * scale_j
    alpha <- D * qnorm(p_j) * scale_j
  } else {
    ltm_coefs <- ltm(y ~ z1)[["coefficients"]]
    beta <- ltm_coefs[, 2, drop = TRUE]
    alpha <- ltm_coefs[, 1, drop = TRUE]
  }

  # initial values of gamma and lambda
  lm_opr <- tcrossprod(solve(crossprod(x)), x)
  gamma <- lm_opr %*% theta_eap
  lambda <- rep(0, q)
  fitted_mean <- as.double(x %*% gamma)
  fitted_var <- rep(1, N)

  # Pre-compute sparse representation for C++ E-step
  # Detect intercept-only case for pattern collapsing
  intercept_only <- p == 1L && q == 1L &&
      all(x[, 1L] == 1) && all(z[, 1L] == 1)
  if (intercept_only) {
    sparse_y <- build_sparse_y_patterns(y)
    N_patterns <- length(sparse_y$freq_weights)
    expand_idx <- sparse_y$expand_idx
    # Scalar fitted_mean/fitted_var replicated to N_patterns (all identical in intercept-only)
    pattern_mean <- rep(fitted_mean[1L], N_patterns)
    pattern_var  <- rep(fitted_var[1L], N_patterns)
  } else {
    sparse_y <- build_sparse_y(y)
  }
  if (profile) {
    timing$init <- proc.time()[["elapsed"]] - t_init_start
    t_em_start <- proc.time()[["elapsed"]]
  }

  # EM fixed-point mapping: takes packed params, returns list(params, log_lik)
  prior_type_int <- match(con[["prior_type"]], c("lognormal", "gaussian"), nomatch = 2L) - 1L
  item_names <- names(y)

  verbose <- isTRUE(con[["verbose"]])

  # Flag: skip constraints inside em_step_fn during SQUAREM acceleration
  squarem_active <- FALSE

  # Standalone constraint application for post-SQUAREM normalization
  apply_constraints_packed <- function(params) {
    a <- params[1:J]
    b <- params[(J+1):(2*J)]
    g <- params[(2*J+1):(2*J+p)]
    l <- params[(2*J+p+1):(2*J+p+q)]
    # location
    loc <- mean(x %*% g)
    a <- unlist(Map(function(ai, bi) ai + loc * bi, a, b))
    g[1L] <- g[1L] - loc
    # scale
    sc <- mean(z %*% l)
    g <- g / exp(sc/2)
    b <- b * exp(sc/2)
    l[1L] <- l[1L] - sc
    # direction
    if (sign_set == (b[beta_set] < 0)) {
      g <- -g
      b <- -b
    }
    c(a, b, as.double(g), as.double(l))
  }

  em_step_fn <- function(params) {
    # Unpack
    a  <- params[1:J]
    b  <- params[(J+1):(2*J)]
    g  <- params[(2*J+1):(2*J+p)]
    l  <- params[(2*J+p+1):(2*J+p+q)]
    fm <- as.double(x %*% g)
    fv <- exp(as.double(z %*% l))

    t_phases <- numeric(4)  # estep, mstep, varreg, constr

    # E-step
    t0 <- proc.time()[["elapsed"]]
    if (intercept_only) {
      pm <- rep(fm[1L], N_patterns)
      pv <- rep(fv[1L], N_patterns)
      es <- compute_estep_ltm_cpp(
          sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
          a, b, theta_ls, qw_ls, pm, pv,
          freq_weights_ = sparse_y$freq_weights
      )
    } else {
      es <- compute_estep_ltm_cpp(
          sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
          a, b, theta_ls, qw_ls, fm, fv
      )
    }
    t_phases[1] <- proc.time()[["elapsed"]] - t0
    if (profile) timing$em$estep <<- timing$em$estep + t_phases[1]
    ll <- es$log_lik
    w_mat <- es$w
    t_eap <- es$theta_eap
    t_vap <- es$theta_vap

    # M-step (with prior warmup: use Inf sigma during warmup to disable prior)
    t0 <- proc.time()[["elapsed"]]
    warmup_n <- con[["prior_warmup"]]
    sigma_eff <- if (warmup_n > 0L && n_eval < warmup_n) Inf else con[["prior_sigma_beta"]]
    ms <- compute_mstep_ltm_cpp(
        sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
        w_mat, theta_ls, a, b,
        mu_prior = con[["prior_mu_beta"]],
        sigma_prior = sigma_eff,
        prior_type = prior_type_int,
        freq_weights_ = if (intercept_only) sparse_y$freq_weights else NULL)
    t_phases[2] <- proc.time()[["elapsed"]] - t0
    if (profile) timing$em$mstep <<- timing$em$mstep + t_phases[2]
    a <- setNames(ms$alpha, item_names)
    b <- setNames(ms$beta, item_names)

    # Expand per-pattern EAP/VAP to full N for variance regression
    if (intercept_only) {
      t_eap <- t_eap[expand_idx]
      t_vap <- t_vap[expand_idx]
    }

    # Variance regression
    t0 <- proc.time()[["elapsed"]]
    n_vr_iter <- 0L
    lazy_vr_threshold <- as.integer(con[["lazy_varreg"]])
    skip_full_vr <- lazy_vr_threshold > 0L && n_eval < lazy_vr_threshold
    g <- lm_opr %*% t_eap
    r2 <- (t_eap - x %*% g)^2 + t_vap

    if (ncol(z) == 1 || skip_full_vr) {
      l <- log(mean(r2))
      if (ncol(z) > 1) l <- rep(l / ncol(z), ncol(z))
    } else {
      s2 <- glm.fit(x = z, y = r2, intercept = FALSE, family = Gamma(link = "log"))[["fitted.values"]]
      loglik_vr <- -0.5 * (log(s2) + r2/s2)
      LL0 <- sum(loglik_vr)
      for (m_vr in seq(1, con[["max_iter2"]])) {
          n_vr_iter <- n_vr_iter + 1L
          g <- lm.wfit(x, t_eap, w = 1/s2)[["coefficients"]]
          r2 <- (t_eap - x %*% g)^2 + t_vap
          var_reg <- glm.fit(x = z, y = r2, intercept = FALSE, family = Gamma(link = "log"))
          s2 <- var_reg[["fitted.values"]]
          loglik_vr <- -0.5 * (log(s2) + r2/s2)
          LL_temp <- sum(loglik_vr)
          if (LL_temp - LL0 < con[["eps2"]]) break
          LL0 <- LL_temp
      }
      l <- var_reg[["coefficients"]]
    }
    t_phases[3] <- proc.time()[["elapsed"]] - t0
    if (profile) timing$em$varreg <<- timing$em$varreg + t_phases[3]

    # Constraints (skipped during SQUAREM to preserve fixed-point map smoothness)
    t0 <- proc.time()[["elapsed"]]
    sign_flipped <- FALSE
    if (!squarem_active) {
      # location
      loc <- mean(x %*% g)
      a <- unlist(Map(function(ai, bi) ai + loc * bi, a, b))
      g[1L] <- g[1L] - loc
      # scale
      sc <- mean(z %*% l)
      g <- g / exp(sc/2)
      b <- b * exp(sc/2)
      l[1L] <- l[1L] - sc
      # direction
      if (sign_set == (b[beta_set] < 0)) {
        g <- -g
        b <- -b
        sign_flipped <- TRUE
      }
    }
    t_phases[4] <- proc.time()[["elapsed"]] - t0
    if (profile) timing$em$constr <<- timing$em$constr + t_phases[4]

    list(params = c(a, b, as.double(g), as.double(l)), log_lik = ll,
         t_phases = t_phases, n_vr_iter = n_vr_iter, sign_flipped = sign_flipped)
  }

  # Pack initial parameters
  params <- c(alpha, beta, as.double(gamma), as.double(lambda))
  n_eval <- 0L

  # EM algorithm (plain or SQUAREM)
  trace_log <- if (verbose) list() else NULL

  if (verbose) {
    cat(sprintf("\n=== hltm EM (N=%d, J=%d, K=%d, NA%%=%.1f%%) ===\n",
                N, J, K, 100 * sum(is.na(as.matrix(y))) / (N * J)))
    cat(sprintf("acceleration=%s, max_iter=%d, eps=%.1e\n",
                con$acceleration, con$max_iter, con$eps))
    cat(sprintf("prior: %s(mu=%.1f, sigma=%s), warmup=%d evals\n",
                con$prior_type, con$prior_mu_beta,
                if (is.infinite(con$prior_sigma_beta)) "Inf" else sprintf("%.1f", con$prior_sigma_beta),
                con$prior_warmup))
    cat(sprintf("constr=%s, beta_set=%d, init=%s\n", constr, beta_set, init))
    cat(sprintf("init params: alpha=[%.3f, %.3f], beta=[%.3f, %.3f], gamma=[%.3f, %.3f], lambda=[%.3f, %.3f]\n",
                min(params[1:J]), max(params[1:J]),
                min(params[(J+1):(2*J)]), max(params[(J+1):(2*J)]),
                min(params[(2*J+1):(2*J+p)]), max(params[(2*J+1):(2*J+p)]),
                min(params[(2*J+p+1):(2*J+p+q)]), max(params[(2*J+p+1):(2*J+p+q)])))
    cat("---\n")
  }

  if (con[["acceleration"]] == "squarem") {
    # SQUAREM S3 (Varadhan & Roland 2008)
    # Positive alpha convention: alpha=1 -> theta_2 (standard 2-step EM),
    # alpha>1 -> acceleration. Formula: theta_prop = theta_0 + 2*alpha*r + alpha^2*v
    step_max <- 1  # grows adaptively on success
    squarem_active <- FALSE  # constrained map: constraints applied every EM step

    converged <- FALSE
    cycle <- 0L
    while (n_eval < con[["max_iter"]]) {
      cycle <- cycle + 1L
      if (verbose) t_cycle_start <- proc.time()[["elapsed"]]

      # Step 1: F(theta_0) -> theta_1, LL(theta_0)
      if (verbose) t0 <- proc.time()[["elapsed"]]
      res1 <- em_step_fn(params)
      n_eval <- n_eval + 1L
      if (verbose) dt1 <- proc.time()[["elapsed"]] - t0
      if (!verbose) cat(".")

      theta_1 <- res1$params
      ll_0 <- res1$log_lik

      # Convergence check on beta RMSE (apply_constraints_packed is idempotent
      # when squarem_active=FALSE, kept for safety)
      params_c <- apply_constraints_packed(params)
      theta_1_c <- apply_constraints_packed(theta_1)
      beta_prev <- params_c[(J+1):(2*J)]
      beta_curr <- theta_1_c[(J+1):(2*J)]
      beta_rmse <- sqrt(mean((beta_curr - beta_prev)^2))
      if (beta_rmse < con[["eps"]]) {
        params <- theta_1
        converged <- TRUE
        if (verbose) {
          cat(sprintf("[Cycle %d] CONVERGED: beta_rmse=%.2e < eps=%.2e (eval=%d, %.1fs total)\n",
                      cycle, beta_rmse, con$eps, n_eval,
                      proc.time()[["elapsed"]] - t_cycle_start))
        } else {
          cat("\n converged at evaluation", n_eval, "\n")
        }
        break
      }

      if (n_eval >= con[["max_iter"]]) {
        params <- theta_1
        break
      }

      # Step 2: F(theta_1) -> theta_2, LL(theta_1)
      if (verbose) t0 <- proc.time()[["elapsed"]]
      res2 <- em_step_fn(theta_1)
      n_eval <- n_eval + 1L
      if (verbose) dt2 <- proc.time()[["elapsed"]] - t0
      if (!verbose) cat(".")

      theta_2 <- res2$params
      ll_1 <- res2$log_lik  # LL evaluated at theta_1

      if (n_eval >= con[["max_iter"]]) {
        params <- theta_2
        break
      }

      # SQUAREM extrapolation
      if (verbose) t0_overhead <- proc.time()[["elapsed"]]
      r <- theta_1 - params
      v <- (theta_2 - theta_1) - r   # = theta_2 - 2*theta_1 + theta_0
      sr2 <- sum(r^2)
      sv2 <- sum(v^2)

      if (sv2 < 1e-30) {
        if (verbose) {
          cat(sprintf("[Cycle %3d | eval %3d] sv2=%.2e ~ 0 -> skip extrap (already near fixed point)\n",
                      cycle, n_eval, sv2))
        }
        params <- theta_2
        next
      }

      alpha_raw <- sqrt(sr2 / sv2)
      alpha_sq <- min(step_max, max(1, alpha_raw))
      cos_rv <- sum(r * v) / (sqrt(sr2) * sqrt(sv2))

      theta_prop <- params + 2 * alpha_sq * r + alpha_sq^2 * v
      n_nonfinite <- sum(!is.finite(theta_prop))
      max_extrap <- max(abs(theta_prop - params))

      # Per-group r norms (RMS change from theta_0 to theta_1)
      r_alpha  <- sqrt(mean(r[1:J]^2))
      r_beta   <- sqrt(mean(r[(J+1):(2*J)]^2))
      r_gamma  <- sqrt(mean(r[(2*J+1):(2*J+p)]^2))
      r_lambda <- sqrt(mean(r[(2*J+p+1):(2*J+p+q)]^2))

      if (verbose) dt_overhead <- proc.time()[["elapsed"]] - t0_overhead

      # Guard: NaN/Inf check
      if (n_nonfinite > 0) {
        if (verbose) {
          cat(sprintf("[Cycle %3d | eval %3d] %d non-finite in theta_prop (alpha=%.2f, max_extrap=%.2e) -> fallback to theta_2\n",
                      cycle, n_eval, n_nonfinite, alpha_sq, max_extrap))
        }
        params <- theta_2
        step_max <- max(1, step_max / 2)
        next
      }

      # Step 3: Stabilize with F(theta_prop)
      if (verbose) t0 <- proc.time()[["elapsed"]]
      res3 <- em_step_fn(theta_prop)
      n_eval <- n_eval + 1L
      if (verbose) dt3 <- proc.time()[["elapsed"]] - t0
      if (!verbose) cat(".")

      ll_prop <- res3$log_lik

      # Monotonicity check
      step_max_before <- step_max
      accepted <- FALSE
      reject_reason <- ""
      if (!is.finite(ll_prop)) {
        reject_reason <- sprintf("ll_prop not finite (%.2f)", ll_prop)
        params <- theta_2
        step_max <- max(1, step_max / 2)
      } else if (ll_prop >= ll_0 - 1e-4) {
        accepted <- TRUE
        params <- res3$params
        if (alpha_sq >= step_max - 0.01) step_max <- 2 * alpha_sq
      } else {
        reject_reason <- sprintf("ll_prop=%.4f < ll_0=%.4f (gap=%.2e)", ll_prop, ll_0, ll_0 - ll_prop)
        params <- theta_2
        step_max <- max(1, step_max / 2)
      }

      if (verbose) {
        dt_cycle <- proc.time()[["elapsed"]] - t_cycle_start
        cat(sprintf(paste0(
          "[Cycle %3d | eval %3d | %.3fs]\n",
          "  LL: %.4f -> %.4f -> %.4f  dLL_em=%.6f\n",
          "  alpha=%.2f (raw=%.2f, max: %.1f->%.1f)  cos(r,v)=%.4f  max_extrap=%.2e\n",
          "  %s%s\n",
          "  beta_rmse=%.2e  beta=[%.4f, %.4f]\n",
          "  F-call times: %.4f + %.4f + %.4fs  overhead=%.5fs\n",
          "  F1 phases: E=%.4f M=%.4f VR=%.4f(n=%d) C=%.4f%s\n",
          "  r_rms: alpha=%.2e beta=%.2e gamma=%.2e lambda=%.2e\n"),
          cycle, n_eval, dt_cycle,
          ll_0, ll_1, ll_prop, ll_1 - ll_0,
          alpha_sq, alpha_raw, step_max_before, step_max, cos_rv, max_extrap,
          if (accepted) "ACCEPT" else "REJECT", if (!accepted) paste0(": ", reject_reason) else "",
          beta_rmse, min(params[(J+1):(2*J)]), max(params[(J+1):(2*J)]),
          dt1, dt2, dt3, dt_overhead,
          res1$t_phases[1], res1$t_phases[2], res1$t_phases[3], res1$n_vr_iter, res1$t_phases[4],
          if (res1$sign_flipped || res2$sign_flipped || res3$sign_flipped)
            sprintf("  *** SIGN FLIP in F%s ***",
                    paste(which(c(res1$sign_flipped, res2$sign_flipped, res3$sign_flipped)), collapse=","))
          else "",
          r_alpha, r_beta, r_gamma, r_lambda
        ))

        trace_log[[cycle]] <- list(
          cycle = cycle, n_eval = n_eval, t_cycle = dt_cycle,
          ll_0 = ll_0, ll_1 = ll_1, ll_prop = ll_prop, dll_em = ll_1 - ll_0,
          alpha_sq = alpha_sq, alpha_raw = alpha_raw,
          step_max_before = step_max_before, step_max_after = step_max,
          sr2 = sr2, sv2 = sv2, cos_rv = cos_rv,
          accepted = accepted, reject_reason = reject_reason,
          beta_rmse = beta_rmse,
          beta_min = min(params[(J+1):(2*J)]), beta_max = max(params[(J+1):(2*J)]),
          t_F1 = dt1, t_F2 = dt2, t_F3 = dt3, t_overhead = dt_overhead,
          t_phases_F1 = res1$t_phases, t_phases_F2 = res2$t_phases, t_phases_F3 = res3$t_phases,
          n_vr_F1 = res1$n_vr_iter, n_vr_F2 = res2$n_vr_iter, n_vr_F3 = res3$n_vr_iter,
          sign_flip_F1 = res1$sign_flipped, sign_flip_F2 = res2$sign_flipped, sign_flip_F3 = res3$sign_flipped,
          r_alpha = r_alpha, r_beta = r_beta, r_gamma = r_gamma, r_lambda = r_lambda,
          max_extrap = max_extrap, n_nonfinite = n_nonfinite
        )
      }
    }

    if (!converged && n_eval >= con[["max_iter"]]) {
      if (verbose) {
        cat(sprintf("\n*** DID NOT CONVERGE after %d evals (%d cycles). Final beta_rmse=%.2e ***\n",
                    n_eval, cycle, beta_rmse))
      }
      stop("algorithm did not converge; try increasing max_iter.")
    }

    # Polish: a few constrained EM steps to ensure final parameters are at
    # the constrained map's fixed point. With squarem_active=FALSE (constrained
    # map), this typically needs 0-2 iterations.
    params <- apply_constraints_packed(params)
    ll_polish <- -Inf
    for (polish_iter in seq_len(con[["max_iter"]] - n_eval)) {
      beta_prev_p <- params[(J+1):(2*J)]
      res_p <- em_step_fn(params)
      n_eval <- n_eval + 1L
      params <- res_p$params
      if (!verbose) cat(".")
      beta_rmse_p <- sqrt(mean((params[(J+1):(2*J)] - beta_prev_p)^2))
      ll_change <- abs(res_p$log_lik - ll_polish)
      ll_polish <- res_p$log_lik
      if (verbose) {
        cat(sprintf("[Polish %d | eval %d] LL=%.4f  beta_rmse=%.2e  dLL=%.2e\n",
                    polish_iter, n_eval, res_p$log_lik, beta_rmse_p, ll_change))
      }
      if (beta_rmse_p < con[["eps"]] || ll_change < 1e-6) break
    }

  } else if (con[["acceleration"]] == "ramsay") {
    # Ramsay (1975) acceleration applied to item parameters only.
    # Every ramsay_freq EM steps, extrapolate: theta += c * delta
    # where c = r / (1 - r) and r is the contraction ratio.
    ramsay_freq <- 3L
    ramsay_max_c <- 10  # cap acceleration factor
    converged <- FALSE
    prev_delta_norm <- NULL
    params_prev_em <- NULL

    for (iter in seq(1, con[["max_iter"]])) {
      if (verbose) t0_iter <- proc.time()[["elapsed"]]
      beta_prev <- params[(J+1):(2*J)]
      params_before <- params
      res <- em_step_fn(params)
      n_eval <- n_eval + 1L
      params <- res$params
      if (!verbose) cat(".")

      # Ramsay extrapolation on item parameters every ramsay_freq iterations
      delta_items <- params[1:(2*J)] - params_before[1:(2*J)]
      delta_norm <- sqrt(sum(delta_items^2))
      ramsay_applied <- FALSE

      if (iter > 1 && iter %% ramsay_freq == 0 && !is.null(prev_delta_norm) && prev_delta_norm > 1e-30) {
        r_ratio <- delta_norm / prev_delta_norm
        if (r_ratio < 1 && r_ratio > 0) {
          acc_c <- min(r_ratio / (1 - r_ratio), ramsay_max_c)
          # Extrapolate only item parameters (alpha, beta)
          params_acc <- params
          params_acc[1:(2*J)] <- params[1:(2*J)] + acc_c * delta_items
          # Verify: stabilize with one EM step
          res_acc <- em_step_fn(params_acc)
          n_eval <- n_eval + 1L
          if (!verbose) cat("*")
          if (is.finite(res_acc$log_lik) && res_acc$log_lik >= res$log_lik - 1e-4) {
            params <- res_acc$params
            ramsay_applied <- TRUE
            res <- res_acc  # update log_lik
          }
          # else: extrapolation made it worse, fall back to plain EM result
        }
      }
      prev_delta_norm <- delta_norm

      beta_curr <- params[(J+1):(2*J)]
      beta_rmse <- sqrt(mean((beta_curr - beta_prev)^2))

      if (verbose) {
        dt_iter <- proc.time()[["elapsed"]] - t0_iter
        ramsay_tag <- if (ramsay_applied) " [Ramsay]" else ""
        cat(sprintf("[Iter %3d | %.3fs] LL=%.4f  beta_rmse=%.2e  phases: E=%.4f M=%.4f VR=%.4f(n=%d) C=%.4f%s%s\n",
                    iter, dt_iter, res$log_lik, beta_rmse,
                    res$t_phases[1], res$t_phases[2], res$t_phases[3], res$n_vr_iter, res$t_phases[4],
                    if (res$sign_flipped) "  *** SIGN FLIP ***" else "",
                    ramsay_tag))
        trace_log[[iter]] <- list(
          iter = iter, n_eval = n_eval, t_iter = dt_iter,
          ll = res$log_lik, beta_rmse = beta_rmse,
          t_phases = res$t_phases, n_vr_iter = res$n_vr_iter,
          sign_flipped = res$sign_flipped,
          ramsay_applied = ramsay_applied,
          beta_min = min(beta_curr), beta_max = max(beta_curr)
        )
      }

      if (beta_rmse < con[["eps"]]) {
        converged <- TRUE
        if (verbose) {
          cat(sprintf("CONVERGED at iter %d (eval %d): beta_rmse=%.2e < eps=%.2e\n",
                      iter, n_eval, beta_rmse, con$eps))
        } else {
          cat("\n converged at iteration", iter, "(eval", n_eval, ")\n")
        }
        break
      }
    }
    if (!converged) {
      stop("algorithm did not converge; try increasing max_iter.")
    }

  } else {
    # Plain EM
    converged <- FALSE
    for (iter in seq(1, con[["max_iter"]])) {
      if (verbose) t0_iter <- proc.time()[["elapsed"]]
      beta_prev <- params[(J+1):(2*J)]
      res <- em_step_fn(params)
      n_eval <- n_eval + 1L
      params <- res$params
      if (!verbose) cat(".")

      beta_curr <- params[(J+1):(2*J)]
      beta_rmse <- sqrt(mean((beta_curr - beta_prev)^2))

      if (verbose) {
        dt_iter <- proc.time()[["elapsed"]] - t0_iter
        cat(sprintf("[Iter %3d | %.3fs] LL=%.4f  beta_rmse=%.2e  phases: E=%.4f M=%.4f VR=%.4f(n=%d) C=%.4f%s\n",
                    iter, dt_iter, res$log_lik, beta_rmse,
                    res$t_phases[1], res$t_phases[2], res$t_phases[3], res$n_vr_iter, res$t_phases[4],
                    if (res$sign_flipped) "  *** SIGN FLIP ***" else ""))
        trace_log[[iter]] <- list(
          iter = iter, n_eval = n_eval, t_iter = dt_iter,
          ll = res$log_lik, beta_rmse = beta_rmse,
          t_phases = res$t_phases, n_vr_iter = res$n_vr_iter,
          sign_flipped = res$sign_flipped,
          beta_min = min(beta_curr), beta_max = max(beta_curr)
        )
      }

      if (beta_rmse < con[["eps"]]) {
        converged <- TRUE
        if (verbose) {
          cat(sprintf("CONVERGED at iter %d: beta_rmse=%.2e < eps=%.2e\n", iter, beta_rmse, con$eps))
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

  if (verbose) {
    cat(sprintf("\n=== EM finished: %d evals, converged=%s ===\n", n_eval, converged))
  }

  # Unpack final parameters
  alpha  <- setNames(params[1:J], item_names)
  beta   <- setNames(params[(J+1):(2*J)], item_names)
  gamma  <- params[(2*J+1):(2*J+p)]
  lambda <- params[(2*J+p+1):(2*J+p+q)]
  fitted_mean <- as.double(x %*% gamma)
  fitted_var  <- exp(as.double(z %*% lambda))

  # Final E-step for theta_eap/theta_vap
  if (intercept_only) {
    pm <- rep(fitted_mean[1L], N_patterns)
    pv <- rep(fitted_var[1L], N_patterns)
    final_estep <- compute_estep_ltm_cpp(
        sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
        alpha, beta, theta_ls, qw_ls, pm, pv,
        freq_weights_ = sparse_y$freq_weights
    )
    theta_eap <- final_estep$theta_eap[expand_idx]
    theta_vap <- final_estep$theta_vap[expand_idx]
  } else {
    final_estep <- compute_estep_ltm_cpp(
        sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
        alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
    )
    theta_eap <- final_estep$theta_eap
    theta_vap <- final_estep$theta_vap
  }

  if (profile) {
    timing$em_total <- proc.time()[["elapsed"]] - t_em_start
    timing$em_evals <- n_eval
  }

  gamma <- setNames(as.double(gamma), paste("x", colnames(x), sep = ""))
  lambda <- setNames(as.double(lambda), paste("z", colnames(z), sep = ""))

  # inference (OPG information matrix via R/BLAS)
  if (profile) t_infer <- proc.time()[["elapsed"]]

  environment(dalpha_ltm) <- environment(sj_ab_ltm) <-
    environment(si_gamma) <- environment(si_lambda) <- environment()

  # Marginal likelihood components
  pik <- matrix(unlist(Map(partial(dnorm, x = theta_ls), mean = fitted_mean,
                           sd = sqrt(fitted_var))),
                N, K, byrow = TRUE) * matrix(qw_ls, N, K, byrow = TRUE)
  Lijk <- lapply(theta_ls, function(theta_k) exp(loglik_ltm(alpha, beta, rep(theta_k, N))))
  Lik <- vapply(Lijk, compose(exp, partial(rowSums, na.rm = TRUE), log), double(N))
  Li <- rowSums(Lik * pik)
  log_Lik <- sum(log(Li))

  names_ab <- paste(rep(names(alpha), each = 2), c("Diff", "Dscrmn"))
  coef_names <- c(names_ab, names(gamma), names(lambda))
  free_coef_names <- c(names_ab[-c(1L, length(names_ab))], names(gamma), names(lambda))

  if (compute_se) {
    # Score vectors: item parameters (BLAS matrix multiply per item)
    dalpha <- dalpha_ltm(alpha, beta)
    s_ab <- unname(Reduce(cbind, lapply(1:J, sj_ab_ltm)))

    # Score vectors: covariate parameters
    s_gamma <- vapply(1:N, si_gamma, double(p))
    s_lambda <- vapply(1:N, si_lambda, double(q))

    # OPG information matrix (BLAS-accelerated tcrossprod)
    s_all <- rbind(t(s_ab)[-c(1L, ncol(s_ab)), , drop = FALSE], s_gamma, s_lambda)
    s_all[is.na(s_all)] <- 0
    covmat <- tryCatch(
      solve(tcrossprod(s_all)),
      error = function(e) {
        warning("The information matrix is singular; SE calculation failed.")
        matrix(NA, nrow(s_all), nrow(s_all))
      }
    )
    rownames(covmat) <- colnames(covmat) <- free_coef_names
    se_free <- sqrt(diag(covmat))

    # reorganize se_all
    sH <- 2 * J
    gamma_indices <- (sH - 1):(sH + p - 2)
    lambda_indices <- (sH + p - 1):(sH + p + q - 2)
    se_all <- c(NA, se_free[1:(sH - 2)], NA, se_free[gamma_indices], se_free[lambda_indices])
    names(se_all) <- coef_names
  } else {
    covmat <- matrix(NA_real_, length(free_coef_names), length(free_coef_names))
    rownames(covmat) <- colnames(covmat) <- free_coef_names
    se_all <- rep(NA_real_, length(coef_names))
    names(se_all) <- coef_names
  }

  if (profile) timing$inference$total <- proc.time()[["elapsed"]] - t_infer

  # item coefficients
  coefs_item <- Map(function(a, b) c(Diff = a, Dscrmn = b), alpha, beta)

  # all coefficients
  coef_all <- c(unlist(coefs_item), gamma, lambda)
  coefs <- data.frame(Estimate = coef_all, Std_Error = se_all, z_value = coef_all/se_all,
                      p_value = 2 * (1 - pnorm(abs(coef_all/se_all))))
  rownames(coefs) <- names(se_all)

  # item constraints
  if (constr == "items"){
    if (profile) t_reparam <- proc.time()[["elapsed"]]

    gamma0_prev <- gamma[1L]

    # location constraint
    alpha_sum <- sum(alpha)
    beta_sum <- sum(beta)
    c1 <- alpha_sum/beta_sum
    gamma[1L] <- gamma[1L] + c1  # adjust gamma0
    alpha <- unlist(Map(function(x, y) x - c1 * y, alpha, beta))

    # scale constraint
    c2 <- 2 * mean(log(abs(beta)))
    gamma <- gamma * exp(c2/2)
    lambda[1L] <- lambda[1L] + c2
    beta <- beta / exp(c2/2)

    # fitted means and variances
    fitted_mean <- as.double(x %*% gamma)
    fitted_var <- exp(as.double(z %*% lambda))

    # theta_eap and theta_vap
    theta_eap <- (theta_eap - gamma0_prev) * exp(c2/2) + gamma[1L]
    theta_vap <- theta_vap * (exp(c2/2))^2

    names_ab <- paste(rep(names(alpha), each = 2), c("Diff", "Dscrmn"))
    coef_names <- c(names_ab, names(gamma), names(lambda))
    free_coef_names <- c(names_ab[-c(1L, length(names_ab))], names(gamma), names(lambda))

    if (compute_se) {
      # covmat for new parameterization
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

      se_free <- sqrt(Matrix::diag(covmat))

      # reorganize se_all
      sH <- 2 * J
      gamma_indices <- (sH - 1):(sH + p - 2)
      lambda_indices <- (sH + p - 1):(sH + p + q - 2)
      se_all <- c(NA, se_free[1:(sH - 2)], NA, se_free[gamma_indices], se_free[lambda_indices])
      names(se_all) <- coef_names
      rownames(covmat) <- colnames(covmat) <- free_coef_names
    } else {
      covmat <- matrix(NA_real_, length(free_coef_names), length(free_coef_names))
      rownames(covmat) <- colnames(covmat) <- free_coef_names
      se_all <- rep(NA_real_, length(coef_names))
      names(se_all) <- coef_names
    }

    # item coefficients
    coefs_item <- Map(function(a, b) c(Diff = a, Dscrmn = b), alpha, beta)

    # all coefficients
    coef_all <- c(unlist(coefs_item), gamma, lambda)
    coefs <- data.frame(Estimate = coef_all, Std_Error = se_all, z_value = coef_all/se_all,
                        p_value = 2 * (1 - pnorm(abs(coef_all/se_all))))
    rownames(coefs) <- names(se_all)
    if (profile) timing$inference$reparam <- proc.time()[["elapsed"]] - t_reparam
  }

  # ability parameter estimates
  theta <- data.frame(post_mean = theta_eap, post_sd = sqrt(theta_vap),
                      prior_mean = fitted_mean, prior_sd = sqrt(fitted_var))

  if (profile) {
    timing$total <- proc.time()[["elapsed"]] - t_total_start
  }

  # output
  out <- list(coefficients = coefs, scores = theta, vcov = covmat, log_Lik = log_Lik, constr = constr,
              N = N, J = J, H = H, ylevels = ylevels, p = p, q = q,
              control = con, se_computed = compute_se, timing = timing,
              trace = trace_log, call = cl)
  class(out) <- c("hltm", "hIRT")
  out
}
