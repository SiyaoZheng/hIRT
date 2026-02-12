#' Hierarchical Latent Trait Models with Known Item Parameters.
#'
#' \code{hltm2} fits a hierarchical latent trait model where the item parameters
#'   are known and supplied by the user.
#'
#' @inheritParams hgrm2
#' @param item_coefs A list of known item parameters. The parameters of item \eqn{j} are given
#'   by the \eqn{j}th element, which should be a vector of length 2, containing
#'   the item difficulty parameter and item discrimination parameter.
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
#'  \item{call}{The matched call.}
#' @importFrom rms lrm.fit
#' @importFrom pryr compose
#' @importFrom pryr partial
#' @import stats
#' @export
#' @examples
#' y <- nes_econ2008[, -(1:3)]
#' x <- model.matrix( ~ party * educ, nes_econ2008)
#' z <- model.matrix( ~ party, nes_econ2008)
#' dichotomize <- function(x) findInterval(x, c(mean(x, na.rm = TRUE)))
#' y_bin <- y
#' y_bin[] <- lapply(y, dichotomize)
#'
#' n <- nrow(nes_econ2008)
#' id_train <- sample.int(n, n/4)
#' id_test <- setdiff(1:n, id_train)
#'
#' y_bin_train <- y_bin[id_train, ]
#' x_train <- x[id_train, ]
#' z_train <- z[id_train, ]
#'
#' mod_train <- hltm(y_bin_train, x_train, z_train)
#'
#' y_bin_test <- y_bin[id_test, ]
#' x_test <- x[id_test, ]
#' z_test <- z[id_test, ]
#'
#' item_coefs <- lapply(coef_item(mod_train), `[[`, "Estimate")
#'
#' model_test <- hltm2(y_bin_test, x_test, z_test, item_coefs = item_coefs)

hltm2 <- function(y, x = NULL, z = NULL, item_coefs, control = list()) {

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

  # extract item parameters
  if(missing(item_coefs))
    stop("`item_coefs` must be supplied.")
  if(!is.list(item_coefs) || length(item_coefs) != J)
    stop("`item_coefs` must be a list of `ncol(y)` elements")
  item_coefs_H <- vapply(item_coefs, length, integer(1L))
  if(!all.equal(item_coefs_H, H))
    stop("`item_coefs` do not match the number of response categories in `y`")
  alpha <- vapply(item_coefs, function(x) x[[1L]], double(1L))
  beta <- vapply(item_coefs, function(x) x[[2L]], double(1L))

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

  # control parameters
  con <- list(max_iter = 150, max_iter2 = 15, eps = 1e-03, eps2 = 1e-03, K = 20, C = 4,
              acceleration = "squarem")
  con[names(control)] <- control

  # set environments for utility functions
  environment(loglik_ltm) <- environment(theta_post_ltm) <- environment(dummy_fun_ltm) <- environment(tab2df_ltm) <- environment()

  # GL points
  K <- con[["K"]]
  theta_ls <- con[["C"]] * GLpoints[[K]][["x"]]
  qw_ls <- con[["C"]] * GLpoints[[K]][["w"]]

  # imputation
  y_imp <- y
  if(anyNA(y)) y_imp[] <- lapply(y, impute)

  # Pre-compute sparse representation for C++ E-step
  sparse_y <- build_sparse_y(y)

  # pca for initial values of theta_eap
  theta_eap <- {
    tmp <- princomp(y_imp, cor = TRUE)$scores[, 1]
    (tmp - mean(tmp, na.rm = TRUE))/sd(tmp, na.rm = TRUE)
  }

  # initial values of gamma and lambda
  lm_opr <- tcrossprod(solve(crossprod(x)), x)
  gamma <- lm_opr %*% theta_eap
  lambda <- rep(0, q)
  fitted_mean <- as.double(x %*% gamma)
  fitted_var <- rep(1, N)

  # EM fixed-point mapping: takes packed params c(gamma, lambda), returns list(params, log_lik)
  em_step_fn <- function(params) {
    g <- params[1:p]
    l <- params[(p+1):(p+q)]
    fm <- as.double(x %*% g)
    fv <- exp(as.double(z %*% l))

    # E-step (C++)
    es <- compute_estep_ltm_cpp(
        sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
        alpha, beta, theta_ls, qw_ls, fm, fv
    )
    ll <- es$log_lik
    t_eap <- es$theta_eap
    t_vap <- es$theta_vap

    # Variance regression
    g <- lm_opr %*% t_eap
    r2 <- (t_eap - x %*% g)^2 + t_vap

    if (ncol(z) == 1) {
      l <- log(mean(r2))
    } else {
      s2 <- glm.fit(x = z, y = r2, intercept = FALSE, family = Gamma(link = "log"))[["fitted.values"]]
      loglik_vr <- -0.5 * (log(s2) + r2/s2)
      LL0 <- sum(loglik_vr)
      for (m_vr in seq(1, con[["max_iter2"]])) {
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

    list(params = c(as.double(g), as.double(l)), log_lik = ll)
  }

  # Pack initial parameters
  params <- c(as.double(gamma), as.double(lambda))
  n_eval <- 0L

  # EM algorithm (plain or SQUAREM)
  if (con[["acceleration"]] == "squarem") {
    # SQUAREM S3 (Varadhan & Roland 2008)
    step_max <- 1
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

      # Convergence check on gamma RMSE
      gamma_prev <- params[1:p]
      gamma_curr <- theta_1[1:p]
      gamma_rmse <- sqrt(mean((gamma_curr - gamma_prev)^2))
      if (gamma_rmse < con[["eps"]]) {
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

      # Guard: NaN/Inf check
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

  } else {
    # Plain EM
    converged <- FALSE
    for (iter in seq(1, con[["max_iter"]])) {
      gamma_prev <- params[1:p]
      res <- em_step_fn(params)
      n_eval <- n_eval + 1L
      params <- res$params
      cat(".")

      gamma_curr <- params[1:p]
      gamma_rmse <- sqrt(mean((gamma_curr - gamma_prev)^2))

      if (gamma_rmse < con[["eps"]]) {
        converged <- TRUE
        cat("\n converged at iteration", iter, "\n")
        break
      }
    }
    if (!converged) {
      stop("algorithm did not converge; try increasing max_iter.")
    }
  }

  # Unpack final parameters
  gamma  <- params[1:p]
  lambda <- params[(p+1):(p+q)]
  fitted_mean <- as.double(x %*% gamma)
  fitted_var  <- exp(as.double(z %*% lambda))

  # Final E-step for theta_eap/theta_vap
  final_estep <- compute_estep_ltm_cpp(
      sparse_y$row_ptr, sparse_y$col_idx, sparse_y$values,
      alpha, beta, theta_ls, qw_ls, fitted_mean, fitted_var
  )
  theta_eap <- final_estep$theta_eap
  theta_vap <- final_estep$theta_vap

  gamma <- setNames(as.double(gamma), paste("x", colnames(x), sep = ""))
  lambda <- setNames(as.double(lambda), paste("z", colnames(z), sep = ""))

  # inference
  pik <- matrix(unlist(Map(partial(dnorm, x = theta_ls), mean = fitted_mean, sd = sqrt(fitted_var))),
                N, K, byrow = TRUE) * matrix(qw_ls, N, K, byrow = TRUE)
  Lijk <- lapply(theta_ls, function(theta_k) exp(loglik_ltm(alpha = alpha, beta = beta, rep(theta_k, N))))  # K-list
  Lik <- vapply(Lijk, compose(exp, partial(rowSums, na.rm = TRUE), log), double(N))
  Li <- rowSums(Lik * pik)

  # log likelihood
  log_Lik <- sum(log(Li))

  # outer product of gradients
  environment(dalpha_ltm) <- environment(sj_ab_ltm) <- environment(si_gamma) <- environment(si_lambda) <- environment()
  dalpha <- dalpha_ltm(alpha, beta)  # K*J matrix
  # s_ab <- unname(Reduce(cbind, lapply(1:J, sj_ab_ltm)))
  s_gamma <- vapply(1:N, si_gamma, double(p))
  s_lambda <- vapply(1:N, si_lambda, double(q))

  s_all <- rbind(s_gamma, s_lambda)
  s_all[is.na(s_all)] <- 0
  covmat <- tryCatch(solve(tcrossprod(s_all)),
                     error = function(e) {warning("The information matrix is singular; SE calculation failed.");
                       matrix(NA, nrow(s_all), nrow(s_all))})
  se_all <- sqrt(diag(covmat))

  # reorganize se_all
  sH <- 2 * J
  gamma_indices <- (sH - 1):(sH + p - 2)
  lambda_indices <- (sH + p - 1):(sH + p + q - 2)
  se_all <- c(rep(0, sH), sqrt(diag(covmat)))

  # name se_all and covmat
  names_ab <- paste(rep(names(alpha), each = 2), c("Diff", "Dscrmn"))
  names(se_all) <- c(names_ab, names(gamma), names(lambda))
  rownames(covmat) <- colnames(covmat) <- c(names(gamma), names(lambda))

  # item coefficients
  coefs_item <- Map(function(a, b) c(Diff = a, Dscrmn = b), alpha, beta)

  # all coefficients
  coef_all <- c(unlist(coefs_item), gamma, lambda)
  coefs <- data.frame(Estimate = coef_all, Std_Error = se_all, z_value = coef_all/se_all,
                      p_value = 2 * (1 - pnorm(abs(coef_all/se_all))))
  rownames(coefs) <- names(se_all)

  # ability parameter estimates
  theta <- data.frame(post_mean = theta_eap, post_sd = sqrt(theta_vap),
                      prior_mean = fitted_mean, prior_sd = sqrt(fitted_var))

  # output
  out <- list(coefficients = coefs, scores = theta, vcov = covmat, log_Lik = log_Lik,
              N = N, J = J, H = H, ylevels = ylevels, p = p, q = q, control = con, call = cl)
  class(out) <- c("hltm", "hIRT")
  out
}
