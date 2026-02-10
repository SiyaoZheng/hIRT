#!/usr/bin/env Rscript
# Instrument hltm by copying its source with timing probes
library(hIRT)

hltm_timed <- function(y, x = NULL, z = NULL, constr = c("latent_scale", "items"),
                       beta_set = 1L, sign_set = TRUE, init = c("naive", "glm", "irt"),
                       control = list()) {
  cl <- match.call()
  timings <- list()
  t0 <- proc.time()

  if(missing(y)) stop("`y` must be provided.")
  if ((!is.data.frame(y) && !is.matrix(y)) || ncol(y) == 1L) stop("bad y")
  if(is.matrix(y)) y <- as.data.frame(y)
  N <- nrow(y); J <- ncol(y)
  y[] <- lapply(y, factor, exclude = c(NA, NaN))
  ylevels <- lapply(y, levels)
  y[] <- lapply(y, function(x) as.integer(x) - 1)
  H <- vapply(y, max, double(1L), na.rm = TRUE) + 1
  x <- x %||% as.matrix(rep(1, N))
  z <- z %||% as.matrix(rep(1, N))
  if (!is.matrix(x)) stop("`x` must be a matrix.")
  if (!is.matrix(z)) stop("`z` must be a matrix.")
  p <- ncol(x); q <- ncol(z)
  colnames(x) <- colnames(x) %||% paste0("x", 1:p)
  colnames(z) <- colnames(z) %||% paste0("x", 1:q)
  constr <- match.arg(constr); init <- match.arg(init)
  con <- list(max_iter = 150, max_iter2 = 15, eps = 1e-03, eps2 = 1e-03, K = 25, C = 4)
  con[names(control)] <- control

  `%||%` <- function(a, b) if (!is.null(a)) a else b
  impute <- function(vec) { vec[is.na(vec)] <- median(vec, na.rm = TRUE); vec }

  loglik_ltm <- function(alpha, beta, theta) {
    util <- matrix(alpha, N, J, byrow = TRUE) + outer(theta, beta)
    log(exp(as.matrix(y) * util)/(1 + exp(util)))
  }
  theta_post_ltm <- function(theta_k, qw_k) {
    wt_k <- dnorm(theta_k - fitted_mean, sd = sqrt(fitted_var)) * qw_k
    loglik <- rowSums(loglik_ltm(alpha, beta, rep(theta_k, N)), na.rm = TRUE)
    exp(loglik + log(wt_k))
  }
  dummy_fun_ltm <- function(y_j) {
    dummy_mat <- outer(y_j, c(0, 1), "=="); dummy_mat[is.na(dummy_mat)] <- 0; w %*% dummy_mat
  }
  tab2df_ltm <- function(tab, theta_ls) {
    data.frame(y = factor(rep(c(0, 1), each = K)), x = rep(theta_ls, 2), wt = as.double(tab))
  }

  K <- con[["K"]]
  theta_ls <- con[["C"]] * hIRT:::GLpoints[[K]][["x"]]
  qw_ls <- con[["C"]] * hIRT:::GLpoints[[K]][["w"]]
  y_imp <- y; if(anyNA(y)) y_imp[] <- lapply(y, impute)
  theta_eap <- { tmp <- princomp(y_imp, cor = TRUE)$scores[, 1]; (tmp - mean(tmp))/sd(tmp) }

  if (init == "naive"){
    alpha <- rep(0, J)
    beta <- vapply(y, function(y) cov(y, theta_eap, use = "complete.obs")/var(theta_eap), double(1L))
  } else if (init == "glm"){
    pseudo_logit <- lapply(y_imp, function(y) glm.fit(cbind(1, theta_eap), y, family = binomial("logit"))$coefficients)
    beta <- vapply(pseudo_logit, function(x) x[2L], double(1L))
    alpha <- vapply(pseudo_logit, function(x) x[1L], double(1L))
  }

  lm_opr <- tcrossprod(solve(crossprod(x)), x)
  gamma <- lm_opr %*% theta_eap; lambda <- rep(0, q)
  fitted_mean <- as.double(x %*% gamma); fitted_var <- rep(1, N)

  timings$init <- (proc.time() - t0)["elapsed"]

  # EM loop
  em_times <- list(estep = 0, mstep = 0, eap = 0, varreg = 0, constr = 0)
  for (iter in seq(1, con[["max_iter"]])) {
    alpha_prev <- alpha; beta_prev <- beta; gamma_prev <- gamma; lambda_prev <- lambda

    tt <- proc.time()
    posterior <- Map(theta_post_ltm, theta_ls, qw_ls)
    w <- { tmp <- matrix(unlist(posterior), N, K); t(sweep(tmp, 1, rowSums(tmp), FUN = "/")) }
    em_times$estep <- em_times$estep + (proc.time() - tt)["elapsed"]

    tt <- proc.time()
    pseudo_tab <- lapply(y, dummy_fun_ltm)
    pseudo_y <- lapply(pseudo_tab, tab2df_ltm, theta_ls = theta_ls)
    pseudo_logit <- lapply(pseudo_y, function(df) glm.fit(cbind(1, df[["x"]]),
      df[["y"]], weights = df[["wt"]], family = quasibinomial("logit"))[["coefficients"]])
    beta <- vapply(pseudo_logit, function(x) x[2L], double(1L))
    alpha <- vapply(pseudo_logit, function(x) x[1L], double(1L))
    em_times$mstep <- em_times$mstep + (proc.time() - tt)["elapsed"]

    tt <- proc.time()
    theta_eap <- t(theta_ls %*% w); theta_vap <- t(theta_ls^2 %*% w) - theta_eap^2
    em_times$eap <- em_times$eap + (proc.time() - tt)["elapsed"]

    tt <- proc.time()
    gamma <- lm_opr %*% theta_eap
    r2 <- (theta_eap - x %*% gamma)^2 + theta_vap
    if (ncol(z)==1) lambda <- log(mean(r2)) else {
      s2 <- glm.fit(x = z, y = r2, intercept = FALSE, family = Gamma(link = "log"))$fitted.values
      loglik <- -0.5 * (log(s2) + r2/s2); LL0 <- sum(loglik); dLL <- 1
      for (m in seq(1, con[["max_iter2"]])) {
        gamma <- lm.wfit(x, theta_eap, w = 1/s2)$coefficients
        r2 <- (theta_eap - x %*% gamma)^2 + theta_vap
        var_reg <- glm.fit(x = z, y = r2, intercept = FALSE, family = Gamma(link = "log"))
        s2 <- var_reg$fitted.values; loglik <- -0.5 * (log(s2) + r2/s2)
        LL_temp <- sum(loglik); dLL <- LL_temp - LL0
        if (dLL < con[["eps2"]]) break; LL0 <- LL_temp
      }
      lambda <- var_reg$coefficients
    }
    em_times$varreg <- em_times$varreg + (proc.time() - tt)["elapsed"]

    tt <- proc.time()
    tmp <- mean(x %*% gamma)
    alpha <- unlist(Map(function(a, b) a + tmp * b, alpha, beta))
    gamma[1L] <- gamma[1L] - tmp
    tmp <- mean(z %*% lambda)
    gamma <- gamma/exp(tmp/2); beta <- beta * exp(tmp/2); lambda[1L] <- lambda[1L] - tmp
    if (sign_set == (beta[beta_set] < 0)) { gamma <- -gamma; beta <- -beta }
    fitted_mean <- as.double(x %*% gamma); fitted_var <- exp(as.double(z %*% lambda))
    em_times$constr <- em_times$constr + (proc.time() - tt)["elapsed"]

    if (sqrt(mean((beta - beta_prev)^2)) < con[["eps"]]) { cat("\n converged at iteration", iter, "\n"); break }
    else if (iter == con[["max_iter"]]) { stop("did not converge"); break }
  }
  timings$em_total <- (proc.time() - t0)["elapsed"] - timings$init
  timings$em_iters <- iter
  timings$em_detail <- em_times

  # SE computation
  tt_se <- proc.time()
  dalpha_ltm <- function(alpha, beta) {
    putil <- plogis(matrix(alpha, K, J, byrow = TRUE) + outer(theta_ls, beta))
    putil * (1 - putil)
  }
  sj_ab_ltm <- function(j) {
    tmp_mat <- (pik * Lik/vapply(Lijk, `[`, 1:N, j, FUN.VALUE = double(N)))
    dalpha_j <- dalpha[, j, drop = FALSE]; dbeta_j <- dalpha_j * theta_ls
    sgn <- .subset2(y, j) * 2 - 1; sgn[is.na(sgn)] <- 0
    drv_alpha <- sgn * (tmp_mat %*% dalpha_j)/Li
    drv_beta <- sgn * (tmp_mat %*% dbeta_j)/Li
    cbind(drv_alpha, drv_beta)
  }
  si_gamma <- function(i) {
    sum(pik[i, ] * Lik[i, ] * (theta_ls - fitted_mean[[i]]))/fitted_var[[i]]/Li[[i]] * x[i, 1:p]
  }
  si_lambda <- function(i) {
    sum(0.5 * pik[i, ] * Lik[i, ] * ((theta_ls - fitted_mean[[i]])^2/fitted_var[[i]] - 1))/Li[[i]] * z[i, 1:q]
  }

  gamma <- setNames(as.double(gamma), paste("x", colnames(x), sep = ""))
  lambda <- setNames(as.double(lambda), paste("z", colnames(z), sep = ""))

  pik <- matrix(unlist(Map(pryr::partial(dnorm, x = theta_ls), mean = fitted_mean, sd = sqrt(fitted_var))),
                N, K, byrow = TRUE) * matrix(qw_ls, N, K, byrow = TRUE)
  Lijk <- lapply(theta_ls, function(theta_k) exp(loglik_ltm(alpha = alpha, beta = beta, rep(theta_k, N))))
  Lik <- vapply(Lijk, pryr::compose(exp, pryr::partial(rowSums, na.rm = TRUE), log), double(N))
  Li <- rowSums(Lik * pik)
  log_Lik <- sum(log(Li))

  tt_scores <- proc.time()
  dalpha <- dalpha_ltm(alpha, beta)
  s_ab <- unname(Reduce(cbind, lapply(1:J, sj_ab_ltm)))
  timings$se_sab <- (proc.time() - tt_scores)["elapsed"]

  tt_sgamma <- proc.time()
  s_gamma <- vapply(1:N, si_gamma, double(p))
  timings$se_sgamma <- (proc.time() - tt_sgamma)["elapsed"]

  tt_slambda <- proc.time()
  s_lambda <- vapply(1:N, si_lambda, double(q))
  timings$se_slambda <- (proc.time() - tt_slambda)["elapsed"]

  tt_solve <- proc.time()
  s_all <- rbind(t(s_ab)[-c(1L, ncol(s_ab)), , drop = FALSE], s_gamma, s_lambda)
  s_all[is.na(s_all)] <- 0
  covmat <- tryCatch(solve(tcrossprod(s_all)),
                     error = function(e) { warning("singular"); matrix(NA, nrow(s_all), nrow(s_all)) })
  timings$se_solve <- (proc.time() - tt_solve)["elapsed"]
  timings$se_total <- (proc.time() - tt_se)["elapsed"]
  timings$total <- (proc.time() - t0)["elapsed"]

  timings
}

# --- Run ---
y <- nes_econ2008[, -(1:3)]
x <- model.matrix(~ party * educ, nes_econ2008)
z <- model.matrix(~ party, nes_econ2008)
dichotomize <- function(v) findInterval(v, c(mean(v, na.rm = TRUE)))
y_bin <- y; y_bin[] <- lapply(y, dichotomize)

cat(sprintf("N=%d, J=%d\n\n", nrow(y_bin), ncol(y_bin)))
timings <- hltm_timed(y_bin, x, z)

cat("=== TIMING BREAKDOWN ===\n\n")
cat(sprintf("Total:     %.3f sec\n", timings$total))
cat(sprintf("Init:      %.3f sec (%.1f%%)\n", timings$init, 100*timings$init/timings$total))
cat(sprintf("EM total:  %.3f sec (%.1f%%), %d iterations\n",
            timings$em_total, 100*timings$em_total/timings$total, timings$em_iters))
em <- timings$em_detail
cat(sprintf("  E-step:    %.3f sec (%.1f%%)\n", em$estep, 100*em$estep/timings$total))
cat(sprintf("  M-step:    %.3f sec (%.1f%%)\n", em$mstep, 100*em$mstep/timings$total))
cat(sprintf("  EAP/VAP:   %.3f sec (%.1f%%)\n", em$eap, 100*em$eap/timings$total))
cat(sprintf("  VarReg:    %.3f sec (%.1f%%)\n", em$varreg, 100*em$varreg/timings$total))
cat(sprintf("  Constr:    %.3f sec (%.1f%%)\n", em$constr, 100*em$constr/timings$total))
cat(sprintf("SE total:  %.3f sec (%.1f%%)\n", timings$se_total, 100*timings$se_total/timings$total))
cat(sprintf("  s_ab:      %.3f sec (%.1f%%)\n", timings$se_sab, 100*timings$se_sab/timings$total))
cat(sprintf("  s_gamma:   %.3f sec (%.1f%%)\n", timings$se_sgamma, 100*timings$se_sgamma/timings$total))
cat(sprintf("  s_lambda:  %.3f sec (%.1f%%)\n", timings$se_slambda, 100*timings$se_slambda/timings$total))
cat(sprintf("  solve:     %.3f sec (%.1f%%)\n", timings$se_solve, 100*timings$se_solve/timings$total))

# --- Repeat with 2x and 4x data ---
for (mult in c(2, 4)) {
  y_m <- do.call(rbind, rep(list(y_bin), mult))
  x_m <- do.call(rbind, rep(list(x), mult))
  z_m <- do.call(rbind, rep(list(z), mult))
  cat(sprintf("\n=== N=%d (x%d) ===\n", nrow(y_m), mult))
  tm <- hltm_timed(y_m, x_m, z_m)
  em_m <- tm$em_detail
  cat(sprintf("Total: %.3f | EM: %.3f (E:%.3f M:%.3f VR:%.3f) | SE: %.3f (sab:%.3f sg:%.3f sl:%.3f solve:%.3f)\n",
              tm$total, tm$em_total, em_m$estep, em_m$mstep, em_m$varreg,
              tm$se_total, tm$se_sab, tm$se_sgamma, tm$se_slambda, tm$se_solve))
}
