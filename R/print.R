#' Printing an object of class \code{hIRT}
#' @param x An object of class \code{hIRT}
#' @param digits The number of significant digits to use when printing
#' @param ... further arguments passed to \code{\link{print}}.
#' @export
print.hIRT <- function(x, digits = 3, ...) {
  cat("\nCall:\n", paste(deparse(x[["call"]]), sep = "\n", collapse = "\n"),
      "\n\n", sep = "")

  if (inherits(x, "mhltm")) {
    D <- x[["D"]]
    p <- x[["p"]]
    q <- x[["q"]]
    mean_coefs <- coef_mean(x, digits)
    var_coefs <- coef_var(x, digits)

    for (d in seq_len(D)) {
      cat(sprintf("Dimension %d Mean Regression:\n", d))
      rows <- ((d - 1) * p + 1):(d * p)
      print(mean_coefs[rows, , drop = FALSE], ...)
      cat("\n")
    }
    for (d in seq_len(D)) {
      cat(sprintf("Dimension %d Variance Regression:\n", d))
      rows <- ((d - 1) * q + 1):(d * q)
      print(var_coefs[rows, , drop = FALSE], ...)
      cat("\n")
    }
    cat("Correlation Matrix:\n")
    print(round(x[["R"]], digits), ...)
    cat("\n")
  } else {
    cat("Mean Regression:\n")
    print(coef_mean(x, digits), ...)
    cat("\n")
    cat("Variance Regression:\n")
    print(coef_var(x, digits), ...)
    cat("\n")
  }

  cat("Log Likelihood:", round(x[["log_Lik"]], digits))
  cat("\n\n")
  invisible(x)
}
