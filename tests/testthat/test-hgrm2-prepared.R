test_that("prepared hgrm2 outcome reproduces the ordinary API", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[seq_len(600L), -(1:3)]
  x <- model.matrix(~ party * educ, nes_econ2008)[seq_len(600L), , drop = FALSE]
  z <- matrix(1, nrow = nrow(y), ncol = 1L)
  colnames(z) <- "(Intercept)"

  item_coefs <- lapply(y, function(values) {
    n_categories <- nlevels(factor(values, exclude = c(NA, NaN)))
    c(seq(1, -1, length.out = n_categories - 1L), 1)
  })
  prepared <- prepare_hgrm2_outcome(y, item_coefs)

  expect_s3_class(prepared, "hgrm2_outcome")
  expect_identical(prepared$schema_version, "hgrm2_outcome_v1")

  ordinary_output <- capture.output({
    ordinary <- hgrm2(y, x, z, item_coefs = item_coefs)
  })
  prepared_output <- capture.output({
    repeated <- hgrm2(prepared, x, z)
  })
  ordinary$call <- NULL
  repeated$call <- NULL

  expect_identical(repeated, ordinary)
  expect_true(length(ordinary_output) > 0L)
  expect_true(length(prepared_output) > 0L)
})

test_that("prepared hgrm2 inputs are validated", {
  data(nes_econ2008, package = "hIRT")
  y <- nes_econ2008[seq_len(100L), -(1:3)]
  item_coefs <- lapply(y, function(values) {
    n_categories <- nlevels(factor(values, exclude = c(NA, NaN)))
    c(seq(1, -1, length.out = n_categories - 1L), 1)
  })
  prepared <- prepare_hgrm2_outcome(y, item_coefs)

  expect_error(
    hgrm2(prepared, item_coefs = item_coefs),
    "must be omitted"
  )
  prepared$schema_version <- "broken"
  expect_error(hgrm2(prepared), "not a valid hgrm2_outcome")
})
