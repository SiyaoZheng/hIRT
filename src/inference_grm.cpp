// Fused inference block for the graded response model (GRM).
//
// Computes the marginal log-likelihood, OPG (outer product of gradients)
// information matrix, and standard errors in a single streaming pass.
// Replaces the R code path in hgrm.R (lines 452-515) and hgrm2.R (lines 218-253)
// that materializes large N-scaled intermediate matrices.
//
// Memory: O(J*K + d^2 + K) — no N-scaled intermediates.
//
// When compute_item_se = true (hgrm): d = sH - 2 + p + q
// When compute_item_se = false (hgrm2): d = p + q

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <vector>
#include <algorithm>

using namespace Rcpp;

// Numerically stable sigmoid
static inline double sigmoid_grm_(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        const double e = std::exp(x);
        return e / (1.0 + e);
    }
}

// [[Rcpp::export]]
Rcpp::List compute_inference_grm_cpp(
    Rcpp::IntegerVector row_ptr,       // CSR row pointers (N+1)
    Rcpp::IntegerVector col_idx,       // CSR column indices (nnz, 0-based)
    Rcpp::IntegerVector values,        // CSR values (nnz, 1-based ordinal responses)
    Rcpp::NumericVector alpha_flat,    // Flattened thresholds (sum(H-1) elements)
    Rcpp::IntegerVector alpha_offsets, // Offsets into alpha_flat per item (J+1)
    Rcpp::IntegerVector H,             // Number of categories per item (J)
    Rcpp::NumericVector beta,          // Item discrimination parameters (J)
    Rcpp::NumericVector theta_ls,      // K quadrature points
    Rcpp::NumericVector qw_ls,         // K quadrature weights
    Rcpp::NumericVector fitted_mean,   // N prior means
    Rcpp::NumericVector fitted_var,    // N prior variances
    Rcpp::NumericMatrix x_mat,         // N x p covariate matrix
    Rcpp::NumericMatrix z_mat,         // N x q covariate matrix
    bool compute_item_se = true,       // false for hgrm2
    bool verbose = false
) {
    const int N = fitted_mean.size();
    const int K = theta_ls.size();
    const int J = beta.size();
    const int p = x_mat.ncol();
    const int q = z_mat.ncol();
    const int n_alpha = alpha_flat.size(); // sum(H-1)
    const size_t Nsz = static_cast<size_t>(N); // for overflow-safe indexing

    // Compute sH = sum(H) and cumulative H offsets for index mapping
    // Full item parameter layout: [tau_1^(1),...,tau_{H1-1}^(1), beta_1,
    //                               tau_1^(2),...,tau_{H2-1}^(2), beta_2, ...]
    // Full index for item j, threshold m: cumH[j] + m  (m = 0..H_j-2)
    // Full index for item j, beta:        cumH[j] + H_j - 1
    // sH = total full item params = sum(H)
    // Constraints: skip full index 0 (1st threshold of item 1) and sH-1 (last beta)
    // Free index = full_index - 1, for full_index in 1..sH-2

    const int* Hv = H.begin();
    std::vector<int> cumH(J + 1, 0);
    for (int j = 0; j < J; j++) {
        cumH[j + 1] = cumH[j] + Hv[j];
    }
    const int sH = cumH[J];

    // Free parameter count
    int d;
    int gamma_start, lambda_start;
    if (compute_item_se) {
        d = sH - 2 + p + q;
        gamma_start = sH - 2;
        lambda_start = sH - 2 + p;
    } else {
        d = p + q;
        gamma_start = 0;
        lambda_start = p;
    }

    // Raw pointers
    const int*    rp   = row_ptr.begin();
    const int*    ci   = col_idx.begin();
    const int*    val  = values.begin();
    const double* af   = alpha_flat.begin();
    const int*    aoff = alpha_offsets.begin();
    const double* b    = beta.begin();
    const double* tls  = theta_ls.begin();
    const double* fm   = fitted_mean.begin();
    const double* fv   = fitted_var.begin();
    const double* xp   = REAL(x_mat);  // column-major N x p
    const double* zp   = REAL(z_mat);  // column-major N x q

    // ---- Phase 1: Precompute sigmoid table ----
    // sig_table[t * K + k] = sigmoid(alpha_flat[t] + beta[j] * theta_ls[k])
    // f_table[t * K + k]   = sig * (1 - sig)  (sigmoid derivative)
    std::vector<double> sig_table(static_cast<size_t>(n_alpha) * K);
    std::vector<double> f_table(static_cast<size_t>(n_alpha) * K);
    for (int j = 0; j < J; j++) {
        for (int t = aoff[j]; t < aoff[j + 1]; t++) {
            for (int k = 0; k < K; k++) {
                double s = sigmoid_grm_(af[t] + b[j] * tls[k]);
                sig_table[t * K + k] = s;
                f_table[t * K + k] = s * (1.0 - s);
            }
        }
    }

    // Log quadrature weights
    std::vector<double> log_qw(K);
    for (int k = 0; k < K; k++) {
        log_qw[k] = std::log(qw_ls[k]);
    }
    const double neg_half_log_2pi = -0.5 * std::log(2.0 * M_PI);

    // ---- Phase 2: Build score matrix & compute log-likelihood ----
    // Strategy: accumulate per-person scores into N×d matrix, then use
    // BLAS-level S^T S for the OPG information matrix. This is much faster
    // than streaming rank-1 updates for moderate d (rank-1: O(N*d^2) with
    // poor cache behavior; S^T S: single blocked BLAS dsyrk call).
    // Use raw column-major buffer for thread-safe parallel writes.
    // S_raw[col * N + row] — each thread writes to distinct rows.
    std::vector<double> S_raw(static_cast<size_t>(N) * d, 0.0);
    double total_log_lik = 0.0;

    #pragma omp parallel reduction(+:total_log_lik)
    {
    std::vector<double> log_joint(K);
    std::vector<double> w_k(K);

    #pragma omp for schedule(dynamic, 512)
    for (int i = 0; i < N; i++) {
        const int start = rp[i];
        const int end   = rp[i + 1];
        const double sd_i = std::sqrt(fv[i]);
        const double log_sd_i = std::log(sd_i);
        const double inv_var_i = 1.0 / fv[i];

        // Step 1: Compute log_joint[k] for each quadrature point
        for (int k = 0; k < K; k++) {
            double loglik = 0.0;
            for (int idx = start; idx < end; idx++) {
                const int j = ci[idx];
                const int h = val[idx];
                const int H_j = Hv[j];
                const int off_j = aoff[j];

                double sig_upper;
                if (h == 1) {
                    sig_upper = 1.0;
                } else {
                    sig_upper = sig_table[(off_j + h - 2) * K + k];
                }

                double sig_lower;
                if (h == H_j) {
                    sig_lower = 0.0;
                } else {
                    sig_lower = sig_table[(off_j + h - 1) * K + k];
                }

                double diff = sig_upper - sig_lower;
                if (diff < 1e-20) diff = 1e-20;
                loglik += std::log(diff);
            }
            const double z = tls[k] - fm[i];
            const double log_prior = neg_half_log_2pi - log_sd_i - 0.5 * z * z * inv_var_i;
            log_joint[k] = loglik + log_prior + log_qw[k];
        }

        // Step 2: Log-sum-exp normalization
        double max_lj = log_joint[0];
        for (int k = 1; k < K; k++) {
            if (log_joint[k] > max_lj) max_lj = log_joint[k];
        }
        double sum_exp = 0.0;
        for (int k = 0; k < K; k++) {
            sum_exp += std::exp(log_joint[k] - max_lj);
        }
        const double log_Li = max_lj + std::log(sum_exp);
        total_log_lik += log_Li;

        // Step 3: Posterior weights
        for (int k = 0; k < K; k++) {
            w_k[k] = std::exp(log_joint[k] - log_Li);
        }

        // Step 4: Score vector (write directly into row i of S)

        // 4a: Item scores (only when computing item SEs)
        if (compute_item_se) {
            for (int idx = start; idx < end; idx++) {
                const int j = ci[idx];
                const int h = val[idx];
                const int H_j = Hv[j];
                const int off_j = aoff[j];

                const bool has_upper = (h >= 2);
                const bool has_lower = (h <= H_j - 1);
                const int upper_m = h - 2;
                const int lower_m = h - 1;

                const int full_idx_upper = has_upper ? (cumH[j] + upper_m) : -1;
                const int full_idx_lower = has_lower ? (cumH[j] + lower_m) : -1;
                const int full_idx_beta  = cumH[j] + H_j - 1;

                double s_upper = 0.0;
                double s_lower = 0.0;
                double s_beta_j = 0.0;

                for (int k = 0; k < K; k++) {
                    double sig_up = (h == 1) ? 1.0 : sig_table[(off_j + h - 2) * K + k];
                    double sig_lo = (h == H_j) ? 0.0 : sig_table[(off_j + h - 1) * K + k];
                    double p_h = sig_up - sig_lo;
                    if (p_h < 1e-20) p_h = 1e-20;

                    const double ratio_k = w_k[k] / p_h;

                    double f_up = has_upper ? f_table[(off_j + upper_m) * K + k] : 0.0;
                    double f_lo = has_lower ? f_table[(off_j + lower_m) * K + k] : 0.0;

                    if (has_upper) s_upper += ratio_k * f_up;
                    if (has_lower) s_lower += ratio_k * (-f_lo);
                    s_beta_j += ratio_k * tls[k] * (f_up - f_lo);
                }

                if (has_upper && full_idx_upper != 0) {
                    S_raw[static_cast<size_t>(full_idx_upper - 1) * Nsz + i] += s_upper;
                }
                if (has_lower && full_idx_lower != 0) {
                    S_raw[static_cast<size_t>(full_idx_lower - 1) * Nsz + i] += s_lower;
                }
                if (full_idx_beta != sH - 1) {
                    S_raw[static_cast<size_t>(full_idx_beta - 1) * Nsz + i] += s_beta_j;
                }
            }
        }

        // 4b: Covariate scores
        double s_gamma_scalar = 0.0;
        double s_lambda_scalar = 0.0;
        for (int k = 0; k < K; k++) {
            const double theta_dev = tls[k] - fm[i];
            s_gamma_scalar += w_k[k] * theta_dev;
            s_lambda_scalar += w_k[k] * 0.5 * (theta_dev * theta_dev * inv_var_i - 1.0);
        }
        s_gamma_scalar *= inv_var_i;

        for (int l = 0; l < p; l++) {
            S_raw[static_cast<size_t>(gamma_start + l) * Nsz + i] = xp[l * N + i] * s_gamma_scalar;
        }
        for (int l = 0; l < q; l++) {
            S_raw[static_cast<size_t>(lambda_start + l) * Nsz + i] = zp[l * N + i] * s_lambda_scalar;
        }
    }
    } // end omp parallel

    // ---- Phase 2b: OPG information matrix via BLAS S^T S ----
    Eigen::Map<Eigen::MatrixXd> S(S_raw.data(), N, d);
    Eigen::MatrixXd info = S.transpose() * S;

    // ---- Phase 3: Matrix inversion ----
    bool singular = false;
    Eigen::MatrixXd covmat(d, d);
    Eigen::VectorXd se_free(d);

    Eigen::LLT<Eigen::MatrixXd> llt(info);
    if (llt.info() == Eigen::Success) {
        covmat = llt.solve(Eigen::MatrixXd::Identity(d, d));
    } else {
        Eigen::FullPivLU<Eigen::MatrixXd> lu(info);
        if (lu.isInvertible()) {
            covmat = lu.inverse();
        } else {
            singular = true;
            covmat.setConstant(NA_REAL);
        }
    }

    if (!singular) {
        for (int i = 0; i < d; i++) {
            se_free(i) = std::sqrt(std::max(covmat(i, i), 0.0));
        }
    } else {
        se_free.setConstant(NA_REAL);
    }

    // ---- Phase 4: Convert to R objects ----
    Rcpp::NumericMatrix covmat_r(d, d);
    Rcpp::NumericVector se_free_r(d);
    for (int i = 0; i < d; i++) {
        se_free_r[i] = se_free(i);
        for (int j = 0; j < d; j++) {
            covmat_r(i, j) = covmat(i, j);
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("covmat")   = covmat_r,
        Rcpp::Named("log_Lik")  = total_log_lik,
        Rcpp::Named("se_free")  = se_free_r,
        Rcpp::Named("singular") = singular,
        Rcpp::Named("d")        = d
    );
}
