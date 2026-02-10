#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace Rcpp;

// Numerically stable log(1 + exp(x)) — Maechler (2012)
// Named softplus_ to avoid collision with R's Rmath.h log1pexp macro.
static inline double softplus_(double x) {
    if (x <= -37.0) return std::exp(x);
    if (x <=  18.0) return std::log1p(std::exp(x));
    if (x <=  33.3) return x + std::exp(-x);
    return x;
}

// Sparse E-step for the latent trait model (LTM).
//
// Computes posterior quadrature weights, EAP and VAP estimates using
// a CSR (Compressed Sparse Row) representation of the response matrix.
// Only observed (non-NA) entries contribute to the log-likelihood,
// giving a speedup proportional to the fraction of missing data.
//
// @param row_ptr  CSR row pointers (length N+1). row_ptr[i]..row_ptr[i+1]-1
//                 index into col_idx/values for observation i.
// @param col_idx  Column indices of observed entries (0-based).
// @param values   Observed binary responses (0 or 1).
// @param alpha    Item difficulty parameters (length J).
// @param beta     Item discrimination parameters (length J).
// @param theta_ls Gauss-Legendre quadrature points (length K).
// @param qw_ls    Gauss-Legendre quadrature weights (length K).
// @param fitted_mean Prior means of latent trait (length N).
// @param fitted_var  Prior variances of latent trait (length N).
//
// @return A list with:
//   w         — K x N matrix of posterior quadrature weights
//   theta_eap — N-vector of EAP (posterior mean) estimates
//   theta_vap — N-vector of VAP (posterior variance) estimates
//
// [[Rcpp::export]]
List compute_estep_ltm_cpp(
    IntegerVector row_ptr,
    IntegerVector col_idx,
    IntegerVector values,
    NumericVector alpha,
    NumericVector beta,
    NumericVector theta_ls,
    NumericVector qw_ls,
    NumericVector fitted_mean,
    NumericVector fitted_var
) {
    const int N = fitted_mean.size();
    const int K = theta_ls.size();

    // Pre-compute log quadrature weights (constant across iterations)
    std::vector<double> log_qw(K);
    for (int k = 0; k < K; k++) {
        log_qw[k] = std::log(qw_ls[k]);
    }

    // Pre-compute -0.5 * log(2 * pi) (constant)
    const double neg_half_log_2pi = -0.5 * std::log(2.0 * M_PI);

    // Output matrices
    NumericMatrix w(K, N);
    NumericVector theta_eap(N);
    NumericVector theta_vap(N);

    // Temporary storage for log-posteriors (one per quadrature point)
    std::vector<double> log_post(K);

    // Raw pointers for tight inner loops
    const int*    rp  = row_ptr.begin();
    const int*    ci  = col_idx.begin();
    const int*    val = values.begin();
    const double* a   = alpha.begin();
    const double* b   = beta.begin();
    const double* tls = theta_ls.begin();
    const double* fm  = fitted_mean.begin();
    const double* fv  = fitted_var.begin();

    for (int i = 0; i < N; i++) {
        const int start = rp[i];
        const int end   = rp[i + 1];
        const double sd_i = std::sqrt(fv[i]);
        const double log_sd_i = std::log(sd_i);
        const double inv_var_i = 1.0 / fv[i];

        // Compute log-posterior for each quadrature point
        for (int k = 0; k < K; k++) {
            const double theta_k = tls[k];

            // Log-likelihood: sum over observed entries only
            double loglik = 0.0;
            for (int idx = start; idx < end; idx++) {
                const double util = a[ci[idx]] + b[ci[idx]] * theta_k;
                loglik += val[idx] * util - softplus_(util);
            }

            // Log-prior: log N(theta_k; fitted_mean_i, sd_i)
            const double z = theta_k - fm[i];
            const double log_prior = neg_half_log_2pi - log_sd_i - 0.5 * z * z * inv_var_i;

            log_post[k] = loglik + log_prior + log_qw[k];
        }

        // Log-sum-exp normalization
        double max_lp = log_post[0];
        for (int k = 1; k < K; k++) {
            if (log_post[k] > max_lp) max_lp = log_post[k];
        }
        double sum_exp = 0.0;
        for (int k = 0; k < K; k++) {
            sum_exp += std::exp(log_post[k] - max_lp);
        }
        const double log_norm = max_lp + std::log(sum_exp);

        // Posterior weights, EAP, VAP
        double eap  = 0.0;
        double eap2 = 0.0;
        for (int k = 0; k < K; k++) {
            const double wki = std::exp(log_post[k] - log_norm);
            w(k, i) = wki;
            eap  += tls[k] * wki;
            eap2 += tls[k] * tls[k] * wki;
        }
        theta_eap[i] = eap;
        theta_vap[i] = eap2 - eap * eap;
    }

    return List::create(
        Named("w")         = w,
        Named("theta_eap") = theta_eap,
        Named("theta_vap") = theta_vap
    );
}

// Sparse M-step for the latent trait model (LTM).
//
// Computes item parameters (alpha, beta) from posterior weights and sparse
// response data.  Replaces the R code path: dummy_fun_ltm → tab2df_ltm →
// glm.fit (quasibinomial logit) with:
//   Phase A  — sparse sufficient statistics (single pass over CSR data)
//   Phase B  — batched 2×2 Newton-Raphson logistic regression per item
//
// @param row_ptr     CSR row pointers (length N+1).
// @param col_idx     Column indices of observed entries (0-based).
// @param values      Observed binary responses (0 or 1).
// @param w           K × N posterior weights from E-step.
// @param theta_ls    K quadrature points.
// @param alpha_init  J starting values for intercept.
// @param beta_init   J starting values for slope.
// @param max_nr_iter Maximum Newton-Raphson iterations (default 25).
// @param nr_tol      Convergence tolerance for parameter change (default 1e-8).
//
// @return A list with:
//   alpha — J-vector of intercept estimates
//   beta  — J-vector of slope (discrimination) estimates
//
// [[Rcpp::export]]
List compute_mstep_ltm_cpp(
    IntegerVector row_ptr,
    IntegerVector col_idx,
    IntegerVector values,
    NumericMatrix w,
    NumericVector theta_ls,
    NumericVector alpha_init,
    NumericVector beta_init,
    int max_nr_iter = 25,
    double nr_tol = 1e-8
) {
    const int N = w.ncol();
    const int K = w.nrow();
    const int J = alpha_init.size();

    // Raw pointers for tight loops
    const int*    rp  = row_ptr.begin();
    const int*    ci  = col_idx.begin();
    const int*    val = values.begin();
    const double* tls = theta_ls.begin();

    // Phase A: Sparse sufficient statistics
    // suff_1[j*K + k] = sum_i w(k,i) * I(y_ij == 1)
    // suff_0[j*K + k] = sum_i w(k,i) * I(y_ij == 0)
    std::vector<double> suff_1(J * K, 0.0);
    std::vector<double> suff_0(J * K, 0.0);

    for (int i = 0; i < N; i++) {
        const int start = rp[i];
        const int end   = rp[i + 1];
        for (int idx = start; idx < end; idx++) {
            const int j = ci[idx];
            const int y_ij = val[idx];
            const int base = j * K;
            if (y_ij == 1) {
                for (int k = 0; k < K; k++) {
                    suff_1[base + k] += w(k, i);
                }
            } else {
                for (int k = 0; k < K; k++) {
                    suff_0[base + k] += w(k, i);
                }
            }
        }
    }

    // Phase B: Newton-Raphson logistic regression per item
    NumericVector alpha_out(J);
    NumericVector beta_out(J);

    for (int j = 0; j < J; j++) {
        double a = alpha_init[j];
        double b = beta_init[j];
        const int base = j * K;

        for (int it = 0; it < max_nr_iter; it++) {
            // Gradient and Hessian accumulation
            double g_a = 0.0, g_b = 0.0;
            double h_aa = 0.0, h_ab = 0.0, h_bb = 0.0;

            for (int k = 0; k < K; k++) {
                const double n1 = suff_1[base + k];
                const double n0 = suff_0[base + k];
                const double n_total = n0 + n1;
                if (n_total < 1e-30) continue;  // skip empty bins

                const double eta = a + b * tls[k];
                // Numerically stable sigmoid
                double p;
                if (eta >= 0.0) {
                    const double e = std::exp(-eta);
                    p = 1.0 / (1.0 + e);
                } else {
                    const double e = std::exp(eta);
                    p = e / (1.0 + e);
                }

                const double residual = n1 - n_total * p;
                const double w_k = n_total * p * (1.0 - p);

                g_a += residual;
                g_b += residual * tls[k];
                h_aa -= w_k;
                h_ab -= w_k * tls[k];
                h_bb -= w_k * tls[k] * tls[k];
            }

            // 2×2 Hessian inverse: [[h_aa, h_ab], [h_ab, h_bb]]
            const double det = h_aa * h_bb - h_ab * h_ab;
            if (std::abs(det) < 1e-30) break;  // singular Hessian

            const double inv_det = 1.0 / det;
            const double da = inv_det * (h_bb * g_a - h_ab * g_b);
            const double db = inv_det * (h_aa * g_b - h_ab * g_a);

            a -= da;
            b -= db;

            if (std::abs(da) + std::abs(db) < nr_tol) break;
        }

        alpha_out[j] = a;
        beta_out[j] = b;
    }

    return List::create(
        Named("alpha") = alpha_out,
        Named("beta")  = beta_out
    );
}
