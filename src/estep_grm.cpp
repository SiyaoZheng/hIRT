#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace Rcpp;

// Numerically stable sigmoid
static inline double sigmoid_(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        const double e = std::exp(x);
        return e / (1.0 + e);
    }
}

// Sparse E-step for the graded response model (GRM).
//
// Computes posterior quadrature weights, EAP and VAP estimates using
// a CSR representation of the ordinal response matrix.
// GRM log-likelihood: log[sigma(tau_upper + beta*theta) - sigma(tau_lower + beta*theta)]
//
// @param row_ptr       CSR row pointers (length N+1).
// @param col_idx       Column indices of observed entries (0-based item index).
// @param values        Observed ordinal responses (1-based: 1..H_j).
// @param alpha_flat    Flattened thresholds: [tau_1^(1),...,tau_{H1-1}^(1), ...].
// @param alpha_offsets Offsets into alpha_flat per item (length J+1).
// @param H             Number of response categories per item (length J).
// @param beta          Item discrimination parameters (length J).
// @param theta_ls      Gauss-Legendre quadrature points (length K).
// @param qw_ls         Gauss-Legendre quadrature weights (length K).
// @param fitted_mean   Prior means of latent trait (length N).
// @param fitted_var    Prior variances of latent trait (length N).
//
// @return A list with:
//   w         - K x N matrix of posterior quadrature weights
//   theta_eap - N-vector of EAP (posterior mean) estimates
//   theta_vap - N-vector of VAP (posterior variance) estimates
//   log_lik   - scalar marginal log-likelihood
//
// [[Rcpp::export]]
List compute_estep_grm_cpp(
    IntegerVector row_ptr,
    IntegerVector col_idx,
    IntegerVector values,
    NumericVector alpha_flat,
    IntegerVector alpha_offsets,
    IntegerVector H,
    NumericVector beta,
    NumericVector theta_ls,
    NumericVector qw_ls,
    NumericVector fitted_mean,
    NumericVector fitted_var,
    Rcpp::Nullable<IntegerVector> freq_weights_ = R_NilValue
) {
    const int N = fitted_mean.size();
    const int K = theta_ls.size();
    const int J = beta.size();
    const int n_alpha = alpha_flat.size();

    const bool use_weights = freq_weights_.isNotNull();
    IntegerVector freq_weights;
    if (use_weights) freq_weights = IntegerVector(freq_weights_);

    // Pre-compute log quadrature weights
    std::vector<double> log_qw(K);
    for (int k = 0; k < K; k++) {
        log_qw[k] = std::log(qw_ls[k]);
    }
    const double neg_half_log_2pi = -0.5 * std::log(2.0 * M_PI);

    // Raw pointers
    const int*    rp   = row_ptr.begin();
    const int*    ci   = col_idx.begin();
    const int*    val  = values.begin();
    const double* af   = alpha_flat.begin();
    const int*    aoff = alpha_offsets.begin();
    const int*    Hv   = H.begin();
    const double* b    = beta.begin();
    const double* tls  = theta_ls.begin();
    const double* fm   = fitted_mean.begin();
    const double* fv   = fitted_var.begin();

    // Precompute sigmoid table: sig_table[t * K + k] = sigma(alpha_flat[t] + beta[j] * theta_ls[k])
    // where t is the flat threshold index belonging to item j.
    std::vector<double> sig_table(n_alpha * K);
    for (int j = 0; j < J; j++) {
        for (int t = aoff[j]; t < aoff[j + 1]; t++) {
            for (int k = 0; k < K; k++) {
                sig_table[t * K + k] = sigmoid_(af[t] + b[j] * tls[k]);
            }
        }
    }

    // Output
    NumericMatrix w(K, N);
    NumericVector theta_eap(N);
    NumericVector theta_vap(N);
    double total_log_lik = 0.0;

    std::vector<double> log_post(K);

    for (int i = 0; i < N; i++) {
        const int start = rp[i];
        const int end   = rp[i + 1];
        const double sd_i = std::sqrt(fv[i]);
        const double log_sd_i = std::log(sd_i);
        const double inv_var_i = 1.0 / fv[i];

        for (int k = 0; k < K; k++) {
            // Log-likelihood: sum over observed items
            double loglik = 0.0;
            for (int idx = start; idx < end; idx++) {
                const int j = ci[idx];
                const int h = val[idx];       // 1-based category
                const int H_j = Hv[j];
                const int off_j = aoff[j];

                // Upper CDF: sigma(tau_{h-1} + beta_j * theta_k)
                double sig_upper;
                if (h == 1) {
                    sig_upper = 1.0;
                } else {
                    sig_upper = sig_table[(off_j + h - 2) * K + k];
                }

                // Lower CDF: sigma(tau_h + beta_j * theta_k)
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

            // Log-prior: log N(theta_k; fitted_mean_i, sd_i)
            const double z = tls[k] - fm[i];
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
        total_log_lik += use_weights ? freq_weights[i] * log_norm : log_norm;

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
        Named("theta_vap") = theta_vap,
        Named("log_lik")   = total_log_lik
    );
}


// Solve d x d dense linear system A*x = b in-place via Gaussian elimination
// with partial pivoting. On exit, b contains the solution.
// Returns false if matrix is singular.
static bool solve_dense(double* A, double* b, int d) {
    for (int k = 0; k < d; k++) {
        // Partial pivoting
        int maxrow = k;
        double maxval = std::abs(A[k * d + k]);
        for (int i = k + 1; i < d; i++) {
            double v = std::abs(A[i * d + k]);
            if (v > maxval) { maxval = v; maxrow = i; }
        }
        if (maxval < 1e-30) return false;

        if (maxrow != k) {
            for (int j = 0; j < d; j++) std::swap(A[k * d + j], A[maxrow * d + j]);
            std::swap(b[k], b[maxrow]);
        }

        // Eliminate
        for (int i = k + 1; i < d; i++) {
            double factor = A[i * d + k] / A[k * d + k];
            for (int j = k + 1; j < d; j++) {
                A[i * d + j] -= factor * A[k * d + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    for (int k = d - 1; k >= 0; k--) {
        for (int j = k + 1; j < d; j++) {
            b[k] -= A[k * d + j] * b[j];
        }
        b[k] /= A[k * d + k];
    }
    return true;
}


// Sparse M-step for the graded response model (GRM).
//
// Computes item parameters (thresholds + discriminations) from posterior
// weights and sparse ordinal response data.
//   Phase A: sparse sufficient statistics (single CSR pass)
//   Phase B: per-item Newton-Raphson (Fisher scoring) for the cumulative logit model
//
// @param row_ptr          CSR row pointers (length N+1).
// @param col_idx          Column indices (0-based).
// @param values           Observed ordinal responses (1-based: 1..H_j).
// @param w                K x N posterior weights from E-step.
// @param theta_ls         K quadrature points.
// @param alpha_flat_init  Starting thresholds (flat).
// @param alpha_offsets    Offsets into alpha_flat per item (length J+1).
// @param H                Number of categories per item (length J).
// @param beta_init        Starting discrimination parameters (length J).
// @param max_nr_iter      Max Newton-Raphson iterations (default 50).
// @param nr_tol           Convergence tolerance (default 1e-8).
// @param mu_prior         Prior mean for beta (default 0).
// @param sigma_prior      Prior SD for beta (default 1e30 = flat).
// @param prior_type       0 = lognormal on |beta|, 1 = Gaussian on beta.
//
// @return A list with:
//   alpha_flat - updated thresholds (flat vector)
//   beta       - updated discrimination parameters (J-vector)
//
// [[Rcpp::export]]
List compute_mstep_grm_cpp(
    IntegerVector row_ptr,
    IntegerVector col_idx,
    IntegerVector values,
    NumericMatrix w,
    NumericVector theta_ls,
    NumericVector alpha_flat_init,
    IntegerVector alpha_offsets,
    IntegerVector H,
    NumericVector beta_init,
    int max_nr_iter = 50,
    double nr_tol = 1e-8,
    double mu_prior = 0.0,
    double sigma_prior = 1e30,
    int prior_type = 1,
    Rcpp::Nullable<IntegerVector> freq_weights_ = R_NilValue
) {
    const int N = w.ncol();
    const int K = w.nrow();
    const int J = beta_init.size();

    const bool use_weights = freq_weights_.isNotNull();
    IntegerVector freq_weights;
    if (use_weights) freq_weights = IntegerVector(freq_weights_);

    // Raw pointers
    const int*    rp  = row_ptr.begin();
    const int*    ci  = col_idx.begin();
    const int*    val = values.begin();
    const double* tls = theta_ls.begin();
    const int*    aoff = alpha_offsets.begin();
    const int*    Hv   = H.begin();

    // Phase A: Sparse sufficient statistics
    // suff layout: for item j, suff[suff_offsets[j] + (h-1)*K + k]
    //   = sum_i w(k,i) * I(y_ij == h) [* freq_weights[i]]
    // h is 1-based (1..H_j), k is 0-based (0..K-1)
    std::vector<int> suff_offsets(J + 1, 0);
    for (int j = 0; j < J; j++) {
        suff_offsets[j + 1] = suff_offsets[j] + Hv[j] * K;
    }
    const int suff_total = suff_offsets[J];
    std::vector<double> suff(suff_total, 0.0);

    for (int i = 0; i < N; i++) {
        const int start = rp[i];
        const int end   = rp[i + 1];
        const double fw = use_weights ? (double)freq_weights[i] : 1.0;
        for (int idx = start; idx < end; idx++) {
            const int j = ci[idx];
            const int h = val[idx];  // 1-based
            const int base = suff_offsets[j] + (h - 1) * K;
            for (int k = 0; k < K; k++) {
                suff[base + k] += fw * w(k, i);
            }
        }
    }

    // Phase B: Per-item Newton-Raphson
    const int n_alpha = alpha_flat_init.size();
    NumericVector alpha_flat_out(n_alpha);
    NumericVector beta_out(J);

    // Copy initial values
    for (int t = 0; t < n_alpha; t++) alpha_flat_out[t] = alpha_flat_init[t];
    for (int j = 0; j < J; j++) beta_out[j] = beta_init[j];

    // Max item dimension (for stack allocation)
    int max_d = 0;
    for (int j = 0; j < J; j++) {
        if (Hv[j] > max_d) max_d = Hv[j];
    }
    // Working arrays (reused across items)
    std::vector<double> cdf(max_d);       // CDF values (nthresh entries)
    std::vector<double> f(max_d);         // sigmoid derivatives
    std::vector<double> p(max_d + 1);     // category probabilities (1-indexed)
    std::vector<double> grad(max_d + 1);  // gradient
    std::vector<double> hess((max_d + 1) * (max_d + 1));  // Hessian (Fisher info, negated)
    std::vector<double> delta(max_d + 1); // NR step

    for (int j = 0; j < J; j++) {
        const int H_j = Hv[j];
        const int nthresh = H_j - 1;
        const int d = H_j;  // nthresh + 1 (thresholds + beta)
        const int off_j = aoff[j];
        const int soff_j = suff_offsets[j];

        // Working parameters: tau[0..nthresh-1], beta_j
        double* tau = alpha_flat_out.begin() + off_j;
        double beta_j = beta_out[j];

        for (int nr_it = 0; nr_it < max_nr_iter; nr_it++) {
            // Zero gradient and Hessian
            std::fill(grad.begin(), grad.begin() + d, 0.0);
            std::fill(hess.begin(), hess.begin() + d * d, 0.0);

            for (int k = 0; k < K; k++) {
                const double theta_k = tls[k];

                // Compute CDFs and derivatives
                for (int m = 0; m < nthresh; m++) {
                    double u = tau[m] + beta_j * theta_k;
                    cdf[m] = sigmoid_(u);
                    f[m] = cdf[m] * (1.0 - cdf[m]);
                }

                // Category probabilities (1-indexed: p[1]..p[H_j])
                for (int h = 1; h <= H_j; h++) {
                    double upper = (h == 1) ? 1.0 : cdf[h - 2];
                    double lower = (h == H_j) ? 0.0 : cdf[h - 1];
                    p[h] = std::max(upper - lower, 1e-20);
                }

                // Total weight at this quadrature point
                double N_k = 0.0;
                for (int h = 1; h <= H_j; h++) {
                    N_k += suff[soff_j + (h - 1) * K + k];
                }

                // Accumulate gradient and Hessian over categories
                for (int h = 1; h <= H_j; h++) {
                    const double n_hk = suff[soff_j + (h - 1) * K + k];
                    const double inv_p = 1.0 / p[h];

                    // Partial derivatives of P(h) w.r.t. parameters
                    const bool has_upper = (h >= 2);
                    const bool has_lower = (h <= H_j - 1);
                    const int upper_m = h - 2;   // threshold index for upper CDF
                    const int lower_m = h - 1;   // threshold index for lower CDF

                    const double d_upper = has_upper ? f[upper_m] : 0.0;
                    const double d_lower = has_lower ? -f[lower_m] : 0.0;
                    const double d_beta = theta_k * ((has_upper ? f[upper_m] : 0.0)
                                                   - (has_lower ? f[lower_m] : 0.0));

                    // Gradient (observed)
                    if (has_upper) grad[upper_m] += n_hk * d_upper * inv_p;
                    if (has_lower) grad[lower_m] += n_hk * d_lower * inv_p;
                    grad[nthresh] += n_hk * d_beta * inv_p;

                    // Hessian: negative expected Fisher information
                    // H[a,b] -= N_k * dp_a * dp_b / P(h)
                    const double w_h = N_k * inv_p;

                    if (has_upper) {
                        hess[upper_m * d + upper_m] -= w_h * d_upper * d_upper;
                        if (has_lower) {
                            double cross = w_h * d_upper * d_lower;
                            hess[upper_m * d + lower_m] -= cross;
                            hess[lower_m * d + upper_m] -= cross;
                        }
                        double cross_b = w_h * d_upper * d_beta;
                        hess[upper_m * d + nthresh] -= cross_b;
                        hess[nthresh * d + upper_m] -= cross_b;
                    }
                    if (has_lower) {
                        hess[lower_m * d + lower_m] -= w_h * d_lower * d_lower;
                        double cross_b = w_h * d_lower * d_beta;
                        hess[lower_m * d + nthresh] -= cross_b;
                        hess[nthresh * d + lower_m] -= cross_b;
                    }
                    hess[nthresh * d + nthresh] -= w_h * d_beta * d_beta;
                }
            } // end quad point loop

            // Prior on beta
            if (sigma_prior < 1e30) {
                const double sig2 = sigma_prior * sigma_prior;
                if (prior_type == 1) {
                    // Gaussian N(0, sigma^2)
                    grad[nthresh] += -beta_j / sig2;
                    hess[nthresh * d + nthresh] += -1.0 / sig2;
                } else {
                    // Lognormal(mu_prior, sigma_prior^2) on |beta|
                    const double abs_b = std::abs(beta_j);
                    if (abs_b > 1e-10) {
                        const double log_abs_b = std::log(abs_b);
                        grad[nthresh] += -(1.0 + (log_abs_b - mu_prior) / sig2) / beta_j;
                        const double h_prior = (sig2 - 1.0 + log_abs_b - mu_prior) / (sig2 * beta_j * beta_j);
                        const double h_safe  = -1.0 / (sig2 * beta_j * beta_j);
                        hess[nthresh * d + nthresh] += std::min(h_prior, h_safe);
                    }
                }
            }

            // Solve Hessian * delta = gradient
            // Copy hess and grad into working arrays (solve_dense modifies in-place)
            std::vector<double> hess_work(hess.begin(), hess.begin() + d * d);
            for (int m = 0; m < d; m++) delta[m] = grad[m];

            if (!solve_dense(hess_work.data(), delta.data(), d)) {
                break;  // Singular Hessian, keep current params
            }

            // Cap step size
            const double max_step = 1.0;
            for (int m = 0; m < d; m++) {
                if (std::abs(delta[m]) > max_step) {
                    delta[m] = std::copysign(max_step, delta[m]);
                }
            }

            // Update: theta -= delta (NR with negative definite Hessian)
            for (int m = 0; m < nthresh; m++) {
                tau[m] -= delta[m];
            }
            beta_j -= delta[nthresh];

            // Enforce threshold ordering: tau[0] > tau[1] > ... > tau[nthresh-1]
            for (int m = 0; m < nthresh - 1; m++) {
                if (tau[m] <= tau[m + 1] + 1e-4) {
                    double avg = 0.5 * (tau[m] + tau[m + 1]);
                    tau[m]     = avg + 5e-5;
                    tau[m + 1] = avg - 5e-5;
                }
            }

            // Check convergence
            double max_delta = 0.0;
            for (int m = 0; m < d; m++) {
                double ad = std::abs(delta[m]);
                if (ad > max_delta) max_delta = ad;
            }
            if (max_delta < nr_tol) break;

        } // end NR loop

        beta_out[j] = beta_j;
    } // end item loop

    return List::create(
        Named("alpha_flat") = alpha_flat_out,
        Named("beta")       = beta_out
    );
}
