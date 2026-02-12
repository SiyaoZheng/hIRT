// Fused inference block for the latent trait model (LTM).
//
// Computes the marginal log-likelihood, OPG (outer product of gradients)
// information matrix, and standard errors in a single streaming pass.
// Replaces the R code path (lines 252-309 of hltm.R) that materializes
// large N-scaled intermediate matrices.
//
// Memory: O(J*K + d^2 + K) — no N-scaled intermediates.

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace Rcpp;
using hrclock = std::chrono::high_resolution_clock;
using dsec = std::chrono::duration<double>;

// Numerically stable log(1 + exp(x))
static inline double softplus_(double x) {
    if (x <= -37.0) return std::exp(x);
    if (x <=  18.0) return std::log1p(std::exp(x));
    if (x <=  33.3) return x + std::exp(-x);
    return x;
}

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

// [[Rcpp::export]]
Rcpp::List compute_inference_ltm_cpp(
    Rcpp::IntegerVector row_ptr,     // CSR row pointers (N+1)
    Rcpp::IntegerVector col_idx,     // CSR column indices (nnz, 0-based)
    Rcpp::IntegerVector values,      // CSR values (nnz, 0/1)
    Rcpp::NumericVector alpha,       // J item difficulties
    Rcpp::NumericVector beta,        // J item discriminations
    Rcpp::NumericVector theta_ls,    // K quadrature points
    Rcpp::NumericVector qw_ls,       // K quadrature weights
    Rcpp::NumericVector fitted_mean, // N prior means
    Rcpp::NumericVector fitted_var,  // N prior variances
    Rcpp::NumericMatrix x_mat,       // N x p covariate matrix
    Rcpp::NumericMatrix z_mat,       // N x q covariate matrix
    Rcpp::IntegerMatrix y_mat,       // N x J response matrix (NA_INTEGER for missing)
    bool verbose = false,            // diagnostic timing output
    Rcpp::Nullable<Rcpp::IntegerVector> freq_weights_ = R_NilValue
) {
    auto t_func_start = hrclock::now();

    const int N = fitted_mean.size();
    const int K = theta_ls.size();
    const int J = alpha.size();
    const int p = x_mat.ncol();
    const int q = z_mat.ncol();

    // Pattern collapsing: if freq_weights is provided, weight OPG rank-1
    // updates and log-likelihood by pattern frequency.
    const bool use_weights = freq_weights_.isNotNull();
    Rcpp::IntegerVector freq_weights;
    if (use_weights) freq_weights = Rcpp::IntegerVector(freq_weights_);

    // Free parameter count: 2J item params - 2 constraints + p + q
    const int d = 2 * J - 2 + p + q;

    // Raw pointers (declared early so verbose block can use them)
    const int*    rp  = row_ptr.begin();
    const int*    ci  = col_idx.begin();
    const int*    val = values.begin();
    const double* a   = alpha.begin();
    const double* b   = beta.begin();
    const double* tls = theta_ls.begin();
    const double* fm  = fitted_mean.begin();
    const double* fv  = fitted_var.begin();

    // --- Data structure analysis ---
    int nnz = row_ptr[N];
    if (verbose) {
        double na_pct = 1.0 - (double)nnz / ((double)N * J);
        double avg_items = (double)nnz / N;

        // Per-obs item count distribution
        int min_items = J, max_items = 0;
        std::vector<int> items_per_obs(N);
        for (int i = 0; i < N; i++) {
            items_per_obs[i] = rp[i+1] - rp[i];
            if (items_per_obs[i] < min_items) min_items = items_per_obs[i];
            if (items_per_obs[i] > max_items) max_items = items_per_obs[i];
        }
        std::sort(items_per_obs.begin(), items_per_obs.end());
        int median_items = items_per_obs[N/2];
        int q25_items = items_per_obs[N/4];
        int q75_items = items_per_obs[3*N/4];

        // Input parameter ranges
        double alpha_min = *std::min_element(a, a+J);
        double alpha_max = *std::max_element(a, a+J);
        double beta_min = *std::min_element(b, b+J);
        double beta_max = *std::max_element(b, b+J);
        double fm_min = *std::min_element(fm, fm+N);
        double fm_max = *std::max_element(fm, fm+N);
        double fv_min = *std::min_element(fv, fv+N);
        double fv_max = *std::max_element(fv, fv+N);
        double theta_min = tls[0], theta_max = tls[K-1];

        // Utility range analysis: how extreme are alpha + beta*theta values?
        double u_min = 1e30, u_max = -1e30;
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++) {
                double u = a[j] + b[j] * tls[k];
                if (u < u_min) u_min = u;
                if (u > u_max) u_max = u;
            }
        }

        Rcpp::Rcout << "\n========================================" << std::endl;
        Rcpp::Rcout << "  INFERENCE DIAGNOSTIC REPORT" << std::endl;
        Rcpp::Rcout << "========================================" << std::endl;

        Rcpp::Rcout << "\n[1] DATA STRUCTURE" << std::endl;
        Rcpp::Rcout << "  N=" << N << " J=" << J << " K=" << K
                    << " d=" << d << " p=" << p << " q=" << q << std::endl;
        Rcpp::Rcout << "  nnz=" << nnz << " NA%=" << (na_pct * 100)
                    << "% avg_items/obs=" << avg_items << std::endl;
        Rcpp::Rcout << "  Items/obs distribution: min=" << min_items
                    << " Q25=" << q25_items << " median=" << median_items
                    << " Q75=" << q75_items << " max=" << max_items << std::endl;

        Rcpp::Rcout << "\n[2] INPUT PARAMETER RANGES" << std::endl;
        Rcpp::Rcout << "  alpha: [" << alpha_min << ", " << alpha_max << "]" << std::endl;
        Rcpp::Rcout << "  beta:  [" << beta_min << ", " << beta_max << "]" << std::endl;
        Rcpp::Rcout << "  fitted_mean: [" << fm_min << ", " << fm_max << "]" << std::endl;
        Rcpp::Rcout << "  fitted_var:  [" << fv_min << ", " << fv_max << "]" << std::endl;
        Rcpp::Rcout << "  theta_ls:    [" << theta_min << ", " << theta_max << "]" << std::endl;
        Rcpp::Rcout << "  utility u=a+b*theta: [" << u_min << ", " << u_max << "]" << std::endl;
        if (u_max > 30 || u_min < -30) {
            Rcpp::Rcout << "  WARNING: extreme utility values -> softplus/sigmoid "
                        << "branches may affect numerical stability" << std::endl;
        }

        Rcpp::Rcout << "\n[3] COMPUTATIONAL COMPLEXITY" << std::endl;
        Rcpp::Rcout << "  info matrix: " << d << "x" << d
                    << " = " << (d*d) << " elements (" << (d*d*8) << " bytes)" << std::endl;
        Rcpp::Rcout << "  rank-1 updates: " << N << " x " << d << "^2 = "
                    << ((long long)N * d * d) << " FMAs" << std::endl;
        Rcpp::Rcout << "  item_score exp(): " << nnz << " x " << K << " = "
                    << ((long long)nnz * K) << " calls" << std::endl;
        Rcpp::Rcout << "  covar_score exp(): " << N << " x " << K << " = "
                    << ((long long)N * K) << " calls" << std::endl;
        Rcpp::Rcout << "  log-sum-exp exp(): " << N << " x " << K << " = "
                    << ((long long)N * K) << " calls" << std::endl;
        Rcpp::Rcout << "  TOTAL exp() calls: "
                    << ((long long)(nnz + 2*N) * K) << std::endl;

        // Compare to R's approach
        Rcpp::Rcout << "\n[4] R ORIGINAL APPROACH (for comparison)" << std::endl;
        Rcpp::Rcout << "  R materializes: pik(N*K)=" << ((long long)N*K*8/1024/1024) << "MB"
                    << " Lik(N*K)=" << ((long long)N*K*8/1024/1024) << "MB"
                    << " Lijk(K*N*J)=" << ((long long)K*N*J*8/1024/1024) << "MB" << std::endl;
        Rcpp::Rcout << "  R uses BLAS tcrossprod(d x N) for info matrix: "
                    << "single dgemm/dsyrk call" << std::endl;
        Rcpp::Rcout << "  C++ uses N=" << N << " rank-1 updates (no BLAS batching)" << std::endl;
        Rcpp::Rcout << "\n[5] STARTING MAIN LOOP..." << std::endl;
    }

    // ---- Phase 1: Precompute J*K tables ----
    auto t_precomp = hrclock::now();

    std::vector<double> log_lik_jk_1(J * K);
    std::vector<double> log_lik_jk_0(J * K);
    std::vector<double> dalpha_jk(J * K);

    for (int j = 0; j < J; j++) {
        for (int k = 0; k < K; k++) {
            const double u = a[j] + b[j] * tls[k];
            const double sp = softplus_(u);
            const double sig = sigmoid_(u);
            log_lik_jk_1[j * K + k] = u - sp;
            log_lik_jk_0[j * K + k] = -sp;
            dalpha_jk[j * K + k] = sig * (1.0 - sig);
        }
    }

    std::vector<double> log_qw(K);
    for (int k = 0; k < K; k++) {
        log_qw[k] = std::log(qw_ls[k]);
    }
    const double neg_half_log_2pi = -0.5 * std::log(2.0 * M_PI);

    double dt_precomp = dsec(hrclock::now() - t_precomp).count();

    // ---- Phase 2: Main streaming loop ----
    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(d, d);
    double total_log_lik = 0.0;

    std::vector<double> log_joint(K);
    std::vector<double> log_lik_ik(K);
    Eigen::VectorXd s_i(d);

    // Timing accumulators for inner phases
    double dt_logjoint = 0.0;    // Step 1: log_joint computation (sparse CSR + prior)
    double dt_logsumexp = 0.0;   // Step 2: log-sum-exp normalization
    double dt_item_scores = 0.0; // Step 3a: item parameter scores (leave-one-out exp)
    double dt_covar_scores = 0.0;// Step 3b: covariate scores (gamma, lambda)
    double dt_rank1 = 0.0;       // Step 4: rank-1 OPG update
    double dt_setzero = 0.0;     // s_i.setZero() overhead
    long long n_exp_item = 0;    // count of exp() calls in item scores
    long long n_exp_covar = 0;   // count of exp() calls in covariate scores
    long long n_exp_lse = 0;     // count of exp() calls in log-sum-exp

    // Intermediate value trackers
    double s_i_norm_sum = 0.0;       // accumulated L2 norm of score vectors
    double s_i_norm_max = 0.0;       // max L2 norm
    double log_Li_min = 1e30, log_Li_max = -1e30;
    double ratio_k_max = 0.0;       // max leave-one-out ratio seen
    int n_ratio_large = 0;           // count of ratio_k > 100
    int n_ratio_huge = 0;            // count of ratio_k > 1e6
    double info_trace_at_1pct = 0.0; // info trace at 1% progress
    double info_trace_at_10pct = 0.0;
    double info_trace_at_50pct = 0.0;

    // Checkpoints for sample observation dumps
    int checkpoint_indices[5] = {0, N/4, N/2, 3*N/4, N-1};

    auto t_loop_start = hrclock::now();

    for (int i = 0; i < N; i++) {
        const int start = rp[i];
        const int end   = rp[i + 1];
        const double sd_i = std::sqrt(fv[i]);
        const double log_sd_i = std::log(sd_i);
        const double inv_var_i = 1.0 / fv[i];

        // Step 1: log_joint
        auto t1 = hrclock::now();
        for (int k = 0; k < K; k++) {
            double ll = 0.0;
            for (int idx = start; idx < end; idx++) {
                const int j = ci[idx];
                const int y_ij = val[idx];
                const int jk = j * K + k;
                ll += (y_ij == 1) ? log_lik_jk_1[jk] : log_lik_jk_0[jk];
            }
            log_lik_ik[k] = ll;
            const double z = tls[k] - fm[i];
            const double log_prior = neg_half_log_2pi - log_sd_i - 0.5 * z * z * inv_var_i;
            log_joint[k] = ll + log_prior + log_qw[k];
        }
        dt_logjoint += dsec(hrclock::now() - t1).count();

        // Step 2: log-sum-exp
        auto t2 = hrclock::now();
        double max_lj = log_joint[0];
        for (int k = 1; k < K; k++) {
            if (log_joint[k] > max_lj) max_lj = log_joint[k];
        }
        double sum_exp = 0.0;
        for (int k = 0; k < K; k++) {
            sum_exp += std::exp(log_joint[k] - max_lj);
        }
        n_exp_lse += K;
        const double log_Li = max_lj + std::log(sum_exp);
        const double fw_i = use_weights ? (double)freq_weights[i] : 1.0;
        total_log_lik += fw_i * log_Li;
        dt_logsumexp += dsec(hrclock::now() - t2).count();

        if (log_Li < log_Li_min) log_Li_min = log_Li;
        if (log_Li > log_Li_max) log_Li_max = log_Li;

        // Step 3: Score vector
        auto t3a_start = hrclock::now();
        s_i.setZero();
        dt_setzero += dsec(hrclock::now() - t3a_start).count();

        // 3a: Item scores
        auto t3a = hrclock::now();
        for (int idx = start; idx < end; idx++) {
            const int j = ci[idx];
            const int y_ij = val[idx];
            const double sgn = (y_ij == 1) ? 1.0 : -1.0;

            double s_alpha_j = 0.0;
            double s_beta_j = 0.0;
            for (int k = 0; k < K; k++) {
                const int jk = j * K + k;
                const double log_lik_jk_y = (y_ij == 1) ? log_lik_jk_1[jk] : log_lik_jk_0[jk];
                const double ratio_k = std::exp(log_joint[k] - log_lik_jk_y - log_Li);
                n_exp_item++;
                if (verbose) {
                    if (ratio_k > ratio_k_max) ratio_k_max = ratio_k;
                    if (ratio_k > 100.0) n_ratio_large++;
                    if (ratio_k > 1e6) n_ratio_huge++;
                }
                const double da = dalpha_jk[jk];
                s_alpha_j += ratio_k * da;
                s_beta_j  += ratio_k * da * tls[k];
            }
            s_alpha_j *= sgn;
            s_beta_j  *= sgn;

            const int full_idx_alpha = 2 * j;
            const int full_idx_beta  = 2 * j + 1;
            if (full_idx_alpha != 0) {
                s_i(full_idx_alpha - 1) = s_alpha_j;
            }
            if (full_idx_beta != 2 * J - 1) {
                s_i(full_idx_beta - 1) = s_beta_j;
            }
        }
        dt_item_scores += dsec(hrclock::now() - t3a).count();

        // 3b: Covariate scores
        auto t3b = hrclock::now();
        double s_gamma_scalar = 0.0;
        double s_lambda_scalar = 0.0;
        for (int k = 0; k < K; k++) {
            const double w_k = std::exp(log_joint[k] - log_Li);
            n_exp_covar++;
            const double theta_dev = tls[k] - fm[i];
            s_gamma_scalar += w_k * theta_dev;
            s_lambda_scalar += w_k * 0.5 * (theta_dev * theta_dev * inv_var_i - 1.0);
        }
        s_gamma_scalar *= inv_var_i;

        const int gamma_start = 2 * J - 2;
        for (int l = 0; l < p; l++) {
            s_i(gamma_start + l) = x_mat(i, l) * s_gamma_scalar;
        }
        const int lambda_start = 2 * J - 2 + p;
        for (int l = 0; l < q; l++) {
            s_i(lambda_start + l) = z_mat(i, l) * s_lambda_scalar;
        }
        dt_covar_scores += dsec(hrclock::now() - t3b).count();

        // Step 4: Rank-1 OPG update (weighted by pattern frequency)
        auto t4 = hrclock::now();
        if (use_weights) {
            info.noalias() += fw_i * s_i * s_i.transpose();
        } else {
            info.noalias() += s_i * s_i.transpose();
        }
        dt_rank1 += dsec(hrclock::now() - t4).count();

        // Intermediate value tracking
        if (verbose) {
            double si_norm = s_i.norm();
            s_i_norm_sum += si_norm;
            if (si_norm > s_i_norm_max) s_i_norm_max = si_norm;

            // Checkpoint dumps for sample observations
            bool is_checkpoint = (i == checkpoint_indices[0] || i == checkpoint_indices[1]
                               || i == checkpoint_indices[2] || i == checkpoint_indices[3]
                               || i == checkpoint_indices[4]);
            if (is_checkpoint) {
                int n_items_i = end - start;
                Rcpp::Rcout << "\n  [Checkpoint obs i=" << i << "] items=" << n_items_i
                            << " log_Li=" << log_Li
                            << " s_i_norm=" << si_norm << std::endl;
                Rcpp::Rcout << "    log_joint range: [" << *std::min_element(log_joint.begin(), log_joint.end())
                            << ", " << *std::max_element(log_joint.begin(), log_joint.end()) << "]" << std::endl;
                // Print first few and last few elements of s_i
                Rcpp::Rcout << "    s_i[0..4]: ";
                for (int si = 0; si < std::min(5, d); si++) Rcpp::Rcout << s_i(si) << " ";
                Rcpp::Rcout << " ... s_i[d-2..d-1]: ";
                for (int si = std::max(0, d-2); si < d; si++) Rcpp::Rcout << s_i(si) << " ";
                Rcpp::Rcout << std::endl;
                Rcpp::Rcout << "    info trace=" << info.trace()
                            << " info max_diag=" << info.diagonal().maxCoeff()
                            << " info min_diag=" << info.diagonal().minCoeff() << std::endl;
            }

            // Track info trace at milestones
            if (i == N/100) info_trace_at_1pct = info.trace();
            if (i == N/10) info_trace_at_10pct = info.trace();
            if (i == N/2) info_trace_at_50pct = info.trace();
        }
    }

    double dt_loop = dsec(hrclock::now() - t_loop_start).count();

    // ---- Phase 3: Matrix inversion ----
    auto t_invert = hrclock::now();

    bool singular = false;
    Eigen::MatrixXd covmat(d, d);
    Eigen::VectorXd se_free(d);
    std::string invert_method = "none";

    if (verbose) {
        Rcpp::Rcout << "\n[6] INFO MATRIX PROPERTIES (before inversion)" << std::endl;
        Rcpp::Rcout << "  trace=" << info.trace()
                    << " max_diag=" << info.diagonal().maxCoeff()
                    << " min_diag=" << info.diagonal().minCoeff() << std::endl;
        Rcpp::Rcout << "  norm (Frobenius)=" << info.norm()
                    << " max_abs=" << info.cwiseAbs().maxCoeff() << std::endl;
        // Check for zeros on diagonal
        int n_zero_diag = 0;
        int n_tiny_diag = 0;
        for (int ii = 0; ii < d; ii++) {
            if (info(ii, ii) == 0.0) n_zero_diag++;
            if (info(ii, ii) < 1e-10) n_tiny_diag++;
        }
        Rcpp::Rcout << "  zero diag entries=" << n_zero_diag
                    << " tiny (<1e-10) diag entries=" << n_tiny_diag << std::endl;
    }

    Eigen::LLT<Eigen::MatrixXd> llt(info);
    if (llt.info() == Eigen::Success) {
        invert_method = "Cholesky (LLT)";
        covmat = llt.solve(Eigen::MatrixXd::Identity(d, d));
    } else {
        if (verbose) {
            Rcpp::Rcout << "  Cholesky FAILED -> trying LU fallback" << std::endl;
        }
        invert_method = "FullPivLU";
        Eigen::FullPivLU<Eigen::MatrixXd> lu(info);
        if (lu.isInvertible()) {
            covmat = lu.inverse();
            if (verbose) {
                Rcpp::Rcout << "  LU rank=" << lu.rank() << "/" << d << std::endl;
            }
        } else {
            singular = true;
            invert_method = "SINGULAR";
            covmat.setConstant(NA_REAL);
            if (verbose) {
                Rcpp::Rcout << "  LU ALSO FAILED: matrix is singular. rank=" << lu.rank() << "/" << d << std::endl;
            }
        }
    }

    if (!singular) {
        for (int i = 0; i < d; i++) {
            se_free(i) = std::sqrt(std::max(covmat(i, i), 0.0));
        }
    } else {
        se_free.setConstant(NA_REAL);
    }

    double dt_invert = dsec(hrclock::now() - t_invert).count();

    if (verbose) {
        Rcpp::Rcout << "  Inversion method: " << invert_method
                    << " (" << dt_invert * 1000 << " ms)" << std::endl;
    }

    // ---- Phase 4: Convert to R objects ----
    auto t_convert = hrclock::now();

    Rcpp::NumericMatrix covmat_r(d, d);
    Rcpp::NumericVector se_free_r(d);
    for (int i = 0; i < d; i++) {
        se_free_r[i] = se_free(i);
        for (int j = 0; j < d; j++) {
            covmat_r(i, j) = covmat(i, j);
        }
    }

    double dt_convert = dsec(hrclock::now() - t_convert).count();
    double dt_total = dsec(hrclock::now() - t_func_start).count();

    // ---- Diagnostic output ----
    if (verbose) {
        double dt_loop_accounted = dt_logjoint + dt_logsumexp + dt_item_scores
                                 + dt_covar_scores + dt_rank1 + dt_setzero;
        double dt_loop_unaccounted = dt_loop - dt_loop_accounted;

        Rcpp::Rcout << "\n[7] TIMING BREAKDOWN" << std::endl;
        Rcpp::Rcout << "  Phase 1 (precompute tables):    "
                    << dt_precomp * 1000 << " ms" << std::endl;
        Rcpp::Rcout << "  Phase 2 (main loop):            "
                    << dt_loop * 1000 << " ms  (100%)" << std::endl;
        Rcpp::Rcout << "    Step 1 (log_joint/CSR+prior): "
                    << dt_logjoint * 1000 << " ms  ("
                    << (dt_logjoint/dt_loop*100) << "%)" << std::endl;
        Rcpp::Rcout << "    Step 2 (log-sum-exp):         "
                    << dt_logsumexp * 1000 << " ms  ("
                    << (dt_logsumexp/dt_loop*100) << "%)" << std::endl;
        Rcpp::Rcout << "    Step 3a (item scores):        "
                    << dt_item_scores * 1000 << " ms  ("
                    << (dt_item_scores/dt_loop*100) << "%)" << std::endl;
        Rcpp::Rcout << "    Step 3b (covariate scores):   "
                    << dt_covar_scores * 1000 << " ms  ("
                    << (dt_covar_scores/dt_loop*100) << "%)" << std::endl;
        Rcpp::Rcout << "    Step 4 (rank-1 OPG update):   "
                    << dt_rank1 * 1000 << " ms  ("
                    << (dt_rank1/dt_loop*100) << "%)" << std::endl;
        Rcpp::Rcout << "    s_i.setZero():                "
                    << dt_setzero * 1000 << " ms  ("
                    << (dt_setzero/dt_loop*100) << "%)" << std::endl;
        Rcpp::Rcout << "    Loop overhead (unaccounted):   "
                    << dt_loop_unaccounted * 1000 << " ms  ("
                    << (dt_loop_unaccounted/dt_loop*100) << "%)" << std::endl;
        Rcpp::Rcout << "  Phase 3 (matrix inversion):     "
                    << dt_invert * 1000 << " ms  [" << invert_method << "]" << std::endl;
        Rcpp::Rcout << "  Phase 4 (R conversion):         "
                    << dt_convert * 1000 << " ms" << std::endl;
        Rcpp::Rcout << "  TOTAL:                          "
                    << dt_total * 1000 << " ms" << std::endl;

        Rcpp::Rcout << "\n[8] EXP() CALL ANALYSIS" << std::endl;
        Rcpp::Rcout << "  log-sum-exp:      " << n_exp_lse
                    << "  (" << (double)n_exp_lse/N << "/obs)" << std::endl;
        Rcpp::Rcout << "  item scores:      " << n_exp_item
                    << "  (" << (double)n_exp_item/N << "/obs)" << std::endl;
        Rcpp::Rcout << "  covariate scores: " << n_exp_covar
                    << "  (" << (double)n_exp_covar/N << "/obs)" << std::endl;
        long long total_exp = n_exp_lse + n_exp_item + n_exp_covar;
        Rcpp::Rcout << "  TOTAL exp():      " << total_exp << std::endl;
        // Estimate exp() cost: ~20ns per call on modern hardware
        Rcpp::Rcout << "  Estimated exp() time @ 20ns/call: "
                    << (total_exp * 20.0 / 1e6) << " ms" << std::endl;
        Rcpp::Rcout << "  Actual time in exp()-heavy steps:  "
                    << (dt_item_scores + dt_covar_scores + dt_logsumexp) * 1000
                    << " ms" << std::endl;

        Rcpp::Rcout << "\n[9] INTERMEDIATE VALUE DIAGNOSTICS" << std::endl;
        Rcpp::Rcout << "  log_Li range: [" << log_Li_min << ", " << log_Li_max << "]" << std::endl;
        Rcpp::Rcout << "  s_i norm: mean=" << (s_i_norm_sum/N)
                    << " max=" << s_i_norm_max << std::endl;
        Rcpp::Rcout << "  Leave-one-out ratio_k: max=" << ratio_k_max
                    << " n(>100)=" << n_ratio_large
                    << " n(>1e6)=" << n_ratio_huge << std::endl;
        Rcpp::Rcout << "  Info trace growth: 1%=" << info_trace_at_1pct
                    << " 10%=" << info_trace_at_10pct
                    << " 50%=" << info_trace_at_50pct
                    << " 100%=" << info.trace() << std::endl;

        Rcpp::Rcout << "\n[10] PER-OBS COST BREAKDOWN" << std::endl;
        Rcpp::Rcout << "  log_joint:   " << (dt_logjoint/N*1e6) << " us/obs" << std::endl;
        Rcpp::Rcout << "  log-sum-exp: " << (dt_logsumexp/N*1e6) << " us/obs" << std::endl;
        Rcpp::Rcout << "  item scores: " << (dt_item_scores/N*1e6) << " us/obs" << std::endl;
        Rcpp::Rcout << "  covar scores:" << (dt_covar_scores/N*1e6) << " us/obs" << std::endl;
        Rcpp::Rcout << "  rank-1 OPG:  " << (dt_rank1/N*1e6) << " us/obs" << std::endl;
        Rcpp::Rcout << "  setZero:     " << (dt_setzero/N*1e6) << " us/obs" << std::endl;
        Rcpp::Rcout << "  TOTAL:       " << (dt_loop/N*1e6) << " us/obs" << std::endl;

        Rcpp::Rcout << "\n[11] TIMING MEASUREMENT OVERHEAD" << std::endl;
        Rcpp::Rcout << "  chrono calls: " << (6 * N) << " pairs "
                    << "(~" << (6*N*20.0/1e6) << " ms est. overhead @ 20ns/call)"
                    << std::endl;

        Rcpp::Rcout << "\n[12] BOTTLENECK HYPOTHESIS" << std::endl;
        // Identify top 3 contributors
        struct Phase { const char* name; double ms; };
        Phase phases[] = {
            {"log_joint", dt_logjoint*1000},
            {"log-sum-exp", dt_logsumexp*1000},
            {"item_scores", dt_item_scores*1000},
            {"covar_scores", dt_covar_scores*1000},
            {"rank-1 OPG", dt_rank1*1000},
            {"setZero", dt_setzero*1000},
            {"mat_inversion", dt_invert*1000},
            {"R_conversion", dt_convert*1000},
            {"loop_overhead", dt_loop_unaccounted*1000}
        };
        // Sort descending
        for (int ii = 0; ii < 8; ii++)
            for (int jj = ii+1; jj < 9; jj++)
                if (phases[jj].ms > phases[ii].ms) std::swap(phases[ii], phases[jj]);

        Rcpp::Rcout << "  Top bottlenecks:" << std::endl;
        for (int ii = 0; ii < 3 && ii < 9; ii++) {
            Rcpp::Rcout << "    #" << (ii+1) << " " << phases[ii].name
                        << ": " << phases[ii].ms << " ms ("
                        << (phases[ii].ms/dt_total/10) << "%)" << std::endl;
        }

        Rcpp::Rcout << "\n========================================\n" << std::endl;
    }

    return Rcpp::List::create(
        Rcpp::Named("covmat")   = covmat_r,
        Rcpp::Named("log_Lik")  = total_log_lik,
        Rcpp::Named("se_free")  = se_free_r,
        Rcpp::Named("singular") = singular
    );
}
