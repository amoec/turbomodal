#include "turbomodal/hermitian_lanczos.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

namespace turbomodal {

// ---- Factorization with pattern caching ----

bool HermitianLanczosEigenSolver::factorize_shifted(
    const SpMatcd& K, const SpMatcd& M, double sigma) {

    SpMatcd mat = K - std::complex<double>(sigma, 0.0) * M;

    if (!m_pattern_set) {
        // First call: full symbolic + numeric factorization
        m_ldlt = std::make_unique<ComplexLDLT>();
        m_ldlt->analyzePattern(mat);
        m_ldlt->factorize(mat);
        if (m_ldlt->info() == Eigen::Success) {
            m_using_ldlt = true;
            m_lu.reset();
            m_pattern_set = true;
            return true;
        }

        // Fallback to SparseLU
        m_ldlt.reset();
        m_lu = std::make_unique<ComplexLU>();
        m_lu->analyzePattern(mat);
        m_lu->factorize(mat);
        if (m_lu->info() != Eigen::Success) {
            return false;
        }
        m_using_ldlt = false;
        m_pattern_set = true;
        return true;
    } else {
        // Reuse symbolic, only do numeric factorization
        if (m_using_ldlt) {
            m_ldlt->factorize(mat);
            if (m_ldlt->info() != Eigen::Success) {
                // Pattern might be stale; retry with full factorization
                m_pattern_set = false;
                return factorize_shifted(K, M, sigma);
            }
        } else {
            m_lu->factorize(mat);
            if (m_lu->info() != Eigen::Success) {
                m_pattern_set = false;
                return factorize_shifted(K, M, sigma);
            }
        }
        return true;
    }
}

Eigen::VectorXcd HermitianLanczosEigenSolver::solve_shifted(
    const Eigen::VectorXcd& rhs) const {
    if (m_using_ldlt)
        return m_ldlt->solve(rhs);
    else
        return m_lu->solve(rhs);
}

// ---- Dense fallback for small systems ----

HermitianLanczosEigenSolver::Result
HermitianLanczosEigenSolver::solve_dense(
    const SpMatcd& K, const SpMatcd& M, int nev) {

    Result result;
    int n = static_cast<int>(K.rows());
    nev = std::min(nev, n);

    Eigen::MatrixXcd K_dense = Eigen::MatrixXcd(K);
    Eigen::MatrixXcd M_dense = Eigen::MatrixXcd(M);

    // Ensure exact Hermitian symmetry
    K_dense = (K_dense + K_dense.adjoint()) * 0.5;
    M_dense = (M_dense + M_dense.adjoint()) * 0.5;

    // Eigen's GeneralizedSelfAdjointEigenSolver: K*x = λ*M*x
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcd> ges(K_dense, M_dense);
    if (ges.info() != Eigen::Success) {
        result.message = "Dense generalized eigensolver failed";
        return result;
    }

    Eigen::VectorXd all_evals = ges.eigenvalues().real();
    Eigen::MatrixXcd all_evecs = ges.eigenvectors();

    // Sort by ascending eigenvalue and take first nev
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return all_evals(a) < all_evals(b);
    });

    result.eigenvalues.resize(nev);
    result.eigenvectors.resize(n, nev);
    for (int i = 0; i < nev; i++) {
        result.eigenvalues(i) = all_evals(idx[i]);
        result.eigenvectors.col(i) = all_evecs.col(idx[i]);
    }
    result.nconv = nev;
    result.converged = true;
    result.message = "Dense solver";
    return result;
}

// ---- Implicit QR restart ----

void HermitianLanczosEigenSolver::implicit_restart(
    Eigen::MatrixXcd& V, Eigen::MatrixXcd& MV,
    Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
    int nev, int ncv, const SpMatcd& M) {

    // 1. Solve the tridiagonal eigenproblem to get Ritz values
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(ncv, ncv);
    for (int i = 0; i < ncv; i++) {
        T(i, i) = alpha(i);
        if (i < ncv - 1) {
            T(i, i + 1) = beta(i);
            T(i + 1, i) = beta(i);
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    Eigen::VectorXd ritz = es.eigenvalues();

    // 2. Sort Ritz values by ascending order.
    // In shift-invert space, the largest θ = 1/(λ-σ) correspond to the
    // smallest λ (the eigenvalues we WANT to keep).  The unwanted
    // eigenvalues are the SMALLEST θ (largest λ).
    std::vector<int> ritz_idx(ncv);
    std::iota(ritz_idx.begin(), ritz_idx.end(), 0);
    std::sort(ritz_idx.begin(), ritz_idx.end(), [&](int a, int b) {
        return ritz(a) < ritz(b);
    });

    // QR shifts = the (ncv - nev) SMALLEST Ritz values (unwanted)
    int n_shifts = ncv - nev;
    std::vector<double> shifts(n_shifts);
    for (int i = 0; i < n_shifts; i++) {
        shifts[i] = ritz(ritz_idx[i]);
    }

    // 3. Apply implicit QR shifts to the tridiagonal T.
    // Each shift: T - μI = QR, T' = RQ + μI = Q^T T Q.
    // Accumulate Q_total = Q_1 * Q_2 * ... * Q_{n_shifts}.
    Eigen::MatrixXd Q_total = Eigen::MatrixXd::Identity(ncv, ncv);

    for (int s = 0; s < n_shifts; s++) {
        // Build current T from alpha, beta
        Eigen::MatrixXd Ts = Eigen::MatrixXd::Zero(ncv, ncv);
        for (int i = 0; i < ncv; i++) {
            Ts(i, i) = alpha(i);
            if (i < ncv - 1) {
                Ts(i, i + 1) = beta(i);
                Ts(i + 1, i) = beta(i);
            }
        }

        // QR factorization of (T - μI)
        Eigen::MatrixXd shifted = Ts - shifts[s] * Eigen::MatrixXd::Identity(ncv, ncv);
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(shifted);
        Eigen::MatrixXd Q = qr.householderQ();
        Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

        // T' = Q^T * T * Q  (equivalent to RQ + μI)
        Eigen::MatrixXd T_new = Q.transpose() * Ts * Q;

        // Extract updated alpha, beta from T_new (it's still tridiagonal)
        for (int i = 0; i < ncv; i++) {
            alpha(i) = T_new(i, i);
            if (i < ncv - 1) {
                beta(i) = T_new(i, i + 1);
            }
        }

        Q_total = Q_total * Q;
    }

    // 4. Update the Lanczos basis V and MV cache.
    //
    // The standard implicit restart continuation vector is:
    //   f = T'(nev,nev-1) * V_old * Q(:,nev) + β_dangling * Q(ncv-1,nev-1) * v_{ncv}
    // where β_dangling is the original beta(ncv-1) (unchanged by QR shifts since
    // it lies outside the tridiagonal), and v_{ncv} = V.col(ncv) is the old
    // residual direction.
    int n = static_cast<int>(V.rows());
    double beta_tridiag = beta(nev - 1);   // T'(nev, nev-1) after QR shifts
    double beta_dangling = beta(ncv - 1);  // original dangling beta (never modified)

    // Compute continuation vector f from OLD V before overwrite.
    // V.col(ncv) is the old residual direction (not touched by basis rotation).
    Eigen::VectorXcd f = Eigen::VectorXcd::Zero(n);
    if (nev < ncv) {
        f = beta_tridiag * (V.leftCols(ncv) * Q_total.col(nev).cast<std::complex<double>>())
          + beta_dangling * Q_total(ncv - 1, nev - 1) * V.col(ncv);
    }

    // Compute the new basis vectors from OLD V
    Eigen::MatrixXcd V_new = V.leftCols(ncv) * Q_total.leftCols(nev).cast<std::complex<double>>();
    V.leftCols(nev) = V_new;

    // Normalize continuation vector and set beta(nev-1)
    if (nev < ncv) {
        Eigen::VectorXcd Mf = M * f;
        double beta_new = std::sqrt(std::abs(std::real(f.dot(Mf))));
        if (beta_new > 1e-14) {
            V.col(nev) = f / beta_new;
            MV.col(nev) = Mf / beta_new;
            beta(nev - 1) = beta_new;
        } else {
            beta(nev - 1) = 0.0;
        }
    }

    // Recompute MV cache for the rotated basis
    for (int j = 0; j < nev; j++) {
        MV.col(j) = M * V.col(j);
    }
}

// ---- Main solver ----

HermitianLanczosEigenSolver::Result
HermitianLanczosEigenSolver::solve(
    const SpMatcd& K, const SpMatcd& M, const Config& cfg) {

    Result result;
    int n = static_cast<int>(K.rows());

    if (n == 0) {
        result.message = "Empty matrices";
        return result;
    }

    int nev = std::min(cfg.nev, n - 1);
    if (nev <= 0) {
        result.message = "System too small for requested number of eigenvalues";
        return result;
    }

    // Dense fallback for small systems
    if (n <= 200) {
        return solve_dense(K, M, nev);
    }

    int ncv = cfg.ncv;
    if (ncv <= 0) {
        ncv = std::min(std::max(4 * nev + 1, 40), n);
    }
    ncv = std::min(ncv, n);

    // Factorize (K - σM)
    if (!factorize_shifted(K, M, cfg.shift)) {
        result.message = "Factorization failed for shift-invert operator";
        return result;
    }

    // Allocate Lanczos vectors and MV cache
    Eigen::MatrixXcd V = Eigen::MatrixXcd::Zero(n, ncv + 1);
    Eigen::MatrixXcd MV = Eigen::MatrixXcd::Zero(n, ncv + 1);
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(ncv);  // Diagonal of tridiagonal
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(ncv);   // Sub-diagonal

    // Initialize with random starting vector, M-normalized
    {
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < n; i++) {
            V(i, 0) = std::complex<double>(dist(gen), dist(gen));
        }
        Eigen::VectorXcd Mv0 = M * V.col(0);
        double m_norm = std::sqrt(std::real(V.col(0).dot(Mv0)));
        V.col(0) /= m_norm;
        MV.col(0) = Mv0 / m_norm;
    }

    int max_restarts = cfg.max_iterations;
    bool converged = false;

    for (int restart = 0; restart <= max_restarts; restart++) {
        // Determine start index: after restart, we already have nev vectors
        int j_start = (restart == 0) ? 0 : nev;

        // Lanczos iteration from j_start to ncv-1
        for (int j = j_start; j < ncv; j++) {
            // w = (K - σM)^{-1} M v_j
            Eigen::VectorXcd w = solve_shifted(MV.col(j));

            // α_j = real(v_j^H M w) — always real for Hermitian
            alpha(j) = std::real(MV.col(j).dot(w));

            // w = w - α_j v_j - β_{j-1} v_{j-1}
            w -= alpha(j) * V.col(j);
            if (j > 0) {
                w -= beta(j - 1) * V.col(j - 1);
            }

            // Full M-orthogonal reorthogonalization (using MV cache — no extra SpMV)
            for (int k = 0; k <= j; k++) {
                std::complex<double> h = MV.col(k).dot(w);
                w -= h * V.col(k);
            }

            // Compute M*w for the norm and cache
            Eigen::VectorXcd Mw = M * w;

            // β_j = sqrt(real(w^H M w))
            double beta_j = std::sqrt(std::abs(std::real(w.dot(Mw))));

            if (beta_j < 1e-14) {
                // Invariant subspace found — restart with random vector
                std::mt19937 gen(42 + restart * 100 + j);
                std::normal_distribution<double> dist(0.0, 1.0);
                for (int i = 0; i < n; i++) {
                    w(i) = std::complex<double>(dist(gen), dist(gen));
                }
                // Orthogonalize against existing basis
                for (int k = 0; k <= j; k++) {
                    std::complex<double> h = MV.col(k).dot(w);
                    w -= h * V.col(k);
                }
                Mw = M * w;
                beta_j = std::sqrt(std::abs(std::real(w.dot(Mw))));
                if (beta_j < 1e-14) {
                    // Truly degenerate — truncate
                    ncv = j + 1;
                    break;
                }
            }

            beta(j) = beta_j;

            if (j + 1 < ncv + 1) {
                V.col(j + 1) = w / beta_j;
                MV.col(j + 1) = Mw / beta_j;
            }
        }

        // Build and solve the tridiagonal eigenproblem
        int active_ncv = std::min(ncv, n);
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(active_ncv, active_ncv);
        for (int i = 0; i < active_ncv; i++) {
            T(i, i) = alpha(i);
            if (i < active_ncv - 1) {
                T(i, i + 1) = beta(i);
                T(i + 1, i) = beta(i);
            }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
        Eigen::VectorXd theta = es.eigenvalues();   // Ritz values (in shift-invert space)
        Eigen::MatrixXd S = es.eigenvectors();       // Ritz vectors in T-space

        // Convert shift-invert eigenvalues back: λ = σ + 1/ν
        // Spectra sorts by LargestMagn in shift-invert, which gives smallest λ.
        // Here θ are the Ritz values of (K-σM)^{-1}M, so λ = σ + 1/θ.
        // Sort by actual eigenvalue λ (ascending).
        std::vector<int> idx(active_ncv);
        std::iota(idx.begin(), idx.end(), 0);

        std::vector<double> lambda_vals(active_ncv);
        for (int i = 0; i < active_ncv; i++) {
            if (std::abs(theta(i)) > 1e-15) {
                lambda_vals[i] = cfg.shift + 1.0 / theta(i);
            } else {
                lambda_vals[i] = 1e30;  // Nearly zero Ritz value → very large eigenvalue
            }
        }

        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return lambda_vals[a] < lambda_vals[b];
        });

        // Check convergence: |β_{ncv-1} * e_{ncv-1}^T * s_i| < tol * |θ_i|
        double beta_last = (active_ncv > 0 && active_ncv - 1 < ncv) ? beta(active_ncv - 1) : 0.0;
        int nconv = 0;
        for (int i = 0; i < std::min(nev, active_ncv); i++) {
            int ii = idx[i];
            double residual = std::abs(beta_last * S(active_ncv - 1, ii));
            double threshold = cfg.tolerance * std::max(1.0, std::abs(theta(ii)));
            if (residual < threshold) {
                nconv++;
            }
        }

        if (nconv >= nev || restart == max_restarts) {
            // Extract results
            int num_extract = std::min(nconv > 0 ? nconv : nev, active_ncv);
            num_extract = std::min(num_extract, nev);
            result.eigenvalues.resize(num_extract);
            result.eigenvectors.resize(n, num_extract);

            for (int i = 0; i < num_extract; i++) {
                int ii = idx[i];
                result.eigenvalues(i) = lambda_vals[ii];
                // Eigenvector in original space: x = V * s
                result.eigenvectors.col(i) =
                    V.leftCols(active_ncv) * S.col(ii).cast<std::complex<double>>();
            }

            result.nconv = num_extract;
            result.converged = (nconv >= nev);
            if (result.converged) {
                result.message = "Converged";
            } else {
                result.message = "Not all eigenvalues converged (" +
                                 std::to_string(nconv) + "/" + std::to_string(nev) +
                                 ") after " + std::to_string(restart + 1) + " restarts";
            }
            return result;
        }

        // Implicit restart: compress from ncv to nev, continue iterating
        implicit_restart(V, MV, alpha, beta, nev, active_ncv, M);
    }

    // Should not reach here
    result.message = "Unexpected exit from Lanczos loop";
    return result;
}

}  // namespace turbomodal
