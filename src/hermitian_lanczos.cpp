#include "turbomodal/hermitian_lanczos.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <stdexcept>

namespace turbomodal {

// ---- Factorization with pattern caching ----

bool HermitianLanczosEigenSolver::factorize_shifted(
    const SpMatcd& K, const SpMatcd& M, double sigma) {

    SpMatcd mat = K - std::complex<double>(sigma, 0.0) * M;

    if (!m_pattern_set) {
        // First call: try LDLT (faster for positive-definite shift)
        m_ldlt = std::make_unique<ComplexLDLT>();
        m_ldlt->analyzePattern(mat);
        m_ldlt->factorize(mat);
        if (m_ldlt->info() == Eigen::Success) {
            m_using_ldlt = true;
            m_pattern_set = true;
            return true;
        }
        // LDLT failed (matrix indefinite at this shift) — fall back to LU
        m_using_ldlt = false;
        m_lu = std::make_unique<ComplexLU>();
        m_lu->analyzePattern(mat);
        m_lu->factorize(mat);
        if (m_lu->info() != Eigen::Success) return false;
        m_pattern_set = true;
        return true;
    }

    if (m_using_ldlt) {
        m_ldlt->factorize(mat);
        if (m_ldlt->info() == Eigen::Success) return true;
        // LDLT failed on refactorization — switch permanently to LU
        m_using_ldlt = false;
        m_lu = std::make_unique<ComplexLU>();
        m_lu->analyzePattern(mat);
        m_lu->factorize(mat);
        if (m_lu->info() != Eigen::Success) { m_pattern_set = false; return false; }
        return true;
    }

    // Already using LU (pattern cached)
    m_lu->factorize(mat);
    if (m_lu->info() != Eigen::Success) {
        m_pattern_set = false;
        return factorize_shifted(K, M, sigma);
    }
    return true;
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

    // 3. Apply implicit QR shifts via Givens rotations on the tridiagonal.
    // Each shift applies a similarity transform T' = Q^T * T * Q where Q
    // is the orthogonal factor of QR(T - mu*I). For a tridiagonal, this is
    // done by chasing the bulge with Givens rotations.
    //
    // We use a sparse representation but apply the rotations on the dense
    // tridiagonal. The Q_total accumulation is O(ncv²) per shift, which
    // dominates, but this is still much cheaper than the O(ncv³) Householder
    // approach since we avoid forming full dense matrices for each QR.
    Eigen::MatrixXd Q_total = Eigen::MatrixXd::Identity(ncv, ncv);

    for (int s = 0; s < n_shifts; s++) {
        double mu = shifts[s];

        // Implicit QR step on tridiagonal with shift mu.
        // We work directly on alpha/beta vectors.
        // Step 1: Givens rotation G(0,1) to zero (alpha(0)-mu, beta(0))
        double bulge;  // fill-in element from rotation

        double x = alpha(0) - mu;
        double z = beta(0);

        for (int i = 0; i < ncv - 1; i++) {
            double r = std::hypot(x, z);
            double c, sn;
            if (r < 1e-300) {
                c = 1.0; sn = 0.0;
            } else {
                c = x / r; sn = z / r;
            }

            // Apply similarity G(i,i+1)^T * T * G(i,i+1) to tridiagonal.
            // Only entries involving rows/cols i and i+1 change.
            // We need to handle 3 entries: alpha(i), alpha(i+1), beta(i),
            // plus beta(i-1) (modified by previous rotation's right-multiply)
            // and beta(i+1) (creates bulge for next step).

            if (i > 0) {
                // beta(i-1) was modified by right-multiply of previous rotation
                // It should now be: r (the length computed above)
                beta(i - 1) = r;
            }

            double a0 = alpha(i);
            double a1 = alpha(i + 1);
            double b0 = beta(i);

            // Diagonal elements after similarity transform
            alpha(i)     = c * c * a0 + 2.0 * c * sn * b0 + sn * sn * a1;
            alpha(i + 1) = sn * sn * a0 - 2.0 * c * sn * b0 + c * c * a1;
            // Off-diagonal beta(i)
            beta(i) = c * sn * (a1 - a0) + (c * c - sn * sn) * b0;

            // Bulge: rotation creates fill at (i+2, i) = sn * beta(i+1)
            if (i + 2 < ncv) {
                bulge = sn * beta(i + 1);
                beta(i + 1) = c * beta(i + 1);
                // Next iteration chases this bulge
                x = beta(i);       // new off-diagonal (will be replaced by r)
                z = bulge;         // bulge element to zero out
            }

            // Accumulate Q_total columns i, i+1
            for (int row = 0; row < ncv; row++) {
                double q0 = Q_total(row, i);
                double q1 = Q_total(row, i + 1);
                Q_total(row, i)     = c * q0 + sn * q1;
                Q_total(row, i + 1) = -sn * q0 + c * q1;
            }
        }
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

    try {
    return solve_lanczos(K, M, cfg, nev, n);
    } catch (const std::exception& e) {
        result.message = std::string("Lanczos solver threw: ") + e.what();
        return result;
    }
}

HermitianLanczosEigenSolver::Result
HermitianLanczosEigenSolver::solve_lanczos(
    const SpMatcd& K, const SpMatcd& M, const Config& cfg, int nev, int n) {

    Result result;

    int ncv = cfg.ncv;
    if (ncv <= 0) {
        ncv = std::min(std::max(4 * nev + 1, 40), n);
    }
    ncv = std::min(ncv, n);

    // Factorize (K - σM)
    try {
        if (!factorize_shifted(K, M, cfg.shift)) {
            result.message = "Factorization failed for shift-invert operator";
            return result;
        }
    } catch (const std::exception& e) {
        result.message = std::string("Factorization threw: ") + e.what();
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

    for (int restart = 0; restart <= max_restarts; restart++) {
        // Determine start index: after restart, we already have nev vectors
        int j_start = (restart == 0) ? 0 : nev;

        // Lanczos iteration from j_start to ncv-1
        bool solve_threw = false;
        for (int j = j_start; j < ncv; j++) {
            // w = (K - σM)^{-1} M v_j
            Eigen::VectorXcd w;
            try {
                w = solve_shifted(MV.col(j));
            } catch (const std::exception& e) {
                result.message = std::string("Shift-invert solve threw: ") + e.what();
                solve_threw = true;
                break;
            }

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

        if (solve_threw) return result;

        // Build and solve the tridiagonal eigenproblem
        int active_ncv = std::min(ncv, n);
        // After early truncation (degenerate subspace), ensure alpha/beta
        // are consistently sized to prevent out-of-bounds access.
        alpha.conservativeResize(active_ncv);
        beta.conservativeResize(active_ncv);

        if (active_ncv < 1) {
            result.nconv = 0;
            result.converged = false;
            result.message = "Lanczos subspace collapsed (active_ncv=0)";
            return result;
        }

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
                if (ii < 0 || ii >= active_ncv) continue;
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
