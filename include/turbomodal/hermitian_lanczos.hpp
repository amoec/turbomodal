#pragma once

#include "turbomodal/common.hpp"
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#ifdef TURBOMODAL_HAS_CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#include <complex>
#include <memory>

namespace turbomodal {

struct HermitianLanczosResult {
    Eigen::VectorXd eigenvalues;       // Real eigenvalues (lambda = omega^2)
    Eigen::MatrixXcd eigenvectors;     // Complex eigenvectors (n x nev)
    int num_converged = 0;
    bool converged = false;
};

// Solve K*x = lambda*M*x where K,M are complex Hermitian sparse.
//
// Uses shift-invert Lanczos operating directly on the n×n complex
// system — no doubling to 2n×2n real.  The Lanczos tridiagonal
// has real entries (guaranteed by Hermitian symmetry), so only
// the basis vectors are complex.
//
// Factorization uses CHOLMOD (supernodal, when available) or
// SimplicialLDLT (half the memory of SparseLU), with SparseLU
// as fallback.  Symbolic factorization is cached and reused
// across calls with the same sparsity pattern.
class HermitianLanczosEigenSolver {
public:
    HermitianLanczosResult solve(
        const SpMatcd& K, const SpMatcd& M,
        int nev, int ncv, double sigma,
        double tol = 1e-8, int max_iter = 1000);

    // Reset cached symbolic factorization (call when matrix size/pattern changes)
    void reset_pattern() { m_pattern_set = false; }

    // Dense solver for reference/testing — Cholesky-based, exact for small systems.
    static HermitianLanczosResult solve_dense_public(
        const SpMatcd& K, const SpMatcd& M, int nev, double sigma) {
        return solve_dense(K, M, nev, sigma);
    }

private:
#ifdef TURBOMODAL_HAS_CHOLMOD
    using LDLTSolver = Eigen::CholmodDecomposition<SpMatcd>;
#else
    using LDLTSolver = Eigen::SimplicialLDLT<SpMatcd, Eigen::Lower>;
#endif
    using LUSolver = Eigen::SparseLU<SpMatcd>;

    void factorize(const SpMatcd& K, const SpMatcd& M, double sigma);
    Eigen::VectorXcd solve_shift(const Eigen::VectorXcd& rhs) const;

    // Dense fallback for small systems where Lanczos has insufficient
    // Krylov subspace room (ncv ≈ n).
    static HermitianLanczosResult solve_dense(
        const SpMatcd& K, const SpMatcd& M, int nev, double sigma);

    const SpMatcd* m_M = nullptr;
    std::unique_ptr<LDLTSolver> m_ldlt;
    std::unique_ptr<LUSolver> m_lu;
    bool m_using_ldlt = true;
    bool m_pattern_set = false;
};

}  // namespace turbomodal
