#pragma once

#include "turbomodal/common.hpp"
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#ifdef TURBOMODAL_HAS_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

namespace turbomodal {

// Native complex Hermitian shift-invert Lanczos eigensolver.
//
// Solves the generalized eigenvalue problem K*x = λ*M*x where K and M
// are complex Hermitian sparse matrices, operating directly on the n×n
// complex system (no 2n×2n real doubling).
//
// Key property: for Hermitian K,M the Lanczos tridiagonal has real α,β,
// so the small eigenproblem is a standard real symmetric tridiagonal solve.
//
// Performance features:
//   - MV cache: maintains M*V alongside V to avoid redundant SpMV in reorthogonalization
//   - Symbolic factorization reuse: cached across calls with same sparsity pattern
//   - Implicit restart via QR shifts on real tridiagonal

class HermitianLanczosEigenSolver {
public:
    struct Config {
        int nev = 20;            // Number of eigenvalues to compute
        int ncv = 0;             // Subspace dimension (0 = auto: max(4*nev+1, 40))
        double shift = 0.0;      // Shift-invert shift σ
        double tolerance = 1e-8; // Convergence tolerance
        int max_iterations = 10; // Max implicit restarts
    };

    struct Result {
        Eigen::VectorXd eigenvalues;     // λ_i (ascending)
        Eigen::MatrixXcd eigenvectors;   // Corresponding eigenvectors (n × nconv)
        int nconv = 0;                   // Number of converged eigenvalues
        bool converged = false;
        std::string message;
    };

    HermitianLanczosEigenSolver() = default;

    // Solve K*x = λ*M*x. K and M must be complex Hermitian.
    Result solve(const SpMatcd& K, const SpMatcd& M, const Config& cfg);

    // Reset cached symbolic factorization (call if sparsity pattern changes)
    void reset_pattern() { m_pattern_set = false; }

    // Dense fallback for small systems (n < threshold).
    // Uses Eigen's GeneralizedSelfAdjointEigenSolver on dense matrices.
    static Result solve_dense(const SpMatcd& K, const SpMatcd& M, int nev);

private:
    // Complex LDLT-based shift-invert operator
#ifdef TURBOMODAL_HAS_CHOLMOD
    using ComplexLDLT = Eigen::CholmodSupernodalLLT<SpMatcd, Eigen::Lower>;
#else
    using ComplexLDLT = Eigen::SimplicialLDLT<SpMatcd, Eigen::Lower>;
#endif
    using ComplexLU = Eigen::SparseLU<SpMatcd>;

    std::unique_ptr<ComplexLDLT> m_ldlt;
    std::unique_ptr<ComplexLU> m_lu;
    bool m_using_ldlt = true;
    bool m_pattern_set = false;

    // Factorize (K - σM) with pattern caching
    bool factorize_shifted(const SpMatcd& K, const SpMatcd& M, double sigma);
    // Solve (K - σM) x = rhs
    Eigen::VectorXcd solve_shifted(const Eigen::VectorXcd& rhs) const;

    // Apply one round of implicit QR restart.
    // Compresses the Lanczos basis from ncv vectors down to nev, using
    // the unwanted Ritz values as QR shifts on the tridiagonal.
    //
    // V:     n × ncv Lanczos basis (modified in-place, first nev columns updated)
    // MV:    n × ncv M*V cache (modified in-place)
    // alpha: ncv diagonal of tridiagonal (modified in-place, first nev entries updated)
    // beta:  ncv-1 sub-diagonal (modified in-place, first nev-1 entries updated)
    // nev:   number of wanted eigenvalues to keep
    // ncv:   current subspace dimension
    // M:     mass matrix (for MV recomputation)
    void implicit_restart(
        Eigen::MatrixXcd& V, Eigen::MatrixXcd& MV,
        Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
        int nev, int ncv, const SpMatcd& M);
};

}  // namespace turbomodal
