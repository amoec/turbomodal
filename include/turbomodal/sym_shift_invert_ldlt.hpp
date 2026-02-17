#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#ifdef TURBOMODAL_HAS_CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#include <memory>
#include <stdexcept>

namespace turbomodal {

// Drop-in replacement for Spectra::SymShiftInvert that uses
// SimplicialLDLT (or CHOLMOD when available) instead of SparseLU.
//
// LDLT stores only L + D (not L + U), roughly halving factorization
// memory for symmetric matrices.  Falls back to SparseLU if the
// LDLT factorization fails (e.g. zero pivot for indefinite K - sigma*M).
//
// Symbolic factorization is cached and reused across set_shift()
// calls when the sparsity pattern is unchanged.
//
// Template interface matches Spectra::SymShiftInvert:
//   Scalar, rows(), cols(), set_shift(sigma), perform_op(x_in, y_out)
template <typename Scalar_ = double>
class SymShiftInvertLDLT {
public:
    using Scalar = Scalar_;

private:
    using Index = Eigen::Index;
    using SpMat = Eigen::SparseMatrix<Scalar>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;

#ifdef TURBOMODAL_HAS_CHOLMOD
    using LDLTSolver = Eigen::CholmodSupernodalLLT<SpMat, Eigen::Lower>;
#else
    using LDLTSolver = Eigen::SimplicialLDLT<SpMat, Eigen::Lower>;
#endif
    using LUSolver = Eigen::SparseLU<SpMat>;

    const SpMat& m_K;
    const SpMat& m_M;
    const Index m_n;

    std::unique_ptr<LDLTSolver> m_ldlt;
    std::unique_ptr<LUSolver> m_lu;
    bool m_using_ldlt = true;
    bool m_pattern_set = false;

public:
    SymShiftInvertLDLT(const SpMat& K, const SpMat& M)
        : m_K(K), m_M(M), m_n(K.rows()) {}

    Index rows() const { return m_n; }
    Index cols() const { return m_n; }

    void reset_pattern() { m_pattern_set = false; }

    void set_shift(const Scalar& sigma) {
        SpMat mat = m_K - sigma * m_M;

        if (!m_pattern_set) {
            // First call: full symbolic + numeric factorization
            m_ldlt = std::make_unique<LDLTSolver>();
            m_ldlt->analyzePattern(mat);
            m_ldlt->factorize(mat);
            if (m_ldlt->info() == Eigen::Success) {
                m_using_ldlt = true;
                m_lu.reset();
                m_pattern_set = true;
                return;
            }

            // Fallback to SparseLU
            m_ldlt.reset();
            m_lu = std::make_unique<LUSolver>();
            m_lu->isSymmetric(true);
            m_lu->analyzePattern(mat);
            m_lu->factorize(mat);
            if (m_lu->info() != Eigen::Success) {
                throw std::invalid_argument(
                    "SymShiftInvertLDLT: both LDLT and LU factorization failed");
            }
            m_using_ldlt = false;
            m_pattern_set = true;
        } else {
            // Reuse symbolic, only do numeric factorization
            if (m_using_ldlt) {
                m_ldlt->factorize(mat);
                if (m_ldlt->info() != Eigen::Success) {
                    m_pattern_set = false;
                    set_shift(sigma);
                    return;
                }
            } else {
                m_lu->factorize(mat);
                if (m_lu->info() != Eigen::Success) {
                    m_pattern_set = false;
                    set_shift(sigma);
                    return;
                }
            }
        }
    }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        MapConstVec x(x_in, m_n);
        MapVec y(y_out, m_n);
        if (m_using_ldlt)
            y.noalias() = m_ldlt->solve(x);
        else
            y.noalias() = m_lu->solve(x);
    }
};

}  // namespace turbomodal
