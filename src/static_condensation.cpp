#include "turbomodal/static_condensation.hpp"

#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#ifdef TURBOMODAL_HAS_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

namespace turbomodal {

int compute_target_dofs(int n_free, int nev, size_t available_bytes) {
    // Invert the cost model: bytes = 800 * n^(4/3) + 2 * n * ncv * 16
    // where ncv = min(4*nev+1, n).  For large n, ncv = 4*nev+1.
    // Dominant term is 800 * n^(4/3) for large n.
    // Solve: available_bytes = 800 * n^(4/3) + 2 * n * (4*nev+1) * 16
    //      = 800 * n^(4/3) + 32 * n * (4*nev+1)
    //
    // Newton iteration starting from the factorization-only estimate:
    //   n0 = (available_bytes / 800)^(3/4)
    double budget = static_cast<double>(available_bytes);
    int ncv = 4 * nev + 1;

    // Initial guess from dominant term
    double n_est = std::pow(budget / 800.0, 3.0 / 4.0);

    // A few Newton iterations to refine
    for (int iter = 0; iter < 10; iter++) {
        double ncv_eff = std::min(static_cast<double>(ncv), n_est);
        double f = 800.0 * std::pow(n_est, 4.0 / 3.0)
                   + 2.0 * n_est * ncv_eff * 16.0 - budget;
        double df = 800.0 * (4.0 / 3.0) * std::pow(n_est, 1.0 / 3.0)
                    + 2.0 * ncv_eff * 16.0;
        if (std::abs(df) < 1e-12) break;
        double n_new = n_est - f / df;
        if (n_new < 1.0) { n_est = 1.0; break; }
        if (std::abs(n_new - n_est) < 1.0) { n_est = n_new; break; }
        n_est = n_new;
    }

    int target = static_cast<int>(std::floor(n_est));
    return std::max(1, std::min(target, n_free));
}

std::vector<int> select_master_dofs(
    const SpMatd& M,
    const std::set<int>& boundary_dofs,
    int n_target) {

    int n = static_cast<int>(M.rows());
    if (n_target >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    // Start with all boundary DOFs (mandatory for cyclic correctness)
    std::set<int> master_set(boundary_dofs.begin(), boundary_dofs.end());

    if (static_cast<int>(master_set.size()) >= n_target) {
        // Even boundary DOFs exceed target — just return them all
        return std::vector<int>(master_set.begin(), master_set.end());
    }

    // Rank interior DOFs by mass participation (diagonal of M)
    std::vector<std::pair<double, int>> interior_ranked;
    interior_ranked.reserve(n - static_cast<int>(boundary_dofs.size()));
    for (int i = 0; i < n; i++) {
        if (boundary_dofs.count(i) == 0) {
            double m_ii = M.coeff(i, i);
            interior_ranked.emplace_back(m_ii, i);
        }
    }

    // Sort descending by mass — keep DOFs with highest mass participation
    std::sort(interior_ranked.begin(), interior_ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int n_remaining = n_target - static_cast<int>(master_set.size());
    for (int i = 0; i < n_remaining && i < static_cast<int>(interior_ranked.size()); i++) {
        master_set.insert(interior_ranked[i].second);
    }

    return std::vector<int>(master_set.begin(), master_set.end());
}

CondensationResult condense(
    const SpMatd& K,
    const SpMatd& M,
    const std::vector<int>& master_dofs) {

    int n = static_cast<int>(K.rows());
    int n_m = static_cast<int>(master_dofs.size());
    int n_s = n - n_m;

    // Build master/slave DOF index maps
    std::set<int> master_set(master_dofs.begin(), master_dofs.end());
    std::vector<int> slave_dofs;
    slave_dofs.reserve(n_s);
    for (int i = 0; i < n; i++) {
        if (master_set.count(i) == 0) {
            slave_dofs.push_back(i);
        }
    }

    // Build permutation maps
    std::vector<int> full_to_master(n, -1);
    std::vector<int> full_to_slave(n, -1);
    for (int i = 0; i < n_m; i++) full_to_master[master_dofs[i]] = i;
    for (int i = 0; i < n_s; i++) full_to_slave[slave_dofs[i]] = i;

    // Extract sub-matrices: K_mm, K_ms, K_ss (and same for M)
    auto extract_block = [&](const SpMatd& A,
                             const std::vector<int>& row_dofs,
                             const std::vector<int>& col_dofs,
                             const std::vector<int>& full_to_row,
                             const std::vector<int>& full_to_col) -> SpMatd {
        int nr = static_cast<int>(row_dofs.size());
        int nc = static_cast<int>(col_dofs.size());
        std::vector<Triplet> trips;
        for (int col = 0; col < A.outerSize(); ++col) {
            int jj = full_to_col[col];
            if (jj < 0) continue;
            for (SpMatd::InnerIterator it(A, col); it; ++it) {
                int ii = full_to_row[static_cast<int>(it.row())];
                if (ii >= 0) {
                    trips.emplace_back(ii, jj, it.value());
                }
            }
        }
        SpMatd block(nr, nc);
        block.setFromTriplets(trips.begin(), trips.end());
        return block;
    };

    SpMatd K_mm = extract_block(K, master_dofs, master_dofs, full_to_master, full_to_master);
    SpMatd K_ms = extract_block(K, master_dofs, slave_dofs, full_to_master, full_to_slave);
    SpMatd K_sm = extract_block(K, slave_dofs, master_dofs, full_to_slave, full_to_master);
    SpMatd K_ss = extract_block(K, slave_dofs, slave_dofs, full_to_slave, full_to_slave);

    SpMatd M_mm = extract_block(M, master_dofs, master_dofs, full_to_master, full_to_master);
    SpMatd M_ms = extract_block(M, master_dofs, slave_dofs, full_to_master, full_to_slave);
    SpMatd M_sm = extract_block(M, slave_dofs, master_dofs, full_to_slave, full_to_master);
    SpMatd M_ss = extract_block(M, slave_dofs, slave_dofs, full_to_slave, full_to_slave);

    // Factorize K_ss and compute K_ss^{-1} * K_sm
    // T_sm = -K_ss^{-1} * K_sm  (slave DOFs as function of master DOFs)
    Eigen::MatrixXd K_sm_dense(K_sm);
    Eigen::MatrixXd T_sm_dense;

#ifdef TURBOMODAL_HAS_CHOLMOD
    Eigen::CholmodSupernodalLLT<SpMatd> solver;
    solver.compute(K_ss);
    if (solver.info() != Eigen::Success) {
        // Fall back to SparseLU if CHOLMOD fails (K_ss may not be SPD)
        Eigen::SparseLU<SpMatd> lu;
        lu.compute(K_ss);
        if (lu.info() != Eigen::Success) {
            throw std::runtime_error("Static condensation: K_ss factorization failed");
        }
        T_sm_dense = -lu.solve(K_sm_dense);
    } else {
        T_sm_dense = -solver.solve(K_sm_dense);
    }
#else
    Eigen::SparseLU<SpMatd> lu;
    lu.compute(K_ss);
    if (lu.info() != Eigen::Success) {
        throw std::runtime_error("Static condensation: K_ss factorization failed");
    }
    T_sm_dense = -lu.solve(K_sm_dense);
#endif

    // Build transformation matrix T_c (n x n_m):
    //   T_c = [I; T_sm]  (master DOFs = identity, slave DOFs = T_sm)
    // In full DOF ordering:
    std::vector<Triplet> tc_trips;
    tc_trips.reserve(n_m + T_sm_dense.rows() * T_sm_dense.cols());

    // Master rows: identity
    for (int i = 0; i < n_m; i++) {
        tc_trips.emplace_back(master_dofs[i], i, 1.0);
    }
    // Slave rows: T_sm
    for (int i = 0; i < n_s; i++) {
        for (int j = 0; j < n_m; j++) {
            double val = T_sm_dense(i, j);
            if (std::abs(val) > 1e-15) {
                tc_trips.emplace_back(slave_dofs[i], j, val);
            }
        }
    }

    SpMatd T_c(n, n_m);
    T_c.setFromTriplets(tc_trips.begin(), tc_trips.end());

    // Reduced matrices: K_r = T_c^T * K * T_c, M_r = T_c^T * M * T_c
    // Equivalent to: K_r = K_mm + K_ms*T_sm + T_sm^T*K_sm + T_sm^T*K_ss*T_sm
    // But T_c^T * K * T_c is simpler and numerically stable.
    SpMatd T_c_T = T_c.transpose();
    SpMatd K_reduced = (T_c_T * K * T_c).pruned(1e-15);
    SpMatd M_reduced = (T_c_T * M * T_c).pruned(1e-15);

    CondensationResult result;
    result.K_reduced = std::move(K_reduced);
    result.M_reduced = std::move(M_reduced);
    result.T_c = std::move(T_c);
    result.master_dofs = master_dofs;
    result.original_size = n;
    return result;
}

Eigen::MatrixXcd expand_modes(
    const SpMatd& T_c,
    const Eigen::MatrixXcd& reduced_modes) {
    // T_c is real, reduced_modes is complex
    int n_full = static_cast<int>(T_c.rows());
    int n_modes = static_cast<int>(reduced_modes.cols());
    Eigen::MatrixXcd full_modes(n_full, n_modes);
    for (int m = 0; m < n_modes; m++) {
        // Real and imaginary parts separately through T_c
        Eigen::VectorXd re = T_c * reduced_modes.col(m).real();
        Eigen::VectorXd im = T_c * reduced_modes.col(m).imag();
        full_modes.col(m) = re.cast<std::complex<double>>()
                           + std::complex<double>(0, 1) * im.cast<std::complex<double>>();
    }
    return full_modes;
}

}  // namespace turbomodal
