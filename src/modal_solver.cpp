#include "turbomodal/modal_solver.hpp"
#include "turbomodal/hermitian_lanczos.hpp"
#include "turbomodal/sym_shift_invert_ldlt.hpp"
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <algorithm>
#include <numeric>
#include <set>

namespace turbomodal {

std::pair<ModalResult, SolverStatus> ModalSolver::solve_real(
    const SpMatd& K, const SpMatd& M,
    const SolverConfig& config) {

    ModalResult result;
    SolverStatus status;

    int n = static_cast<int>(K.rows());
    if (n == 0) {
        status.message = "Empty matrices";
        return {result, status};
    }

    int nev = std::min(config.nev, n - 1);
    if (nev <= 0) {
        status.message = "System too small for requested number of eigenvalues";
        return {result, status};
    }

    int ncv = config.ncv;
    if (ncv <= 0) {
        ncv = std::min(std::max(2 * nev + 1, 20), n);
    }
    ncv = std::min(ncv, n);

    // Shift-invert mode: solve (K - sigma*M)^{-1} M x = nu x
    // where nu = 1/(lambda - sigma)
    using OpType = SymShiftInvertLDLT<double>;
    using BOpType = Spectra::SparseSymMatProd<double>;

    OpType op(K, M);
    BOpType Bop(M);

    Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert>
        solver(op, Bop, nev, ncv, config.shift);

    solver.init();
    int nconv = solver.compute(Spectra::SortRule::LargestMagn,
                               config.max_iterations, config.tolerance);

    status.num_converged = nconv;
    status.converged = (solver.info() == Spectra::CompInfo::Successful);

    if (solver.info() == Spectra::CompInfo::Successful) {
        status.message = "Converged";
    } else if (solver.info() == Spectra::CompInfo::NotConverging) {
        status.message = "Not all eigenvalues converged (" +
                         std::to_string(nconv) + "/" + std::to_string(nev) + ")";
    } else {
        status.message = "Numerical issue in solver";
        return {result, status};
    }

    // Extract eigenvalues (these are lambda = omega^2)
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();

    // Sort by ascending eigenvalue (Spectra returns in shift-invert order)
    std::vector<int> idx(eigenvalues.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return eigenvalues(a) < eigenvalues(b);
    });

    int num_modes = static_cast<int>(eigenvalues.size());
    result.frequencies.resize(num_modes);
    result.mode_shapes.resize(n, num_modes);

    for (int i = 0; i < num_modes; i++) {
        double lambda = eigenvalues(idx[i]);
        // lambda = omega^2; f = sqrt(lambda) / (2*pi)
        if (lambda > 0.0) {
            result.frequencies(i) = std::sqrt(lambda) / (2.0 * PI);
        } else {
            result.frequencies(i) = 0.0;  // Rigid body mode or numerical noise
        }
        result.mode_shapes.col(i) = eigenvectors.col(idx[i]).cast<std::complex<double>>();
    }

    result.whirl_direction = Eigen::VectorXi::Zero(num_modes);

    return {result, status};
}

std::pair<ModalResult, SolverStatus> ModalSolver::solve_complex_hermitian(
    const SpMatcd& K_complex, const SpMatcd& M_complex,
    const SolverConfig& config) {

    // Solve complex Hermitian K*x = λ*M*x using native Hermitian Lanczos.
    // Operates directly on n×n complex system (no 2n×2n doubling).

    ModalResult result;
    SolverStatus status;

    int n = static_cast<int>(K_complex.rows());
    if (n == 0) {
        status.message = "Empty matrices";
        return {result, status};
    }

    int nev = std::min(config.nev, n - 1);
    if (nev <= 0) {
        status.message = "System too small for requested number of eigenvalues";
        return {result, status};
    }

    HermitianLanczosEigenSolver::Config lanczos_cfg;
    lanczos_cfg.nev = nev;
    lanczos_cfg.ncv = config.ncv;
    lanczos_cfg.shift = config.shift;
    lanczos_cfg.tolerance = config.tolerance;
    lanczos_cfg.max_iterations = 10;

    auto lanczos_result = m_hermitian_lanczos.solve(K_complex, M_complex, lanczos_cfg);

    int num_modes = lanczos_result.nconv;
    result.frequencies.resize(num_modes);
    result.mode_shapes.resize(n, num_modes);

    for (int i = 0; i < num_modes; i++) {
        double lambda = lanczos_result.eigenvalues(i);
        if (lambda > 0.0) {
            result.frequencies(i) = std::sqrt(lambda) / (2.0 * PI);
        } else {
            result.frequencies(i) = 0.0;
        }
        result.mode_shapes.col(i) = lanczos_result.eigenvectors.col(i);
    }

    status.num_converged = num_modes;
    status.converged = lanczos_result.converged;
    status.message = lanczos_result.message;
    result.whirl_direction = Eigen::VectorXi::Zero(num_modes);

    return {result, status};
}

// --- Lancaster QEP block matrix helpers ---

static SpMatcd build_lancaster_L1(const SpMatcd& D, const SpMatcd& K, int n) {
    // L1 = [D  K]
    //      [K  0]
    std::vector<TripletC> trips;
    trips.reserve(D.nonZeros() + 2 * K.nonZeros());

    // Upper-left block: D (rows 0..n-1, cols 0..n-1)
    for (int col = 0; col < D.outerSize(); ++col)
        for (SpMatcd::InnerIterator it(D, col); it; ++it)
            trips.emplace_back(it.row(), it.col(), it.value());

    // Upper-right block: K (rows 0..n-1, cols n..2n-1)
    for (int col = 0; col < K.outerSize(); ++col)
        for (SpMatcd::InnerIterator it(K, col); it; ++it)
            trips.emplace_back(it.row(), it.col() + n, it.value());

    // Lower-left block: K (rows n..2n-1, cols 0..n-1)
    for (int col = 0; col < K.outerSize(); ++col)
        for (SpMatcd::InnerIterator it(K, col); it; ++it)
            trips.emplace_back(it.row() + n, it.col(), it.value());

    // Lower-right block: 0 (nothing)

    SpMatcd L1(2 * n, 2 * n);
    L1.setFromTriplets(trips.begin(), trips.end());
    return L1;
}

static SpMatcd build_lancaster_L2(const SpMatcd& M, const SpMatcd& K, int n) {
    // L2 = [M  0]
    //      [0  K]
    std::vector<TripletC> trips;
    trips.reserve(M.nonZeros() + K.nonZeros());

    // Upper-left block: M
    for (int col = 0; col < M.outerSize(); ++col)
        for (SpMatcd::InnerIterator it(M, col); it; ++it)
            trips.emplace_back(it.row(), it.col(), it.value());

    // Lower-right block: K
    for (int col = 0; col < K.outerSize(); ++col)
        for (SpMatcd::InnerIterator it(K, col); it; ++it)
            trips.emplace_back(it.row() + n, it.col() + n, it.value());

    SpMatcd L2(2 * n, 2 * n);
    L2.setFromTriplets(trips.begin(), trips.end());
    return L2;
}

std::pair<ModalResult, SolverStatus> ModalSolver::solve_lancaster_qep(
    const SpMatcd& K, const SpMatcd& M, const SpMatcd& D,
    int num_modes,
    const SolverConfig& config) {

    ModalResult result;
    SolverStatus status;

    int n = static_cast<int>(K.rows());
    if (n == 0) {
        status.message = "Empty matrices";
        return {result, status};
    }

    // Build 2n×2n Lancaster system: L1*z = omega*L2*z
    int n2 = 2 * n;
    SpMatcd L1 = build_lancaster_L1(D, K, n);
    SpMatcd L2 = build_lancaster_L2(M, K, n);

    // Lancaster eigenvalues ω span both positive (FW) and negative (BW).
    // For small systems (n2 <= 200), the dense fallback returns eigenvalues in
    // ascending algebraic order, which would miss all positive (FW) eigenvalues.
    // Request all eigenvalues for the dense path so our FW/BW selection sees
    // the full spectrum.  For large systems, Lanczos shift-invert naturally
    // finds eigenvalues near the shift from both sides.
    int nev_request;
    if (n2 <= 200) {
        nev_request = n2 - 1;  // all eigenvalues (dense is fast for small n)
    } else {
        nev_request = std::min(4 * num_modes, n2 - 1);
    }
    if (nev_request <= 0) {
        status.message = "System too small for requested number of eigenvalues";
        return {result, status};
    }

    HermitianLanczosEigenSolver::Config lanczos_cfg;
    lanczos_cfg.nev = nev_request;
    lanczos_cfg.ncv = config.ncv;
    if (lanczos_cfg.ncv <= 0) {
        lanczos_cfg.ncv = std::min(std::max(4 * nev_request + 1, 40), n2);
    }
    // Shift: eigenvalues are ω (rad/s), not ω². Use sqrt of the standard
    // shift (which targets ω²) for consistent behavior.
    lanczos_cfg.shift = (config.shift > 0) ? std::sqrt(config.shift) : 0.1;
    lanczos_cfg.tolerance = config.tolerance;
    lanczos_cfg.max_iterations = 10;

    auto lanczos_result = m_lancaster_lanczos.solve(L1, L2, lanczos_cfg);

    int nconv = lanczos_result.nconv;
    if (nconv == 0) {
        status.message = lanczos_result.message;
        return {result, status};
    }

    // Eigenvalues are omega (rad/s). Positive = FW, negative = BW.
    // Collect into mode entries sorted by |omega|.
    struct ModeEntry {
        double omega;
        int orig_idx;
        int whirl;  // +1 FW, -1 BW
    };
    std::vector<ModeEntry> entries;
    entries.reserve(nconv);

    for (int i = 0; i < nconv; i++) {
        double omega_i = lanczos_result.eigenvalues(i);
        if (std::abs(omega_i) < 1e-6) continue;  // skip near-zero (rigid body)
        int whirl = (omega_i > 0) ? 1 : -1;
        entries.push_back({omega_i, i, whirl});
    }

    // Sort by ascending |omega|
    std::sort(entries.begin(), entries.end(), [](const ModeEntry& a, const ModeEntry& b) {
        return std::abs(a.omega) < std::abs(b.omega);
    });

    // Keep up to num_modes FW + num_modes BW, sorted by |frequency|
    int n_fw = 0, n_bw = 0;
    std::vector<ModeEntry> selected;
    for (const auto& e : entries) {
        if (e.whirl == 1 && n_fw < num_modes) {
            selected.push_back(e);
            n_fw++;
        } else if (e.whirl == -1 && n_bw < num_modes) {
            selected.push_back(e);
            n_bw++;
        }
        if (n_fw >= num_modes && n_bw >= num_modes) break;
    }

    // Re-sort selected by |omega|
    std::sort(selected.begin(), selected.end(), [](const ModeEntry& a, const ModeEntry& b) {
        return std::abs(a.omega) < std::abs(b.omega);
    });

    int total_modes = static_cast<int>(selected.size());
    result.frequencies.resize(total_modes);
    result.mode_shapes.resize(n, total_modes);  // first n rows of 2n eigenvector
    result.whirl_direction.resize(total_modes);

    for (int i = 0; i < total_modes; i++) {
        result.frequencies(i) = std::abs(selected[i].omega) / (2.0 * PI);
        result.mode_shapes.col(i) = lanczos_result.eigenvectors.col(selected[i].orig_idx).head(n);
        result.whirl_direction(i) = selected[i].whirl;
    }

    status.num_converged = total_modes;
    status.converged = lanczos_result.converged;
    status.message = lanczos_result.message;

    return {result, status};
}

std::tuple<SpMatd, SpMatd, std::vector<int>> ModalSolver::eliminate_dofs(
    const SpMatd& K, const SpMatd& M,
    const std::vector<int>& constrained_dofs) {

    int n = static_cast<int>(K.rows());
    std::set<int> constrained_set(constrained_dofs.begin(), constrained_dofs.end());

    // Build free DOF map
    std::vector<int> free_dof_map;
    free_dof_map.reserve(n - constrained_set.size());
    std::vector<int> full_to_reduced(n, -1);

    for (int i = 0; i < n; i++) {
        if (constrained_set.find(i) == constrained_set.end()) {
            full_to_reduced[i] = static_cast<int>(free_dof_map.size());
            free_dof_map.push_back(i);
        }
    }

    int n_red = static_cast<int>(free_dof_map.size());

    // Build reduced matrices
    std::vector<Triplet> k_trips, m_trips;
    for (int k = 0; k < K.outerSize(); ++k) {
        for (SpMatd::InnerIterator it(K, k); it; ++it) {
            int ri = full_to_reduced[it.row()];
            int rj = full_to_reduced[it.col()];
            if (ri >= 0 && rj >= 0) {
                k_trips.emplace_back(ri, rj, it.value());
            }
        }
    }
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SpMatd::InnerIterator it(M, k); it; ++it) {
            int ri = full_to_reduced[it.row()];
            int rj = full_to_reduced[it.col()];
            if (ri >= 0 && rj >= 0) {
                m_trips.emplace_back(ri, rj, it.value());
            }
        }
    }

    SpMatd K_red(n_red, n_red), M_red(n_red, n_red);
    K_red.setFromTriplets(k_trips.begin(), k_trips.end());
    M_red.setFromTriplets(m_trips.begin(), m_trips.end());

    return {K_red, M_red, free_dof_map};
}

Eigen::MatrixXcd ModalSolver::expand_mode_shapes(
    const Eigen::MatrixXcd& reduced_modes,
    int full_ndof,
    const std::vector<int>& free_dof_map) {

    int num_modes = static_cast<int>(reduced_modes.cols());
    Eigen::MatrixXcd full_modes = Eigen::MatrixXcd::Zero(full_ndof, num_modes);

    for (int m = 0; m < num_modes; m++) {
        for (int i = 0; i < static_cast<int>(free_dof_map.size()); i++) {
            full_modes(free_dof_map[i], m) = reduced_modes(i, m);
        }
    }

    return full_modes;
}

}  // namespace turbomodal
