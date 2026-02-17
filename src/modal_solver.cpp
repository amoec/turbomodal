#include "turbomodal/modal_solver.hpp"
#include "turbomodal/sym_shift_invert_ldlt.hpp"
#include "turbomodal/hermitian_lanczos.hpp"
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

    // Native complex Hermitian Lanczos — works directly on n×n complex
    // system instead of inflating to 2n×2n doubled-real.

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

    int ncv = config.ncv;
    if (ncv <= 0) {
        ncv = std::min(std::max(2 * nev + 1, 20), n);
    }
    ncv = std::min(ncv, n);

    HermitianLanczosResult lanczos_result = m_lanczos.solve(
        K_complex, M_complex, nev, ncv, config.shift,
        config.tolerance, config.max_iterations);

    status.num_converged = lanczos_result.num_converged;
    status.converged = lanczos_result.converged;
    status.message = lanczos_result.converged ? "Converged" :
        "Not all eigenvalues converged (" +
        std::to_string(lanczos_result.num_converged) + "/" +
        std::to_string(nev) + ")";

    int num_modes = static_cast<int>(lanczos_result.eigenvalues.size());
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

    result.whirl_direction = Eigen::VectorXi::Zero(num_modes);

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
