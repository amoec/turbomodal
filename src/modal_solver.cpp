#include "turbomodal/modal_solver.hpp"
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SymShiftInvert.h>
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
    using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
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

    // Doubled-real-system approach:
    // [K_real  -K_imag] [x_real]       [M_real  -M_imag] [x_real]
    // [K_imag   K_real] [x_imag] = λ   [M_imag   M_real] [x_imag]
    //
    // This converts the complex Hermitian eigenproblem to a real symmetric one
    // of double the size, which Spectra can solve directly.

    int n = static_cast<int>(K_complex.rows());

    // Extract real and imaginary parts
    // For Hermitian matrices: K_real is symmetric, K_imag is skew-symmetric
    SpMatd K_real(n, n), K_imag(n, n);
    SpMatd M_real(n, n), M_imag(n, n);

    {
        std::vector<Triplet> kr_trips, ki_trips, mr_trips, mi_trips;
        for (int k = 0; k < K_complex.outerSize(); ++k) {
            for (SpMatcd::InnerIterator it(K_complex, k); it; ++it) {
                if (std::abs(it.value().real()) > 1e-15)
                    kr_trips.emplace_back(it.row(), it.col(), it.value().real());
                if (std::abs(it.value().imag()) > 1e-15)
                    ki_trips.emplace_back(it.row(), it.col(), it.value().imag());
            }
        }
        for (int k = 0; k < M_complex.outerSize(); ++k) {
            for (SpMatcd::InnerIterator it(M_complex, k); it; ++it) {
                if (std::abs(it.value().real()) > 1e-15)
                    mr_trips.emplace_back(it.row(), it.col(), it.value().real());
                if (std::abs(it.value().imag()) > 1e-15)
                    mi_trips.emplace_back(it.row(), it.col(), it.value().imag());
            }
        }
        K_real.setFromTriplets(kr_trips.begin(), kr_trips.end());
        K_imag.setFromTriplets(ki_trips.begin(), ki_trips.end());
        M_real.setFromTriplets(mr_trips.begin(), mr_trips.end());
        M_imag.setFromTriplets(mi_trips.begin(), mi_trips.end());
    }

    // Build the doubled system (2n x 2n)
    int n2 = 2 * n;
    std::vector<Triplet> kd_trips, md_trips;
    kd_trips.reserve(K_real.nonZeros() * 2 + K_imag.nonZeros() * 2);
    md_trips.reserve(M_real.nonZeros() * 2 + M_imag.nonZeros() * 2);

    // K_doubled = [K_real, -K_imag; K_imag, K_real]
    for (int k = 0; k < K_real.outerSize(); ++k) {
        for (SpMatd::InnerIterator it(K_real, k); it; ++it) {
            kd_trips.emplace_back(it.row(), it.col(), it.value());           // top-left
            kd_trips.emplace_back(it.row() + n, it.col() + n, it.value());  // bottom-right
        }
    }
    for (int k = 0; k < K_imag.outerSize(); ++k) {
        for (SpMatd::InnerIterator it(K_imag, k); it; ++it) {
            kd_trips.emplace_back(it.row(), it.col() + n, -it.value());     // top-right: -K_imag
            kd_trips.emplace_back(it.row() + n, it.col(), it.value());      // bottom-left: K_imag
        }
    }

    // M_doubled = [M_real, -M_imag; M_imag, M_real]
    for (int k = 0; k < M_real.outerSize(); ++k) {
        for (SpMatd::InnerIterator it(M_real, k); it; ++it) {
            md_trips.emplace_back(it.row(), it.col(), it.value());
            md_trips.emplace_back(it.row() + n, it.col() + n, it.value());
        }
    }
    for (int k = 0; k < M_imag.outerSize(); ++k) {
        for (SpMatd::InnerIterator it(M_imag, k); it; ++it) {
            md_trips.emplace_back(it.row(), it.col() + n, -it.value());
            md_trips.emplace_back(it.row() + n, it.col(), it.value());
        }
    }

    SpMatd K_doubled(n2, n2), M_doubled(n2, n2);
    K_doubled.setFromTriplets(kd_trips.begin(), kd_trips.end());
    M_doubled.setFromTriplets(md_trips.begin(), md_trips.end());

    // Solve the doubled real system
    // Request 2*nev because eigenvalues come in conjugate pairs
    SolverConfig doubled_config = config;
    doubled_config.nev = std::min(2 * config.nev, n2 - 1);
    if (doubled_config.ncv <= 0) {
        doubled_config.ncv = std::min(std::max(2 * doubled_config.nev + 1, 20), n2);
    }

    auto [doubled_result, doubled_status] = solve_real(K_doubled, M_doubled, doubled_config);

    // Deduplicate conjugate pairs
    // In the doubled system, each physical eigenvalue appears twice
    ModalResult result;
    SolverStatus status = doubled_status;
    result.harmonic_index = 0;
    result.rpm = 0.0;

    int num_doubled = static_cast<int>(doubled_result.frequencies.size());
    std::vector<double> unique_freqs;
    std::vector<Eigen::VectorXcd> unique_modes;

    for (int i = 0; i < num_doubled; i++) {
        double f = doubled_result.frequencies(i);

        // Check if this frequency is already in our unique list
        bool is_duplicate = false;
        for (double uf : unique_freqs) {
            if (std::abs(f - uf) < 1e-6 * std::max(f, 1.0)) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate && static_cast<int>(unique_freqs.size()) < config.nev) {
            unique_freqs.push_back(f);
            // Reconstruct complex mode shape: x = x_real + i*x_imag
            Eigen::VectorXcd mode(n);
            Eigen::VectorXcd doubled_mode = doubled_result.mode_shapes.col(i);
            for (int j = 0; j < n; j++) {
                mode(j) = std::complex<double>(doubled_mode(j).real(),
                                                doubled_mode(j + n).real());
            }
            unique_modes.push_back(mode);
        }
    }

    int num_modes = static_cast<int>(unique_freqs.size());
    result.frequencies.resize(num_modes);
    result.mode_shapes.resize(n, num_modes);
    for (int i = 0; i < num_modes; i++) {
        result.frequencies(i) = unique_freqs[i];
        result.mode_shapes.col(i) = unique_modes[i];
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
