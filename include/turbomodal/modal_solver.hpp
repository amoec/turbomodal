#pragma once

#include "turbomodal/common.hpp"

namespace turbomodal {

struct ModalResult {
    int harmonic_index = 0;
    double rpm = 0.0;
    Eigen::VectorXd frequencies;       // Natural frequencies (Hz)
    Eigen::MatrixXcd mode_shapes;      // Complex mode shape vectors
    Eigen::VectorXi whirl_direction;   // +1 = FW, -1 = BW, 0 = standing

    // Compute circumferential wave propagation velocity for each mode (m/s)
    // radius: characteristic radius (m), e.g. mean blade radius
    // v_prop = 2*pi*f*R / ND  (returns 0 for ND=0)
    Eigen::VectorXd wave_propagation_velocity(double radius) const {
        int n = static_cast<int>(frequencies.size());
        Eigen::VectorXd v(n);
        for (int i = 0; i < n; i++) {
            if (harmonic_index > 0) {
                v(i) = 2.0 * PI * frequencies(i) * radius /
                       static_cast<double>(harmonic_index);
            } else {
                v(i) = 0.0;
            }
        }
        return v;
    }
};

struct SolverConfig {
    int nev = 20;            // Number of eigenvalues to compute
    int ncv = 0;             // Lanczos vectors (0 = auto: max(2*nev+1, 20))
    double shift = 0.0;      // Shift-invert shift sigma
    double tolerance = 1e-8;
    int max_iterations = 1000;
};

struct SolverStatus {
    bool converged = false;
    int num_converged = 0;
    int iterations = 0;
    double max_residual = 0.0;
    std::string message;
};

class ModalSolver {
public:
    // Solve real symmetric generalized eigenvalue problem: K*x = lambda*M*x
    // Returns eigenvalues as frequencies (Hz) and mode shapes
    std::pair<ModalResult, SolverStatus> solve_real(
        const SpMatd& K, const SpMatd& M,
        const SolverConfig& config = SolverConfig());

    // Solve complex Hermitian generalized eigenvalue problem via doubled-real-system
    // K and M are complex Hermitian sparse matrices
    std::pair<ModalResult, SolverStatus> solve_complex_hermitian(
        const SpMatcd& K, const SpMatcd& M,
        const SolverConfig& config = SolverConfig());

    // Eliminate constrained DOFs from K and M matrices
    // constrained_dofs: sorted vector of DOF indices to remove
    // Returns reduced K, M and the mapping from reduced to full DOF indices
    static std::tuple<SpMatd, SpMatd, std::vector<int>> eliminate_dofs(
        const SpMatd& K, const SpMatd& M,
        const std::vector<int>& constrained_dofs);

    // Expand reduced mode shapes back to full DOF space (zeros at constrained DOFs)
    static Eigen::MatrixXcd expand_mode_shapes(
        const Eigen::MatrixXcd& reduced_modes,
        int full_ndof,
        const std::vector<int>& free_dof_map);
};

}  // namespace turbomodal
