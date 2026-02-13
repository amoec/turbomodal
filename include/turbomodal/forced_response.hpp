#pragma once

#include "turbomodal/common.hpp"
#include "turbomodal/modal_solver.hpp"
#include "turbomodal/damping.hpp"
#include "turbomodal/mesh.hpp"

namespace turbomodal {

struct ForcedResponseConfig {
    int engine_order = 1;
    double force_amplitude = 1.0;  // N or Pa

    enum class ExcitationType {
        UNIFORM_PRESSURE,        // Uniform on blade surfaces
        POINT_FORCE,             // Force at specific node
        SPATIAL_DISTRIBUTION,    // User-supplied force vector
    };
    ExcitationType excitation_type = ExcitationType::UNIFORM_PRESSURE;

    // For POINT_FORCE
    int force_node_id = -1;
    Eigen::Vector3d force_direction = Eigen::Vector3d(0, 0, 1);

    // For SPATIAL_DISTRIBUTION
    Eigen::VectorXcd force_vector;

    // Frequency sweep parameters
    double freq_min = 0.0;       // Hz (0 = auto: 0.5 * min natural freq)
    double freq_max = 0.0;       // Hz (0 = auto: 1.5 * max natural freq)
    int num_freq_points = 500;
};

struct ForcedResponseResult {
    int engine_order = 0;
    double rpm = 0.0;

    // Modal quantities
    Eigen::VectorXd natural_frequencies;          // Hz, (n_modes,)
    Eigen::VectorXcd modal_forces;                // Complex modal force, (n_modes,)
    Eigen::VectorXd modal_damping_ratios;         // Effective zeta, (n_modes,)
    Eigen::VectorXd participation_factors;        // Modal participation, (n_modes,)
    Eigen::VectorXd effective_modal_mass;         // Effective mass, (n_modes,)

    // FRF data
    Eigen::VectorXd sweep_frequencies;            // Hz, (n_freq,)
    Eigen::MatrixXcd modal_amplitudes;            // (n_modes, n_freq) complex
    Eigen::VectorXd max_response_amplitude;       // Peak over sweep, (n_modes,)
    Eigen::VectorXd resonance_frequencies;        // Hz at peak, (n_modes,)
};

class ForcedResponseSolver {
public:
    ForcedResponseSolver(const Mesh& mesh,
                         const DampingConfig& damping = DampingConfig());

    // Compute forced response from modal results at a given RPM.
    // Only modes matching the EO aliasing condition are excited:
    //   EO = k + m*N for integer m (where k = harmonic_index, N = num_sectors)
    ForcedResponseResult solve(
        const std::vector<ModalResult>& modal_results,
        double rpm,
        const ForcedResponseConfig& config) const;

    // Compute modal force: Q_r = phi_r^H * F
    static Eigen::VectorXcd compute_modal_forces(
        const Eigen::MatrixXcd& mode_shapes,
        const Eigen::VectorXcd& force_vector);

    // Single-DOF FRF: H_r(omega) = Q_r / (omega_r^2 - omega^2 + 2i*zeta_r*omega_r*omega)
    static std::complex<double> modal_frf(
        double omega,
        double omega_r,
        std::complex<double> Q_r,
        double zeta_r);

    // Modal participation factors: Gamma_r = phi_r^T * M * D
    static Eigen::VectorXd compute_participation_factors(
        const Eigen::MatrixXcd& mode_shapes,
        const SpMatd& M,
        const Eigen::Vector3d& direction);

    // Effective modal mass: m_eff_r = Gamma_r^2 / (phi_r^T * M * phi_r)
    static Eigen::VectorXd compute_effective_modal_mass(
        const Eigen::MatrixXcd& mode_shapes,
        const SpMatd& M,
        const Eigen::Vector3d& direction);

    // Build EO excitation force vector
    Eigen::VectorXcd build_eo_excitation(
        int engine_order,
        double amplitude,
        ForcedResponseConfig::ExcitationType type,
        int force_node_id = -1,
        const Eigen::Vector3d& force_dir = Eigen::Vector3d(0, 0, 1)) const;

private:
    const Mesh& mesh_;
    DampingConfig damping_;

    // Check if engine order EO excites harmonic index k for N sectors
    static bool eo_excites_harmonic(int eo, int k, int num_sectors);
};

}  // namespace turbomodal
