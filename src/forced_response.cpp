#include "turbomodal/forced_response.hpp"
#include <algorithm>

namespace turbomodal {

ForcedResponseSolver::ForcedResponseSolver(
    const Mesh& mesh, const DampingConfig& damping)
    : mesh_(mesh), damping_(damping) {}

bool ForcedResponseSolver::eo_excites_harmonic(int eo, int k, int N) {
    // EO excitation at harmonic k if EO = k + m*N for some integer m
    // Also: aliased harmonics, EO = -k + m*N (backward)
    // Equivalently: (EO - k) % N == 0 or (EO + k) % N == 0
    if (N <= 0) return false;
    int eo_pos = ((eo % N) + N) % N;
    int k_pos = ((k % N) + N) % N;
    int k_neg = (((-k) % N) + N) % N;
    return (eo_pos == k_pos) || (eo_pos == k_neg);
}

ForcedResponseResult ForcedResponseSolver::solve(
    const std::vector<ModalResult>& modal_results,
    double rpm,
    const ForcedResponseConfig& config) const
{
    ForcedResponseResult result;
    result.engine_order = config.engine_order;
    result.rpm = rpm;

    int N = mesh_.num_sectors;

    // Find which harmonic indices are excited by this EO
    std::vector<int> excited_indices;
    for (size_t h = 0; h < modal_results.size(); h++) {
        int k = modal_results[h].harmonic_index;
        if (eo_excites_harmonic(config.engine_order, k, N)) {
            excited_indices.push_back(static_cast<int>(h));
        }
    }

    // Count total modes across excited harmonics
    int total_modes = 0;
    for (int hi : excited_indices) {
        total_modes += static_cast<int>(modal_results[hi].frequencies.size());
    }

    if (total_modes == 0) {
        // No modes excited — return empty result
        return result;
    }

    // Build excitation force vector
    Eigen::VectorXcd F = build_eo_excitation(
        config.engine_order, config.force_amplitude,
        config.excitation_type, config.force_node_id, config.force_direction);
    if (config.excitation_type == ForcedResponseConfig::ExcitationType::SPATIAL_DISTRIBUTION &&
        config.force_vector.size() > 0) {
        F = config.force_vector;
    }

    // Allocate result arrays
    result.natural_frequencies.resize(total_modes);
    result.modal_forces.resize(total_modes);
    result.modal_damping_ratios.resize(total_modes);
    result.participation_factors.resize(total_modes);
    result.effective_modal_mass.resize(total_modes);
    result.max_response_amplitude.resize(total_modes);
    result.resonance_frequencies.resize(total_modes);

    // Determine frequency sweep range
    double f_min = config.freq_min;
    double f_max = config.freq_max;

    // Collect all natural frequencies for auto-range
    if (f_min <= 0.0 || f_max <= 0.0) {
        double min_f = 1e30, max_f = 0.0;
        for (int hi : excited_indices) {
            for (int m = 0; m < modal_results[hi].frequencies.size(); m++) {
                double f = modal_results[hi].frequencies(m);
                min_f = std::min(min_f, f);
                max_f = std::max(max_f, f);
            }
        }
        if (f_min <= 0.0) f_min = 0.5 * min_f;
        if (f_max <= 0.0) f_max = 1.5 * max_f;
    }

    int n_freq = config.num_freq_points;
    result.sweep_frequencies.resize(n_freq);
    for (int i = 0; i < n_freq; i++) {
        result.sweep_frequencies(i) = f_min + (f_max - f_min) * i / std::max(n_freq - 1, 1);
    }

    result.modal_amplitudes.resize(total_modes, n_freq);
    result.modal_amplitudes.setZero();

    // Process each excited harmonic
    int mode_offset = 0;
    for (int hi : excited_indices) {
        const auto& mr = modal_results[hi];
        int n_modes = static_cast<int>(mr.frequencies.size());

        // Compute modal forces
        Eigen::VectorXcd Q;
        if (F.size() == mr.mode_shapes.rows()) {
            Q = compute_modal_forces(mr.mode_shapes, F);
        } else {
            // Force vector size mismatch — use unit modal force
            Q = Eigen::VectorXcd::Ones(n_modes);
        }

        for (int m = 0; m < n_modes; m++) {
            double f_n = mr.frequencies(m);
            double omega_n = 2.0 * PI * f_n;
            double zeta = damping_.effective_damping(m, omega_n);

            result.natural_frequencies(mode_offset + m) = f_n;
            result.modal_forces(mode_offset + m) = Q(m);
            result.modal_damping_ratios(mode_offset + m) = zeta;

            // Compute FRF over frequency sweep
            double max_amp = 0.0;
            double res_freq = f_n;
            for (int fi = 0; fi < n_freq; fi++) {
                double omega = 2.0 * PI * result.sweep_frequencies(fi);
                auto h = modal_frf(omega, omega_n, Q(m), zeta);
                result.modal_amplitudes(mode_offset + m, fi) = h;

                double amp = std::abs(h);
                if (amp > max_amp) {
                    max_amp = amp;
                    res_freq = result.sweep_frequencies(fi);
                }
            }

            result.max_response_amplitude(mode_offset + m) = max_amp;
            result.resonance_frequencies(mode_offset + m) = res_freq;
        }

        mode_offset += n_modes;
    }

    return result;
}

Eigen::VectorXcd ForcedResponseSolver::compute_modal_forces(
    const Eigen::MatrixXcd& mode_shapes,
    const Eigen::VectorXcd& force_vector)
{
    int n_modes = static_cast<int>(mode_shapes.cols());
    Eigen::VectorXcd Q(n_modes);
    for (int m = 0; m < n_modes; m++) {
        Q(m) = mode_shapes.col(m).conjugate().dot(force_vector);
    }
    return Q;
}

std::complex<double> ForcedResponseSolver::modal_frf(
    double omega, double omega_r,
    std::complex<double> Q_r, double zeta_r)
{
    // H_r(omega) = Q_r / (omega_r^2 - omega^2 + 2i*zeta*omega_r*omega)
    std::complex<double> denom(omega_r * omega_r - omega * omega,
                                2.0 * zeta_r * omega_r * omega);
    if (std::abs(denom) < 1e-30) {
        return Q_r * 1e30;  // Near-infinite response at exact resonance with zero damping
    }
    return Q_r / denom;
}

Eigen::VectorXd ForcedResponseSolver::compute_participation_factors(
    const Eigen::MatrixXcd& mode_shapes,
    const SpMatd& M,
    const Eigen::Vector3d& direction)
{
    int ndof = static_cast<int>(mode_shapes.rows());
    int n_modes = static_cast<int>(mode_shapes.cols());
    int n_nodes = ndof / 3;

    // Build direction vector D: repeat direction for each node
    Eigen::VectorXd D(ndof);
    for (int i = 0; i < n_nodes; i++) {
        D(3 * i)     = direction(0);
        D(3 * i + 1) = direction(1);
        D(3 * i + 2) = direction(2);
    }

    // Gamma_r = |phi_r^T * M * D|
    Eigen::VectorXd M_D = M * D;
    Eigen::VectorXd gamma(n_modes);
    for (int m = 0; m < n_modes; m++) {
        // Use real part of phi^H * M * D since M is real and D is real
        std::complex<double> g = mode_shapes.col(m).conjugate().dot(
            M_D.cast<std::complex<double>>());
        gamma(m) = std::abs(g);
    }
    return gamma;
}

Eigen::VectorXd ForcedResponseSolver::compute_effective_modal_mass(
    const Eigen::MatrixXcd& mode_shapes,
    const SpMatd& M,
    const Eigen::Vector3d& direction)
{
    Eigen::VectorXd gamma = compute_participation_factors(mode_shapes, M, direction);
    int n_modes = static_cast<int>(mode_shapes.cols());

    Eigen::VectorXd m_eff(n_modes);
    for (int m = 0; m < n_modes; m++) {
        // m_eff = Gamma^2 / (phi^T * M * phi)
        Eigen::VectorXcd M_phi = M.cast<std::complex<double>>() * mode_shapes.col(m);
        double gen_mass = std::abs(mode_shapes.col(m).conjugate().dot(M_phi));
        if (gen_mass > 1e-30) {
            m_eff(m) = gamma(m) * gamma(m) / gen_mass;
        } else {
            m_eff(m) = 0.0;
        }
    }
    return m_eff;
}

Eigen::VectorXcd ForcedResponseSolver::build_eo_excitation(
    int engine_order,
    double amplitude,
    ForcedResponseConfig::ExcitationType type,
    int force_node_id,
    const Eigen::Vector3d& force_dir) const
{
    int ndof = mesh_.num_dof();
    Eigen::VectorXcd F = Eigen::VectorXcd::Zero(ndof);

    if (type == ForcedResponseConfig::ExcitationType::POINT_FORCE) {
        if (force_node_id >= 0 && force_node_id < mesh_.num_nodes()) {
            for (int d = 0; d < 3; d++) {
                F(3 * force_node_id + d) = amplitude * force_dir(d);
            }
        }
        return F;
    }

    // UNIFORM_PRESSURE: apply force proportional to exp(i*EO*theta) at each node
    double sector_angle = 2.0 * PI / mesh_.num_sectors;
    for (int i = 0; i < mesh_.num_nodes(); i++) {
        double x = mesh_.nodes(i, 0);
        double y = mesh_.nodes(i, 1);
        double theta = std::atan2(y, x);

        // Phase from engine order
        double phase = engine_order * theta;
        std::complex<double> eo_phase(std::cos(phase), std::sin(phase));

        // Apply in z-direction (axial) for pressure loading
        F(3 * i + 2) = amplitude * eo_phase;
    }

    return F;
}

}  // namespace turbomodal
