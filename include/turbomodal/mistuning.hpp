#pragma once

#include "turbomodal/common.hpp"

namespace turbomodal {

struct MistuningConfig {
    // Fractional blade frequency deviations: delta_omega_b / omega_ref
    // Length = num_sectors. Positive = stiffer, negative = softer.
    Eigen::VectorXd blade_frequency_deviations;

    // Tuned frequencies for ND=0..N/2 of one mode family (Hz)
    Eigen::VectorXd tuned_frequencies;

    // Which mode family index (0-based within each ND)
    int mode_family_index = 0;
};

struct MistuningResult {
    // Mistuned eigenfrequencies (N values for N blades)
    Eigen::VectorXd frequencies;

    // Mistuned mode shapes in blade amplitude coordinates (N x N complex)
    Eigen::MatrixXcd blade_amplitudes;

    // Amplitude magnification per mode: max_blade_amp / tuned_blade_amp
    Eigen::VectorXd amplitude_magnification;

    // Peak magnification across all modes
    double peak_magnification = 1.0;

    // Localization metric: inverse participation ratio (IPR)
    // IPR = N * sum(|a_b|^4) / (sum(|a_b|^2))^2, ranges from 1 (extended) to N (localized)
    Eigen::VectorXd localization_ipr;
};

class FMMSolver {
public:
    // Solve the Fundamental Mistuning Model for a single mode family.
    // num_sectors: number of blades N
    // tuned_frequencies: natural frequencies for ND=0..N/2 of this family (Hz)
    //   Length should be N/2+1 (or N/2 for odd N)
    // blade_frequency_deviations: fractional frequency deviations per blade (N values)
    static MistuningResult solve(
        int num_sectors,
        const Eigen::VectorXd& tuned_frequencies,
        const Eigen::VectorXd& blade_frequency_deviations);

    // Generate random mistuning pattern drawn from N(0, sigma)
    static Eigen::VectorXd random_mistuning(
        int num_sectors, double sigma, unsigned int seed = 42);
};

}  // namespace turbomodal
