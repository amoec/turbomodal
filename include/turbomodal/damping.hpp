#pragma once

#include "turbomodal/common.hpp"

namespace turbomodal {

struct DampingConfig {
    enum class Type { NONE, MODAL, RAYLEIGH };
    Type type = Type::NONE;

    // Modal damping: per-mode damping ratios (zeta_r)
    // If fewer values than modes, the last value is reused.
    std::vector<double> modal_damping_ratios;

    // Rayleigh damping: C = alpha*M + beta*K
    // zeta_r = alpha/(2*omega_r) + beta*omega_r/2
    double rayleigh_alpha = 0.0;
    double rayleigh_beta = 0.0;

    // Aerodynamic damping: additional per-mode damping ratio
    // Added on top of structural damping. If empty, no aero damping.
    std::vector<double> aero_damping_ratios;

    // Get effective damping ratio for mode at index mode_index with
    // natural frequency omega_r (rad/s).
    double effective_damping(int mode_index, double omega_r) const;
};

}  // namespace turbomodal
