#include "turbomodal/damping.hpp"

namespace turbomodal {

double DampingConfig::effective_damping(int mode_index, double omega_r) const {
    double zeta = 0.0;

    switch (type) {
        case Type::NONE:
            break;
        case Type::MODAL:
            if (!modal_damping_ratios.empty()) {
                int idx = std::min(mode_index,
                                   static_cast<int>(modal_damping_ratios.size()) - 1);
                zeta = modal_damping_ratios[idx];
            }
            break;
        case Type::RAYLEIGH:
            if (omega_r > 0.0) {
                zeta = rayleigh_alpha / (2.0 * omega_r) +
                       rayleigh_beta * omega_r / 2.0;
            }
            break;
    }

    // Add aerodynamic damping
    if (!aero_damping_ratios.empty()) {
        int idx = std::min(mode_index,
                           static_cast<int>(aero_damping_ratios.size()) - 1);
        zeta += aero_damping_ratios[idx];
    }

    return zeta;
}

}  // namespace turbomodal
