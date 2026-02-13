#include "turbomodal/added_mass.hpp"

namespace turbomodal {

// Kwak (1991) Gamma_n coefficients for clamped-center free-edge circular disk
const std::array<double, 11> AddedMassModel::gamma_coefficients_ = {{
    0.6526, 0.3910, 0.2737, 0.2078, 0.1660,
    0.1372, 0.1163, 0.1003, 0.0878, 0.0777,
    0.0695
}};

double AddedMassModel::gamma_n(int n) {
    if (n < 0)
        throw std::invalid_argument("Nodal diameter must be non-negative");
    if (n < 11)
        return gamma_coefficients_[n];
    // Approximate formula for n > 10
    return 0.65 / std::pow(n + 0.05, 0.85);
}

double AddedMassModel::kwak_avmi(int nodal_diameter, double rho_fluid,
                                  double rho_structure, double thickness,
                                  double radius) {
    return gamma_n(nodal_diameter) * (rho_fluid / rho_structure) * (radius / thickness);
}

double AddedMassModel::frequency_ratio(int nodal_diameter, double rho_fluid,
                                        double rho_structure, double thickness,
                                        double radius) {
    double avmi = kwak_avmi(nodal_diameter, rho_fluid, rho_structure, thickness, radius);
    return 1.0 / std::sqrt(1.0 + avmi);
}

SpMatd AddedMassModel::corrected_mass_matrix(
    const SpMatd& M_dry,
    int harmonic_index, double rho_fluid,
    double rho_structure, double thickness, double radius) {
    double avmi = kwak_avmi(harmonic_index, rho_fluid, rho_structure, thickness, radius);
    return M_dry * (1.0 + avmi);
}

}  // namespace turbomodal
