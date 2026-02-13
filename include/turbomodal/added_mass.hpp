#pragma once

#include "turbomodal/common.hpp"

namespace turbomodal {

class AddedMassModel {
public:
    static double kwak_avmi(int nodal_diameter, double rho_fluid,
                            double rho_structure, double thickness,
                            double radius);

    static double frequency_ratio(int nodal_diameter, double rho_fluid,
                                  double rho_structure, double thickness,
                                  double radius);

    static SpMatd corrected_mass_matrix(
        const SpMatd& M_dry,
        int harmonic_index, double rho_fluid,
        double rho_structure, double thickness, double radius);

private:
    static const std::array<double, 11> gamma_coefficients_;
    static double gamma_n(int n);
};

}  // namespace turbomodal
