#pragma once

#include "turbomodal/common.hpp"

namespace turbomodal {

struct Material {
    double E;       // Young's modulus (Pa)
    double nu;      // Poisson's ratio
    double rho;     // Density (kg/m^3)

    double T_ref = 293.15;   // Reference temperature (K)
    double E_slope = 0.0;    // dE/dT (Pa/K, typically negative)

    Material(double E, double nu, double rho);
    Material(double E, double nu, double rho, double T_ref, double E_slope);

    Matrix6d constitutive_matrix() const;
    Material at_temperature(double T) const;
    void validate() const;
};

}  // namespace turbomodal
