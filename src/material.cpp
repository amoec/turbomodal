#include "turbomodal/material.hpp"

namespace turbomodal {

Material::Material(double E, double nu, double rho)
    : E(E), nu(nu), rho(rho) {
    validate();
}

Material::Material(double E, double nu, double rho, double T_ref, double E_slope)
    : E(E), nu(nu), rho(rho), T_ref(T_ref), E_slope(E_slope) {
    validate();
}

void Material::validate() const {
    if (E <= 0.0)
        throw std::invalid_argument("Young's modulus E must be positive");
    if (nu <= 0.0 || nu >= 0.5)
        throw std::invalid_argument("Poisson's ratio nu must be in (0, 0.5)");
    if (rho <= 0.0)
        throw std::invalid_argument("Density rho must be positive");
}

Matrix6d Material::constitutive_matrix() const {
    double c = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    Matrix6d D = Matrix6d::Zero();
    D(0, 0) = D(1, 1) = D(2, 2) = c * (1.0 - nu);
    D(0, 1) = D(0, 2) = D(1, 0) = D(1, 2) = D(2, 0) = D(2, 1) = c * nu;
    D(3, 3) = D(4, 4) = D(5, 5) = c * (1.0 - 2.0 * nu) / 2.0;
    return D;
}

Material Material::at_temperature(double T) const {
    double E_adjusted = E + E_slope * (T - T_ref);
    if (E_adjusted <= 0.0)
        throw std::runtime_error("Temperature-adjusted E is non-positive");
    return Material(E_adjusted, nu, rho, T_ref, E_slope);
}

}  // namespace turbomodal
