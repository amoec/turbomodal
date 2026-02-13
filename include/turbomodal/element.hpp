#pragma once

#include "turbomodal/common.hpp"
#include "turbomodal/material.hpp"

namespace turbomodal {

class TET10Element {
public:
    Matrix10x3d node_coords;  // 10 x 3 (x, y, z per node)

    Matrix30d stiffness(const Material& mat) const;
    Matrix30d mass(const Material& mat) const;

    // Shape functions at natural coordinates
    Vector10d shape_functions(double xi, double eta, double zeta) const;

    // Shape function derivatives w.r.t. natural coordinates (10 x 3)
    Matrix10x3d shape_derivatives(double xi, double eta, double zeta) const;

    // Jacobian matrix at a point (3 x 3)
    Eigen::Matrix3d jacobian(double xi, double eta, double zeta) const;

    // B matrix (strain-displacement) at a point (6 x 30)
    Matrix6x30d B_matrix(double xi, double eta, double zeta) const;

    // Gauss quadrature data
    static const std::array<Eigen::Vector3d, 4> gauss_points;
    static const std::array<double, 4> gauss_weights;
};

}  // namespace turbomodal
