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

    // 4-point Gauss quadrature (degree 2) — exact for stiffness on straight-sided TET10
    static const std::array<Eigen::Vector3d, 4> gauss_points;
    static const std::array<double, 4> gauss_weights;

    // 5-point Stroud quadrature (degree 3) — DEPRECATED: has negative centroid
    // weight (-2/15) that can produce indefinite element mass matrices.
    // Kept for reference only; use mass_gauss_points_14 for mass-type integrals.
    static const std::array<Eigen::Vector3d, 5> mass_gauss_points;
    static const std::array<double, 5> mass_gauss_weights;

    // Keast Rule 6: 14-point quadrature (degree 4) — for mass-type integrals
    // (N^T*N, degree 4).  All weights positive, guarantees positive-definite
    // element mass matrices.
    static const std::array<Eigen::Vector3d, 14> mass_gauss_points_14;
    static const std::array<double, 14> mass_gauss_weights_14;
};

}  // namespace turbomodal
