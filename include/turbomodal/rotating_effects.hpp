#pragma once

#include "turbomodal/common.hpp"
#include "turbomodal/element.hpp"
#include "turbomodal/material.hpp"

namespace turbomodal {

class RotatingEffects {
public:
    static Vector30d centrifugal_load(
        const TET10Element& elem, const Material& mat,
        double omega, const Eigen::Vector3d& axis);

    static Matrix30d stress_stiffening(
        const TET10Element& elem,
        const std::array<Vector6d, 4>& prestress);

    static Matrix30d spin_softening(
        const TET10Element& elem, const Material& mat, double omega);

    static Matrix30d gyroscopic(
        const TET10Element& elem, const Material& mat);
};

}  // namespace turbomodal
