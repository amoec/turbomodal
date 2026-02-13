#include "turbomodal/rotating_effects.hpp"

namespace turbomodal {

Vector30d RotatingEffects::centrifugal_load(
    const TET10Element& elem, const Material& mat,
    double omega, const Eigen::Vector3d& axis) {

    // F_centrifugal = omega^2 * integral(rho * N_mat^T * r_perp dV)
    // where r_perp is the radial position vector perpendicular to rotation axis.
    // For z-axis rotation: r_perp = (x, y, 0) at each point.
    //
    // More generally, r_perp = pos - (pos . axis_hat) * axis_hat
    // where pos is the physical position at the Gauss point.

    Eigen::Vector3d axis_hat = axis.normalized();
    Vector30d Fe = Vector30d::Zero();

    for (int gp = 0; gp < 4; gp++) {
        double xi   = TET10Element::gauss_points[gp](0);
        double eta  = TET10Element::gauss_points[gp](1);
        double zeta = TET10Element::gauss_points[gp](2);
        double w    = TET10Element::gauss_weights[gp];

        Vector10d N = elem.shape_functions(xi, eta, zeta);
        Eigen::Matrix3d J = elem.jacobian(xi, eta, zeta);
        double detJ = J.determinant();

        // Physical position at Gauss point
        Eigen::Vector3d pos = elem.node_coords.transpose() * N;  // 3x10 * 10x1

        // Perpendicular distance vector from rotation axis
        Eigen::Vector3d r_perp = pos - pos.dot(axis_hat) * axis_hat;

        // Build 3x30 shape function matrix N_mat
        Matrix3x30d N_mat = Matrix3x30d::Zero();
        for (int i = 0; i < 10; i++) {
            N_mat(0, 3 * i)     = N(i);
            N_mat(1, 3 * i + 1) = N(i);
            N_mat(2, 3 * i + 2) = N(i);
        }

        // F += rho * omega^2 * N_mat^T * r_perp * detJ * w
        Fe.noalias() += mat.rho * omega * omega * N_mat.transpose() * r_perp * (detJ * w);
    }

    return Fe;
}

Matrix30d RotatingEffects::stress_stiffening(
    const TET10Element& elem,
    const std::array<Vector6d, 4>& prestress) {

    // K_sigma = integral(B_NL^T * S * B_NL dV)
    //
    // B_NL is 9x30 geometric strain-displacement matrix:
    //   For node i (columns 3i, 3i+1, 3i+2):
    //   Row 0: [dNi/dx,  0,       0      ]   du/dx
    //   Row 1: [dNi/dy,  0,       0      ]   du/dy
    //   Row 2: [dNi/dz,  0,       0      ]   du/dz
    //   Row 3: [0,       dNi/dx,  0      ]   dv/dx
    //   Row 4: [0,       dNi/dy,  0      ]   dv/dy
    //   Row 5: [0,       dNi/dz,  0      ]   dv/dz
    //   Row 6: [0,       0,       dNi/dx ]   dw/dx
    //   Row 7: [0,       0,       dNi/dy ]   dw/dy
    //   Row 8: [0,       0,       dNi/dz ]   dw/dz
    //
    // S = blkdiag(sigma_mat, sigma_mat, sigma_mat) where:
    //   sigma_mat = [s_xx  s_xy  s_xz]
    //               [s_xy  s_yy  s_yz]
    //               [s_xz  s_yz  s_zz]
    //
    // Voigt: sigma_0 = {s_xx, s_yy, s_zz, s_xy, s_yz, s_xz}

    Matrix30d Ks = Matrix30d::Zero();

    for (int gp = 0; gp < 4; gp++) {
        double xi   = TET10Element::gauss_points[gp](0);
        double eta  = TET10Element::gauss_points[gp](1);
        double zeta = TET10Element::gauss_points[gp](2);
        double w    = TET10Element::gauss_weights[gp];

        // Compute physical derivatives dN/dx
        Matrix10x3d dNdxi = elem.shape_derivatives(xi, eta, zeta);
        Eigen::Matrix3d J = dNdxi.transpose() * elem.node_coords;
        double detJ = J.determinant();
        Eigen::Matrix3d Jinv = J.inverse();
        Matrix10x3d dNdx = dNdxi * Jinv.transpose();

        // Build stress matrix (3x3) from Voigt notation
        const Vector6d& s = prestress[gp];
        Eigen::Matrix3d sigma_mat;
        sigma_mat(0, 0) = s(0);  // s_xx
        sigma_mat(1, 1) = s(1);  // s_yy
        sigma_mat(2, 2) = s(2);  // s_zz
        sigma_mat(0, 1) = s(3);  sigma_mat(1, 0) = s(3);  // s_xy
        sigma_mat(1, 2) = s(4);  sigma_mat(2, 1) = s(4);  // s_yz
        sigma_mat(0, 2) = s(5);  sigma_mat(2, 0) = s(5);  // s_xz

        // Build S = blkdiag(sigma_mat, sigma_mat, sigma_mat) — 9x9
        Matrix9d S = Matrix9d::Zero();
        S.block<3, 3>(0, 0) = sigma_mat;
        S.block<3, 3>(3, 3) = sigma_mat;
        S.block<3, 3>(6, 6) = sigma_mat;

        // Build B_NL (9x30)
        Matrix9x30d B_NL = Matrix9x30d::Zero();
        for (int i = 0; i < 10; i++) {
            double dNdx_i = dNdx(i, 0);
            double dNdy_i = dNdx(i, 1);
            double dNdz_i = dNdx(i, 2);
            int c = 3 * i;

            // du/dx, du/dy, du/dz
            B_NL(0, c) = dNdx_i;
            B_NL(1, c) = dNdy_i;
            B_NL(2, c) = dNdz_i;

            // dv/dx, dv/dy, dv/dz
            B_NL(3, c + 1) = dNdx_i;
            B_NL(4, c + 1) = dNdy_i;
            B_NL(5, c + 1) = dNdz_i;

            // dw/dx, dw/dy, dw/dz
            B_NL(6, c + 2) = dNdx_i;
            B_NL(7, c + 2) = dNdy_i;
            B_NL(8, c + 2) = dNdz_i;
        }

        // K_sigma += B_NL^T * S * B_NL * detJ * w
        Ks.noalias() += B_NL.transpose() * S * B_NL * (detJ * w);
    }

    return Ks;
}

Matrix30d RotatingEffects::spin_softening(
    const TET10Element& elem, const Material& mat, double omega) {

    // K_omega = integral(rho * N_mat^T * Omega_mat * N_mat * dV)
    // where Omega_mat = [[omega^2, 0, 0], [0, omega^2, 0], [0, 0, 0]]
    // for z-axis rotation.

    double omega2 = omega * omega;
    Matrix30d Kw = Matrix30d::Zero();

    for (int gp = 0; gp < 4; gp++) {
        double xi   = TET10Element::gauss_points[gp](0);
        double eta  = TET10Element::gauss_points[gp](1);
        double zeta = TET10Element::gauss_points[gp](2);
        double w    = TET10Element::gauss_weights[gp];

        Vector10d N = elem.shape_functions(xi, eta, zeta);
        Eigen::Matrix3d J = elem.jacobian(xi, eta, zeta);
        double detJ = J.determinant();

        // Build N_mat (3x30)
        Matrix3x30d N_mat = Matrix3x30d::Zero();
        for (int i = 0; i < 10; i++) {
            N_mat(0, 3 * i)     = N(i);
            N_mat(1, 3 * i + 1) = N(i);
            N_mat(2, 3 * i + 2) = N(i);
        }

        // Omega_mat = diag(omega^2, omega^2, 0)
        Eigen::Matrix3d Omega_mat = Eigen::Matrix3d::Zero();
        Omega_mat(0, 0) = omega2;
        Omega_mat(1, 1) = omega2;

        // K_omega += rho * N_mat^T * Omega_mat * N_mat * detJ * w
        Kw.noalias() += mat.rho * N_mat.transpose() * Omega_mat * N_mat * (detJ * w);
    }

    return Kw;
}

Matrix30d RotatingEffects::gyroscopic(
    const TET10Element& elem, const Material& mat) {

    // G = integral(rho * N_mat^T * Omega_cross * N_mat * dV)
    // where Omega_cross = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
    // for z-axis rotation.
    //
    // The full Coriolis term in the equation of motion is 2*omega*G,
    // but this function returns the base G matrix without the 2*omega factor.

    Matrix30d Ge = Matrix30d::Zero();

    for (int gp = 0; gp < 4; gp++) {
        double xi   = TET10Element::gauss_points[gp](0);
        double eta  = TET10Element::gauss_points[gp](1);
        double zeta = TET10Element::gauss_points[gp](2);
        double w    = TET10Element::gauss_weights[gp];

        Vector10d N = elem.shape_functions(xi, eta, zeta);
        Eigen::Matrix3d J = elem.jacobian(xi, eta, zeta);
        double detJ = J.determinant();

        // Build N_mat (3x30)
        Matrix3x30d N_mat = Matrix3x30d::Zero();
        for (int i = 0; i < 10; i++) {
            N_mat(0, 3 * i)     = N(i);
            N_mat(1, 3 * i + 1) = N(i);
            N_mat(2, 3 * i + 2) = N(i);
        }

        // Omega_cross for z-axis rotation
        Eigen::Matrix3d Omega_cross = Eigen::Matrix3d::Zero();
        Omega_cross(0, 1) = -1.0;
        Omega_cross(1, 0) =  1.0;

        // G += rho * N_mat^T * Omega_cross * N_mat * detJ * w
        Ge.noalias() += mat.rho * N_mat.transpose() * Omega_cross * N_mat * (detJ * w);
    }

    return Ge;
}

}  // namespace turbomodal
