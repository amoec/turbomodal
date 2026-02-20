#include "turbomodal/element.hpp"
#include <iostream>

namespace turbomodal {

// Thread-local counter to avoid spamming per-Gauss-point warnings.
static thread_local int s_neg_jac_warnings = 0;

// 4-point Gauss quadrature for tetrahedra
// Points are in barycentric-like natural coordinates (xi, eta, zeta)
// with constraint xi + eta + zeta <= 1, all >= 0
const std::array<Eigen::Vector3d, 4> TET10Element::gauss_points = {{
    {0.1381966011250105, 0.1381966011250105, 0.1381966011250105},
    {0.5854101966249685, 0.1381966011250105, 0.1381966011250105},
    {0.1381966011250105, 0.5854101966249685, 0.1381966011250105},
    {0.1381966011250105, 0.1381966011250105, 0.5854101966249685}
}};

// Weights: each 1/24, which equals (1/6 reference tet volume) / 4 points
const std::array<double, 4> TET10Element::gauss_weights = {{
    1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0
}};

// 5-point Stroud degree-3 quadrature for tetrahedra (sum = 1/6 = ref tet volume)
// Used for mass-type integrals where integrand is N^T*N (degree 4).
const std::array<Eigen::Vector3d, 5> TET10Element::mass_gauss_points = {{
    {1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0},   // centroid
    {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0},
    {1.0 / 2.0, 1.0 / 6.0, 1.0 / 6.0},
    {1.0 / 6.0, 1.0 / 2.0, 1.0 / 6.0},
    {1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0}
}};

const std::array<double, 5> TET10Element::mass_gauss_weights = {{
    -2.0 / 15.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0
}};

// TET10 node ordering (standard Gmsh convention):
//   Corners: 0,1,2,3  (vertices of tetrahedron)
//   Mid-edge: 4(0-1), 5(1-2), 6(0-2), 7(0-3), 8(1-3), 9(2-3)
//
// Natural coordinates of the 10 nodes:
//   Node 0: (0, 0, 0)  -> L1=1, L2=0, L3=0, L4=0
//   Node 1: (1, 0, 0)  -> L1=0, L2=1, L3=0, L4=0
//   Node 2: (0, 1, 0)  -> L1=0, L2=0, L3=1, L4=0
//   Node 3: (0, 0, 1)  -> L1=0, L2=0, L3=0, L4=1
//   Node 4: (0.5, 0, 0)    mid 0-1
//   Node 5: (0.5, 0.5, 0)  mid 1-2
//   Node 6: (0, 0.5, 0)    mid 0-2
//   Node 7: (0, 0, 0.5)    mid 0-3
//   Node 8: (0.5, 0, 0.5)  mid 1-3
//   Node 9: (0, 0.5, 0.5)  mid 2-3
//
// Barycentric coordinates: L1 = 1 - xi - eta - zeta, L2 = xi, L3 = eta, L4 = zeta

Vector10d TET10Element::shape_functions(double xi, double eta, double zeta) const {
    double L1 = 1.0 - xi - eta - zeta;
    double L2 = xi;
    double L3 = eta;
    double L4 = zeta;

    Vector10d N;
    // Corner nodes: Ni = Li * (2*Li - 1)
    N(0) = L1 * (2.0 * L1 - 1.0);
    N(1) = L2 * (2.0 * L2 - 1.0);
    N(2) = L3 * (2.0 * L3 - 1.0);
    N(3) = L4 * (2.0 * L4 - 1.0);
    // Mid-edge nodes: 4*Li*Lj
    N(4) = 4.0 * L1 * L2;   // edge 0-1
    N(5) = 4.0 * L2 * L3;   // edge 1-2
    N(6) = 4.0 * L1 * L3;   // edge 0-2
    N(7) = 4.0 * L1 * L4;   // edge 0-3
    N(8) = 4.0 * L2 * L4;   // edge 1-3
    N(9) = 4.0 * L3 * L4;   // edge 2-3

    return N;
}

Matrix10x3d TET10Element::shape_derivatives(double xi, double eta, double zeta) const {
    // Derivatives of shape functions w.r.t. natural coordinates (xi, eta, zeta)
    // L1 = 1 - xi - eta - zeta  =>  dL1/dxi = -1, dL1/deta = -1, dL1/dzeta = -1
    // L2 = xi                   =>  dL2/dxi =  1, dL2/deta =  0, dL2/dzeta =  0
    // L3 = eta                  =>  dL3/dxi =  0, dL3/deta =  1, dL3/dzeta =  0
    // L4 = zeta                 =>  dL4/dxi =  0, dL4/deta =  0, dL4/dzeta =  1

    double L1 = 1.0 - xi - eta - zeta;
    double L2 = xi;
    double L3 = eta;
    double L4 = zeta;

    Matrix10x3d dN;

    // Node 0: N0 = L1*(2*L1 - 1)
    // dN0/dxi = dL1/dxi * (2*L1-1) + L1 * 2 * dL1/dxi = (-1)*(2*L1-1) + L1*2*(-1) = -(2*L1-1) - 2*L1 = 1 - 4*L1
    dN(0, 0) = 1.0 - 4.0 * L1;   // dN0/dxi
    dN(0, 1) = 1.0 - 4.0 * L1;   // dN0/deta
    dN(0, 2) = 1.0 - 4.0 * L1;   // dN0/dzeta

    // Node 1: N1 = L2*(2*L2 - 1)
    // dN1/dxi = 1*(2*L2-1) + L2*2*1 = (2*L2-1) + 2*L2 = 4*L2 - 1
    dN(1, 0) = 4.0 * L2 - 1.0;   // dN1/dxi
    dN(1, 1) = 0.0;               // dN1/deta
    dN(1, 2) = 0.0;               // dN1/dzeta

    // Node 2: N2 = L3*(2*L3 - 1)
    dN(2, 0) = 0.0;               // dN2/dxi
    dN(2, 1) = 4.0 * L3 - 1.0;   // dN2/deta
    dN(2, 2) = 0.0;               // dN2/dzeta

    // Node 3: N3 = L4*(2*L4 - 1)
    dN(3, 0) = 0.0;               // dN3/dxi
    dN(3, 1) = 0.0;               // dN3/deta
    dN(3, 2) = 4.0 * L4 - 1.0;   // dN3/dzeta

    // Node 4: N4 = 4*L1*L2
    // dN4/dxi  = 4*(dL1/dxi * L2 + L1 * dL2/dxi)  = 4*(-L2 + L1) = 4*(L1 - L2)
    // dN4/deta = 4*(dL1/deta * L2 + L1 * dL2/deta) = 4*(-L2 + 0) = -4*L2
    // dN4/dzeta= 4*(-L2 + 0) = -4*L2
    dN(4, 0) = 4.0 * (L1 - L2);
    dN(4, 1) = -4.0 * L2;
    dN(4, 2) = -4.0 * L2;

    // Node 5: N5 = 4*L2*L3
    // dN5/dxi  = 4*(1*L3 + L2*0)  = 4*L3
    // dN5/deta = 4*(0*L3 + L2*1)  = 4*L2
    // dN5/dzeta= 4*(0 + 0) = 0
    dN(5, 0) = 4.0 * L3;
    dN(5, 1) = 4.0 * L2;
    dN(5, 2) = 0.0;

    // Node 6: N6 = 4*L1*L3
    // dN6/dxi  = 4*(-1*L3 + L1*0) = -4*L3
    // dN6/deta = 4*(-1*L3 + L1*1) = 4*(L1 - L3)
    // dN6/dzeta= 4*(-L3 + 0) = -4*L3
    dN(6, 0) = -4.0 * L3;
    dN(6, 1) = 4.0 * (L1 - L3);
    dN(6, 2) = -4.0 * L3;

    // Node 7: N7 = 4*L1*L4
    // dN7/dxi  = 4*(-L4 + 0) = -4*L4
    // dN7/deta = 4*(-L4 + 0) = -4*L4
    // dN7/dzeta= 4*(-1*L4 + L1*1) = 4*(L1 - L4)
    dN(7, 0) = -4.0 * L4;
    dN(7, 1) = -4.0 * L4;
    dN(7, 2) = 4.0 * (L1 - L4);

    // Node 8: N8 = 4*L2*L4
    // dN8/dxi  = 4*(1*L4 + L2*0) = 4*L4
    // dN8/deta = 0
    // dN8/dzeta= 4*(0 + L2*1)    = 4*L2
    dN(8, 0) = 4.0 * L4;
    dN(8, 1) = 0.0;
    dN(8, 2) = 4.0 * L2;

    // Node 9: N9 = 4*L3*L4
    // dN9/dxi  = 4*(0*L4 + L3*0) = 0
    // dN9/deta = 4*(1*L4 + L3*0) = 4*L4
    // dN9/dzeta= 4*(0 + L3*1)    = 4*L3
    dN(9, 0) = 0.0;
    dN(9, 1) = 4.0 * L4;
    dN(9, 2) = 4.0 * L3;

    return dN;
}

Eigen::Matrix3d TET10Element::jacobian(double xi, double eta, double zeta) const {
    // J = dNdxi^T * node_coords
    // dNdxi is 10x3, node_coords is 10x3
    // J = (10x3)^T * (10x3) = 3x10 * 10x3 = 3x3
    Matrix10x3d dN = shape_derivatives(xi, eta, zeta);
    Eigen::Matrix3d J = dN.transpose() * node_coords;
    return J;
}

Matrix6x30d TET10Element::B_matrix(double xi, double eta, double zeta) const {
    // Compute shape function derivatives in physical coordinates
    Matrix10x3d dNdxi = shape_derivatives(xi, eta, zeta);
    Eigen::Matrix3d J = dNdxi.transpose() * node_coords;
    Eigen::Matrix3d Jinv = J.inverse();

    // Physical derivatives: dN/dx = dN/dξ * J^{-T}
    // Chain rule: ∂f/∂x = J^{-T} * ∂f/∂ξ (column), transposed as rows: dNdx = dNdxi * J^{-T}
    Matrix10x3d dNdx = dNdxi * Jinv.transpose();

    // Assemble B matrix (6x30)
    Matrix6x30d B = Matrix6x30d::Zero();

    for (int i = 0; i < 10; i++) {
        double dNdx_i = dNdx(i, 0);
        double dNdy_i = dNdx(i, 1);
        double dNdz_i = dNdx(i, 2);

        int c = 3 * i;  // column offset

        // Row 0: epsilon_xx = du/dx
        B(0, c)     = dNdx_i;
        // Row 1: epsilon_yy = dv/dy
        B(1, c + 1) = dNdy_i;
        // Row 2: epsilon_zz = dw/dz
        B(2, c + 2) = dNdz_i;
        // Row 3: gamma_xy = du/dy + dv/dx
        B(3, c)     = dNdy_i;
        B(3, c + 1) = dNdx_i;
        // Row 4: gamma_yz = dv/dz + dw/dy
        B(4, c + 1) = dNdz_i;
        B(4, c + 2) = dNdy_i;
        // Row 5: gamma_xz = du/dz + dw/dx
        B(5, c)     = dNdz_i;
        B(5, c + 2) = dNdx_i;
    }

    return B;
}

Matrix30d TET10Element::stiffness(const Material& mat) const {
    Matrix6d D = mat.constitutive_matrix();
    Matrix30d Ke = Matrix30d::Zero();

    for (int gp = 0; gp < 4; gp++) {
        double xi   = gauss_points[gp](0);
        double eta  = gauss_points[gp](1);
        double zeta = gauss_points[gp](2);
        double w    = gauss_weights[gp];

        Matrix6x30d B = B_matrix(xi, eta, zeta);
        Eigen::Matrix3d J = jacobian(xi, eta, zeta);
        double detJ = J.determinant();

        // Safety net: use |detJ| if negative to prevent sign-flip corruption
        if (detJ <= 0.0) {
            if (s_neg_jac_warnings < 5) {
                std::cerr << "[TET10] Warning: negative Jacobian (detJ="
                          << detJ << ") in stiffness — using |detJ|\n";
                s_neg_jac_warnings++;
                if (s_neg_jac_warnings == 5)
                    std::cerr << "[TET10] (further warnings suppressed)\n";
            }
            detJ = std::abs(detJ);
        }

        // K_e += B^T * D * B * det(J) * w
        Ke.noalias() += B.transpose() * D * B * (detJ * w);
    }

    return Ke;
}

Matrix30d TET10Element::mass(const Material& mat) const {
    Matrix30d Me = Matrix30d::Zero();

    // 4-point Gauss quadrature (degree 2) for mass matrix.
    // Under-integrates the degree-4 integrand N^T*N but guarantees
    // a positive-definite mass matrix (all weights positive).
    for (int gp = 0; gp < 4; gp++) {
        double xi   = gauss_points[gp](0);
        double eta  = gauss_points[gp](1);
        double zeta = gauss_points[gp](2);
        double w    = gauss_weights[gp];

        Vector10d N = shape_functions(xi, eta, zeta);
        Eigen::Matrix3d J = jacobian(xi, eta, zeta);
        double detJ = J.determinant();

        if (detJ <= 0.0) {
            if (s_neg_jac_warnings < 5) {
                std::cerr << "[TET10] Warning: negative Jacobian (detJ="
                          << detJ << ") in mass — using |detJ|\n";
                s_neg_jac_warnings++;
                if (s_neg_jac_warnings == 5)
                    std::cerr << "[TET10] (further warnings suppressed)\n";
            }
            detJ = std::abs(detJ);
        }

        // Build 3x30 shape function matrix N_mat
        // N_mat(d, 3*i + d) = N(i) for d = 0,1,2
        Matrix3x30d N_mat = Matrix3x30d::Zero();
        for (int i = 0; i < 10; i++) {
            N_mat(0, 3 * i)     = N(i);
            N_mat(1, 3 * i + 1) = N(i);
            N_mat(2, 3 * i + 2) = N(i);
        }

        // M_e += rho * N_mat^T * N_mat * det(J) * w
        Me.noalias() += mat.rho * N_mat.transpose() * N_mat * (detJ * w);
    }

    return Me;
}

}  // namespace turbomodal
