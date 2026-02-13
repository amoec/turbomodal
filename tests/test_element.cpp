#include <gtest/gtest.h>
#include "turbomodal/element.hpp"
#include <Eigen/Eigenvalues>

using namespace turbomodal;

// Helper: create a TET10 element with a regular tetrahedron
// Vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
// Mid-edge nodes at midpoints
static TET10Element make_reference_tet10() {
    TET10Element elem;
    // Corner nodes
    elem.node_coords.row(0) = Eigen::Vector3d(0.0, 0.0, 0.0);
    elem.node_coords.row(1) = Eigen::Vector3d(1.0, 0.0, 0.0);
    elem.node_coords.row(2) = Eigen::Vector3d(0.0, 1.0, 0.0);
    elem.node_coords.row(3) = Eigen::Vector3d(0.0, 0.0, 1.0);
    // Mid-edge nodes (midpoints)
    elem.node_coords.row(4) = Eigen::Vector3d(0.5, 0.0, 0.0);  // 0-1
    elem.node_coords.row(5) = Eigen::Vector3d(0.5, 0.5, 0.0);  // 1-2
    elem.node_coords.row(6) = Eigen::Vector3d(0.0, 0.5, 0.0);  // 0-2
    elem.node_coords.row(7) = Eigen::Vector3d(0.0, 0.0, 0.5);  // 0-3
    elem.node_coords.row(8) = Eigen::Vector3d(0.5, 0.0, 0.5);  // 1-3
    elem.node_coords.row(9) = Eigen::Vector3d(0.0, 0.5, 0.5);  // 2-3
    return elem;
}

// Helper: create a scaled/translated TET10
static TET10Element make_scaled_tet10(double scale, Eigen::Vector3d offset) {
    TET10Element elem = make_reference_tet10();
    for (int i = 0; i < 10; i++) {
        elem.node_coords.row(i) = elem.node_coords.row(i) * scale + offset.transpose();
    }
    return elem;
}

// Natural coordinates of the 10 nodes in the reference element
static const double node_nat_coords[10][3] = {
    {0.0, 0.0, 0.0},      // Node 0
    {1.0, 0.0, 0.0},      // Node 1
    {0.0, 1.0, 0.0},      // Node 2
    {0.0, 0.0, 1.0},      // Node 3
    {0.5, 0.0, 0.0},      // Node 4 (mid 0-1)
    {0.5, 0.5, 0.0},      // Node 5 (mid 1-2)
    {0.0, 0.5, 0.0},      // Node 6 (mid 0-2)
    {0.0, 0.0, 0.5},      // Node 7 (mid 0-3)
    {0.5, 0.0, 0.5},      // Node 8 (mid 1-3)
    {0.0, 0.5, 0.5},      // Node 9 (mid 2-3)
};

// ==========================================================
// Shape Function Tests
// ==========================================================

TEST(TET10, PartitionOfUnity) {
    TET10Element elem = make_reference_tet10();
    // Test at several random interior points
    double test_points[][3] = {
        {0.25, 0.25, 0.25},
        {0.1, 0.1, 0.1},
        {0.5, 0.1, 0.1},
        {0.1, 0.5, 0.1},
        {0.1, 0.1, 0.5},
        {0.2, 0.3, 0.1},
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
    };
    for (auto& pt : test_points) {
        Vector10d N = elem.shape_functions(pt[0], pt[1], pt[2]);
        EXPECT_NEAR(N.sum(), 1.0, 1e-14)
            << "Failed at (" << pt[0] << "," << pt[1] << "," << pt[2] << ")";
    }
}

TEST(TET10, ShapeFunctionsAtNodes) {
    TET10Element elem = make_reference_tet10();
    for (int i = 0; i < 10; i++) {
        Vector10d N = elem.shape_functions(
            node_nat_coords[i][0], node_nat_coords[i][1], node_nat_coords[i][2]);
        for (int j = 0; j < 10; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(N(j), expected, 1e-14)
                << "N_" << j << " at node " << i;
        }
    }
}

TEST(TET10, ShapeFunctionsQuadraticAtCentroid) {
    // Note: Quadratic TET10 shape functions CAN be negative at interior points.
    // Corner nodes have N = -1/8 at centroid, mid-edge nodes have N = +1/4.
    // This is expected and correct for quadratic elements.
    TET10Element elem = make_reference_tet10();
    Vector10d N = elem.shape_functions(0.25, 0.25, 0.25);
    // Corner nodes at centroid: Li = 0.25, Ni = 0.25*(2*0.25-1) = 0.25*(-0.5) = -0.125
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(N(i), -0.125, 1e-14) << "Corner node " << i;
    }
    // Mid-edge nodes at centroid: 4*0.25*0.25 = 0.25
    for (int i = 4; i < 10; i++) {
        EXPECT_NEAR(N(i), 0.25, 1e-14) << "Mid-edge node " << i;
    }
}

// ==========================================================
// Shape Derivative Tests
// ==========================================================

TEST(TET10, ShapeDerivativesFiniteDifference) {
    // Verify analytical derivatives against finite differences
    TET10Element elem = make_reference_tet10();
    double xi = 0.2, eta = 0.15, zeta = 0.1;
    double h = 1e-7;

    Matrix10x3d dN_analytical = elem.shape_derivatives(xi, eta, zeta);

    // Finite difference for dN/dxi
    Vector10d Np = elem.shape_functions(xi + h, eta, zeta);
    Vector10d Nm = elem.shape_functions(xi - h, eta, zeta);
    Eigen::VectorXd dNdxi_fd = (Np - Nm) / (2.0 * h);

    // Finite difference for dN/deta
    Np = elem.shape_functions(xi, eta + h, zeta);
    Nm = elem.shape_functions(xi, eta - h, zeta);
    Eigen::VectorXd dNdeta_fd = (Np - Nm) / (2.0 * h);

    // Finite difference for dN/dzeta
    Np = elem.shape_functions(xi, eta, zeta + h);
    Nm = elem.shape_functions(xi, eta, zeta - h);
    Eigen::VectorXd dNdzeta_fd = (Np - Nm) / (2.0 * h);

    for (int i = 0; i < 10; i++) {
        EXPECT_NEAR(dN_analytical(i, 0), dNdxi_fd(i), 1e-6)
            << "dN" << i << "/dxi";
        EXPECT_NEAR(dN_analytical(i, 1), dNdeta_fd(i), 1e-6)
            << "dN" << i << "/deta";
        EXPECT_NEAR(dN_analytical(i, 2), dNdzeta_fd(i), 1e-6)
            << "dN" << i << "/dzeta";
    }
}

TEST(TET10, ShapeDerivativesSumToZero) {
    // Since sum(Ni) = 1, sum(dNi/dxi) = 0 for any xi,eta,zeta
    TET10Element elem = make_reference_tet10();
    double test_points[][3] = {
        {0.25, 0.25, 0.25},
        {0.1, 0.2, 0.3},
        {0.5, 0.1, 0.1},
    };
    for (auto& pt : test_points) {
        Matrix10x3d dN = elem.shape_derivatives(pt[0], pt[1], pt[2]);
        for (int col = 0; col < 3; col++) {
            EXPECT_NEAR(dN.col(col).sum(), 0.0, 1e-14);
        }
    }
}

// ==========================================================
// Jacobian Tests
// ==========================================================

TEST(TET10, JacobianReferenceElement) {
    // For the reference tet with vertices at (0,0,0),(1,0,0),(0,1,0),(0,0,1)
    // and mid-edge nodes at exact midpoints, the Jacobian should be identity
    TET10Element elem = make_reference_tet10();
    Eigen::Matrix3d J = elem.jacobian(0.25, 0.25, 0.25);
    EXPECT_TRUE(J.isApprox(Eigen::Matrix3d::Identity(), 1e-13))
        << "J =\n" << J;
}

TEST(TET10, JacobianScaledElement) {
    // Scaling by factor s should give J = s*I
    double s = 2.5;
    TET10Element elem = make_scaled_tet10(s, Eigen::Vector3d::Zero());
    Eigen::Matrix3d J = elem.jacobian(0.25, 0.25, 0.25);
    EXPECT_TRUE(J.isApprox(s * Eigen::Matrix3d::Identity(), 1e-12))
        << "J =\n" << J;
}

TEST(TET10, JacobianDeterminantPositive) {
    TET10Element elem = make_reference_tet10();
    for (const auto& gp : TET10Element::gauss_points) {
        Eigen::Matrix3d J = elem.jacobian(gp(0), gp(1), gp(2));
        EXPECT_GT(J.determinant(), 0.0);
    }
}

TEST(TET10, JacobianDeterminantVolume) {
    // Volume of reference tet = 1/6
    // integral det(J) dV_ref = integral 1 * det(J) * (1/6 per tet ref)
    // Using Gauss quadrature: V = sum(det(J) * w)
    TET10Element elem = make_reference_tet10();
    double vol = 0.0;
    for (int gp = 0; gp < 4; gp++) {
        Eigen::Matrix3d J = elem.jacobian(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        vol += J.determinant() * TET10Element::gauss_weights[gp];
    }
    // Reference tet volume = 1/6
    EXPECT_NEAR(vol, 1.0 / 6.0, 1e-14);
}

TEST(TET10, ScaledElementVolume) {
    double s = 3.0;
    TET10Element elem = make_scaled_tet10(s, Eigen::Vector3d(1, 2, 3));
    double vol = 0.0;
    for (int gp = 0; gp < 4; gp++) {
        Eigen::Matrix3d J = elem.jacobian(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        vol += J.determinant() * TET10Element::gauss_weights[gp];
    }
    // Scaled volume = s^3 / 6
    EXPECT_NEAR(vol, s * s * s / 6.0, 1e-12);
}

// ==========================================================
// Stiffness Matrix Tests
// ==========================================================

TEST(TET10, StiffnessSymmetry) {
    TET10Element elem = make_reference_tet10();
    Material mat(200e9, 0.3, 7850);
    Matrix30d K = elem.stiffness(mat);
    EXPECT_TRUE(K.isApprox(K.transpose(), 1e-4))
        << "Max asymmetry: " << (K - K.transpose()).cwiseAbs().maxCoeff();
}

TEST(TET10, StiffnessRigidBodyModes) {
    // A free element should have 6 zero eigenvalues (3 translations + 3 rotations)
    TET10Element elem = make_reference_tet10();
    Material mat(200e9, 0.3, 7850);
    Matrix30d K = elem.stiffness(mat);

    // Symmetrize for cleaner eigenvalues
    K = (K + K.transpose()) / 2.0;

    Eigen::SelfAdjointEigenSolver<Matrix30d> es(K);
    Eigen::VectorXd evals = es.eigenvalues();

    // Sort eigenvalues
    int n_zero = 0;
    for (int i = 0; i < 30; i++) {
        if (std::abs(evals(i)) < 1e-4 * evals(29)) {
            n_zero++;
        }
    }
    EXPECT_EQ(n_zero, 6) << "Expected 6 rigid body modes, got " << n_zero
        << "\nSmallest eigenvalues: " << evals.head(8).transpose();
}

TEST(TET10, StiffnessNonNegativeEigenvalues) {
    TET10Element elem = make_reference_tet10();
    Material mat(200e9, 0.3, 7850);
    Matrix30d K = elem.stiffness(mat);
    K = (K + K.transpose()) / 2.0;

    Eigen::SelfAdjointEigenSolver<Matrix30d> es(K);
    // All eigenvalues should be >= 0 (positive semi-definite)
    EXPECT_GE(es.eigenvalues().minCoeff(), -1e-4 * es.eigenvalues().maxCoeff());
}

TEST(TET10, StiffnessScalesWithE) {
    TET10Element elem = make_reference_tet10();
    Material mat1(100e9, 0.3, 7850);
    Material mat2(200e9, 0.3, 7850);
    Matrix30d K1 = elem.stiffness(mat1);
    Matrix30d K2 = elem.stiffness(mat2);
    // K should scale linearly with E
    EXPECT_TRUE(K2.isApprox(2.0 * K1, 1e-6));
}

// ==========================================================
// Mass Matrix Tests
// ==========================================================

TEST(TET10, MassSymmetry) {
    TET10Element elem = make_reference_tet10();
    Material mat(200e9, 0.3, 7850);
    Matrix30d M = elem.mass(mat);
    EXPECT_TRUE(M.isApprox(M.transpose(), 1e-10))
        << "Max asymmetry: " << (M - M.transpose()).cwiseAbs().maxCoeff();
}

TEST(TET10, MassPositiveSemiDefinite) {
    // 4-point Gauss under-integrates the quartic mass integrand (N^T*N),
    // so eigenvalues may be very slightly negative (~1e-14). Accept this.
    TET10Element elem = make_reference_tet10();
    Material mat(200e9, 0.3, 7850);
    Matrix30d M = elem.mass(mat);
    M = (M + M.transpose()) / 2.0;
    Eigen::SelfAdjointEigenSolver<Matrix30d> es(M);
    double max_eval = es.eigenvalues().maxCoeff();
    EXPECT_GT(es.eigenvalues().minCoeff(), -1e-10 * max_eval)
        << "Smallest eigenvalue: " << es.eigenvalues().minCoeff();
}

TEST(TET10, MassTotalMass) {
    // For consistent mass: u^T * M * u = rho * V for a rigid body translation u.
    // This works because integral(sum_i Ni * sum_j Nj) = integral(1) = V
    // (partition of unity, exact even with low-order quadrature).
    TET10Element elem = make_reference_tet10();
    double rho = 7850.0;
    Material mat(200e9, 0.3, rho);
    Matrix30d M = elem.mass(mat);

    // Unit translation in x: u = [1,0,0, 1,0,0, ...]
    Eigen::VectorXd ux = Eigen::VectorXd::Zero(30);
    for (int i = 0; i < 10; i++) ux(3 * i) = 1.0;

    double vol = 1.0 / 6.0;
    double total_mass = ux.transpose() * M * ux;
    EXPECT_NEAR(total_mass, rho * vol, rho * vol * 1e-10);
}

TEST(TET10, MassTotalMassScaled) {
    double s = 2.0;
    TET10Element elem = make_scaled_tet10(s, Eigen::Vector3d(5, 5, 5));
    double rho = 4430.0;
    Material mat(110e9, 0.34, rho);
    Matrix30d M = elem.mass(mat);

    Eigen::VectorXd ux = Eigen::VectorXd::Zero(30);
    for (int i = 0; i < 10; i++) ux(3 * i) = 1.0;

    double vol = s * s * s / 6.0;
    double total_mass = ux.transpose() * M * ux;
    EXPECT_NEAR(total_mass, rho * vol, rho * vol * 1e-10);
}

TEST(TET10, MassScalesWithDensity) {
    TET10Element elem = make_reference_tet10();
    Material mat1(200e9, 0.3, 1000);
    Material mat2(200e9, 0.3, 2000);
    Matrix30d M1 = elem.mass(mat1);
    Matrix30d M2 = elem.mass(mat2);
    EXPECT_TRUE(M2.isApprox(2.0 * M1, 1e-10));
}

TEST(TET10, MassIndependentOfE) {
    TET10Element elem = make_reference_tet10();
    Material mat1(100e9, 0.3, 7850);
    Material mat2(200e9, 0.3, 7850);
    Matrix30d M1 = elem.mass(mat1);
    Matrix30d M2 = elem.mass(mat2);
    EXPECT_TRUE(M1.isApprox(M2, 1e-15));
}

// ==========================================================
// Patch Test (CRITICAL)
// ==========================================================

TEST(TET10, PatchTestConstantStrain) {
    // Patch test: impose a linear displacement field on a single element.
    // For a quadratic element, linear displacements should produce EXACT
    // constant strain (no approximation error).
    //
    // Displacement field: u = a*x + b*y, v = c*z, w = d*x
    // Strain: eps_xx = a, eps_yy = 0, eps_zz = 0
    //         gamma_xy = b, gamma_yz = c, gamma_xz = d

    TET10Element elem = make_reference_tet10();
    Material mat(200e9, 0.3, 7850);

    double a = 1e-3, b = 2e-3, c = -0.5e-3, d = 1.5e-3;

    // Compute nodal displacements from the linear field
    Eigen::VectorXd u_e(30);
    for (int i = 0; i < 10; i++) {
        double x = elem.node_coords(i, 0);
        double y = elem.node_coords(i, 1);
        double z = elem.node_coords(i, 2);
        u_e(3 * i)     = a * x + b * y;     // u
        u_e(3 * i + 1) = c * z;              // v
        u_e(3 * i + 2) = d * x;              // w
    }

    // Expected constant strain
    Vector6d eps_expected;
    eps_expected << a, 0.0, 0.0, b, c, d;

    // Check strain at each Gauss point: eps = B * u_e
    for (int gp = 0; gp < 4; gp++) {
        Matrix6x30d B = elem.B_matrix(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));

        Vector6d eps = B * u_e;

        for (int k = 0; k < 6; k++) {
            EXPECT_NEAR(eps(k), eps_expected(k), 1e-12)
                << "Strain component " << k << " at GP " << gp;
        }
    }
}

TEST(TET10, PatchTestPureShear) {
    // Pure shear: u = gamma * y, v = 0, w = 0
    // Expected: eps_xx = 0, gamma_xy = gamma, rest = 0
    TET10Element elem = make_reference_tet10();

    double gamma = 1e-3;
    Eigen::VectorXd u_e(30);
    for (int i = 0; i < 10; i++) {
        double y = elem.node_coords(i, 1);
        u_e(3 * i)     = gamma * y;
        u_e(3 * i + 1) = 0.0;
        u_e(3 * i + 2) = 0.0;
    }

    Vector6d eps_expected;
    eps_expected << 0.0, 0.0, 0.0, gamma, 0.0, 0.0;

    for (int gp = 0; gp < 4; gp++) {
        Matrix6x30d B = elem.B_matrix(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        Vector6d eps = B * u_e;

        for (int k = 0; k < 6; k++) {
            EXPECT_NEAR(eps(k), eps_expected(k), 1e-12)
                << "Component " << k << " at GP " << gp;
        }
    }
}

TEST(TET10, PatchTestUniformExtension) {
    // Uniform extension: u = eps0*x, v = eps0*y, w = eps0*z
    // Expected: eps = (eps0, eps0, eps0, 0, 0, 0)
    TET10Element elem = make_reference_tet10();

    double eps0 = 5e-4;
    Eigen::VectorXd u_e(30);
    for (int i = 0; i < 10; i++) {
        double x = elem.node_coords(i, 0);
        double y = elem.node_coords(i, 1);
        double z = elem.node_coords(i, 2);
        u_e(3 * i)     = eps0 * x;
        u_e(3 * i + 1) = eps0 * y;
        u_e(3 * i + 2) = eps0 * z;
    }

    Vector6d eps_expected;
    eps_expected << eps0, eps0, eps0, 0.0, 0.0, 0.0;

    for (int gp = 0; gp < 4; gp++) {
        Matrix6x30d B = elem.B_matrix(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        Vector6d eps = B * u_e;

        for (int k = 0; k < 6; k++) {
            EXPECT_NEAR(eps(k), eps_expected(k), 1e-12)
                << "Component " << k << " at GP " << gp;
        }
    }
}

TEST(TET10, PatchTestOnScaledElement) {
    // Same patch test but on a scaled + translated element
    TET10Element elem = make_scaled_tet10(3.0, Eigen::Vector3d(10, -5, 2));
    double a = 1e-3;

    Eigen::VectorXd u_e(30);
    for (int i = 0; i < 10; i++) {
        double x = elem.node_coords(i, 0);
        u_e(3 * i)     = a * x;
        u_e(3 * i + 1) = 0.0;
        u_e(3 * i + 2) = 0.0;
    }

    Vector6d eps_expected;
    eps_expected << a, 0.0, 0.0, 0.0, 0.0, 0.0;

    for (int gp = 0; gp < 4; gp++) {
        Matrix6x30d B = elem.B_matrix(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        Vector6d eps = B * u_e;

        for (int k = 0; k < 6; k++) {
            EXPECT_NEAR(eps(k), eps_expected(k), 1e-11)
                << "Component " << k << " at GP " << gp;
        }
    }
}

TEST(TET10, PatchTestOnSkewedElement) {
    // Patch test on a sheared/skewed element where J is NOT symmetric.
    // This catches bugs like using J^{-1} instead of J^{-T} for physical derivatives.
    TET10Element elem;
    // Skewed tet: sheared in x-y plane
    Eigen::Vector3d v0(0, 0, 0), v1(2, 1, 0), v2(0.5, 3, 0), v3(0.3, 0.2, 1.5);
    elem.node_coords.row(0) = v0;
    elem.node_coords.row(1) = v1;
    elem.node_coords.row(2) = v2;
    elem.node_coords.row(3) = v3;
    // Mid-edge nodes at midpoints
    elem.node_coords.row(4) = (v0 + v1) / 2.0;
    elem.node_coords.row(5) = (v1 + v2) / 2.0;
    elem.node_coords.row(6) = (v0 + v2) / 2.0;
    elem.node_coords.row(7) = (v0 + v3) / 2.0;
    elem.node_coords.row(8) = (v1 + v3) / 2.0;
    elem.node_coords.row(9) = (v2 + v3) / 2.0;

    // Test: u_x = x should give eps_xx = 1, all others 0
    double a = 1.0;
    Eigen::VectorXd u_e(30);
    for (int i = 0; i < 10; i++) {
        u_e(3 * i)     = a * elem.node_coords(i, 0);
        u_e(3 * i + 1) = 0.0;
        u_e(3 * i + 2) = 0.0;
    }

    Vector6d eps_expected;
    eps_expected << a, 0.0, 0.0, 0.0, 0.0, 0.0;

    for (int gp = 0; gp < 4; gp++) {
        Matrix6x30d B = elem.B_matrix(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        Vector6d eps = B * u_e;

        for (int k = 0; k < 6; k++) {
            EXPECT_NEAR(eps(k), eps_expected(k), 1e-10)
                << "Component " << k << " at GP " << gp
                << " (skewed element patch test)";
        }
    }

    // Also test: u_y = y should give eps_yy = 1, all others 0
    for (int i = 0; i < 10; i++) {
        u_e(3 * i)     = 0.0;
        u_e(3 * i + 1) = a * elem.node_coords(i, 1);
        u_e(3 * i + 2) = 0.0;
    }
    eps_expected << 0.0, a, 0.0, 0.0, 0.0, 0.0;

    for (int gp = 0; gp < 4; gp++) {
        Matrix6x30d B = elem.B_matrix(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        Vector6d eps = B * u_e;

        for (int k = 0; k < 6; k++) {
            EXPECT_NEAR(eps(k), eps_expected(k), 1e-10)
                << "Component " << k << " at GP " << gp
                << " (skewed element eps_yy test)";
        }
    }
}

// ==========================================================
// Internal Force Consistency
// ==========================================================

TEST(TET10, InternalForceFromConstantStress) {
    // If we apply a displacement that produces constant stress sigma,
    // then f_int = K * u should equal the consistent nodal forces
    // from that stress field: f_int_i = integral(B^T * sigma * dV)
    TET10Element elem = make_reference_tet10();
    Material mat(200e9, 0.3, 7850);
    Matrix30d K = elem.stiffness(mat);
    Matrix6d D = mat.constitutive_matrix();

    // Uniform uniaxial strain eps_xx = 1e-3
    double eps0 = 1e-3;
    Eigen::VectorXd u_e(30);
    for (int i = 0; i < 10; i++) {
        u_e(3 * i)     = eps0 * elem.node_coords(i, 0);
        u_e(3 * i + 1) = 0.0;
        u_e(3 * i + 2) = 0.0;
    }

    // f_int from K
    Eigen::VectorXd f_K = K * u_e;

    // f_int from direct integration of B^T * sigma
    Vector6d eps;
    eps << eps0, 0.0, 0.0, 0.0, 0.0, 0.0;
    Vector6d sigma = D * eps;

    Eigen::VectorXd f_direct = Eigen::VectorXd::Zero(30);
    for (int gp = 0; gp < 4; gp++) {
        Matrix6x30d B = elem.B_matrix(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        Eigen::Matrix3d J = elem.jacobian(
            TET10Element::gauss_points[gp](0),
            TET10Element::gauss_points[gp](1),
            TET10Element::gauss_points[gp](2));
        double detJ = J.determinant();
        f_direct += B.transpose() * sigma * detJ * TET10Element::gauss_weights[gp];
    }

    EXPECT_TRUE(f_K.isApprox(f_direct, 1e-4))
        << "Max diff: " << (f_K - f_direct).cwiseAbs().maxCoeff();
}

// ==========================================================
// Gauss Data
// ==========================================================

TEST(TET10, GaussPointCount) {
    EXPECT_EQ(TET10Element::gauss_points.size(), 4u);
    EXPECT_EQ(TET10Element::gauss_weights.size(), 4u);
}

TEST(TET10, GaussWeightsSum) {
    // Sum of weights should equal 1/6 (reference tet volume)
    double sum = 0.0;
    for (double w : TET10Element::gauss_weights) sum += w;
    EXPECT_NEAR(sum, 1.0 / 6.0, 1e-15);
}

TEST(TET10, GaussPointsInsideElement) {
    for (const auto& gp : TET10Element::gauss_points) {
        EXPECT_GE(gp(0), 0.0);
        EXPECT_GE(gp(1), 0.0);
        EXPECT_GE(gp(2), 0.0);
        EXPECT_LE(gp(0) + gp(1) + gp(2), 1.0 + 1e-15);
    }
}
