#include <gtest/gtest.h>
#include "turbomodal/potential_flow.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace turbomodal;

// Helper: build a sphere (semicircle) meridional mesh
static MeridionalMesh sphere_mesh(double R, int np) {
    Eigen::MatrixXd points(np + 1, 2);
    for (int i = 0; i <= np; i++) {
        double theta = PI * i / np;  // 0 to pi
        points(i, 0) = R * std::sin(theta);  // r
        points(i, 1) = R * std::cos(theta);  // z
    }
    return MeridionalMesh::from_points(points);
}

// ========================================================================
// Elliptic integral tests
// ========================================================================

TEST(PotentialFlow, EllipticK_KnownValues) {
    // K(0) = pi/2
    EXPECT_NEAR(elliptic_K(0.0), PI / 2.0, 1e-12);
    // K(1/sqrt(2)) = Gamma(1/4)^2 / (4*sqrt(pi))
    EXPECT_NEAR(elliptic_K(1.0 / std::sqrt(2.0)), 1.8540746773013719, 1e-10);
    // K(0.5)
    EXPECT_NEAR(elliptic_K(0.5), 1.6857503548325898, 1e-10);
}

TEST(PotentialFlow, EllipticE_KnownValues) {
    // E(0) = pi/2
    EXPECT_NEAR(elliptic_E(0.0), PI / 2.0, 1e-12);
    // E(1) = 1
    EXPECT_NEAR(elliptic_E(1.0), 1.0, 1e-12);
    // E(1/sqrt(2))
    EXPECT_NEAR(elliptic_E(1.0 / std::sqrt(2.0)), 1.3506438810476755, 1e-10);
    // E(0.5) ≈ 1.46746...
    EXPECT_NEAR(elliptic_E(0.5), 1.4674622093394272, 1e-10);
}

TEST(PotentialFlow, EllipticLegendreIdentity) {
    // Legendre relation: K(k)*E(k') + E(k)*K(k') - K(k)*K(k') = pi/2
    // where k' = sqrt(1 - k^2)
    for (double k = 0.1; k < 0.95; k += 0.1) {
        double kp = std::sqrt(1.0 - k * k);
        double lhs = elliptic_K(k) * elliptic_E(kp) +
                     elliptic_E(k) * elliptic_K(kp) -
                     elliptic_K(k) * elliptic_K(kp);
        EXPECT_NEAR(lhs, PI / 2.0, 1e-10)
            << "Legendre identity failed for k=" << k;
    }
}

// ========================================================================
// Green's function tests
// ========================================================================

TEST(PotentialFlow, GreenFunction_N0_Positive) {
    double r = 0.2, z = 0.0;
    double rp = 0.15, zp = 0.01;
    double G1 = AxiBEMSolver::green_function(0, r, z, rp, zp);
    EXPECT_GT(G1, 0.0) << "Green's function should be positive";
}

TEST(PotentialFlow, GreenFunction_N0_DecayWithDistance) {
    double r = 0.2, z = 0.0;
    double G_near = AxiBEMSolver::green_function(0, r, z, 0.2, 0.01);
    double G_far = AxiBEMSolver::green_function(0, r, z, 0.2, 0.5);
    EXPECT_GT(G_near, G_far)
        << "Green's function should decay with distance";
}

TEST(PotentialFlow, GreenFunction_N1_QuadratureConvergence) {
    double r = 0.2, z = 0.0;
    double rp = 0.15, zp = 0.05;

    double G_32 = AxiBEMSolver::green_function(1, r, z, rp, zp, 32);
    double G_64 = AxiBEMSolver::green_function(1, r, z, rp, zp, 64);

    EXPECT_NEAR(G_32, G_64, 1e-10 * std::abs(G_64))
        << "32 vs 64 points should agree for n=1";
}

TEST(PotentialFlow, GreenFunction_DecreaseWithND) {
    // For fixed geometry, G_n should generally decrease with increasing n
    double r = 0.2, z = 0.0;
    double rp = 0.15, zp = 0.05;

    double G0 = std::abs(AxiBEMSolver::green_function(0, r, z, rp, zp));
    double G1 = std::abs(AxiBEMSolver::green_function(1, r, z, rp, zp));
    double G2 = std::abs(AxiBEMSolver::green_function(2, r, z, rp, zp));
    double G3 = std::abs(AxiBEMSolver::green_function(3, r, z, rp, zp));

    EXPECT_GT(G0, G1) << "G_0 > G_1";
    EXPECT_GT(G1, G2) << "G_1 > G_2";
    EXPECT_GT(G2, G3) << "G_2 > G_3";
}

// ========================================================================
// Self-influence tests
// ========================================================================

TEST(PotentialFlow, SelfInfluence_N0_Positive) {
    double G_self = AxiBEMSolver::green_self(0, 0.2, 0.0, 0.01);
    EXPECT_GT(G_self, 0.0) << "Self-influence should be positive";
}

TEST(PotentialFlow, SelfInfluence_N0_IncreasesWithPanelSize) {
    double G_small = AxiBEMSolver::green_self(0, 0.2, 0.0, 0.001);
    double G_large = AxiBEMSolver::green_self(0, 0.2, 0.0, 0.01);
    EXPECT_LT(G_small, G_large)
        << "Larger panels should have larger self-influence";
}

// ========================================================================
// BEM solve: sphere added mass (ND=1, translating sphere)
// ========================================================================

TEST(PotentialFlow, SphereAddedMass_ND1_Translation) {
    // A sphere of radius R translating in x-direction has added mass = (2/3)*pi*rho*R^3.
    // For x-translation (ND=1 mode), the normal velocity is v_n = V*nr*cos(theta),
    // where nr is the radial component of the outward surface normal.
    // M_x = (1/2) * nr^T * M_panel(ND=1) * nr  (factor 1/2 from cos^2 average)
    // The ND=0 (pulsation) case is skipped because the BIE is singular for
    // a closed body at n=0 (constant null space in the Neumann exterior problem).
    double R = 0.1;
    double rho_f = 1000.0;
    double M_ref = (2.0 / 3.0) * PI * rho_f * R * R * R;

    std::cout << "\n=== Sphere Added Mass (ND=1, x-translation) ===\n";
    std::cout << "  Reference: " << M_ref << " kg\n";

    for (int np : {16, 32, 64}) {
        auto mesh = sphere_mesh(R, np);
        AxiBEMSolver bem;
        auto result = bem.solve_nd(1, rho_f, mesh);

        // Build nr vector: on a sphere, nr = r / R
        Eigen::VectorXd nr(np);
        for (int i = 0; i < np; i++) {
            nr(i) = mesh.panels[i].r_mid / R;
        }

        // M_x = (1/2) * nr^T * M_panel * nr
        double M_x = 0.5 * nr.dot(result.M_panel * nr);

        double error_pct = 100.0 * std::abs(M_x - M_ref) / M_ref;
        std::cout << "  N=" << std::setw(3) << np << " panels: M_x="
                  << std::setw(12) << std::setprecision(6) << M_x
                  << " kg, error=" << std::setprecision(1) << std::fixed
                  << error_pct << "%\n" << std::defaultfloat;

        if (np >= 64) {
            EXPECT_LT(error_pct, 10.0)
                << "Sphere translation added mass with " << np
                << " panels should be within 10%";
        }
    }
    std::cout << "\n";
}

// ========================================================================
// BEM matrix properties
// ========================================================================

TEST(PotentialFlow, BEMMatrixSymmetry) {
    // The added mass matrix should be approximately symmetric.
    // Exact symmetry requires consistent quadrature; the panel-midpoint
    // approximation introduces O(h^2) asymmetry.
    auto mesh = sphere_mesh(0.1, 16);
    AxiBEMSolver bem;

    // Only test ND >= 1 (ND=0 on a closed body is singular)
    for (int nd = 1; nd <= 3; nd++) {
        auto result = bem.solve_nd(nd, 1000.0, mesh);
        double asym = (result.M_panel - result.M_panel.transpose()).norm();
        double scale = result.M_panel.norm();
        if (scale > 0) {
            EXPECT_LT(asym / scale, 1e-2)
                << "M_panel should be approximately symmetric for ND=" << nd;
        }
    }
}

TEST(PotentialFlow, BEMAddedMassDecreasesWithND) {
    // For a sphere, added mass should decrease with increasing ND (n >= 1)
    auto mesh = sphere_mesh(0.1, 32);
    AxiBEMSolver bem;

    double prev_norm = std::numeric_limits<double>::max();
    // Start at ND=1 to avoid ND=0 singularity on closed body
    for (int nd = 1; nd <= 4; nd++) {
        auto result = bem.solve_nd(nd, 1000.0, mesh);
        double norm = result.M_panel.norm();
        EXPECT_LT(norm, prev_norm)
            << "Added mass norm should decrease from ND=" << (nd - 1) << " to " << nd;
        prev_norm = norm;
    }
}

// ========================================================================
// Meridional mesh construction
// ========================================================================

TEST(PotentialFlow, MeridionalMeshFromPoints) {
    Eigen::MatrixXd points(4, 2);
    points << 0.1, 0.0,
              0.2, 0.0,
              0.3, 0.0,
              0.3, -0.01;

    auto mesh = MeridionalMesh::from_points(points);
    ASSERT_EQ(mesh.panels.size(), 3);
    EXPECT_NEAR(mesh.panels[0].length, 0.1, 1e-12);
    EXPECT_NEAR(mesh.panels[1].length, 0.1, 1e-12);
    // Panel normals should be consistent
    for (const auto& p : mesh.panels) {
        double nn = std::sqrt(p.nr * p.nr + p.nz * p.nz);
        EXPECT_NEAR(nn, 1.0, 1e-12) << "Normal should be unit length";
    }
}
