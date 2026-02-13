#include <gtest/gtest.h>
#include "turbomodal/material.hpp"

using namespace turbomodal;

// --- Construction ---

TEST(Material, ConstructionThreeArg) {
    Material mat(200e9, 0.3, 7850);
    EXPECT_DOUBLE_EQ(mat.E, 200e9);
    EXPECT_DOUBLE_EQ(mat.nu, 0.3);
    EXPECT_DOUBLE_EQ(mat.rho, 7850);
    EXPECT_DOUBLE_EQ(mat.T_ref, 293.15);
    EXPECT_DOUBLE_EQ(mat.E_slope, 0.0);
}

TEST(Material, ConstructionFiveArg) {
    Material mat(110e9, 0.34, 4430, 300.0, -50e6);
    EXPECT_DOUBLE_EQ(mat.E, 110e9);
    EXPECT_DOUBLE_EQ(mat.nu, 0.34);
    EXPECT_DOUBLE_EQ(mat.rho, 4430);
    EXPECT_DOUBLE_EQ(mat.T_ref, 300.0);
    EXPECT_DOUBLE_EQ(mat.E_slope, -50e6);
}

// --- Validation ---

TEST(Material, RejectsNegativeE) {
    EXPECT_THROW(Material(-1e9, 0.3, 7850), std::invalid_argument);
}

TEST(Material, RejectsZeroE) {
    EXPECT_THROW(Material(0.0, 0.3, 7850), std::invalid_argument);
}

TEST(Material, RejectsNuZero) {
    EXPECT_THROW(Material(200e9, 0.0, 7850), std::invalid_argument);
}

TEST(Material, RejectsNuNegative) {
    EXPECT_THROW(Material(200e9, -0.1, 7850), std::invalid_argument);
}

TEST(Material, RejectsNuHalf) {
    EXPECT_THROW(Material(200e9, 0.5, 7850), std::invalid_argument);
}

TEST(Material, RejectsNuAboveHalf) {
    EXPECT_THROW(Material(200e9, 0.6, 7850), std::invalid_argument);
}

TEST(Material, RejectsNegativeRho) {
    EXPECT_THROW(Material(200e9, 0.3, -100), std::invalid_argument);
}

TEST(Material, RejectsZeroRho) {
    EXPECT_THROW(Material(200e9, 0.3, 0.0), std::invalid_argument);
}

// --- Constitutive Matrix: Structure ---

TEST(Material, ConstitutiveMatrixSymmetry) {
    Material mat(200e9, 0.3, 7850);
    auto D = mat.constitutive_matrix();
    EXPECT_TRUE(D.isApprox(D.transpose(), 1e-10));
}

TEST(Material, ConstitutiveMatrixPositiveDefinite) {
    Material mat(200e9, 0.3, 7850);
    auto D = mat.constitutive_matrix();
    Eigen::SelfAdjointEigenSolver<Matrix6d> es(D);
    EXPECT_GT(es.eigenvalues().minCoeff(), 0.0);
}

TEST(Material, ConstitutiveMatrixZeroOffDiagonalInShearBlock) {
    Material mat(200e9, 0.3, 7850);
    auto D = mat.constitutive_matrix();
    // Shear block (rows/cols 3-5) should be diagonal
    EXPECT_DOUBLE_EQ(D(3, 4), 0.0);
    EXPECT_DOUBLE_EQ(D(3, 5), 0.0);
    EXPECT_DOUBLE_EQ(D(4, 5), 0.0);
    // Normal-shear coupling should be zero
    EXPECT_DOUBLE_EQ(D(0, 3), 0.0);
    EXPECT_DOUBLE_EQ(D(1, 4), 0.0);
    EXPECT_DOUBLE_EQ(D(2, 5), 0.0);
}

// --- Constitutive Matrix: Known Values (Steel) ---

TEST(Material, ConstitutiveMatrixSteelKnownValues) {
    // Steel: E=200 GPa, nu=0.3, rho=7850
    Material mat(200e9, 0.3, 7850);
    auto D = mat.constitutive_matrix();

    // c = E / ((1+nu)(1-2*nu)) = 200e9 / (1.3 * 0.4) = 384.6154e9
    double c = 200e9 / (1.3 * 0.4);

    // D(0,0) = c * (1-nu) = c * 0.7
    EXPECT_NEAR(D(0, 0), c * 0.7, 1e-2);
    EXPECT_NEAR(D(0, 0), 269.2308e9, 1e5);  // ~269.23 GPa

    // D(0,1) = c * nu = c * 0.3
    EXPECT_NEAR(D(0, 1), c * 0.3, 1e-2);
    EXPECT_NEAR(D(0, 1), 115.3846e9, 1e5);  // ~115.38 GPa

    // D(3,3) = c * (1-2*nu)/2 = shear modulus G = E/(2(1+nu))
    double G = 200e9 / (2.0 * 1.3);  // 76.923 GPa
    EXPECT_NEAR(D(3, 3), G, 1e-2);
    EXPECT_NEAR(D(3, 3), 76.9231e9, 1e5);

    // All three normal stiffnesses equal
    EXPECT_DOUBLE_EQ(D(0, 0), D(1, 1));
    EXPECT_DOUBLE_EQ(D(0, 0), D(2, 2));

    // All three shear stiffnesses equal
    EXPECT_DOUBLE_EQ(D(3, 3), D(4, 4));
    EXPECT_DOUBLE_EQ(D(3, 3), D(5, 5));
}

// --- Constitutive Matrix: Known Values (Ti-6Al-4V) ---

TEST(Material, ConstitutiveMatrixTitaniumKnownValues) {
    // Ti-6Al-4V: E=110 GPa, nu=0.34, rho=4430
    Material mat(110e9, 0.34, 4430);
    auto D = mat.constitutive_matrix();

    // c = E / ((1+nu)(1-2*nu)) = 110e9 / (1.34 * 0.32) = 256.506e9
    double c = 110e9 / (1.34 * 0.32);

    // D(0,0) = c * (1-nu) = c * 0.66 = ~169.29 GPa
    EXPECT_NEAR(D(0, 0), c * 0.66, 1e-2);
    EXPECT_NEAR(D(0, 0), 169.29e9, 0.1e9);

    // D(3,3) = G = E/(2(1+nu)) = 110e9 / 2.68 = 41.045 GPa
    double G = 110e9 / (2.0 * 1.34);
    EXPECT_NEAR(D(3, 3), G, 1e-2);
}

// --- Constitutive Matrix: Shear Modulus Identity ---

TEST(Material, ShearModulusEqualsGFormula) {
    // For any valid material: D(3,3) = G = E / (2*(1+nu))
    // Test with several materials
    std::vector<std::pair<double, double>> materials = {
        {200e9, 0.3},   // Steel
        {110e9, 0.34},  // Titanium
        {72e9, 0.33},   // Aluminum
        {200e9, 0.29},  // Inconel
    };
    for (auto [E, nu] : materials) {
        Material mat(E, nu, 1000);  // rho doesn't affect D
        auto D = mat.constitutive_matrix();
        double G_expected = E / (2.0 * (1.0 + nu));
        EXPECT_NEAR(D(3, 3), G_expected, 1e-2)
            << "Failed for E=" << E << ", nu=" << nu;
    }
}

// --- Constitutive Matrix: D is independent of rho ---

TEST(Material, ConstitutiveMatrixIndependentOfDensity) {
    Material mat1(200e9, 0.3, 7850);
    Material mat2(200e9, 0.3, 1000);
    EXPECT_TRUE(mat1.constitutive_matrix().isApprox(mat2.constitutive_matrix(), 1e-15));
}

// --- Temperature Dependence ---

TEST(Material, AtTemperatureDecreaseE) {
    // Negative slope: E decreases with temperature
    Material mat(110e9, 0.34, 4430, 300.0, -50e6);
    Material hot = mat.at_temperature(500.0);  // 200K above ref

    double expected_E = 110e9 + (-50e6) * (500.0 - 300.0);  // 110e9 - 10e9 = 100e9
    EXPECT_NEAR(hot.E, expected_E, 1e-2);
    EXPECT_DOUBLE_EQ(hot.nu, mat.nu);
    EXPECT_DOUBLE_EQ(hot.rho, mat.rho);
}

TEST(Material, AtTemperatureReferenceUnchanged) {
    Material mat(110e9, 0.34, 4430, 300.0, -50e6);
    Material same = mat.at_temperature(300.0);  // At reference temperature
    EXPECT_NEAR(same.E, mat.E, 1e-2);
}

TEST(Material, AtTemperatureIncreaseE) {
    // Below reference temperature: E increases (with negative slope)
    Material mat(110e9, 0.34, 4430, 300.0, -50e6);
    Material cold = mat.at_temperature(200.0);  // 100K below ref
    EXPECT_GT(cold.E, mat.E);
}

TEST(Material, AtTemperatureThrowsWhenENonPositive) {
    // Slope would drive E negative at extreme temperature
    Material mat(110e9, 0.34, 4430, 300.0, -50e6);
    // E would be: 110e9 + (-50e6) * (2500 - 300) = 110e9 - 110e9 = 0 -> throws
    EXPECT_THROW(mat.at_temperature(2500.0), std::runtime_error);
}

TEST(Material, AtTemperatureZeroSlopeNoChange) {
    Material mat(200e9, 0.3, 7850);  // Default E_slope = 0
    Material hot = mat.at_temperature(1000.0);
    EXPECT_DOUBLE_EQ(hot.E, mat.E);
}

// --- Constitutive Matrix: Positive Definite for Multiple Materials ---

TEST(Material, ConstitutiveMatrixPDForAllAppendixBMaterials) {
    // All materials from spec Appendix B
    struct MatData { double E; double nu; double rho; };
    std::vector<MatData> materials = {
        {110e9, 0.34, 4430},   // Ti-6Al-4V
        {200e9, 0.29, 8190},   // Inconel 718
        {196e9, 0.30, 7750},   // Steel 17-4PH
        {72e9,  0.33, 2810},   // Aluminum 7075
        {211e9, 0.28, 8190},   // Waspaloy
        {200e9, 0.30, 8530},   // MAR-M-247
    };
    for (const auto& md : materials) {
        Material mat(md.E, md.nu, md.rho);
        auto D = mat.constitutive_matrix();
        Eigen::SelfAdjointEigenSolver<Matrix6d> es(D);
        EXPECT_GT(es.eigenvalues().minCoeff(), 0.0)
            << "Failed for E=" << md.E << ", nu=" << md.nu;
        EXPECT_TRUE(D.isApprox(D.transpose(), 1e-10));
    }
}
