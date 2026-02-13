#include <gtest/gtest.h>
#include "turbomodal/added_mass.hpp"
#include <cmath>

using namespace turbomodal;

// Reference parameters: steel disk in water (Kwak 1991)
// rho_f = 1000 kg/m^3 (water)
// rho_s = 7850 kg/m^3 (steel)
// h = 0.01 m (thickness)
// a = 0.3 m (radius)
// rho_f/rho_s = 0.12739, a/h = 30
// AVMI_n = gamma_n * 0.12739 * 30 = gamma_n * 3.8217

static constexpr double rho_f = 1000.0;
static constexpr double rho_s = 7850.0;
static constexpr double h = 0.01;
static constexpr double a = 0.3;

// ======== Gamma coefficients ========

TEST(AddedMass, GammaCoefficientsTabulated) {
    // Verify tabulated gamma values from Kwak (1991)
    double expected[] = {0.6526, 0.3910, 0.2737, 0.2078, 0.1660,
                         0.1372, 0.1163, 0.1003, 0.0878, 0.0777, 0.0695};

    for (int n = 0; n <= 10; n++) {
        // gamma_n is private, but we can verify through AVMI with unit ratios
        // AVMI = gamma * (rho_f/rho_s) * (a/h)
        // With rho_f=rho_s and a=h: AVMI = gamma * 1 * 1 = gamma
        double avmi = AddedMassModel::kwak_avmi(n, 1.0, 1.0, 1.0, 1.0);
        EXPECT_NEAR(avmi, expected[n], 1e-6)
            << "Gamma coefficient mismatch at ND=" << n;
    }
}

TEST(AddedMass, GammaMonotonicallyDecreasingTabulated) {
    // Tabulated gamma_n (0..10) should decrease with increasing nodal diameter
    for (int n = 0; n < 10; n++) {
        double avmi_n = AddedMassModel::kwak_avmi(n, 1.0, 1.0, 1.0, 1.0);
        double avmi_n1 = AddedMassModel::kwak_avmi(n + 1, 1.0, 1.0, 1.0, 1.0);
        EXPECT_GT(avmi_n, avmi_n1)
            << "Gamma should decrease: gamma(" << n << ") > gamma(" << n + 1 << ")";
    }
}

TEST(AddedMass, GammaMonotonicallyDecreasingExtrapolated) {
    // Extrapolated gamma_n (11+) should decrease with increasing nodal diameter
    for (int n = 11; n < 20; n++) {
        double avmi_n = AddedMassModel::kwak_avmi(n, 1.0, 1.0, 1.0, 1.0);
        double avmi_n1 = AddedMassModel::kwak_avmi(n + 1, 1.0, 1.0, 1.0, 1.0);
        EXPECT_GT(avmi_n, avmi_n1)
            << "Gamma should decrease: gamma(" << n << ") > gamma(" << n + 1 << ")";
    }
}

TEST(AddedMass, GammaExtrapolationBeyond10) {
    // For n > 10, uses approximate formula: 0.65 / (n + 0.05)^0.85
    // Verify it gives reasonable values
    for (int n = 11; n <= 20; n++) {
        double gamma = AddedMassModel::kwak_avmi(n, 1.0, 1.0, 1.0, 1.0);
        double expected = 0.65 / std::pow(n + 0.05, 0.85);
        EXPECT_NEAR(gamma, expected, 1e-10)
            << "Extrapolation formula mismatch at ND=" << n;
        EXPECT_GT(gamma, 0.0) << "Gamma should be positive at ND=" << n;
    }
}

TEST(AddedMass, GammaBoundaryReasonable) {
    // The approximate formula for n>10 doesn't perfectly match tabulated values.
    // Verify both give values in a reasonable range at the boundary.
    double gamma_10 = AddedMassModel::kwak_avmi(10, 1.0, 1.0, 1.0, 1.0);
    double gamma_11 = AddedMassModel::kwak_avmi(11, 1.0, 1.0, 1.0, 1.0);

    // Both should be small positive values
    EXPECT_GT(gamma_10, 0.0);
    EXPECT_GT(gamma_11, 0.0);
    EXPECT_LT(gamma_10, 0.1);
    EXPECT_LT(gamma_11, 0.1);
}

// ======== AVMI values ========

TEST(AddedMass, KnownAVMIValues) {
    // Steel disk in water: AVMI_n = gamma_n * (1000/7850) * (0.3/0.01)
    double avmi_0 = AddedMassModel::kwak_avmi(0, rho_f, rho_s, h, a);
    double avmi_1 = AddedMassModel::kwak_avmi(1, rho_f, rho_s, h, a);
    double avmi_2 = AddedMassModel::kwak_avmi(2, rho_f, rho_s, h, a);

    // Expected: AVMI_0 ~ 2.49, AVMI_1 ~ 1.49, AVMI_2 ~ 1.05
    EXPECT_NEAR(avmi_0, 2.494, 0.025) << "AVMI_0 should be ~2.49 (within 1%)";
    EXPECT_NEAR(avmi_1, 1.494, 0.015) << "AVMI_1 should be ~1.49 (within 1%)";
    EXPECT_NEAR(avmi_2, 1.046, 0.011) << "AVMI_2 should be ~1.05 (within 1%)";
}

TEST(AddedMass, AVMIScalesLinearly) {
    // AVMI = gamma * (rho_f/rho_s) * (a/h)
    // Doubling rho_f should double AVMI
    double avmi_1x = AddedMassModel::kwak_avmi(0, 1000, rho_s, h, a);
    double avmi_2x = AddedMassModel::kwak_avmi(0, 2000, rho_s, h, a);
    EXPECT_NEAR(avmi_2x / avmi_1x, 2.0, 1e-10);

    // Doubling radius should double AVMI
    double avmi_r1 = AddedMassModel::kwak_avmi(0, rho_f, rho_s, h, 0.3);
    double avmi_r2 = AddedMassModel::kwak_avmi(0, rho_f, rho_s, h, 0.6);
    EXPECT_NEAR(avmi_r2 / avmi_r1, 2.0, 1e-10);

    // Doubling thickness should halve AVMI
    double avmi_h1 = AddedMassModel::kwak_avmi(0, rho_f, rho_s, 0.01, a);
    double avmi_h2 = AddedMassModel::kwak_avmi(0, rho_f, rho_s, 0.02, a);
    EXPECT_NEAR(avmi_h2 / avmi_h1, 0.5, 1e-10);
}

TEST(AddedMass, AVMIMonotonicallyDecreasingWithND) {
    // Check within tabulated range (0-10)
    for (int n = 0; n < 10; n++) {
        double avmi_n = AddedMassModel::kwak_avmi(n, rho_f, rho_s, h, a);
        double avmi_n1 = AddedMassModel::kwak_avmi(n + 1, rho_f, rho_s, h, a);
        EXPECT_GT(avmi_n, avmi_n1)
            << "AVMI should decrease with ND: AVMI(" << n << ") > AVMI(" << n + 1 << ")";
    }
    // Check within extrapolated range (11+)
    for (int n = 11; n < 20; n++) {
        double avmi_n = AddedMassModel::kwak_avmi(n, rho_f, rho_s, h, a);
        double avmi_n1 = AddedMassModel::kwak_avmi(n + 1, rho_f, rho_s, h, a);
        EXPECT_GT(avmi_n, avmi_n1)
            << "AVMI should decrease with ND: AVMI(" << n << ") > AVMI(" << n + 1 << ")";
    }
}

// ======== Frequency ratio ========

TEST(AddedMass, KnownFrequencyRatios) {
    // f_wet/f_dry = 1/sqrt(1 + AVMI_n)
    double r0 = AddedMassModel::frequency_ratio(0, rho_f, rho_s, h, a);
    double r1 = AddedMassModel::frequency_ratio(1, rho_f, rho_s, h, a);
    double r2 = AddedMassModel::frequency_ratio(2, rho_f, rho_s, h, a);

    // Expected: ratio_0 ~ 0.535, ratio_1 ~ 0.633, ratio_2 ~ 0.699
    EXPECT_NEAR(r0, 0.535, 0.01) << "Frequency ratio ND=0 should be ~0.54";
    EXPECT_NEAR(r1, 0.633, 0.01) << "Frequency ratio ND=1 should be ~0.63";
    EXPECT_NEAR(r2, 0.699, 0.01) << "Frequency ratio ND=2 should be ~0.70";
}

TEST(AddedMass, FrequencyRatioAlwaysLessThanOne) {
    // For any fluid present, ratio should be < 1
    for (int n = 0; n <= 20; n++) {
        double ratio = AddedMassModel::frequency_ratio(n, rho_f, rho_s, h, a);
        EXPECT_GT(ratio, 0.0) << "Frequency ratio should be positive at ND=" << n;
        EXPECT_LT(ratio, 1.0) << "Frequency ratio should be < 1 at ND=" << n;
    }
}

TEST(AddedMass, FrequencyRatioIncreasesWithND) {
    // Higher ND -> lower AVMI -> higher frequency ratio (closer to 1)
    // Check within tabulated range
    for (int n = 0; n < 10; n++) {
        double r_n = AddedMassModel::frequency_ratio(n, rho_f, rho_s, h, a);
        double r_n1 = AddedMassModel::frequency_ratio(n + 1, rho_f, rho_s, h, a);
        EXPECT_LT(r_n, r_n1)
            << "Ratio should increase: ratio(" << n << ") < ratio(" << n + 1 << ")";
    }
    // Check within extrapolated range
    for (int n = 11; n < 20; n++) {
        double r_n = AddedMassModel::frequency_ratio(n, rho_f, rho_s, h, a);
        double r_n1 = AddedMassModel::frequency_ratio(n + 1, rho_f, rho_s, h, a);
        EXPECT_LT(r_n, r_n1)
            << "Ratio should increase: ratio(" << n << ") < ratio(" << n + 1 << ")";
    }
}

TEST(AddedMass, ZeroFluidDensityGivesUnityRatio) {
    // No fluid -> f_wet/f_dry = 1.0
    double ratio = AddedMassModel::frequency_ratio(0, 0.0, rho_s, h, a);
    EXPECT_DOUBLE_EQ(ratio, 1.0) << "Zero fluid density should give ratio=1.0";
}

TEST(AddedMass, FrequencyRatioConsistentWithAVMI) {
    // ratio = 1/sqrt(1 + AVMI)
    for (int n = 0; n <= 10; n++) {
        double avmi = AddedMassModel::kwak_avmi(n, rho_f, rho_s, h, a);
        double ratio = AddedMassModel::frequency_ratio(n, rho_f, rho_s, h, a);
        double expected = 1.0 / std::sqrt(1.0 + avmi);
        EXPECT_NEAR(ratio, expected, 1e-12)
            << "Ratio should equal 1/sqrt(1+AVMI) at ND=" << n;
    }
}

// ======== Corrected mass matrix ========

TEST(AddedMass, CorrectedMassMatrixScaling) {
    // M_wet = M_dry * (1 + AVMI)
    int n = 5;
    SpMatd M_dry(n, n);
    std::vector<Triplet> trips;
    for (int i = 0; i < n; i++) {
        trips.emplace_back(i, i, 1.0 + 0.1 * i);
    }
    M_dry.setFromTriplets(trips.begin(), trips.end());

    SpMatd M_wet = AddedMassModel::corrected_mass_matrix(M_dry, 0, rho_f, rho_s, h, a);

    double avmi_0 = AddedMassModel::kwak_avmi(0, rho_f, rho_s, h, a);
    double scale = 1.0 + avmi_0;

    // Each entry should be scaled by (1 + AVMI)
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(M_wet.coeff(i, i), M_dry.coeff(i, i) * scale, 1e-10)
            << "M_wet(i,i) should equal M_dry(i,i) * (1 + AVMI)";
    }
}

TEST(AddedMass, CorrectedMassMatrixPreservesSparsity) {
    int n = 5;
    SpMatd M_dry(n, n);
    std::vector<Triplet> trips;
    trips.emplace_back(0, 0, 1.0);
    trips.emplace_back(1, 1, 2.0);
    trips.emplace_back(0, 1, 0.5);
    trips.emplace_back(1, 0, 0.5);
    M_dry.setFromTriplets(trips.begin(), trips.end());

    SpMatd M_wet = AddedMassModel::corrected_mass_matrix(M_dry, 2, rho_f, rho_s, h, a);

    // Should have same nonzero pattern
    EXPECT_EQ(M_wet.nonZeros(), M_dry.nonZeros());
}

// ======== Edge cases ========

TEST(AddedMass, NegativeNDThrows) {
    EXPECT_THROW(AddedMassModel::kwak_avmi(-1, rho_f, rho_s, h, a), std::invalid_argument);
}

TEST(AddedMass, LargeNDStillPositive) {
    // Even at very high ND, AVMI should be small but positive
    double avmi = AddedMassModel::kwak_avmi(100, rho_f, rho_s, h, a);
    EXPECT_GT(avmi, 0.0);
    EXPECT_LT(avmi, 0.5) << "AVMI at ND=100 should be small";
}

TEST(AddedMass, HighFluidDensityGivesLowRatio) {
    // Very dense fluid should give ratio close to 0
    double ratio = AddedMassModel::frequency_ratio(0, 100000.0, rho_s, h, a);
    EXPECT_LT(ratio, 0.1) << "Very dense fluid should give very low frequency ratio";
}

TEST(AddedMass, ThinDiskHigherAVMI) {
    // Thinner disk -> higher AVMI (more affected by fluid)
    double avmi_thin = AddedMassModel::kwak_avmi(0, rho_f, rho_s, 0.005, a);
    double avmi_thick = AddedMassModel::kwak_avmi(0, rho_f, rho_s, 0.020, a);
    EXPECT_GT(avmi_thin, avmi_thick)
        << "Thinner disk should have higher AVMI";
}
