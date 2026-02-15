#include <gtest/gtest.h>
#include "turbomodal/damping.hpp"
#include <cmath>

using namespace turbomodal;

// ---- Default construction ----

TEST(Damping, DefaultConfig_TypeNone) {
    DampingConfig cfg;
    EXPECT_EQ(cfg.type, DampingConfig::Type::NONE);
    EXPECT_TRUE(cfg.modal_damping_ratios.empty());
    EXPECT_DOUBLE_EQ(cfg.rayleigh_alpha, 0.0);
    EXPECT_DOUBLE_EQ(cfg.rayleigh_beta, 0.0);
    EXPECT_TRUE(cfg.aero_damping_ratios.empty());
}

// ---- Type::NONE ----

TEST(Damping, NoDamping_ReturnsZero) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::NONE;
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 1000.0), 0.0);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(5, 5000.0), 0.0);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 0.0), 0.0);
}

// ---- Type::MODAL ----

TEST(Damping, ModalDamping_SingleRatio) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::MODAL;
    cfg.modal_damping_ratios = {0.02};
    // Single ratio reused for all mode indices
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 1000.0), 0.02);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(1, 1000.0), 0.02);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(10, 5000.0), 0.02);
}

TEST(Damping, ModalDamping_MultipleRatios) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::MODAL;
    cfg.modal_damping_ratios = {0.01, 0.02, 0.03};
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 1000.0), 0.01);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(1, 1000.0), 0.02);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(2, 1000.0), 0.03);
}

TEST(Damping, ModalDamping_IndexClamped) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::MODAL;
    cfg.modal_damping_ratios = {0.01, 0.02, 0.03};
    // mode_index beyond size reuses last value
    EXPECT_DOUBLE_EQ(cfg.effective_damping(3, 1000.0), 0.03);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(100, 1000.0), 0.03);
}

TEST(Damping, ModalDamping_EmptyRatios) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::MODAL;
    cfg.modal_damping_ratios = {};
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 1000.0), 0.0);
}

// ---- Type::RAYLEIGH ----

TEST(Damping, RayleighDamping_Formula) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::RAYLEIGH;
    cfg.rayleigh_alpha = 10.0;
    cfg.rayleigh_beta = 1e-5;
    double omega = 1000.0;
    // zeta = alpha/(2*omega) + beta*omega/2 = 10/2000 + 1e-5*500 = 0.005 + 0.005 = 0.01
    double expected = 10.0 / (2.0 * omega) + 1e-5 * omega / 2.0;
    EXPECT_NEAR(cfg.effective_damping(0, omega), expected, 1e-15);
}

TEST(Damping, RayleighDamping_ZeroOmega) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::RAYLEIGH;
    cfg.rayleigh_alpha = 10.0;
    cfg.rayleigh_beta = 1e-5;
    // omega_r=0 is guarded: returns 0
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 0.0), 0.0);
}

TEST(Damping, RayleighDamping_TwoPointFit) {
    // Fit Rayleigh damping to give zeta=0.02 at omega1 and omega2
    double omega1 = 2.0 * PI * 100.0;   // 100 Hz
    double omega2 = 2.0 * PI * 1000.0;  // 1000 Hz
    double zeta_target = 0.02;

    // Solve: zeta = alpha/(2*omega) + beta*omega/2
    // At two points: [1/(2*omega1), omega1/2] [alpha] = [zeta]
    //                [1/(2*omega2), omega2/2] [beta ]   [zeta]
    double a11 = 1.0 / (2.0 * omega1), a12 = omega1 / 2.0;
    double a21 = 1.0 / (2.0 * omega2), a22 = omega2 / 2.0;
    double det = a11 * a22 - a12 * a21;
    double alpha = (a22 * zeta_target - a12 * zeta_target) / det;
    double beta  = (-a21 * zeta_target + a11 * zeta_target) / det;

    DampingConfig cfg;
    cfg.type = DampingConfig::Type::RAYLEIGH;
    cfg.rayleigh_alpha = alpha;
    cfg.rayleigh_beta = beta;

    EXPECT_NEAR(cfg.effective_damping(0, omega1), zeta_target, 1e-12);
    EXPECT_NEAR(cfg.effective_damping(0, omega2), zeta_target, 1e-12);
}

// ---- Aerodynamic damping ----

TEST(Damping, AeroDamping_AddsToStructural) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::MODAL;
    cfg.modal_damping_ratios = {0.01};
    cfg.aero_damping_ratios = {0.005};
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 1000.0), 0.015);
}

TEST(Damping, AeroDamping_OnlyAero) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::NONE;
    cfg.aero_damping_ratios = {0.005};
    // Type::NONE gives 0 structural, but aero still adds
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 1000.0), 0.005);
}

TEST(Damping, AeroDamping_IndexClamped) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::NONE;
    cfg.aero_damping_ratios = {0.01, 0.02, 0.03};
    EXPECT_DOUBLE_EQ(cfg.effective_damping(0, 1000.0), 0.01);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(2, 1000.0), 0.03);
    EXPECT_DOUBLE_EQ(cfg.effective_damping(10, 1000.0), 0.03);  // clamped to last
}

TEST(Damping, AeroDamping_WithRayleigh) {
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::RAYLEIGH;
    cfg.rayleigh_alpha = 0.0;
    cfg.rayleigh_beta = 2e-5;  // pure stiffness-proportional
    cfg.aero_damping_ratios = {0.003};
    double omega = 2000.0;
    double rayleigh_zeta = cfg.rayleigh_beta * omega / 2.0;  // 1e-5 * 1000 = 0.02
    EXPECT_NEAR(cfg.effective_damping(0, omega), rayleigh_zeta + 0.003, 1e-15);
}

// ---- Property test ----

TEST(Damping, EffectiveDamping_Nonnegative) {
    // With all positive inputs, effective damping should be non-negative
    DampingConfig cfg;
    cfg.type = DampingConfig::Type::RAYLEIGH;
    cfg.rayleigh_alpha = 5.0;
    cfg.rayleigh_beta = 1e-4;
    cfg.aero_damping_ratios = {0.001, 0.002};

    for (double omega : {100.0, 500.0, 1000.0, 5000.0, 10000.0}) {
        for (int m = 0; m < 5; m++) {
            EXPECT_GE(cfg.effective_damping(m, omega), 0.0);
        }
    }
}
