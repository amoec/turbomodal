#include <gtest/gtest.h>
#include "turbomodal/mistuning.hpp"
#include <cmath>
#include <set>
#include <algorithm>
#include <numeric>

using namespace turbomodal;

// ---- random_mistuning ----

TEST(Mistuning, RandomMistuning_CorrectSize) {
    auto dev = FMMSolver::random_mistuning(24, 0.05, 42);
    EXPECT_EQ(dev.size(), 24);
}

TEST(Mistuning, RandomMistuning_ZeroMean) {
    int N = 1000;
    auto dev = FMMSolver::random_mistuning(N, 0.05, 42);
    double mean = dev.sum() / N;
    // Mean should be near 0 within statistical bounds
    EXPECT_NEAR(mean, 0.0, 3.0 * 0.05 / std::sqrt(static_cast<double>(N)));
}

TEST(Mistuning, RandomMistuning_StdDev) {
    int N = 1000;
    double sigma = 0.05;
    auto dev = FMMSolver::random_mistuning(N, sigma, 42);
    double mean = dev.sum() / N;
    double var = 0.0;
    for (int i = 0; i < N; i++) {
        var += (dev(i) - mean) * (dev(i) - mean);
    }
    var /= (N - 1);
    double std_dev = std::sqrt(var);
    EXPECT_NEAR(std_dev, sigma, sigma * 0.2);  // within 20%
}

TEST(Mistuning, RandomMistuning_Reproducible) {
    auto dev1 = FMMSolver::random_mistuning(24, 0.05, 42);
    auto dev2 = FMMSolver::random_mistuning(24, 0.05, 42);
    for (int i = 0; i < 24; i++) {
        EXPECT_DOUBLE_EQ(dev1(i), dev2(i));
    }
}

TEST(Mistuning, RandomMistuning_DifferentSeeds) {
    auto dev1 = FMMSolver::random_mistuning(24, 0.05, 42);
    auto dev2 = FMMSolver::random_mistuning(24, 0.05, 99);
    bool all_equal = true;
    for (int i = 0; i < 24; i++) {
        if (dev1(i) != dev2(i)) { all_equal = false; break; }
    }
    EXPECT_FALSE(all_equal);
}

// ---- FMMSolver::solve input validation ----

TEST(Mistuning, Solve_WrongSize_Throws) {
    Eigen::VectorXd tuned(13);  // N/2+1 = 13 for N=24
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 100.0;

    Eigen::VectorXd dev_wrong(10);  // wrong size, should be 24
    dev_wrong.setZero();
    EXPECT_THROW(FMMSolver::solve(24, tuned, dev_wrong), std::invalid_argument);
}

TEST(Mistuning, Solve_SmallSystem_N4) {
    int N = 4;
    Eigen::VectorXd tuned(3);  // N/2+1 = 3
    tuned << 1000.0, 1200.0, 1500.0;
    Eigen::VectorXd dev = Eigen::VectorXd::Zero(N);

    auto result = FMMSolver::solve(N, tuned, dev);
    EXPECT_EQ(result.frequencies.size(), N);
    for (int i = 0; i < N; i++) {
        EXPECT_GT(result.frequencies(i), 0.0);
    }
}

TEST(Mistuning, Solve_EvenOddN) {
    // Even N=24
    {
        Eigen::VectorXd tuned(13);
        for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
        Eigen::VectorXd dev = Eigen::VectorXd::Zero(24);
        auto result = FMMSolver::solve(24, tuned, dev);
        EXPECT_EQ(result.frequencies.size(), 24);
    }
    // Odd N=23
    {
        Eigen::VectorXd tuned(12);  // N/2+1 = 12 for N=23
        for (int i = 0; i < 12; i++) tuned(i) = 1000.0 + i * 50.0;
        Eigen::VectorXd dev = Eigen::VectorXd::Zero(23);
        auto result = FMMSolver::solve(23, tuned, dev);
        EXPECT_EQ(result.frequencies.size(), 23);
    }
}

TEST(Mistuning, Solve_LargeMistuning) {
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
    auto dev = FMMSolver::random_mistuning(N, 0.20, 42);
    auto result = FMMSolver::solve(N, tuned, dev);
    EXPECT_EQ(result.frequencies.size(), N);
    // All frequencies should still be non-negative
    for (int i = 0; i < N; i++) {
        EXPECT_GE(result.frequencies(i), 0.0);
    }
}

// ---- Mistuned system properties ----

TEST(Mistuning, MistunedSystem_FrequencySplit) {
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
    auto dev = FMMSolver::random_mistuning(N, 0.05, 42);
    auto result = FMMSolver::solve(N, tuned, dev);

    // Frequencies should not all be identical
    double f_min = result.frequencies.minCoeff();
    double f_max = result.frequencies.maxCoeff();
    EXPECT_GT(f_max - f_min, 1.0);  // at least 1 Hz spread
}

TEST(Mistuning, MistunedSystem_MagnificationAboveOne) {
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
    auto dev = FMMSolver::random_mistuning(N, 0.05, 42);
    auto result = FMMSolver::solve(N, tuned, dev);
    EXPECT_GE(result.peak_magnification, 1.0);
}

TEST(Mistuning, MistunedSystem_IPRIncreases) {
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
    auto dev = FMMSolver::random_mistuning(N, 0.05, 42);
    auto result = FMMSolver::solve(N, tuned, dev);
    // At least one mode should have IPR > 1 (localized)
    EXPECT_GT(result.localization_ipr.maxCoeff(), 1.0);
}

// ---- Validation: tuned system analytical properties ----

TEST(Mistuning, TunedSystem_FrequenciesPreserved) {
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 500.0 + i * 100.0;
    Eigen::VectorXd dev = Eigen::VectorXd::Zero(N);

    auto result = FMMSolver::solve(N, tuned, dev);

    // Build set of expected frequencies (each tuned freq appears for cos/sin pairs)
    std::vector<double> expected;
    for (int k = 0; k < 13; k++) {
        expected.push_back(tuned(k));
    }

    // Every result frequency should be close to one of the tuned frequencies
    for (int i = 0; i < N; i++) {
        double f = result.frequencies(i);
        bool found = false;
        for (double ef : expected) {
            if (std::abs(f - ef) < 0.1) {  // within 0.1 Hz
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Frequency " << f << " Hz not in tuned set";
    }
}

TEST(Mistuning, TunedSystem_EigenvectorNormalization) {
    // Each eigenvector column should have unit norm
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
    Eigen::VectorXd dev = Eigen::VectorXd::Zero(N);

    auto result = FMMSolver::solve(N, tuned, dev);

    for (int m = 0; m < N; m++) {
        double norm = 0.0;
        for (int b = 0; b < N; b++) {
            norm += std::norm(result.blade_amplitudes(b, m));
        }
        EXPECT_NEAR(norm, 1.0, 1e-10)
            << "Mode " << m << " eigenvector should have unit norm";
    }
}

TEST(Mistuning, TunedSystem_MagnificationBounded) {
    // For a tuned system, magnification is bounded. Due to degenerate eigenvector
    // arbitrariness, it may exceed 1.0 but should not exceed sqrt(2).
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
    Eigen::VectorXd dev = Eigen::VectorXd::Zero(N);

    auto result = FMMSolver::solve(N, tuned, dev);
    EXPECT_LE(result.peak_magnification, std::sqrt(2.0) + 0.01);
}

TEST(Mistuning, TunedSystem_IPRBounded) {
    // For a tuned system, IPR should be bounded.
    // Degenerate modes can have IPR up to 2.0.
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
    Eigen::VectorXd dev = Eigen::VectorXd::Zero(N);

    auto result = FMMSolver::solve(N, tuned, dev);
    for (int m = 0; m < N; m++) {
        EXPECT_LE(result.localization_ipr(m), 2.0)
            << "Mode " << m << " IPR should be bounded for tuned system";
        EXPECT_GE(result.localization_ipr(m), 0.5)
            << "Mode " << m << " IPR should be at least 0.5";
    }
}

TEST(Mistuning, WhiteheadBound_NotExceeded) {
    // Whitehead (1966): peak magnification <= 1 + sqrt((N-1)/2)
    int N = 24;
    double whitehead_bound = 1.0 + std::sqrt((N - 1) / 2.0);

    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;

    for (unsigned int seed = 0; seed < 100; seed++) {
        auto dev = FMMSolver::random_mistuning(N, 0.05, seed);
        auto result = FMMSolver::solve(N, tuned, dev);
        EXPECT_LE(result.peak_magnification, whitehead_bound)
            << "Whitehead bound violated for seed " << seed;
    }
}

TEST(Mistuning, HermitianMatrix_RealEigenvalues) {
    // All frequencies should be real (non-negative) since FMM matrix is forced Hermitian
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 1000.0 + i * 50.0;
    auto dev = FMMSolver::random_mistuning(N, 0.05, 42);
    auto result = FMMSolver::solve(N, tuned, dev);

    for (int i = 0; i < N; i++) {
        EXPECT_GE(result.frequencies(i), 0.0)
            << "Frequency " << i << " should be non-negative";
    }
}
