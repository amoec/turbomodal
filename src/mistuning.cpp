#include "turbomodal/mistuning.hpp"
#include <random>
#include <Eigen/Eigenvalues>

namespace turbomodal {

MistuningResult FMMSolver::solve(
    int num_sectors,
    const Eigen::VectorXd& tuned_frequencies,
    const Eigen::VectorXd& blade_frequency_deviations)
{
    const int N = num_sectors;
    if (blade_frequency_deviations.size() != N) {
        throw std::invalid_argument(
            "blade_frequency_deviations must have length num_sectors");
    }

    // Build tuned eigenvalue vector in harmonic coordinates (omega^2)
    // For N blades: harmonic indices go 0, 1, ..., N/2
    // Each harmonic index k has two modes (cos, sin) except k=0 and k=N/2
    // Total: N modes
    int max_k = N / 2;
    int n_tuned = static_cast<int>(tuned_frequencies.size());

    // Build omega^2 for all N harmonic modes
    Eigen::VectorXd omega2_harmonic(N);
    for (int idx = 0; idx < N; idx++) {
        // Map blade index to harmonic index via DFT ordering
        // DFT indices: 0, 1, ..., N/2, ..., N-1
        // Harmonic k corresponds to DFT indices k and N-k
        int k;
        if (idx <= N / 2) {
            k = idx;
        } else {
            k = N - idx;
        }
        k = std::min(k, n_tuned - 1);
        double f = tuned_frequencies(k);
        omega2_harmonic(idx) = (2.0 * PI * f) * (2.0 * PI * f);
    }

    // Build DFT matrix E: E(b,n) = (1/sqrt(N)) * exp(i*2*pi*b*n/N)
    Eigen::MatrixXcd E(N, N);
    for (int b = 0; b < N; b++) {
        for (int n = 0; n < N; n++) {
            double phase = 2.0 * PI * b * n / N;
            E(b, n) = std::complex<double>(std::cos(phase), std::sin(phase)) /
                       std::sqrt(static_cast<double>(N));
        }
    }

    // Tuned system in blade coordinates: Omega_tuned = E * diag(omega2) * E^H
    Eigen::MatrixXcd Omega_tuned = E * omega2_harmonic.asDiagonal().toDenseMatrix().cast<std::complex<double>>() * E.adjoint();

    // Reference frequency for mistuning perturbation
    // Use the average of tuned frequencies
    double omega_ref = 0.0;
    for (int k = 0; k < n_tuned; k++) {
        omega_ref += tuned_frequencies(k);
    }
    omega_ref = 2.0 * PI * omega_ref / n_tuned;

    // Mistuning perturbation in blade coordinates: diagonal
    // delta_Omega(b,b) = 2 * omega_ref^2 * delta_b
    Eigen::MatrixXcd delta_Omega = Eigen::MatrixXcd::Zero(N, N);
    for (int b = 0; b < N; b++) {
        delta_Omega(b, b) = 2.0 * omega_ref * omega_ref *
                             blade_frequency_deviations(b);
    }

    // Solve the N x N eigenvalue problem: (Omega_tuned + delta_Omega) v = lambda v
    Eigen::MatrixXcd H = Omega_tuned + delta_Omega;

    // Force Hermitian
    H = (H + H.adjoint()) / 2.0;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eig(H);
    if (eig.info() != Eigen::Success) {
        throw std::runtime_error("FMM eigenvalue solve failed");
    }

    MistuningResult result;
    result.frequencies.resize(N);
    result.blade_amplitudes = eig.eigenvectors();
    result.amplitude_magnification.resize(N);
    result.localization_ipr.resize(N);

    // Convert eigenvalues to frequencies
    for (int i = 0; i < N; i++) {
        double lambda = eig.eigenvalues()(i);
        if (lambda > 0) {
            result.frequencies(i) = std::sqrt(lambda) / (2.0 * PI);
        } else {
            result.frequencies(i) = 0.0;
        }
    }

    // Compute amplitude magnification and localization
    // Reference: uniform tuned blade amplitude = 1/sqrt(N)
    double tuned_amp = 1.0 / std::sqrt(static_cast<double>(N));

    result.peak_magnification = 1.0;
    for (int m = 0; m < N; m++) {
        // Max blade amplitude for this mode
        double max_amp = 0.0;
        double sum2 = 0.0;
        double sum4 = 0.0;
        for (int b = 0; b < N; b++) {
            double a = std::abs(result.blade_amplitudes(b, m));
            max_amp = std::max(max_amp, a);
            sum2 += a * a;
            sum4 += a * a * a * a;
        }

        // Magnification relative to tuned
        result.amplitude_magnification(m) = max_amp / std::max(tuned_amp, 1e-15);

        // Inverse participation ratio
        if (sum2 > 1e-30) {
            result.localization_ipr(m) = N * sum4 / (sum2 * sum2);
        } else {
            result.localization_ipr(m) = 1.0;
        }

        result.peak_magnification = std::max(result.peak_magnification,
                                              result.amplitude_magnification(m));
    }

    return result;
}

Eigen::VectorXd FMMSolver::random_mistuning(
    int num_sectors, double sigma, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, sigma);

    Eigen::VectorXd deviations(num_sectors);
    for (int i = 0; i < num_sectors; i++) {
        deviations(i) = dist(gen);
    }
    return deviations;
}

}  // namespace turbomodal
