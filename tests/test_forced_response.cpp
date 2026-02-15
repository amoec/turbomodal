#include <gtest/gtest.h>
#include "turbomodal/forced_response.hpp"
#include <cmath>
#include <complex>

using namespace turbomodal;

// Helper: build a minimal mesh with proper boundary node sets
static Mesh make_test_mesh(int N = 24) {
    Mesh mesh;
    // 4 nodes at sector boundaries
    double angle = 2.0 * PI / N;
    double r = 0.1;
    Eigen::MatrixXd coords(4, 3);
    coords << r, 0.0, 0.0,      // node 0 on left boundary
              r, 0.0, 0.01,     // node 1 on left boundary
              r * std::cos(angle), r * std::sin(angle), 0.0,   // node 2 on right boundary
              r * std::cos(angle), r * std::sin(angle), 0.01;  // node 3 on right boundary

    Eigen::MatrixXi conn(1, 10);
    conn << 0, 1, 2, 3, 0, 1, 2, 3, 0, 1;

    NodeSet left_ns, right_ns;
    left_ns.name = "left_boundary";
    left_ns.node_ids = {0, 1};
    right_ns.name = "right_boundary";
    right_ns.node_ids = {2, 3};

    std::vector<NodeSet> ns = {left_ns, right_ns};
    mesh.load_from_arrays(coords, conn, ns, N);
    return mesh;
}

// ---- modal_frf static method ----

TEST(ForcedResponse, ModalFRF_AtResonance) {
    // At resonance (omega = omega_r), H = Q / (2i*zeta*omega_r^2)
    double omega_r = 2.0 * PI * 1000.0;  // 1000 Hz
    double zeta = 0.02;
    std::complex<double> Q(1.0, 0.0);

    auto H = ForcedResponseSolver::modal_frf(omega_r, omega_r, Q, zeta);
    // |H| = |Q| / (2*zeta*omega_r^2)
    double expected_mag = 1.0 / (2.0 * zeta * omega_r * omega_r);
    EXPECT_NEAR(std::abs(H), expected_mag, expected_mag * 1e-10);
}

TEST(ForcedResponse, ModalFRF_ZeroFrequency) {
    // Static response: H(0) = Q / omega_r^2
    double omega_r = 2.0 * PI * 500.0;
    double zeta = 0.02;
    std::complex<double> Q(3.0, 0.0);

    auto H = ForcedResponseSolver::modal_frf(0.0, omega_r, Q, zeta);
    double expected = 3.0 / (omega_r * omega_r);
    EXPECT_NEAR(std::abs(H), expected, expected * 1e-10);
}

TEST(ForcedResponse, ModalFRF_HighFrequency) {
    // Far above resonance: H ≈ -Q / omega^2
    double omega_r = 2.0 * PI * 100.0;
    double zeta = 0.01;
    std::complex<double> Q(1.0, 0.0);
    double omega = 2.0 * PI * 10000.0;  // 100x resonance

    auto H = ForcedResponseSolver::modal_frf(omega, omega_r, Q, zeta);
    double expected_mag = 1.0 / (omega * omega - omega_r * omega_r);
    EXPECT_NEAR(std::abs(H), expected_mag, expected_mag * 0.01);
}

TEST(ForcedResponse, ModalFRF_PhaseAtResonance) {
    // At resonance with real Q: phase = -pi/2
    double omega_r = 2.0 * PI * 1000.0;
    double zeta = 0.05;
    std::complex<double> Q(1.0, 0.0);

    auto H = ForcedResponseSolver::modal_frf(omega_r, omega_r, Q, zeta);
    double phase = std::arg(H);
    EXPECT_NEAR(phase, -PI / 2.0, 1e-10);
}

TEST(ForcedResponse, ModalFRF_ZeroDamping) {
    // Zero damping at exact resonance: very large response
    double omega_r = 2.0 * PI * 1000.0;
    std::complex<double> Q(1.0, 0.0);

    auto H = ForcedResponseSolver::modal_frf(omega_r, omega_r, Q, 0.0);
    // Response should be very large (exact value depends on floating-point precision)
    EXPECT_GT(std::abs(H), 1e5);
}

TEST(ForcedResponse, ModalFRF_AnalyticalSweep) {
    // FRF at 100 points matches the formula exactly
    double omega_r = 2.0 * PI * 500.0;
    double zeta = 0.03;
    std::complex<double> Q(2.0, -1.0);

    for (int i = 0; i < 100; i++) {
        double omega = 2.0 * PI * (100.0 + i * 10.0);
        auto H = ForcedResponseSolver::modal_frf(omega, omega_r, Q, zeta);

        std::complex<double> denom(omega_r * omega_r - omega * omega,
                                    2.0 * zeta * omega_r * omega);
        std::complex<double> expected = Q / denom;

        EXPECT_NEAR(H.real(), expected.real(), std::abs(expected) * 1e-12)
            << "Mismatch at omega=" << omega;
        EXPECT_NEAR(H.imag(), expected.imag(), std::abs(expected) * 1e-12)
            << "Mismatch at omega=" << omega;
    }
}

// ---- compute_modal_forces static method ----

TEST(ForcedResponse, ComputeModalForces_IdentityModes) {
    // Identity mode shapes with real values: Q_r = F_r
    // (conj(I) . F = F for real identity, since dot does conj internally)
    int ndof = 6;
    int n_modes = 3;
    Eigen::MatrixXcd phi = Eigen::MatrixXcd::Identity(ndof, n_modes);
    Eigen::VectorXcd F(ndof);
    F << std::complex<double>(1.0, 0.0), std::complex<double>(2.0, 0.0),
         std::complex<double>(3.0, 0.0), std::complex<double>(4.0, 0.0),
         std::complex<double>(5.0, 0.0), std::complex<double>(6.0, 0.0);

    auto Q = ForcedResponseSolver::compute_modal_forces(phi, F);
    EXPECT_EQ(Q.size(), n_modes);
    for (int i = 0; i < n_modes; i++) {
        EXPECT_NEAR(Q(i).real(), F(i).real(), 1e-12);
        EXPECT_NEAR(Q(i).imag(), F(i).imag(), 1e-12);
    }
}

TEST(ForcedResponse, ComputeModalForces_ZeroForce) {
    int ndof = 9;
    int n_modes = 3;
    Eigen::MatrixXcd phi = Eigen::MatrixXcd::Random(ndof, n_modes);
    Eigen::VectorXcd F = Eigen::VectorXcd::Zero(ndof);

    auto Q = ForcedResponseSolver::compute_modal_forces(phi, F);
    for (int i = 0; i < n_modes; i++) {
        EXPECT_NEAR(std::abs(Q(i)), 0.0, 1e-12);
    }
}

TEST(ForcedResponse, ComputeModalForces_ComplexModes) {
    // Implementation: Q(m) = col(m).conjugate().dot(F)
    // Eigen's .dot() does conj(first) * second, so:
    //   col.conjugate().dot(F) = conj(conj(col))^T * F = col^T * F
    int ndof = 4;
    int n_modes = 2;
    Eigen::MatrixXcd phi(ndof, n_modes);
    phi.setZero();
    phi(0, 0) = std::complex<double>(1.0, 1.0);
    phi(1, 1) = std::complex<double>(1.0, -2.0);

    Eigen::VectorXcd F(ndof);
    F << std::complex<double>(3.0, 4.0), std::complex<double>(1.0, 2.0),
         std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0);

    auto Q = ForcedResponseSolver::compute_modal_forces(phi, F);

    // Q(0) = phi(0,0)^T * F = (1+i)(3+4i) = 3+4i+3i-4 = -1+7i
    std::complex<double> expected0 = phi(0, 0) * F(0);
    EXPECT_NEAR(Q(0).real(), expected0.real(), 1e-12);
    EXPECT_NEAR(Q(0).imag(), expected0.imag(), 1e-12);

    // Q(1) = phi(1,1)^T * F(1) = (1-2i)(1+2i) = 1+2i-2i+4 = 5+0i
    std::complex<double> expected1 = phi(1, 1) * F(1);
    EXPECT_NEAR(Q(1).real(), expected1.real(), 1e-12);
    EXPECT_NEAR(Q(1).imag(), expected1.imag(), 1e-12);
}

// ---- EO aliasing (tested indirectly through solve) ----

TEST(ForcedResponse, EOAliasing_ViaSolve) {
    Mesh mesh = make_test_mesh(24);
    int n_nodes = mesh.num_nodes();

    DampingConfig damping;
    damping.type = DampingConfig::Type::MODAL;
    damping.modal_damping_ratios = {0.02};
    ForcedResponseSolver solver(mesh, damping);

    // Create ModalResults for harmonics k=0..5
    std::vector<ModalResult> results;
    for (int k = 0; k <= 5; k++) {
        ModalResult mr;
        mr.harmonic_index = k;
        mr.frequencies.resize(1);
        mr.frequencies(0) = 1000.0 + k * 100.0;
        mr.mode_shapes = Eigen::MatrixXcd::Identity(3 * n_nodes, 1);
        mr.whirl_direction.resize(1);
        mr.whirl_direction(0) = 0;
        results.push_back(mr);
    }

    // EO=3, N=24: should excite k=3
    ForcedResponseConfig cfg;
    cfg.engine_order = 3;
    cfg.excitation_type = ForcedResponseConfig::ExcitationType::POINT_FORCE;
    cfg.force_node_id = 0;
    cfg.num_freq_points = 10;

    auto result = solver.solve(results, 3000.0, cfg);
    EXPECT_EQ(result.natural_frequencies.size(), 1);
    if (result.natural_frequencies.size() > 0) {
        EXPECT_NEAR(result.natural_frequencies(0), 1300.0, 0.01);
    }

    // EO=5, N=24: should excite k=5
    cfg.engine_order = 5;
    result = solver.solve(results, 3000.0, cfg);
    EXPECT_EQ(result.natural_frequencies.size(), 1);

    // EO=7, N=24: nothing in our results
    cfg.engine_order = 7;
    result = solver.solve(results, 3000.0, cfg);
    EXPECT_EQ(result.natural_frequencies.size(), 0);
}

TEST(ForcedResponse, EOAliasing_BackwardAlias) {
    Mesh mesh = make_test_mesh(24);
    int n_nodes = mesh.num_nodes();

    DampingConfig damping;
    damping.type = DampingConfig::Type::MODAL;
    damping.modal_damping_ratios = {0.02};
    ForcedResponseSolver solver(mesh, damping);

    ModalResult mr;
    mr.harmonic_index = 3;
    mr.frequencies.resize(1);
    mr.frequencies(0) = 1300.0;
    mr.mode_shapes = Eigen::MatrixXcd::Identity(3 * n_nodes, 1);
    mr.whirl_direction.resize(1);
    mr.whirl_direction(0) = 0;

    std::vector<ModalResult> results = {mr};

    ForcedResponseConfig cfg;
    cfg.engine_order = 21;  // N-k = 24-3 = 21
    cfg.excitation_type = ForcedResponseConfig::ExcitationType::POINT_FORCE;
    cfg.force_node_id = 0;
    cfg.num_freq_points = 10;

    auto result = solver.solve(results, 3000.0, cfg);
    EXPECT_EQ(result.natural_frequencies.size(), 1);
}

// ---- FRF peak properties ----

TEST(ForcedResponse, ModalFRF_PeakAmplitude) {
    double f_r = 1000.0;
    double omega_r = 2.0 * PI * f_r;
    double zeta = 0.01;
    std::complex<double> Q(5.0, 0.0);

    double max_amp = 0.0;
    for (int i = 0; i < 1000; i++) {
        double f = f_r * (0.95 + 0.1 * i / 999.0);
        double omega = 2.0 * PI * f;
        auto H = ForcedResponseSolver::modal_frf(omega, omega_r, Q, zeta);
        max_amp = std::max(max_amp, std::abs(H));
    }

    double expected_peak = std::abs(Q) / (2.0 * zeta * omega_r * omega_r);
    EXPECT_NEAR(max_amp, expected_peak, expected_peak * 0.01);
}

// ---- compute_participation_factors and effective_modal_mass ----

TEST(ForcedResponse, ParticipationFactors_UnitTranslation) {
    int ndof = 6;
    int n_modes = 2;

    Eigen::MatrixXcd phi(ndof, n_modes);
    phi.setZero();
    phi(0, 0) = 1.0;
    phi(3, 0) = 1.0;
    phi(1, 1) = 1.0;
    phi(4, 1) = 1.0;

    SpMatd M(ndof, ndof);
    std::vector<Triplet> triplets;
    for (int i = 0; i < ndof; i++) {
        triplets.push_back(Triplet(i, i, 1.0));
    }
    M.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::Vector3d dir(1.0, 0.0, 0.0);
    auto gamma = ForcedResponseSolver::compute_participation_factors(phi, M, dir);

    EXPECT_EQ(gamma.size(), n_modes);
    EXPECT_NEAR(gamma(0), 2.0, 1e-10);
    EXPECT_NEAR(gamma(1), 0.0, 1e-10);
}

TEST(ForcedResponse, EffectiveModalMass_Nonnegative) {
    int ndof = 9;
    int n_modes = 3;

    Eigen::MatrixXcd phi = Eigen::MatrixXcd::Random(ndof, n_modes);

    SpMatd M(ndof, ndof);
    std::vector<Triplet> triplets;
    for (int i = 0; i < ndof; i++) {
        triplets.push_back(Triplet(i, i, 1.0 + 0.1 * i));
    }
    M.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::Vector3d dir(0.0, 0.0, 1.0);
    auto m_eff = ForcedResponseSolver::compute_effective_modal_mass(phi, M, dir);

    EXPECT_EQ(m_eff.size(), n_modes);
    for (int i = 0; i < n_modes; i++) {
        EXPECT_GE(m_eff(i), 0.0);
    }
}

// ---- Config defaults ----

TEST(ForcedResponse, ConfigDefaults) {
    ForcedResponseConfig cfg;
    EXPECT_EQ(cfg.engine_order, 1);
    EXPECT_DOUBLE_EQ(cfg.force_amplitude, 1.0);
    EXPECT_EQ(cfg.excitation_type, ForcedResponseConfig::ExcitationType::UNIFORM_PRESSURE);
    EXPECT_EQ(cfg.force_node_id, -1);
    EXPECT_DOUBLE_EQ(cfg.freq_min, 0.0);
    EXPECT_DOUBLE_EQ(cfg.freq_max, 0.0);
    EXPECT_EQ(cfg.num_freq_points, 500);
}

TEST(ForcedResponse, ResultDefaults) {
    ForcedResponseResult result;
    EXPECT_EQ(result.engine_order, 0);
    EXPECT_DOUBLE_EQ(result.rpm, 0.0);
    EXPECT_EQ(result.natural_frequencies.size(), 0);
    EXPECT_EQ(result.sweep_frequencies.size(), 0);
}

// ---- Solve with empty results ----

TEST(ForcedResponse, Solve_NoExcitedModes_EmptyResult) {
    Mesh mesh = make_test_mesh(24);
    ForcedResponseSolver solver(mesh);

    std::vector<ModalResult> results;
    ForcedResponseConfig cfg;
    cfg.engine_order = 3;
    auto result = solver.solve(results, 3000.0, cfg);
    EXPECT_EQ(result.natural_frequencies.size(), 0);
}

// ---- FRF symmetry property ----

TEST(ForcedResponse, ModalFRF_SymmetricDamping) {
    double omega_r = 2.0 * PI * 1000.0;
    double zeta = 0.001;
    std::complex<double> Q(1.0, 0.0);
    double delta = omega_r * 0.0001;

    auto H_lo = ForcedResponseSolver::modal_frf(omega_r - delta, omega_r, Q, zeta);
    auto H_hi = ForcedResponseSolver::modal_frf(omega_r + delta, omega_r, Q, zeta);

    EXPECT_NEAR(std::abs(H_lo), std::abs(H_hi), std::abs(H_lo) * 0.01);
}
