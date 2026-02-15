#include <gtest/gtest.h>
#include "turbomodal/cyclic_solver.hpp"
#include "turbomodal/mode_identification.hpp"
#include "turbomodal/forced_response.hpp"
#include "turbomodal/mistuning.hpp"
#include <cmath>
#include <fstream>

using namespace turbomodal;

static std::string test_data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

// ---- End-to-end: Load → Solve → Identify ----

TEST(Integration, LoadSolveIdentify) {
    // Load wedge mesh
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;
    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();

    Material mat(200e9, 0.3, 7800);
    CyclicSymmetrySolver solver(mesh, mat);
    auto results = solver.solve_at_rpm(0.0, 5);

    ASSERT_FALSE(results.empty());

    // Identify modes for each harmonic
    for (const auto& r : results) {
        auto ids = identify_modes(r, mesh);
        EXPECT_EQ(static_cast<int>(ids.size()), static_cast<int>(r.frequencies.size()));
        for (const auto& id : ids) {
            EXPECT_GT(id.frequency, 0.0);
            EXPECT_FALSE(id.family_label.empty());
            EXPECT_EQ(id.nodal_diameter, r.harmonic_index);
        }
    }
}

// ---- End-to-end: Load → Solve → Forced Response ----

TEST(Integration, FullForcedResponsePipeline) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;
    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();

    Material mat(200e9, 0.3, 7800);
    CyclicSymmetrySolver solver(mesh, mat);
    auto results = solver.solve_at_rpm(0.0, 5);
    ASSERT_FALSE(results.empty());

    // Set up modal damping
    DampingConfig damping;
    damping.type = DampingConfig::Type::MODAL;
    damping.modal_damping_ratios = {0.02};

    ForcedResponseSolver fr_solver(mesh, damping);
    ForcedResponseConfig cfg;
    cfg.engine_order = 3;
    cfg.excitation_type = ForcedResponseConfig::ExcitationType::UNIFORM_PRESSURE;
    cfg.force_amplitude = 1000.0;
    cfg.num_freq_points = 200;

    auto fr_result = fr_solver.solve(results, 3000.0, cfg);

    // Should have excited at least one mode from harmonic k=3
    bool has_k3 = false;
    for (const auto& r : results) {
        if (r.harmonic_index == 3) { has_k3 = true; break; }
    }
    if (has_k3) {
        EXPECT_GT(fr_result.natural_frequencies.size(), 0);
        // Peak response should be near the natural frequency
        if (fr_result.natural_frequencies.size() > 0) {
            double f_nat = fr_result.natural_frequencies(0);
            double f_peak = fr_result.resonance_frequencies(0);
            EXPECT_NEAR(f_peak, f_nat, f_nat * 0.05)
                << "Peak should be within 5% of natural frequency";
        }
    }
}

// ---- End-to-end: Solve → FMM Mistuning ----

TEST(Integration, MistuningPipeline) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;
    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();

    Material mat(200e9, 0.3, 7800);
    CyclicSymmetrySolver solver(mesh, mat);
    auto results = solver.solve_at_rpm(0.0, 5);
    ASSERT_FALSE(results.empty());

    // Extract tuned frequencies (first mode at each harmonic)
    int N = mesh.num_sectors;
    int n_harmonics = static_cast<int>(results.size());
    Eigen::VectorXd tuned_freqs(n_harmonics);
    for (int i = 0; i < n_harmonics; i++) {
        tuned_freqs(i) = results[i].frequencies.size() > 0 ?
                         results[i].frequencies(0) : 0.0;
    }

    // Apply FMM mistuning
    auto dev = FMMSolver::random_mistuning(N, 0.03, 42);
    auto fmm_result = FMMSolver::solve(N, tuned_freqs, dev);

    EXPECT_EQ(fmm_result.frequencies.size(), N);
    EXPECT_GE(fmm_result.peak_magnification, 1.0);

    // All frequencies should be positive
    for (int i = 0; i < N; i++) {
        EXPECT_GE(fmm_result.frequencies(i), 0.0);
    }
}
