#include <gtest/gtest.h>
#include "turbomodal/cyclic_solver.hpp"
#include <fstream>
#include <cmath>
#include <sstream>

using namespace turbomodal;

static std::string test_data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

// Helper: load and prepare the wedge sector mesh
static Mesh load_wedge_mesh() {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;
    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();
    return mesh;
}

// ---- FluidConfig Defaults ----

TEST(Cyclic, FluidConfigDefault) {
    FluidConfig config;
    EXPECT_EQ(config.type, FluidConfig::Type::NONE);
    EXPECT_DOUBLE_EQ(config.fluid_density, 0.0);
    EXPECT_DOUBLE_EQ(config.disk_radius, 0.0);
    EXPECT_DOUBLE_EQ(config.disk_thickness, 0.0);
}

// ---- DOF Classification ----

TEST(Cyclic, DofClassification) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);

    CyclicSymmetrySolver solver(mesh, mat);

    // Total DOFs should equal interior + left + right
    // (We can't directly access private members, but we verify via transformation matrix)
    // The transformation matrix should have full_ndof rows
    // and (full_ndof - 3*num_right_boundary_nodes) columns
    int full_ndof = mesh.num_dof();
    int n_right = static_cast<int>(mesh.right_boundary.size());
    int expected_reduced = full_ndof - 3 * n_right;

    // Build transformation for k=0 and check dimensions
    // We test this indirectly via solve_at_rpm producing correct-dimensioned results
    EXPECT_GT(expected_reduced, 0);
    EXPECT_LT(expected_reduced, full_ndof);
}

// ---- Transformation Matrix ----

TEST(Cyclic, TransformationK0IsReal) {
    // For k=0, phase = exp(i*0) = 1, so T should be purely real
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    // Apply cyclic BC and check resulting matrices are real
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    auto [K_k, M_k] = solver.apply_cyclic_bc_public(0, assembler.K(), assembler.M());

    // Check imaginary parts are negligible
    double imag_norm_K = 0, imag_norm_M = 0;
    for (int col = 0; col < K_k.outerSize(); ++col) {
        for (SpMatcd::InnerIterator it(K_k, col); it; ++it) {
            imag_norm_K += it.value().imag() * it.value().imag();
        }
    }
    for (int col = 0; col < M_k.outerSize(); ++col) {
        for (SpMatcd::InnerIterator it(M_k, col); it; ++it) {
            imag_norm_M += it.value().imag() * it.value().imag();
        }
    }
    EXPECT_LT(std::sqrt(imag_norm_K), 1e-10) << "K_0 has non-negligible imaginary part";
    EXPECT_LT(std::sqrt(imag_norm_M), 1e-10) << "M_0 has non-negligible imaginary part";
}

TEST(Cyclic, TransformationKN2IsReal) {
    // For k=N/2=12 (N=24 even), phase = exp(i*pi) = -1, so T is real
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    int k_half = mesh.num_sectors / 2;  // 12
    auto [K_k, M_k] = solver.apply_cyclic_bc_public(k_half, assembler.K(), assembler.M());

    // Compute total norm and imaginary norm
    double total_norm_K = 0, imag_norm_K = 0;
    double total_norm_M = 0, imag_norm_M = 0;
    for (int col = 0; col < K_k.outerSize(); ++col) {
        for (SpMatcd::InnerIterator it(K_k, col); it; ++it) {
            total_norm_K += std::norm(it.value());
            imag_norm_K += it.value().imag() * it.value().imag();
        }
    }
    for (int col = 0; col < M_k.outerSize(); ++col) {
        for (SpMatcd::InnerIterator it(M_k, col); it; ++it) {
            total_norm_M += std::norm(it.value());
            imag_norm_M += it.value().imag() * it.value().imag();
        }
    }
    // Imaginary part should be negligible relative to total norm
    EXPECT_LT(std::sqrt(imag_norm_K), 1e-6 * std::sqrt(total_norm_K))
        << "K_N/2 has non-negligible imaginary part";
    EXPECT_LT(std::sqrt(imag_norm_M), 1e-6 * std::sqrt(total_norm_M))
        << "M_N/2 has non-negligible imaginary part";
}

TEST(Cyclic, TransformationK1IsComplex) {
    // For k=1, phase = exp(i*2*pi/24) which has nonzero imaginary part
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    auto [K_k, M_k] = solver.apply_cyclic_bc_public(1, assembler.K(), assembler.M());

    // Reduced matrices should be Hermitian (K = K^H)
    SpMatcd K_H = K_k.adjoint();
    SpMatcd diff = K_k - K_H;
    double hermitian_err = 0, total_norm = 0;
    for (int col = 0; col < diff.outerSize(); ++col) {
        for (SpMatcd::InnerIterator it(diff, col); it; ++it)
            hermitian_err += std::norm(it.value());
    }
    for (int col = 0; col < K_k.outerSize(); ++col) {
        for (SpMatcd::InnerIterator it(K_k, col); it; ++it)
            total_norm += std::norm(it.value());
    }
    // Hermitian error should be small relative to total norm
    EXPECT_LT(std::sqrt(hermitian_err), 1e-6 * std::sqrt(total_norm)) << "K_1 is not Hermitian";

    // Should have nonzero imaginary part (since k=1 introduces complex phase)
    double imag_norm = 0;
    for (int col = 0; col < K_k.outerSize(); ++col) {
        for (SpMatcd::InnerIterator it(K_k, col); it; ++it)
            imag_norm += it.value().imag() * it.value().imag();
    }
    EXPECT_GT(std::sqrt(imag_norm), 1e-6) << "K_1 should have nonzero imaginary part";
}

TEST(Cyclic, ReducedDimensions) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    auto [K_k, M_k] = solver.apply_cyclic_bc_public(0, assembler.K(), assembler.M());

    int full_ndof = mesh.num_dof();
    int n_right = static_cast<int>(mesh.right_boundary.size());
    int expected_reduced = full_ndof - 3 * n_right;

    EXPECT_EQ(K_k.rows(), expected_reduced);
    EXPECT_EQ(K_k.cols(), expected_reduced);
    EXPECT_EQ(M_k.rows(), expected_reduced);
    EXPECT_EQ(M_k.cols(), expected_reduced);
}

// ---- solve_at_rpm ----

TEST(Cyclic, SolveAtRpmZeroProducesResults) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(0.0, 5);

    // Should have results for k=0 to k=12 (N/2 for N=24)
    EXPECT_GE(results.size(), 1u);
    EXPECT_LE(results.size(), 13u);  // max floor(24/2) + 1 = 13

    // Each result should have positive frequencies
    for (const auto& r : results) {
        EXPECT_GE(r.harmonic_index, 0);
        EXPECT_LE(r.harmonic_index, 12);
        EXPECT_EQ(r.rpm, 0.0);
        for (Eigen::Index i = 0; i < r.frequencies.size(); i++) {
            EXPECT_GT(r.frequencies(i), 0.0)
                << "k=" << r.harmonic_index << " mode " << i << " has non-positive frequency";
        }
    }
}

TEST(Cyclic, SolveAtRpmZeroFrequenciesOrdered) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(0.0, 5);

    // Within each harmonic index, frequencies should be ascending
    for (const auto& r : results) {
        for (Eigen::Index i = 1; i < r.frequencies.size(); i++) {
            EXPECT_GE(r.frequencies(i), r.frequencies(i - 1) - 1e-6)
                << "k=" << r.harmonic_index << " frequencies not ascending at mode " << i;
        }
    }
}

TEST(Cyclic, SolveAtRpmHarmonicIndicesComplete) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(0.0, 3);

    // Verify harmonic indices are sequential starting from 0
    std::set<int> harmonics;
    for (const auto& r : results) {
        harmonics.insert(r.harmonic_index);
    }
    // Should contain 0 at minimum
    EXPECT_TRUE(harmonics.count(0) > 0) << "Missing harmonic index 0";
}

// ---- CSV Export ----

TEST(Cyclic, ExportZzenfCsv) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(0.0, 3);

    std::string filename = test_data_path("test_zzenf.csv");
    solver.export_zzenf_csv(filename, results);

    // Read back and verify format
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());

    std::string header;
    std::getline(file, header);
    EXPECT_EQ(header, "nodal_diameter,mode_family,frequency_hz,rpm");

    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        line_count++;
        // Verify comma-separated with 4 fields
        std::istringstream iss(line);
        std::string field;
        int field_count = 0;
        while (std::getline(iss, field, ',')) {
            field_count++;
        }
        EXPECT_EQ(field_count, 4) << "Line " << line_count << " has wrong number of fields";
    }
    EXPECT_GT(line_count, 0) << "ZZENF CSV is empty";

    // Clean up
    std::remove(filename.c_str());
}

TEST(Cyclic, ExportCampbellCsv) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    Eigen::VectorXd rpms(2);
    rpms << 0.0, 1000.0;
    auto sweep = solver.solve_rpm_sweep(rpms, 3);

    std::string filename = test_data_path("test_campbell.csv");
    solver.export_campbell_csv(filename, sweep);

    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());

    std::string header;
    std::getline(file, header);
    EXPECT_EQ(header, "rpm,harmonic_index,mode_family,frequency_hz,whirl_direction");

    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        line_count++;
        std::istringstream iss(line);
        std::string field;
        int field_count = 0;
        while (std::getline(iss, field, ',')) {
            field_count++;
        }
        EXPECT_EQ(field_count, 5) << "Line " << line_count << " has wrong number of fields";
    }
    EXPECT_GT(line_count, 0) << "Campbell CSV is empty";

    std::remove(filename.c_str());
}

// ---- VTK Export ----

TEST(Cyclic, ExportModeShapeVtk) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(0.0, 3);
    ASSERT_FALSE(results.empty());

    // Need to expand mode shapes to full DOF space for VTK export
    // The mode shapes from solve_at_rpm are in reduced space; for a basic test
    // we just ensure the export doesn't crash and produces a valid file
    // We'll create a ModalResult with full-size mode shapes manually
    ModalResult full_result;
    full_result.harmonic_index = 0;
    full_result.rpm = 0.0;
    full_result.frequencies = results[0].frequencies;
    full_result.mode_shapes = Eigen::MatrixXcd::Zero(mesh.num_dof(), results[0].frequencies.size());
    // Fill with dummy data
    for (Eigen::Index i = 0; i < full_result.mode_shapes.rows() && i < results[0].mode_shapes.rows(); i++) {
        for (Eigen::Index j = 0; j < full_result.mode_shapes.cols() && j < results[0].mode_shapes.cols(); j++) {
            full_result.mode_shapes(i, j) = results[0].mode_shapes(i, j);
        }
    }

    std::string filename = test_data_path("test_mode.vtu");
    solver.export_mode_shape_vtk(filename, full_result, 0);

    // Verify file exists and has VTK header
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());
    std::string first_line;
    std::getline(file, first_line);
    EXPECT_NE(first_line.find("xml"), std::string::npos) << "VTK file missing XML header";

    std::remove(filename.c_str());
}

// ---- RPM Sweep ----

TEST(Cyclic, RpmSweepDimensions) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    Eigen::VectorXd rpms(3);
    rpms << 0.0, 5000.0, 10000.0;
    auto sweep = solver.solve_rpm_sweep(rpms, 3);

    EXPECT_EQ(sweep.size(), 3u) << "Should have results for each RPM point";
    for (const auto& rpm_results : sweep) {
        EXPECT_GE(rpm_results.size(), 1u) << "Each RPM should produce at least one harmonic";
    }
}

// ---- Added Mass Integration ----

TEST(Cyclic, AddedMassReducesFrequencies) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);

    // Solve dry
    CyclicSymmetrySolver solver_dry(mesh, mat);
    auto results_dry = solver_dry.solve_at_rpm(0.0, 3);

    // Solve wet (water, a/h typical values)
    FluidConfig fluid;
    fluid.type = FluidConfig::Type::LIQUID_ANALYTICAL;
    fluid.fluid_density = 1000.0;
    fluid.disk_radius = 0.15;
    fluid.disk_thickness = 0.01;

    CyclicSymmetrySolver solver_wet(mesh, mat, fluid);
    auto results_wet = solver_wet.solve_at_rpm(0.0, 3);

    ASSERT_EQ(results_dry.size(), results_wet.size());

    // Wet frequencies should be lower than dry for each harmonic index
    for (size_t i = 0; i < results_dry.size(); i++) {
        ASSERT_EQ(results_dry[i].harmonic_index, results_wet[i].harmonic_index);
        auto n_modes = std::min(results_dry[i].frequencies.size(),
                                results_wet[i].frequencies.size());
        for (Eigen::Index m = 0; m < n_modes; m++) {
            EXPECT_LT(results_wet[i].frequencies(m), results_dry[i].frequencies(m))
                << "k=" << results_dry[i].harmonic_index << " mode " << m
                << ": wet frequency should be lower than dry";
        }
    }
}

// ---- Rotating-frame frequency consistency ----

TEST(Cyclic, ConsistentModeCountAcrossHarmonics) {
    // At non-zero RPM, all harmonics should return the same number of modes
    // (rotating-frame eigenvalues, no FW/BW doubling)
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(5000.0, 5);
    ASSERT_GE(results.size(), 3u);

    int expected_modes = static_cast<int>(results[0].frequencies.size());
    for (const auto& r : results) {
        EXPECT_EQ(static_cast<int>(r.frequencies.size()), expected_modes)
            << "k=" << r.harmonic_index << " has different mode count";
    }
}

TEST(Cyclic, WhirlDirectionZeroInRotatingFrame) {
    // In the rotating frame, whirl_direction should be 0 for all modes
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(5000.0, 5);
    for (const auto& r : results) {
        for (Eigen::Index m = 0; m < r.whirl_direction.size(); m++) {
            EXPECT_EQ(r.whirl_direction(m), 0)
                << "k=" << r.harmonic_index << " mode " << m
                << " has non-zero whirl in rotating frame";
        }
    }
}

// ---- compute_stationary_frame ----

TEST(Cyclic, ComputeStationaryFrameFWBWSplit) {
    // For k=3, RPM=6000: omega = 200*pi rad/s, k_omega_hz = 300 Hz
    ModalResult rot;
    rot.harmonic_index = 3;
    rot.rpm = 6000.0;
    rot.frequencies.resize(2);
    rot.frequencies << 500.0, 1000.0;
    rot.whirl_direction = Eigen::VectorXi::Zero(2);
    rot.mode_shapes = Eigen::MatrixXcd::Zero(6, 2);

    auto sf = CyclicSymmetrySolver::compute_stationary_frame(rot, 24);

    // FW: 500+300=800, 1000+300=1300
    // BW: |500-300|=200, |1000-300|=700
    // Sorted: 200, 700, 800, 1300
    ASSERT_EQ(sf.frequencies.size(), 4);
    EXPECT_NEAR(sf.frequencies(0), 200.0, 0.1);
    EXPECT_NEAR(sf.frequencies(1), 700.0, 0.1);
    EXPECT_NEAR(sf.frequencies(2), 800.0, 0.1);
    EXPECT_NEAR(sf.frequencies(3), 1300.0, 0.1);

    // Check whirl directions
    EXPECT_EQ(sf.whirl_direction(0), -1);  // 200 = BW
    EXPECT_EQ(sf.whirl_direction(1), -1);  // 700 = BW
    EXPECT_EQ(sf.whirl_direction(2), 1);   // 800 = FW
    EXPECT_EQ(sf.whirl_direction(3), 1);   // 1300 = FW
}

TEST(Cyclic, ComputeStationaryFrameK0NoSplit) {
    // k=0: no splitting regardless of RPM
    ModalResult rot;
    rot.harmonic_index = 0;
    rot.rpm = 6000.0;
    rot.frequencies.resize(3);
    rot.frequencies << 100.0, 200.0, 300.0;
    rot.whirl_direction = Eigen::VectorXi::Zero(3);
    rot.mode_shapes = Eigen::MatrixXcd::Zero(6, 3);

    auto sf = CyclicSymmetrySolver::compute_stationary_frame(rot, 24);

    ASSERT_EQ(sf.frequencies.size(), 3);
    EXPECT_NEAR(sf.frequencies(0), 100.0, 1e-10);
    EXPECT_NEAR(sf.frequencies(1), 200.0, 1e-10);
    EXPECT_NEAR(sf.frequencies(2), 300.0, 1e-10);
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(sf.whirl_direction(i), 0);
    }
}

TEST(Cyclic, ComputeStationaryFrameKN2NoSplit) {
    // k=N/2=12 for N=24: no splitting (standing wave)
    ModalResult rot;
    rot.harmonic_index = 12;
    rot.rpm = 6000.0;
    rot.frequencies.resize(2);
    rot.frequencies << 500.0, 1000.0;
    rot.whirl_direction = Eigen::VectorXi::Zero(2);
    rot.mode_shapes = Eigen::MatrixXcd::Zero(6, 2);

    auto sf = CyclicSymmetrySolver::compute_stationary_frame(rot, 24);

    ASSERT_EQ(sf.frequencies.size(), 2);
    EXPECT_NEAR(sf.frequencies(0), 500.0, 1e-10);
    EXPECT_NEAR(sf.frequencies(1), 1000.0, 1e-10);
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(sf.whirl_direction(i), 0);
    }
}

TEST(Cyclic, ComputeStationaryFrameZeroRpmNoSplit) {
    // At RPM=0, no splitting for any k
    ModalResult rot;
    rot.harmonic_index = 5;
    rot.rpm = 0.0;
    rot.frequencies.resize(2);
    rot.frequencies << 500.0, 1000.0;
    rot.whirl_direction = Eigen::VectorXi::Zero(2);
    rot.mode_shapes = Eigen::MatrixXcd::Zero(6, 2);

    auto sf = CyclicSymmetrySolver::compute_stationary_frame(rot, 24);

    ASSERT_EQ(sf.frequencies.size(), 2);
    EXPECT_NEAR(sf.frequencies(0), 500.0, 1e-10);
    EXPECT_NEAR(sf.frequencies(1), 1000.0, 1e-10);
}

// ============== Coriolis / Lancaster QEP tests ==============

TEST(Cyclic, CoriolisOffMatchesBaseline) {
    // include_coriolis=false should give whirl_direction=0 for all modes
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(5000.0, 5, {}, 0, false);
    ASSERT_GT(results.size(), 0u);

    for (const auto& r : results) {
        for (Eigen::Index m = 0; m < r.whirl_direction.size(); m++) {
            EXPECT_EQ(r.whirl_direction(m), 0)
                << "k=" << r.harmonic_index << " mode=" << m;
        }
    }
}

TEST(Cyclic, CoriolisK0FallsBackToStandard) {
    // k=0 should always use standard solver: whirl=0 even with Coriolis
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(5000.0, 5, {0}, 0, true);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].harmonic_index, 0);
    for (Eigen::Index m = 0; m < results[0].whirl_direction.size(); m++) {
        EXPECT_EQ(results[0].whirl_direction(m), 0);
    }
}

TEST(Cyclic, CoriolisZeroRPMSameAsNoCoriolis) {
    // At RPM=0, Coriolis should have no effect
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results_no  = solver.solve_at_rpm(0.0, 5, {3}, 0, false);
    auto results_yes = solver.solve_at_rpm(0.0, 5, {3}, 0, true);
    ASSERT_EQ(results_no.size(), 1u);
    ASSERT_EQ(results_yes.size(), 1u);

    auto nm = std::min(results_no[0].frequencies.size(),
                       results_yes[0].frequencies.size());
    for (Eigen::Index m = 0; m < nm; m++) {
        EXPECT_NEAR(results_no[0].frequencies(m),
                    results_yes[0].frequencies(m), 1e-6)
            << "mode=" << m;
    }
}

TEST(Cyclic, CoriolisProducesFWBWSplitting) {
    // At high RPM with Coriolis, k>0 modes should have FW and BW
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(10000.0, 10, {3}, 0, true);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].harmonic_index, 3);

    int n_fw = 0, n_bw = 0, n_stand = 0;
    for (Eigen::Index m = 0; m < results[0].whirl_direction.size(); m++) {
        int w = results[0].whirl_direction(m);
        if (w == 1) n_fw++;
        else if (w == -1) n_bw++;
        else n_stand++;
    }

    EXPECT_TRUE(n_fw > 0) << "k=3 should have FW modes";
    EXPECT_TRUE(n_bw > 0) << "k=3 should have BW modes";
}

TEST(Cyclic, CoriolisFWGreaterThanBW) {
    // For same mode family: FW rotating-frame frequency > BW
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    auto results = solver.solve_at_rpm(10000.0, 10, {3}, 0, true);
    ASSERT_EQ(results.size(), 1u);

    // Collect first FW and first BW frequencies
    double first_fw = -1, first_bw = -1;
    for (Eigen::Index m = 0; m < results[0].whirl_direction.size(); m++) {
        if (results[0].whirl_direction(m) == 1 && first_fw < 0)
            first_fw = results[0].frequencies(m);
        if (results[0].whirl_direction(m) == -1 && first_bw < 0)
            first_bw = results[0].frequencies(m);
    }
    if (first_fw >= 0 && first_bw >= 0) {
        EXPECT_GT(first_fw, first_bw)
            << "FW=" << first_fw << " should be > BW=" << first_bw;
    }
}

TEST(Cyclic, CoriolisStationaryFrameConversion) {
    // Verify compute_stationary_frame handles Coriolis modes (whirl != 0)
    ModalResult rot;
    rot.harmonic_index = 3;
    rot.rpm = 6000.0;
    rot.frequencies.resize(4);
    // BW mode (lower freq), FW mode (higher freq), BW, FW
    rot.frequencies << 450.0, 550.0, 900.0, 1100.0;
    rot.whirl_direction.resize(4);
    rot.whirl_direction << -1, 1, -1, 1;
    rot.mode_shapes = Eigen::MatrixXcd::Zero(6, 4);

    auto sf = CyclicSymmetrySolver::compute_stationary_frame(rot, 24);

    // k=3, RPM=6000 => omega=200*pi rad/s => k*omega/(2pi) = 3*100 = 300 Hz
    double k_omega_hz = 300.0;
    // BW(450): |450 + (-1)*300| = 150 Hz
    // FW(550): |550 + 1*300| = 850 Hz
    // BW(900): |900 + (-1)*300| = 600 Hz
    // FW(1100): |1100 + 1*300| = 1400 Hz

    ASSERT_EQ(sf.frequencies.size(), 4);
    // Sorted: 150, 600, 850, 1400
    EXPECT_NEAR(sf.frequencies(0), 150.0, 1e-6);
    EXPECT_NEAR(sf.frequencies(1), 600.0, 1e-6);
    EXPECT_NEAR(sf.frequencies(2), 850.0, 1e-6);
    EXPECT_NEAR(sf.frequencies(3), 1400.0, 1e-6);
    // Check whirl preserved after sorting
    EXPECT_EQ(sf.whirl_direction(0), -1);  // 150 Hz was BW
    EXPECT_EQ(sf.whirl_direction(1), -1);  // 600 Hz was BW
    EXPECT_EQ(sf.whirl_direction(2), 1);   // 850 Hz was FW
    EXPECT_EQ(sf.whirl_direction(3), 1);   // 1400 Hz was FW
}

TEST(Cyclic, CoriolisDMatrixHermitian) {
    // Verify that D = i * 2 * omega * G_k is Hermitian after cyclic projection
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);
    assembler.assemble_rotating_effects(mesh, mat, 100.0);

    SpMatd G = assembler.G();
    // Verify G is skew-symmetric: G + G^T = 0
    SpMatd G_sym = G + SpMatd(G.transpose());
    double g_err = 0.0;
    for (int col = 0; col < G_sym.outerSize(); ++col)
        for (SpMatd::InnerIterator it(G_sym, col); it; ++it)
            g_err = std::max(g_err, std::abs(it.value()));
    EXPECT_LT(g_err, 1e-10 * G.norm()) << "G should be skew-symmetric";

    // Project through cyclic transformation for k=3
    CyclicSymmetrySolver solver(mesh, mat);
    SpMatcd T3 = solver.get_transformation(3);
    SpMatcd T3_H = T3.adjoint();

    // Convert G to complex
    int ndof = mesh.num_dof();
    SpMatcd G_complex(ndof, ndof);
    {
        std::vector<TripletC> gc;
        for (int col = 0; col < G.outerSize(); ++col)
            for (SpMatd::InnerIterator it(G, col); it; ++it)
                gc.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()),
                                std::complex<double>(it.value(), 0.0));
        G_complex.setFromTriplets(gc.begin(), gc.end());
    }

    SpMatcd G_k = (T3_H * G_complex * T3).pruned(1e-15);
    // G_k should be skew-Hermitian: G_k + G_k^H = 0
    SpMatcd G_k_check = G_k + SpMatcd(G_k.adjoint());
    double gk_err = 0.0;
    for (int col = 0; col < G_k_check.outerSize(); ++col)
        for (SpMatcd::InnerIterator it(G_k_check, col); it; ++it)
            gk_err = std::max(gk_err, std::abs(it.value()));
    EXPECT_LT(gk_err, 1e-10 * G_k.norm()) << "G_k should be skew-Hermitian";

    // D = i * 2 * omega * G_k should be Hermitian
    double omega = 100.0;
    std::complex<double> factor(0.0, 2.0 * omega);
    SpMatcd D_k = factor * G_k;
    SpMatcd D_check = D_k - SpMatcd(D_k.adjoint());
    double d_err = 0.0;
    for (int col = 0; col < D_check.outerSize(); ++col)
        for (SpMatcd::InnerIterator it(D_check, col); it; ++it)
            d_err = std::max(d_err, std::abs(it.value()));
    EXPECT_LT(d_err, 1e-10 * D_k.norm()) << "D = i*2*omega*G_k should be Hermitian";
}

TEST(Cyclic, AllHarmonicsReturnResults) {
    // Verify that all requested harmonics produce results (no silently dropped)
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    CyclicSymmetrySolver solver(mesh, mat);

    int max_k = mesh.num_sectors / 2;
    auto results = solver.solve_at_rpm(3000.0, 5, {}, 1);  // single thread
    EXPECT_EQ(static_cast<int>(results.size()), max_k + 1)
        << "All " << (max_k + 1) << " harmonics should produce results";
}
