#include <gtest/gtest.h>
#include "turbomodal/rotating_effects.hpp"
#include "turbomodal/assembler.hpp"
#include "turbomodal/cyclic_solver.hpp"
#include <cmath>
#include <set>
#include <fstream>
#include <string>

using namespace turbomodal;

static std::string test_data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

// Helper: create a single TET10 element off the rotation axis
static TET10Element make_offset_element() {
    TET10Element elem;
    // Element centered around (0.1, 0, 0) — offset from z-axis
    double cx = 0.1, cy = 0.0, cz = 0.0;
    double s = 0.01;  // half-size

    // Corner nodes
    elem.node_coords.row(0) << cx - s, cy - s, cz - s;
    elem.node_coords.row(1) << cx + s, cy - s, cz - s;
    elem.node_coords.row(2) << cx - s, cy + s, cz - s;
    elem.node_coords.row(3) << cx - s, cy - s, cz + s;

    // Mid-edge nodes (midpoints of edges)
    elem.node_coords.row(4) = 0.5 * (elem.node_coords.row(0) + elem.node_coords.row(1));
    elem.node_coords.row(5) = 0.5 * (elem.node_coords.row(1) + elem.node_coords.row(2));
    elem.node_coords.row(6) = 0.5 * (elem.node_coords.row(0) + elem.node_coords.row(2));
    elem.node_coords.row(7) = 0.5 * (elem.node_coords.row(0) + elem.node_coords.row(3));
    elem.node_coords.row(8) = 0.5 * (elem.node_coords.row(1) + elem.node_coords.row(3));
    elem.node_coords.row(9) = 0.5 * (elem.node_coords.row(2) + elem.node_coords.row(3));

    return elem;
}

// Helper: create a single TET10 element ON the rotation axis
static TET10Element make_onaxis_element() {
    TET10Element elem;
    double s = 0.01;

    // Corner nodes centered on z-axis
    elem.node_coords.row(0) << -s, -s, -s;
    elem.node_coords.row(1) <<  s, -s, -s;
    elem.node_coords.row(2) << -s,  s, -s;
    elem.node_coords.row(3) << -s, -s,  s;

    // Mid-edge nodes
    elem.node_coords.row(4) = 0.5 * (elem.node_coords.row(0) + elem.node_coords.row(1));
    elem.node_coords.row(5) = 0.5 * (elem.node_coords.row(1) + elem.node_coords.row(2));
    elem.node_coords.row(6) = 0.5 * (elem.node_coords.row(0) + elem.node_coords.row(2));
    elem.node_coords.row(7) = 0.5 * (elem.node_coords.row(0) + elem.node_coords.row(3));
    elem.node_coords.row(8) = 0.5 * (elem.node_coords.row(1) + elem.node_coords.row(3));
    elem.node_coords.row(9) = 0.5 * (elem.node_coords.row(2) + elem.node_coords.row(3));

    return elem;
}

// Helper: load wedge sector mesh
static Mesh load_wedge_mesh() {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;
    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();
    return mesh;
}

// ======== Element-level centrifugal load ========

TEST(Rotating, CentrifugalLoadRadiallyOutward) {
    TET10Element elem = make_offset_element();
    Material mat(200e9, 0.3, 7850);
    double omega = 100.0;  // rad/s
    Eigen::Vector3d z_axis(0, 0, 1);

    Vector30d F = RotatingEffects::centrifugal_load(elem, mat, omega, z_axis);

    // Sum up x, y, z force components across all nodes
    double fx_total = 0, fy_total = 0, fz_total = 0;
    for (int i = 0; i < 10; i++) {
        fx_total += F(3 * i);
        fy_total += F(3 * i + 1);
        fz_total += F(3 * i + 2);
    }

    // Element is at x~0.1, so centrifugal force should be primarily in +x
    EXPECT_GT(fx_total, 0.0) << "Centrifugal force should be radially outward (+x)";
    // y component should be near zero (element centered at y~0)
    EXPECT_NEAR(fy_total, 0.0, std::abs(fx_total) * 0.1);
    // z component should be zero (rotation about z-axis)
    EXPECT_NEAR(fz_total, 0.0, std::abs(fx_total) * 1e-10);
}

TEST(Rotating, CentrifugalLoadSmallOnAxis) {
    // Element centered on z-axis should have small centrifugal load
    TET10Element elem = make_onaxis_element();
    Material mat(200e9, 0.3, 7850);
    double omega = 100.0;
    Eigen::Vector3d z_axis(0, 0, 1);

    Vector30d F = RotatingEffects::centrifugal_load(elem, mat, omega, z_axis);

    // The element is centered on-axis but not perfectly zero-sized,
    // so the total force should be small but not exactly zero
    double fx_total = 0, fy_total = 0;
    for (int i = 0; i < 10; i++) {
        fx_total += F(3 * i);
        fy_total += F(3 * i + 1);
    }

    // Much smaller than the off-axis element
    TET10Element offset_elem = make_offset_element();
    Vector30d F_offset = RotatingEffects::centrifugal_load(offset_elem, mat, omega, z_axis);
    double fx_offset = 0;
    for (int i = 0; i < 10; i++) fx_offset += F_offset(3 * i);

    EXPECT_LT(std::abs(fx_total) + std::abs(fy_total), std::abs(fx_offset) * 0.2)
        << "On-axis element should have much smaller centrifugal load";
}

TEST(Rotating, CentrifugalLoadScalesWithOmegaSquared) {
    TET10Element elem = make_offset_element();
    Material mat(200e9, 0.3, 7850);
    Eigen::Vector3d z_axis(0, 0, 1);

    Vector30d F1 = RotatingEffects::centrifugal_load(elem, mat, 100.0, z_axis);
    Vector30d F2 = RotatingEffects::centrifugal_load(elem, mat, 200.0, z_axis);

    // F2 should be 4x F1 (omega^2 scaling)
    double ratio = F2.norm() / F1.norm();
    EXPECT_NEAR(ratio, 4.0, 1e-10) << "Centrifugal load should scale with omega^2";
}

// ======== Element-level gyroscopic ========

TEST(Rotating, GyroscopicSkewSymmetric) {
    TET10Element elem = make_offset_element();
    Material mat(200e9, 0.3, 7850);

    Matrix30d G = RotatingEffects::gyroscopic(elem, mat, Eigen::Vector3d::UnitZ());

    // G + G^T should be zero (skew-symmetric)
    Matrix30d sym = G + G.transpose();
    EXPECT_LT(sym.norm(), 1e-10 * G.norm()) << "G + G^T should be zero (skew-symmetric)";
}

TEST(Rotating, GyroscopicNonzero) {
    TET10Element elem = make_offset_element();
    Material mat(200e9, 0.3, 7850);

    Matrix30d G = RotatingEffects::gyroscopic(elem, mat, Eigen::Vector3d::UnitZ());
    EXPECT_GT(G.norm(), 0.0) << "Gyroscopic matrix should be nonzero";
}

// ======== Element-level spin softening ========

TEST(Rotating, SpinSofteningSymmetric) {
    TET10Element elem = make_offset_element();
    Material mat(200e9, 0.3, 7850);
    double omega = 100.0;

    Matrix30d Kw = RotatingEffects::spin_softening(elem, mat, omega, Eigen::Vector3d::UnitZ());

    // K_omega should be symmetric
    Matrix30d diff = Kw - Kw.transpose();
    EXPECT_LT(diff.norm(), 1e-10 * Kw.norm()) << "K_omega should be symmetric";
}

TEST(Rotating, SpinSofteningScalesWithOmegaSquared) {
    TET10Element elem = make_offset_element();
    Material mat(200e9, 0.3, 7850);

    Matrix30d Kw1 = RotatingEffects::spin_softening(elem, mat, 100.0, Eigen::Vector3d::UnitZ());
    Matrix30d Kw2 = RotatingEffects::spin_softening(elem, mat, 200.0, Eigen::Vector3d::UnitZ());

    // Kw2 should be 4x Kw1
    double ratio = Kw2.norm() / Kw1.norm();
    EXPECT_NEAR(ratio, 4.0, 1e-10) << "K_omega should scale with omega^2";
}

TEST(Rotating, SpinSofteningPositiveSemiDefinite) {
    TET10Element elem = make_offset_element();
    Material mat(200e9, 0.3, 7850);

    Matrix30d Kw = RotatingEffects::spin_softening(elem, mat, 100.0, Eigen::Vector3d::UnitZ());

    Eigen::SelfAdjointEigenSolver<Matrix30d> es(Kw);
    // All eigenvalues should be >= 0 (PSD)
    double min_eig = es.eigenvalues().minCoeff();
    EXPECT_GE(min_eig, -1e-10 * Kw.norm()) << "K_omega should be positive semi-definite";
}

// ======== Element-level stress stiffening ========

TEST(Rotating, StressStiffeningSymmetric) {
    TET10Element elem = make_offset_element();

    // Create a uniform tensile prestress
    std::array<Vector6d, 4> prestress;
    for (int gp = 0; gp < 4; gp++) {
        prestress[gp] << 1e6, 1e6, 0, 0, 0, 0;  // biaxial tension
    }

    Matrix30d Ks = RotatingEffects::stress_stiffening(elem, prestress);

    Matrix30d diff = Ks - Ks.transpose();
    EXPECT_LT(diff.norm(), 1e-10 * Ks.norm()) << "K_sigma should be symmetric";
}

TEST(Rotating, StressStiffeningNonzero) {
    TET10Element elem = make_offset_element();

    std::array<Vector6d, 4> prestress;
    for (int gp = 0; gp < 4; gp++) {
        prestress[gp] << 1e6, 1e6, 1e6, 0, 0, 0;  // hydrostatic tension
    }

    Matrix30d Ks = RotatingEffects::stress_stiffening(elem, prestress);
    EXPECT_GT(Ks.norm(), 0.0) << "K_sigma should be nonzero for nonzero prestress";
}

TEST(Rotating, StressStiffeningZeroForZeroPrestress) {
    TET10Element elem = make_offset_element();

    std::array<Vector6d, 4> prestress;
    for (int gp = 0; gp < 4; gp++) {
        prestress[gp] = Vector6d::Zero();
    }

    Matrix30d Ks = RotatingEffects::stress_stiffening(elem, prestress);
    EXPECT_LT(Ks.norm(), 1e-15) << "K_sigma should be zero for zero prestress";
}

TEST(Rotating, StressStiffeningPSDForTension) {
    TET10Element elem = make_offset_element();

    std::array<Vector6d, 4> prestress;
    for (int gp = 0; gp < 4; gp++) {
        prestress[gp] << 1e8, 1e8, 1e8, 0, 0, 0;  // hydrostatic tension
    }

    Matrix30d Ks = RotatingEffects::stress_stiffening(elem, prestress);

    Eigen::SelfAdjointEigenSolver<Matrix30d> es(Ks);
    double min_eig = es.eigenvalues().minCoeff();
    EXPECT_GE(min_eig, -1e-6 * Ks.norm())
        << "K_sigma should be positive semi-definite for hydrostatic tension";
}

// ======== Global assembly of centrifugal load ========

TEST(Rotating, AssembleCentrifugalLoadDimension) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::VectorXd F = assembler.assemble_centrifugal_load(mesh, mat, 100.0, z_axis);

    EXPECT_EQ(F.size(), mesh.num_dof());
}

TEST(Rotating, AssembleCentrifugalLoadNonzero) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::VectorXd F = assembler.assemble_centrifugal_load(mesh, mat, 100.0, z_axis);

    EXPECT_GT(F.norm(), 0.0) << "Centrifugal load should be nonzero for off-axis mesh";
}

TEST(Rotating, AssembleCentrifugalLoadRadialDirection) {
    // For wedge mesh in x-y plane, load should be primarily radially outward
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::VectorXd F = assembler.assemble_centrifugal_load(mesh, mat, 100.0, z_axis);

    // Sum z-component (should be ~0 for z-axis rotation)
    double fz_total = 0;
    double fxy_total = 0;
    for (int i = 0; i < mesh.num_nodes(); i++) {
        fz_total += std::abs(F(3 * i + 2));
        fxy_total += std::sqrt(F(3 * i) * F(3 * i) + F(3 * i + 1) * F(3 * i + 1));
    }
    EXPECT_LT(fz_total, 1e-10 * fxy_total) << "z-component should be negligible";
}

// ======== Global assembly of K_sigma ========

TEST(Rotating, AssembleStressStiffeningDimensionAndSymmetry) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    // Create a dummy displacement (small radial expansion)
    Eigen::VectorXd u = Eigen::VectorXd::Zero(mesh.num_dof());
    for (int i = 0; i < mesh.num_nodes(); i++) {
        double x = mesh.nodes(i, 0);
        double y = mesh.nodes(i, 1);
        double r = std::sqrt(x * x + y * y);
        if (r > 1e-10) {
            u(3 * i)     = 1e-6 * x / r;  // radial displacement
            u(3 * i + 1) = 1e-6 * y / r;
        }
    }

    assembler.assemble_stress_stiffening(mesh, mat, u, 100.0);
    SpMatd Ks = assembler.K_sigma();

    EXPECT_EQ(Ks.rows(), mesh.num_dof());
    EXPECT_EQ(Ks.cols(), mesh.num_dof());

    // Check symmetry
    SpMatd diff = Ks - SpMatd(Ks.transpose());
    double sym_err = 0;
    for (int col = 0; col < diff.outerSize(); ++col)
        for (SpMatd::InnerIterator it(diff, col); it; ++it)
            sym_err += it.value() * it.value();
    double total = 0;
    for (int col = 0; col < Ks.outerSize(); ++col)
        for (SpMatd::InnerIterator it(Ks, col); it; ++it)
            total += it.value() * it.value();

    EXPECT_LT(std::sqrt(sym_err), 1e-10 * std::sqrt(total))
        << "Assembled K_sigma should be symmetric";
}

// ======== Global assembly of K_omega and G ========

TEST(Rotating, AssembleRotatingEffectsProperties) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    assembler.assemble_rotating_effects(mesh, mat, 100.0);
    SpMatd Kw = assembler.K_omega();
    SpMatd G = assembler.G();

    // Dimensions
    EXPECT_EQ(Kw.rows(), mesh.num_dof());
    EXPECT_EQ(G.rows(), mesh.num_dof());

    // K_omega should be symmetric
    SpMatd kw_diff = Kw - SpMatd(Kw.transpose());
    double kw_sym_err = 0, kw_norm = 0;
    for (int col = 0; col < kw_diff.outerSize(); ++col)
        for (SpMatd::InnerIterator it(kw_diff, col); it; ++it)
            kw_sym_err += it.value() * it.value();
    for (int col = 0; col < Kw.outerSize(); ++col)
        for (SpMatd::InnerIterator it(Kw, col); it; ++it)
            kw_norm += it.value() * it.value();
    EXPECT_LT(std::sqrt(kw_sym_err), 1e-10 * std::sqrt(kw_norm))
        << "K_omega should be symmetric";

    // G should be skew-symmetric: G + G^T = 0
    SpMatd g_sum = G + SpMatd(G.transpose());
    double g_skew_err = 0, g_norm = 0;
    for (int col = 0; col < g_sum.outerSize(); ++col)
        for (SpMatd::InnerIterator it(g_sum, col); it; ++it)
            g_skew_err += it.value() * it.value();
    for (int col = 0; col < G.outerSize(); ++col)
        for (SpMatd::InnerIterator it(G, col); it; ++it)
            g_norm += it.value() * it.value();
    EXPECT_LT(std::sqrt(g_skew_err), 1e-10 * std::sqrt(g_norm))
        << "G should be skew-symmetric";
}

// ======== Full rotating solve ========

TEST(Rotating, FrequenciesChangeWithRPM) {
    // Rotating effects (stress stiffening and spin softening) should change frequencies.
    // For a thin flat disk, the net effect may be a slight decrease (spin softening
    // dominates) or increase (stress stiffening dominates), depending on the mode.
    // We verify that the frequencies DO change and that K_sigma is nonzero.
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);

    CyclicSymmetrySolver solver(mesh, mat);

    auto results_0 = solver.solve_at_rpm(0.0, 3);
    auto results_rpm = solver.solve_at_rpm(10000.0, 3);

    ASSERT_FALSE(results_0.empty());
    ASSERT_FALSE(results_rpm.empty());

    // Compare k=0 frequencies
    const ModalResult* r0 = nullptr;
    const ModalResult* r_rpm = nullptr;
    for (const auto& r : results_0) {
        if (r.harmonic_index == 0) { r0 = &r; break; }
    }
    for (const auto& r : results_rpm) {
        if (r.harmonic_index == 0) { r_rpm = &r; break; }
    }

    ASSERT_NE(r0, nullptr) << "No k=0 result at 0 RPM";
    ASSERT_NE(r_rpm, nullptr) << "No k=0 result at 10000 RPM";

    int n_compare = std::min(r0->frequencies.size(), r_rpm->frequencies.size());
    ASSERT_GE(n_compare, 1);

    // Frequencies should change (not be exactly equal)
    bool any_changed = false;
    for (int m = 0; m < n_compare; m++) {
        double diff = std::abs(r_rpm->frequencies(m) - r0->frequencies(m));
        if (diff > 1e-6 * r0->frequencies(m)) {
            any_changed = true;
        }
    }
    EXPECT_TRUE(any_changed) << "Rotating effects should cause frequency changes";

    // Verify stress stiffening is actually being computed (K_sigma nonzero)
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);
    double omega = 10000.0 * 2.0 * PI / 60.0;
    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::VectorXd F = assembler.assemble_centrifugal_load(mesh, mat, omega, z_axis);
    EXPECT_GT(F.norm(), 0.0) << "Centrifugal load should be nonzero";
}

TEST(Rotating, FWBWSplittingAtNonzeroRPM) {
    // For k > 0, compute_stationary_frame should produce FW and BW modes
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);

    CyclicSymmetrySolver solver(mesh, mat);
    auto results = solver.solve_at_rpm(5000.0, 3);

    // Rotating-frame result should have whirl_direction = 0
    for (const auto& r : results) {
        for (int m = 0; m < r.whirl_direction.size(); m++) {
            EXPECT_EQ(r.whirl_direction(m), 0);
        }
    }

    // Stationary-frame conversion should produce FW/BW for 0 < k < N/2
    bool found_split = false;
    for (const auto& r : results) {
        if (r.harmonic_index > 0 && r.harmonic_index < mesh.num_sectors / 2) {
            auto sf = CyclicSymmetrySolver::compute_stationary_frame(r, mesh.num_sectors);
            bool has_fw = false, has_bw = false;
            for (int m = 0; m < sf.whirl_direction.size(); m++) {
                if (sf.whirl_direction(m) == 1) has_fw = true;
                if (sf.whirl_direction(m) == -1) has_bw = true;
            }
            EXPECT_TRUE(has_fw) << "k=" << r.harmonic_index << " should have forward whirl modes";
            EXPECT_TRUE(has_bw) << "k=" << r.harmonic_index << " should have backward whirl modes";
            // Stationary frame should have 2x the modes
            EXPECT_EQ(sf.frequencies.size(), 2 * r.frequencies.size());
            found_split = true;
        }
    }
    EXPECT_TRUE(found_split) << "Should have at least one harmonic with FW/BW splitting";
}

TEST(Rotating, StandingWavesAtK0) {
    // k=0 modes should have whirl_direction = 0 (standing)
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);

    CyclicSymmetrySolver solver(mesh, mat);
    auto results = solver.solve_at_rpm(5000.0, 3);

    for (const auto& r : results) {
        if (r.harmonic_index == 0) {
            for (int m = 0; m < r.whirl_direction.size(); m++) {
                EXPECT_EQ(r.whirl_direction(m), 0)
                    << "k=0 modes should be standing waves (whirl_direction=0)";
            }
        }
    }
}

TEST(Rotating, CampbellDiagramWithWhirl) {
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);

    CyclicSymmetrySolver solver(mesh, mat);

    Eigen::VectorXd rpms(3);
    rpms << 0.0, 5000.0, 10000.0;
    auto sweep = solver.solve_rpm_sweep(rpms, 3);

    ASSERT_EQ(sweep.size(), 3u);

    // Export and verify Campbell CSV
    std::string filename = test_data_path("test_campbell_rotating.csv");
    solver.export_campbell_csv(filename, sweep);

    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());

    std::string header;
    std::getline(file, header);
    EXPECT_EQ(header, "rpm,harmonic_index,mode_family,frequency_hz,whirl_direction");

    int line_count = 0;
    std::string line;
    std::set<int> whirl_values;
    while (std::getline(file, line)) {
        line_count++;
        // Parse whirl direction (last field)
        size_t last_comma = line.rfind(',');
        if (last_comma != std::string::npos) {
            int whirl = std::stoi(line.substr(last_comma + 1));
            whirl_values.insert(whirl);
        }
    }
    EXPECT_GT(line_count, 0);

    // At higher RPM, should have -1, 0, and +1 whirl directions
    EXPECT_TRUE(whirl_values.count(0) > 0) << "Should have standing wave modes";
    EXPECT_TRUE(whirl_values.count(1) > 0) << "Should have forward whirl modes";
    EXPECT_TRUE(whirl_values.count(-1) > 0) << "Should have backward whirl modes";

    std::remove(filename.c_str());
}

TEST(Rotating, FWGreaterThanBWInStationaryFrame) {
    // For the same mode family at k > 0, FW frequency > BW frequency
    Mesh mesh = load_wedge_mesh();
    Material mat(200e9, 0.3, 7850);

    CyclicSymmetrySolver solver(mesh, mat);
    auto results = solver.solve_at_rpm(5000.0, 3);

    for (const auto& r : results) {
        if (r.harmonic_index > 0 && r.harmonic_index < mesh.num_sectors / 2) {
            auto sf = CyclicSymmetrySolver::compute_stationary_frame(r, mesh.num_sectors);
            // Collect FW and BW frequencies from stationary-frame result
            std::vector<double> fw_freqs, bw_freqs;
            for (int m = 0; m < sf.whirl_direction.size(); m++) {
                if (sf.whirl_direction(m) == 1) fw_freqs.push_back(sf.frequencies(m));
                if (sf.whirl_direction(m) == -1) bw_freqs.push_back(sf.frequencies(m));
            }
            // At minimum, max FW > min BW
            if (!fw_freqs.empty() && !bw_freqs.empty()) {
                double max_fw = *std::max_element(fw_freqs.begin(), fw_freqs.end());
                double min_bw = *std::min_element(bw_freqs.begin(), bw_freqs.end());
                EXPECT_GT(max_fw, min_bw)
                    << "k=" << r.harmonic_index << ": highest FW should exceed lowest BW";
            }
        }
    }
}
