#include <gtest/gtest.h>
#include "turbomodal/mode_identification.hpp"
#include "turbomodal/cyclic_solver.hpp"
#include <cmath>
#include <regex>

using namespace turbomodal;

static std::string test_data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

// ---- Struct defaults ----

TEST(ModeIdentification, StructDefaults) {
    ModeIdentification id;
    EXPECT_EQ(id.nodal_diameter, 0);
    EXPECT_EQ(id.nodal_circle, 0);
    EXPECT_EQ(id.whirl_direction, 0);
    EXPECT_DOUBLE_EQ(id.frequency, 0.0);
    EXPECT_DOUBLE_EQ(id.wave_velocity, 0.0);
    EXPECT_DOUBLE_EQ(id.participation_factor, 0.0);
    EXPECT_TRUE(id.family_label.empty());
}

TEST(ModeIdentification, GroundTruthLabel_Defaults) {
    GroundTruthLabel gt;
    EXPECT_DOUBLE_EQ(gt.rpm, 0.0);
    EXPECT_DOUBLE_EQ(gt.temperature, 293.15);
    EXPECT_DOUBLE_EQ(gt.pressure_ratio, 1.0);
    EXPECT_DOUBLE_EQ(gt.inlet_distortion, 0.0);
    EXPECT_DOUBLE_EQ(gt.tip_clearance, 0.0);
    EXPECT_EQ(gt.nodal_diameter, 0);
    EXPECT_EQ(gt.nodal_circle, 0);
    EXPECT_EQ(gt.whirl_direction, 0);
    EXPECT_DOUBLE_EQ(gt.frequency, 0.0);
    EXPECT_DOUBLE_EQ(gt.amplitude_magnification, 1.0);
    EXPECT_FALSE(gt.is_localized);
}

// Helper: build a minimal mesh with proper boundary node sets
static Mesh make_test_mesh_mi() {
    Mesh mesh;
    double angle = 2.0 * PI / 24.0;
    double r = 0.1;
    Eigen::MatrixXd coords(4, 3);
    coords << r, 0.0, 0.0,
              r, 0.0, 0.01,
              r * std::cos(angle), r * std::sin(angle), 0.0,
              r * std::cos(angle), r * std::sin(angle), 0.01;
    Eigen::MatrixXi conn(1, 10);
    conn << 0, 1, 2, 3, 0, 1, 2, 3, 0, 1;
    NodeSet left_ns, right_ns;
    left_ns.name = "left_boundary";
    left_ns.node_ids = {0, 1};
    right_ns.name = "right_boundary";
    right_ns.node_ids = {2, 3};
    std::vector<NodeSet> ns = {left_ns, right_ns};
    mesh.load_from_arrays(coords, conn, ns, 24);
    return mesh;
}

// ---- classify_mode_family with synthetic mode shapes ----

TEST(ModeIdentification, ClassifyModeFamily_AxialDominated) {
    Mesh mesh = make_test_mesh_mi();
    int n_nodes = mesh.num_nodes();

    // Mode shape: only z-displacement (axial/bending)
    Eigen::VectorXcd mode = Eigen::VectorXcd::Zero(3 * n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        mode(3 * i + 2) = 1.0;
    }

    std::string family = classify_mode_family(mode, mesh);
    EXPECT_EQ(family, "B");
}

TEST(ModeIdentification, ClassifyModeFamily_ZeroModeShape) {
    Mesh mesh = make_test_mesh_mi();
    int n_nodes = mesh.num_nodes();

    Eigen::VectorXcd mode = Eigen::VectorXcd::Zero(3 * n_nodes);
    std::string family = classify_mode_family(mode, mesh);
    EXPECT_EQ(family, "B");
}

TEST(ModeIdentification, ClassifyModeFamily_TangentialDominated) {
    Mesh mesh = make_test_mesh_mi();
    int n_nodes = mesh.num_nodes();

    // Mode shape: tangential displacement
    Eigen::VectorXcd mode = Eigen::VectorXcd::Zero(3 * n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        double x = mesh.nodes(i, 0);
        double y = mesh.nodes(i, 1);
        double r_val = std::sqrt(x * x + y * y);
        if (r_val > 1e-10) {
            mode(3 * i)     = -y / r_val;
            mode(3 * i + 1) =  x / r_val;
        }
    }

    std::string family = classify_mode_family(mode, mesh);
    EXPECT_EQ(family, "T");
}

// ---- identify_nodal_circles ----

TEST(ModeIdentification, IdentifyNodalCircles_EmptyMesh) {
    Mesh mesh;  // empty mesh (num_nodes = 0 after default construct)
    Eigen::VectorXcd mode(3);
    mode << 1.0, 0.0, 0.0;
    // Should return 0 for empty mesh
    int nc = identify_nodal_circles(mode, mesh);
    EXPECT_EQ(nc, 0);
}

TEST(ModeIdentification, IdentifyNodalCircles_ShortModeShape) {
    Mesh mesh = make_test_mesh_mi();

    Eigen::VectorXcd mode(2);  // too short
    mode << 1.0, 0.0;
    int nc = identify_nodal_circles(mode, mesh);
    EXPECT_GE(nc, 0);
}

// ---- Integration tests requiring wedge mesh ----

class ModeIdIntegration : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        mesh_ = std::make_unique<Mesh>();
        mesh_->load_from_gmsh(test_data_path("wedge_sector.msh"));
        mesh_->num_sectors = 24;
        mesh_->identify_cyclic_boundaries();
        mesh_->match_boundary_nodes();

        Material mat(200e9, 0.3, 7800);
        CyclicSymmetrySolver solver(*mesh_, mat);
        results_ = solver.solve_at_rpm(0.0, 5);
    }

    static std::unique_ptr<Mesh> mesh_;
    static std::vector<ModalResult> results_;
};

std::unique_ptr<Mesh> ModeIdIntegration::mesh_;
std::vector<ModalResult> ModeIdIntegration::results_;

TEST_F(ModeIdIntegration, IdentifyModes_ResultSize) {
    ASSERT_FALSE(results_.empty());
    for (const auto& r : results_) {
        auto ids = identify_modes(r, *mesh_);
        EXPECT_EQ(static_cast<int>(ids.size()), static_cast<int>(r.frequencies.size()));
    }
}

TEST_F(ModeIdIntegration, IdentifyModes_NDFromHarmonicIndex) {
    for (const auto& r : results_) {
        auto ids = identify_modes(r, *mesh_);
        for (const auto& id : ids) {
            EXPECT_EQ(id.nodal_diameter, r.harmonic_index);
        }
    }
}

TEST_F(ModeIdIntegration, IdentifyModes_FamilyLabelFormat) {
    std::regex pattern("[0-9]+[BTA]");
    for (const auto& r : results_) {
        auto ids = identify_modes(r, *mesh_);
        for (const auto& id : ids) {
            EXPECT_TRUE(std::regex_match(id.family_label, pattern))
                << "Label '" << id.family_label << "' doesn't match [0-9]+[BTA]";
        }
    }
}

TEST_F(ModeIdIntegration, IdentifyModes_FrequencyPositive) {
    for (const auto& r : results_) {
        auto ids = identify_modes(r, *mesh_);
        for (const auto& id : ids) {
            EXPECT_GT(id.frequency, 0.0);
        }
    }
}

TEST_F(ModeIdIntegration, IdentifyModes_WaveVelocity) {
    for (const auto& r : results_) {
        auto ids = identify_modes(r, *mesh_);
        for (const auto& id : ids) {
            if (id.nodal_diameter == 0) {
                EXPECT_DOUBLE_EQ(id.wave_velocity, 0.0);
            } else {
                EXPECT_GT(id.wave_velocity, 0.0);
            }
        }
    }
}

TEST_F(ModeIdIntegration, IdentifyModes_WhirlFromResult) {
    for (const auto& r : results_) {
        auto ids = identify_modes(r, *mesh_);
        for (int m = 0; m < static_cast<int>(ids.size()); m++) {
            if (m < r.whirl_direction.size()) {
                EXPECT_EQ(ids[m].whirl_direction, r.whirl_direction(m));
            }
        }
    }
}
