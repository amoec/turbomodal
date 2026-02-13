#include <gtest/gtest.h>
#include "turbomodal/mesh.hpp"
#include "turbomodal/element.hpp"
#include "turbomodal/material.hpp"
#include <fstream>
#include <cmath>
#include <set>

using namespace turbomodal;

// Helper to get the path to the test data directory
static std::string test_data_path(const std::string& filename) {
    // CMake builds in turbomodal/build, tests run from there
    // test_data is at turbomodal/tests/test_data/
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

// ---- Construction ----

TEST(Mesh, DefaultConstruction) {
    Mesh mesh;
    EXPECT_EQ(mesh.num_nodes(), 0);
    EXPECT_EQ(mesh.num_elements(), 0);
    EXPECT_EQ(mesh.num_dof(), 0);
    EXPECT_EQ(mesh.num_sectors, 0);
    EXPECT_TRUE(mesh.node_sets.empty());
    EXPECT_TRUE(mesh.left_boundary.empty());
    EXPECT_TRUE(mesh.right_boundary.empty());
    EXPECT_TRUE(mesh.matched_pairs.empty());
}

// ---- Gmsh Parser ----

TEST(Mesh, LoadWedgeSector) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    EXPECT_EQ(mesh.num_nodes(), 43);
    EXPECT_EQ(mesh.num_elements(), 10);  // Only TET10 elements stored
    EXPECT_EQ(mesh.num_dof(), 43 * 3);
}

TEST(Mesh, LoadWedgeSectorNodeSets) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    // Should have 4 physical groups: volume, left_boundary, right_boundary, hub_constraint
    EXPECT_GE(mesh.node_sets.size(), 4u);

    const NodeSet* vol = mesh.find_node_set("volume");
    const NodeSet* left = mesh.find_node_set("left_boundary");
    const NodeSet* right = mesh.find_node_set("right_boundary");
    const NodeSet* hub = mesh.find_node_set("hub_constraint");

    ASSERT_NE(vol, nullptr);
    ASSERT_NE(left, nullptr);
    ASSERT_NE(right, nullptr);
    ASSERT_NE(hub, nullptr);

    // Volume should contain all nodes referenced by TET10 elements
    EXPECT_GT(vol->node_ids.size(), 0u);

    // Left and right boundaries should have the same number of nodes
    EXPECT_EQ(left->node_ids.size(), right->node_ids.size());
    EXPECT_GT(left->node_ids.size(), 0u);

    // Hub should have some nodes
    EXPECT_GT(hub->node_ids.size(), 0u);
}

TEST(Mesh, NodeCoordinatesValid) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    // All z-coordinates should be between 0 and 0.01 (disk thickness)
    for (int i = 0; i < mesh.num_nodes(); i++) {
        EXPECT_GE(mesh.nodes(i, 2), -1e-10) << "Node " << i << " has z < 0";
        EXPECT_LE(mesh.nodes(i, 2), 0.01 + 1e-10) << "Node " << i << " has z > 0.01";
    }

    // All radii should be between inner (0.05) and outer (0.15)
    // Midside nodes on curved boundaries may be slightly inside the arc
    // (chord midpoint vs arc midpoint), so allow ~1% tolerance
    for (int i = 0; i < mesh.num_nodes(); i++) {
        double r = std::sqrt(mesh.nodes(i, 0) * mesh.nodes(i, 0) +
                             mesh.nodes(i, 1) * mesh.nodes(i, 1));
        EXPECT_GE(r, 0.05 * 0.99) << "Node " << i << " has r < inner radius";
        EXPECT_LE(r, 0.15 * 1.01) << "Node " << i << " has r > outer radius";
    }

    // All angular positions should be between 0 and 15 degrees
    for (int i = 0; i < mesh.num_nodes(); i++) {
        double theta = std::atan2(mesh.nodes(i, 1), mesh.nodes(i, 0));
        EXPECT_GE(theta, -1e-6) << "Node " << i << " has negative angle";
        EXPECT_LE(theta, PI / 12.0 + 1e-6) << "Node " << i << " exceeds 15 degrees";
    }
}

TEST(Mesh, ElementConnectivityValid) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    for (int e = 0; e < mesh.num_elements(); e++) {
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            EXPECT_GE(node_id, 0) << "Element " << e << " has negative node ID at position " << n;
            EXPECT_LT(node_id, mesh.num_nodes())
                << "Element " << e << " references out-of-range node " << node_id;
        }
    }
}

TEST(Mesh, LeftBoundaryNodesAtThetaZero) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    const NodeSet* left = mesh.find_node_set("left_boundary");
    ASSERT_NE(left, nullptr);

    // All left boundary nodes should have y ≈ 0 (theta = 0)
    for (int id : left->node_ids) {
        EXPECT_NEAR(mesh.nodes(id, 1), 0.0, 1e-10)
            << "Left boundary node " << id << " has y = " << mesh.nodes(id, 1);
    }
}

TEST(Mesh, RightBoundaryNodesAtThetaAlpha) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    const NodeSet* right = mesh.find_node_set("right_boundary");
    ASSERT_NE(right, nullptr);

    double alpha = PI / 12.0;  // 15 degrees

    // All right boundary nodes should be at theta ≈ 15 degrees
    for (int id : right->node_ids) {
        double theta = std::atan2(mesh.nodes(id, 1), mesh.nodes(id, 0));
        EXPECT_NEAR(theta, alpha, 1e-6)
            << "Right boundary node " << id << " has theta = " << theta << " (expected " << alpha << ")";
    }
}

// ---- Cyclic Boundary Identification ----

TEST(Mesh, IdentifyCyclicBoundaries) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    mesh.identify_cyclic_boundaries();

    EXPECT_FALSE(mesh.left_boundary.empty());
    EXPECT_FALSE(mesh.right_boundary.empty());
    EXPECT_EQ(mesh.left_boundary.size(), mesh.right_boundary.size());
}

TEST(Mesh, IdentifyCyclicBoundariesThrowsWithoutGroups) {
    Mesh mesh;
    // Empty mesh has no physical groups
    EXPECT_THROW(mesh.identify_cyclic_boundaries(), std::runtime_error);
}

// ---- Node Matching ----

TEST(Mesh, MatchBoundaryNodes) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;

    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();

    EXPECT_EQ(mesh.matched_pairs.size(), mesh.left_boundary.size());

    // Verify each matched pair: rotating the right node by -alpha should give the left node position
    double alpha = 2.0 * PI / 24.0;
    double cos_a = std::cos(-alpha);
    double sin_a = std::sin(-alpha);

    for (const auto& [left_id, right_id] : mesh.matched_pairs) {
        double lx = mesh.nodes(left_id, 0);
        double ly = mesh.nodes(left_id, 1);
        double lz = mesh.nodes(left_id, 2);

        double rx = mesh.nodes(right_id, 0);
        double ry = mesh.nodes(right_id, 1);
        double rz = mesh.nodes(right_id, 2);

        double rx_rot = cos_a * rx - sin_a * ry;
        double ry_rot = sin_a * rx + cos_a * ry;

        EXPECT_NEAR(lx, rx_rot, 1e-8)
            << "Left node " << left_id << " x doesn't match rotated right node " << right_id;
        EXPECT_NEAR(ly, ry_rot, 1e-8)
            << "Left node " << left_id << " y doesn't match rotated right node " << right_id;
        EXPECT_NEAR(lz, rz, 1e-8)
            << "Left node " << left_id << " z doesn't match right node " << right_id;
    }
}

TEST(Mesh, MatchBoundaryNodesSameRadii) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;

    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();

    // Each matched pair should have the same radial distance
    for (const auto& [left_id, right_id] : mesh.matched_pairs) {
        double r_left = std::sqrt(mesh.nodes(left_id, 0) * mesh.nodes(left_id, 0) +
                                  mesh.nodes(left_id, 1) * mesh.nodes(left_id, 1));
        double r_right = std::sqrt(mesh.nodes(right_id, 0) * mesh.nodes(right_id, 0) +
                                   mesh.nodes(right_id, 1) * mesh.nodes(right_id, 1));
        EXPECT_NEAR(r_left, r_right, 1e-8)
            << "Pair (" << left_id << ", " << right_id << ") radii mismatch";
    }
}

TEST(Mesh, MatchBoundaryNodesSameZ) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;

    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();

    for (const auto& [left_id, right_id] : mesh.matched_pairs) {
        EXPECT_NEAR(mesh.nodes(left_id, 2), mesh.nodes(right_id, 2), 1e-10)
            << "Pair (" << left_id << ", " << right_id << ") z mismatch";
    }
}

TEST(Mesh, MatchBoundaryNodesUniquePairs) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;

    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();

    // All left IDs should be unique
    std::set<int> left_ids, right_ids;
    for (const auto& [l, r] : mesh.matched_pairs) {
        EXPECT_TRUE(left_ids.insert(l).second) << "Duplicate left node " << l;
        EXPECT_TRUE(right_ids.insert(r).second) << "Duplicate right node " << r;
    }
}

TEST(Mesh, MatchBoundaryNodesThrowsWithoutSectors) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.identify_cyclic_boundaries();
    // num_sectors = 0 by default
    EXPECT_THROW(mesh.match_boundary_nodes(), std::runtime_error);
}

TEST(Mesh, MatchBoundaryNodesThrowsWithoutIdentify) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;
    // Don't call identify_cyclic_boundaries
    EXPECT_THROW(mesh.match_boundary_nodes(), std::runtime_error);
}

// ---- Element Volume ----

TEST(Mesh, ElementVolumesPositive) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    Material mat(200e9, 0.3, 7850);

    for (int e = 0; e < mesh.num_elements(); e++) {
        TET10Element elem;
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
        }

        // Check Jacobian determinant at centroid is positive
        Eigen::Matrix3d J = elem.jacobian(0.25, 0.25, 0.25);
        double detJ = J.determinant();
        EXPECT_GT(detJ, 0.0) << "Element " << e << " has negative Jacobian determinant";
    }
}

TEST(Mesh, TotalVolumeApproxCorrect) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    // Analytical volume of an annular sector:
    // V = (1/2) * (R_outer^2 - R_inner^2) * alpha * thickness
    // R_outer = 0.15, R_inner = 0.05, alpha = pi/12, thickness = 0.01
    double R_outer = 0.15;
    double R_inner = 0.05;
    double alpha = PI / 12.0;
    double thickness = 0.01;
    double V_analytical = 0.5 * (R_outer * R_outer - R_inner * R_inner) * alpha * thickness;

    // Compute total volume from elements using Gauss quadrature
    double V_total = 0.0;
    for (int e = 0; e < mesh.num_elements(); e++) {
        TET10Element elem;
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
        }

        for (int gp = 0; gp < 4; gp++) {
            double xi   = TET10Element::gauss_points[gp](0);
            double eta  = TET10Element::gauss_points[gp](1);
            double zeta = TET10Element::gauss_points[gp](2);
            double w    = TET10Element::gauss_weights[gp];

            Eigen::Matrix3d J = elem.jacobian(xi, eta, zeta);
            V_total += J.determinant() * w;
        }
    }

    // Straight-edged TET10 will have some geometric error on curved boundaries
    // Allow 5% tolerance for the arc approximation
    EXPECT_NEAR(V_total, V_analytical, 0.05 * V_analytical)
        << "Total volume " << V_total << " vs analytical " << V_analytical;
}

// ---- find_node_set ----

TEST(Mesh, FindNodeSetReturnsNullForMissing) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    EXPECT_EQ(mesh.find_node_set("nonexistent"), nullptr);
}

TEST(Mesh, FindNodeSetReturnsCorrectSet) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));

    const NodeSet* vol = mesh.find_node_set("volume");
    ASSERT_NE(vol, nullptr);
    EXPECT_EQ(vol->name, "volume");
}

// ---- Error Handling ----

TEST(Mesh, LoadThrowsForMissingFile) {
    Mesh mesh;
    EXPECT_THROW(mesh.load_from_gmsh("nonexistent_file.msh"), std::runtime_error);
}

TEST(Mesh, LoadThrowsForInvalidFormat) {
    // Create a temporary file with invalid format version
    std::string tmpfile = test_data_path("invalid_format.msh");
    {
        std::ofstream f(tmpfile);
        f << "$MeshFormat\n";
        f << "4.1 0 8\n";
        f << "$EndMeshFormat\n";
    }
    Mesh mesh;
    EXPECT_THROW(mesh.load_from_gmsh(tmpfile), std::runtime_error);
    std::remove(tmpfile.c_str());
}

TEST(Mesh, LoadThrowsForBinaryFormat) {
    std::string tmpfile = test_data_path("binary_format.msh");
    {
        std::ofstream f(tmpfile);
        f << "$MeshFormat\n";
        f << "2.2 1 8\n";  // file_type=1 means binary
        f << "$EndMeshFormat\n";
    }
    Mesh mesh;
    EXPECT_THROW(mesh.load_from_gmsh(tmpfile), std::runtime_error);
    std::remove(tmpfile.c_str());
}

// ---- Integration: Full Workflow ----

TEST(Mesh, FullWorkflowLoadIdentifyMatch) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    mesh.num_sectors = 24;

    mesh.identify_cyclic_boundaries();
    EXPECT_FALSE(mesh.left_boundary.empty());
    EXPECT_EQ(mesh.left_boundary.size(), mesh.right_boundary.size());

    mesh.match_boundary_nodes();
    EXPECT_EQ(mesh.matched_pairs.size(), mesh.left_boundary.size());

    // Verify right_boundary is reordered to match left_boundary
    for (size_t i = 0; i < mesh.matched_pairs.size(); i++) {
        EXPECT_EQ(mesh.matched_pairs[i].first, mesh.left_boundary[i]);
        EXPECT_EQ(mesh.matched_pairs[i].second, mesh.right_boundary[i]);
    }
}
