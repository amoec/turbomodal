#include <gtest/gtest.h>
#include "turbomodal/assembler.hpp"
#include "turbomodal/element.hpp"
#include "turbomodal/mesh.hpp"
#include "turbomodal/material.hpp"
#include <Eigen/Eigenvalues>

using namespace turbomodal;

static std::string test_data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

// Helper: create a simple 1-element mesh programmatically
static Mesh make_single_element_mesh() {
    Mesh mesh;
    mesh.nodes.resize(10, 3);
    // Reference tetrahedron with mid-edge nodes
    mesh.nodes.row(0) = Eigen::Vector3d(0, 0, 0);
    mesh.nodes.row(1) = Eigen::Vector3d(1, 0, 0);
    mesh.nodes.row(2) = Eigen::Vector3d(0, 1, 0);
    mesh.nodes.row(3) = Eigen::Vector3d(0, 0, 1);
    mesh.nodes.row(4) = Eigen::Vector3d(0.5, 0, 0);     // mid 0-1
    mesh.nodes.row(5) = Eigen::Vector3d(0.5, 0.5, 0);   // mid 1-2
    mesh.nodes.row(6) = Eigen::Vector3d(0, 0.5, 0);     // mid 0-2
    mesh.nodes.row(7) = Eigen::Vector3d(0, 0, 0.5);     // mid 0-3
    mesh.nodes.row(8) = Eigen::Vector3d(0.5, 0, 0.5);   // mid 1-3
    mesh.nodes.row(9) = Eigen::Vector3d(0, 0.5, 0.5);   // mid 2-3

    mesh.elements.resize(1, 10);
    mesh.elements.row(0) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;

    return mesh;
}

// Helper: create a 2-element mesh (two tets sharing a face)
static Mesh make_two_element_mesh() {
    Mesh mesh;
    // Two tetrahedra sharing face (0,1,2) with mid-edge nodes
    // Tet 1: corners 0,1,2,3 (apex above)
    // Tet 2: corners 0,2,1,4 (apex below, note reversed face winding for positive volume)
    mesh.nodes.resize(14, 3);
    // Corners
    mesh.nodes.row(0) = Eigen::Vector3d(0, 0, 0);
    mesh.nodes.row(1) = Eigen::Vector3d(1, 0, 0);
    mesh.nodes.row(2) = Eigen::Vector3d(0, 1, 0);
    mesh.nodes.row(3) = Eigen::Vector3d(0.3, 0.3, 1);
    mesh.nodes.row(4) = Eigen::Vector3d(0.3, 0.3, -1);

    // Mid-edge nodes for tet 1 (0,1,2,3)
    mesh.nodes.row(5)  = Eigen::Vector3d(0.5, 0, 0);        // mid 0-1
    mesh.nodes.row(6)  = Eigen::Vector3d(0.5, 0.5, 0);      // mid 1-2
    mesh.nodes.row(7)  = Eigen::Vector3d(0, 0.5, 0);        // mid 0-2
    mesh.nodes.row(8)  = Eigen::Vector3d(0.15, 0.15, 0.5);  // mid 0-3
    mesh.nodes.row(9)  = Eigen::Vector3d(0.65, 0.15, 0.5);  // mid 1-3
    mesh.nodes.row(10) = Eigen::Vector3d(0.15, 0.65, 0.5);  // mid 2-3

    // Mid-edge nodes for tet 2 (0,2,1,4) — note winding reversal
    // Reuses shared face mid-edge nodes but with swapped edge order:
    // mid 0-2 = node 7, mid 2-1 = node 6, mid 0-1 = node 5
    mesh.nodes.row(11) = Eigen::Vector3d(0.15, 0.15, -0.5);  // mid 0-4
    mesh.nodes.row(12) = Eigen::Vector3d(0.65, 0.15, -0.5);  // mid 1-4
    mesh.nodes.row(13) = Eigen::Vector3d(0.15, 0.65, -0.5);  // mid 2-4

    mesh.elements.resize(2, 10);
    // Tet 1: (0,1,2,3) with mid-edge (0-1, 1-2, 0-2, 0-3, 1-3, 2-3)
    mesh.elements.row(0) << 0, 1, 2, 3, 5, 6, 7, 8, 9, 10;
    // Tet 2: (0,2,1,4) with mid-edge (0-2, 2-1, 0-1, 0-4, 2-4, 1-4)
    mesh.elements.row(1) << 0, 2, 1, 4, 7, 6, 5, 11, 13, 12;

    return mesh;
}

// ---- Construction ----

TEST(Assembler, Construction) {
    GlobalAssembler assembler;
    EXPECT_EQ(assembler.K().rows(), 0);
    EXPECT_EQ(assembler.M().rows(), 0);
}

// ---- Single Element Assembly ----

TEST(Assembler, SingleElementDimensions) {
    Mesh mesh = make_single_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    EXPECT_EQ(assembler.K().rows(), 30);
    EXPECT_EQ(assembler.K().cols(), 30);
    EXPECT_EQ(assembler.M().rows(), 30);
    EXPECT_EQ(assembler.M().cols(), 30);
}

TEST(Assembler, SingleElementMatchesDirectComputation) {
    Mesh mesh = make_single_element_mesh();
    Material mat(200e9, 0.3, 7850);

    // Direct element computation
    TET10Element elem;
    for (int n = 0; n < 10; n++) {
        elem.node_coords.row(n) = mesh.nodes.row(n);
    }
    Matrix30d Ke_direct = elem.stiffness(mat);
    Matrix30d Me_direct = elem.mass(mat);

    // Assembly
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    // Convert sparse to dense for comparison
    Eigen::MatrixXd K_dense = Eigen::MatrixXd(assembler.K());
    Eigen::MatrixXd M_dense = Eigen::MatrixXd(assembler.M());

    EXPECT_TRUE(K_dense.isApprox(Ke_direct, 1e-10))
        << "Assembled K differs from direct element K";
    EXPECT_TRUE(M_dense.isApprox(Me_direct, 1e-10))
        << "Assembled M differs from direct element M";
}

// ---- Multi-Element Assembly ----

TEST(Assembler, TwoElementDimensions) {
    Mesh mesh = make_two_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    int ndof = mesh.num_dof();
    EXPECT_EQ(assembler.K().rows(), ndof);
    EXPECT_EQ(assembler.K().cols(), ndof);
    EXPECT_EQ(assembler.M().rows(), ndof);
    EXPECT_EQ(assembler.M().cols(), ndof);
}

TEST(Assembler, KSymmetric) {
    Mesh mesh = make_two_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::MatrixXd K_dense = Eigen::MatrixXd(assembler.K());
    EXPECT_TRUE(K_dense.isApprox(K_dense.transpose(), 1e-10))
        << "Global K is not symmetric";
}

TEST(Assembler, MSymmetric) {
    Mesh mesh = make_two_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::MatrixXd M_dense = Eigen::MatrixXd(assembler.M());
    EXPECT_TRUE(M_dense.isApprox(M_dense.transpose(), 1e-10))
        << "Global M is not symmetric";
}

TEST(Assembler, SharedNodesHaveOverlap) {
    // When two elements share nodes, the assembled matrix at shared DOFs
    // should differ from the sum of individual element matrices at those DOFs
    Mesh mesh = make_two_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::MatrixXd K_dense = Eigen::MatrixXd(assembler.K());

    // The shared face nodes are 0,1,2 and mid-edge nodes 5,6,7
    // K(0,0) in global should be sum of contributions from both elements
    // Compute element 0 contribution to DOF(0,0)
    TET10Element elem0;
    for (int n = 0; n < 10; n++) {
        elem0.node_coords.row(n) = mesh.nodes.row(mesh.elements(0, n));
    }
    Matrix30d Ke0 = elem0.stiffness(mat);

    TET10Element elem1;
    for (int n = 0; n < 10; n++) {
        elem1.node_coords.row(n) = mesh.nodes.row(mesh.elements(1, n));
    }
    Matrix30d Ke1 = elem1.stiffness(mat);

    // Node 0 is the first node in both elements, so DOF 0 maps to global DOF 0
    // K_global(0,0) should equal Ke0(0,0) + Ke1(0,0)
    EXPECT_NEAR(K_dense(0, 0), Ke0(0, 0) + Ke1(0, 0), 1e-6)
        << "Shared DOF assembly incorrect";
}

// ---- Rigid Body Modes (Single Element) ----

TEST(Assembler, SingleElementRigidBodyModes) {
    Mesh mesh = make_single_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::MatrixXd K_dense = Eigen::MatrixXd(assembler.K());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K_dense);
    Eigen::VectorXd evals = es.eigenvalues();

    // Count near-zero eigenvalues (rigid body modes)
    int num_zero = 0;
    double threshold = 1e-6 * evals.tail(1)(0);  // Relative to largest
    for (int i = 0; i < evals.size(); i++) {
        if (std::abs(evals(i)) < threshold) num_zero++;
    }
    EXPECT_EQ(num_zero, 6) << "Expected 6 rigid body modes for a single free element";
}

// ---- Total Mass Check ----

TEST(Assembler, SingleElementTotalMass) {
    Mesh mesh = make_single_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    // Total mass via rigid body translation: u^T M u = rho * V
    // For unit x-translation: u = [1,0,0, 1,0,0, ...]
    Eigen::VectorXd u = Eigen::VectorXd::Zero(30);
    for (int i = 0; i < 10; i++) u(3 * i) = 1.0;

    Eigen::VectorXd Mu = assembler.M() * u;
    double total_mass = u.dot(Mu);

    // Volume of reference tet = 1/6
    double V = 1.0 / 6.0;
    double expected_mass = mat.rho * V;

    EXPECT_NEAR(total_mass, expected_mass, 1e-4 * expected_mass)
        << "Total mass " << total_mass << " vs expected " << expected_mass;
}

TEST(Assembler, TwoElementTotalMass) {
    Mesh mesh = make_two_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    int ndof = mesh.num_dof();
    // Unit x-translation
    Eigen::VectorXd u = Eigen::VectorXd::Zero(ndof);
    for (int i = 0; i < mesh.num_nodes(); i++) u(3 * i) = 1.0;

    double total_mass = u.dot(assembler.M() * u);

    // Compute expected total volume from both elements
    double V_total = 0.0;
    for (int e = 0; e < mesh.num_elements(); e++) {
        TET10Element elem;
        for (int n = 0; n < 10; n++) {
            elem.node_coords.row(n) = mesh.nodes.row(mesh.elements(e, n));
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

    double expected_mass = mat.rho * V_total;
    EXPECT_NEAR(total_mass, expected_mass, 1e-4 * expected_mass)
        << "Two-element total mass " << total_mass << " vs expected " << expected_mass;
}

// ---- Wedge Sector (real mesh from file) ----

TEST(Assembler, WedgeSectorDimensions) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    Material mat(200e9, 0.3, 7850);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    int ndof = mesh.num_dof();
    EXPECT_EQ(assembler.K().rows(), ndof);
    EXPECT_EQ(assembler.K().cols(), ndof);
    EXPECT_EQ(assembler.M().rows(), ndof);
    EXPECT_EQ(assembler.M().cols(), ndof);
}

TEST(Assembler, WedgeSectorKSymmetric) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    Material mat(200e9, 0.3, 7850);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::MatrixXd K_dense = Eigen::MatrixXd(assembler.K());
    double asym = (K_dense - K_dense.transpose()).norm();
    EXPECT_LT(asym, 1e-6 * K_dense.norm()) << "Wedge K asymmetry: " << asym;
}

TEST(Assembler, WedgeSectorMSymmetric) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    Material mat(200e9, 0.3, 7850);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::MatrixXd M_dense = Eigen::MatrixXd(assembler.M());
    double asym = (M_dense - M_dense.transpose()).norm();
    EXPECT_LT(asym, 1e-6 * M_dense.norm()) << "Wedge M asymmetry: " << asym;
}

TEST(Assembler, WedgeSectorTotalMass) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    Material mat(200e9, 0.3, 7850);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    int ndof = mesh.num_dof();
    Eigen::VectorXd u = Eigen::VectorXd::Zero(ndof);
    for (int i = 0; i < mesh.num_nodes(); i++) u(3 * i) = 1.0;

    double total_mass = u.dot(assembler.M() * u);

    // Analytical sector volume
    double R_outer = 0.15, R_inner = 0.05;
    double alpha = PI / 12.0, thickness = 0.01;
    double V_analytical = 0.5 * (R_outer * R_outer - R_inner * R_inner) * alpha * thickness;
    double expected_mass = mat.rho * V_analytical;

    // Allow 5% for geometric approximation (straight-edged tets on curved arc)
    EXPECT_NEAR(total_mass, expected_mass, 0.05 * expected_mass)
        << "Wedge sector total mass " << total_mass << " vs expected " << expected_mass;
}

TEST(Assembler, WedgeSectorMPositiveSemiDefinite) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    Material mat(200e9, 0.3, 7850);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::MatrixXd M_dense = Eigen::MatrixXd(assembler.M());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M_dense);
    double min_eval = es.eigenvalues().minCoeff();
    double max_eval = es.eigenvalues().maxCoeff();

    // Allow small numerical noise
    EXPECT_GT(min_eval, -1e-10 * max_eval)
        << "M has negative eigenvalue: " << min_eval;
}

TEST(Assembler, WedgeSectorKNonNegativeEigenvalues) {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
    Material mat(200e9, 0.3, 7850);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    Eigen::MatrixXd K_dense = Eigen::MatrixXd(assembler.K());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K_dense);
    double min_eval = es.eigenvalues().minCoeff();
    double max_eval = es.eigenvalues().maxCoeff();

    // K should be positive semi-definite (6 rigid body zero eigenvalues)
    EXPECT_GT(min_eval, -1e-6 * max_eval)
        << "K has unexpected negative eigenvalue: " << min_eval;
}

// ---- Scaling Tests ----

TEST(Assembler, KScalesWithE) {
    Mesh mesh = make_single_element_mesh();
    Material mat1(200e9, 0.3, 7850);
    Material mat2(400e9, 0.3, 7850);  // 2x Young's modulus

    GlobalAssembler asm1, asm2;
    asm1.assemble(mesh, mat1);
    asm2.assemble(mesh, mat2);

    Eigen::MatrixXd K1 = Eigen::MatrixXd(asm1.K());
    Eigen::MatrixXd K2 = Eigen::MatrixXd(asm2.K());

    EXPECT_TRUE(K2.isApprox(2.0 * K1, 1e-10))
        << "K should scale linearly with E";
}

TEST(Assembler, MScalesWithRho) {
    Mesh mesh = make_single_element_mesh();
    Material mat1(200e9, 0.3, 7850);
    Material mat2(200e9, 0.3, 15700);  // 2x density

    GlobalAssembler asm1, asm2;
    asm1.assemble(mesh, mat1);
    asm2.assemble(mesh, mat2);

    Eigen::MatrixXd M1 = Eigen::MatrixXd(asm1.M());
    Eigen::MatrixXd M2 = Eigen::MatrixXd(asm2.M());

    EXPECT_TRUE(M2.isApprox(2.0 * M1, 1e-10))
        << "M should scale linearly with rho";
}

TEST(Assembler, MIndependentOfE) {
    Mesh mesh = make_single_element_mesh();
    Material mat1(200e9, 0.3, 7850);
    Material mat2(400e9, 0.3, 7850);

    GlobalAssembler asm1, asm2;
    asm1.assemble(mesh, mat1);
    asm2.assemble(mesh, mat2);

    Eigen::MatrixXd M1 = Eigen::MatrixXd(asm1.M());
    Eigen::MatrixXd M2 = Eigen::MatrixXd(asm2.M());

    EXPECT_TRUE(M1.isApprox(M2, 1e-15))
        << "M should be independent of E";
}

// ---- Nonzero Structure ----

TEST(Assembler, SparseNonzeroCount) {
    Mesh mesh = make_single_element_mesh();
    Material mat(200e9, 0.3, 7850);
    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);

    // Single element: K and M should have same sparsity pattern
    EXPECT_GT(assembler.K().nonZeros(), 0);
    EXPECT_GT(assembler.M().nonZeros(), 0);
}
