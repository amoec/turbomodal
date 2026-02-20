#include "turbomodal/assembler.hpp"
#include "turbomodal/element.hpp"
#include "turbomodal/hermitian_lanczos.hpp"
#include "turbomodal/material.hpp"
#include "turbomodal/mesh.hpp"
#include "turbomodal/modal_solver.hpp"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <gtest/gtest.h>

using namespace turbomodal;

static std::string test_data_path(const std::string &filename) {
  return std::string(TEST_DATA_DIR) + "/" + filename;
}

// Helper: build a small sparse matrix from dense
static SpMatd dense_to_sparse(const Eigen::MatrixXd &D) {
  int n = static_cast<int>(D.rows());
  std::vector<Triplet> trips;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (std::abs(D(i, j)) > 1e-15) {
        trips.emplace_back(i, j, D(i, j));
      }
    }
  }
  SpMatd S(n, n);
  S.setFromTriplets(trips.begin(), trips.end());
  return S;
}

// ---- Known 2x2 System ----

TEST(Solver, Known4x4System) {
  // 4x4 tridiagonal: K = [2 -1 0 0; -1 2 -1 0; 0 -1 2 -1; 0 0 -1 2], M = I
  // Eigenvalues: 2 - 2*cos(k*pi/5), k=1,2,3,4 (but we'll request 3)
  Eigen::MatrixXd K_dense(4, 4);
  K_dense << 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2;
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(4, 4);

  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  SolverConfig config;
  config.nev = 3; // n-1 = 3 maximum for Spectra

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K, M, config);

  EXPECT_TRUE(status.converged) << status.message;
  ASSERT_EQ(result.frequencies.size(), 3);

  // Analytical eigenvalues for tridiagonal n=4: lambda_k = 2 - 2*cos(k*pi/5)
  for (int i = 0; i < 3; i++) {
    double lambda = 2.0 - 2.0 * std::cos((i + 1) * PI / 5.0);
    double f_expected = std::sqrt(lambda) / (2.0 * PI);
    EXPECT_NEAR(result.frequencies(i), f_expected, 1e-6)
        << "Frequency " << i << ": " << result.frequencies(i) << " vs expected "
        << f_expected;
  }
}

TEST(Solver, ModeShapesOrthogonalSmallSystem) {
  // 4x4 system - request 3 modes, check orthogonality
  Eigen::MatrixXd K_dense(4, 4);
  K_dense << 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2;
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(4, 4);

  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  SolverConfig config;
  config.nev = 3;

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K, M, config);

  ASSERT_TRUE(status.converged);
  ASSERT_EQ(result.mode_shapes.cols(), 3);

  // Check pairwise orthogonality
  for (int i = 0; i < 3; i++) {
    for (int j = i + 1; j < 3; j++) {
      double dot =
          std::abs(result.mode_shapes.col(i).dot(result.mode_shapes.col(j)));
      EXPECT_NEAR(dot, 0.0, 1e-6)
          << "Modes " << i << " and " << j << " not orthogonal";
    }
  }
}

// ---- Known 3x3 System ----

TEST(Solver, Known5x5System) {
  // 5x5 tridiagonal: eigenvalues = 2 - 2*cos(k*pi/6), k=1..5
  int n = 5;
  Eigen::MatrixXd K_dense = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = 2.0;
    if (i > 0)
      K_dense(i, i - 1) = -1.0;
    if (i < n - 1)
      K_dense(i, i + 1) = -1.0;
  }
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(n, n);

  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  SolverConfig config;
  config.nev = 4; // max n-1 = 4

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K, M, config);

  ASSERT_TRUE(status.converged) << status.message;
  ASSERT_EQ(result.frequencies.size(), 4);

  for (int i = 0; i < 4; i++) {
    double lambda = 2.0 - 2.0 * std::cos((i + 1) * PI / 6.0);
    double f_expected = std::sqrt(lambda) / (2.0 * PI);
    EXPECT_NEAR(result.frequencies(i), f_expected, 1e-6) << "Frequency " << i;
  }
}

// ---- Frequency Ordering ----

TEST(Solver, FrequenciesAscending) {
  // 8x8 tridiagonal system
  int n = 8;
  Eigen::MatrixXd K_dense = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = 2.0;
    if (i > 0)
      K_dense(i, i - 1) = -1.0;
    if (i < n - 1)
      K_dense(i, i + 1) = -1.0;
  }
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(n, n);

  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  SolverConfig config;
  config.nev = 6; // request fewer than n-1=7

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K, M, config);

  ASSERT_TRUE(status.converged);
  for (int i = 1; i < result.frequencies.size(); i++) {
    EXPECT_GE(result.frequencies(i), result.frequencies(i - 1) - 1e-10)
        << "Frequencies not in ascending order at index " << i;
  }
}

// ---- DOF Elimination ----

TEST(Solver, EliminateDofsDimensions) {
  int n = 6;
  Eigen::MatrixXd K_dense = Eigen::MatrixXd::Identity(n, n) * 2.0;
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(n, n);
  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  std::vector<int> constrained = {0, 3};
  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(K, M, constrained);

  EXPECT_EQ(K_red.rows(), 4);
  EXPECT_EQ(K_red.cols(), 4);
  EXPECT_EQ(M_red.rows(), 4);
  EXPECT_EQ(M_red.cols(), 4);
  EXPECT_EQ(free_map.size(), 4u);

  // Free DOFs should be {1, 2, 4, 5}
  EXPECT_EQ(free_map[0], 1);
  EXPECT_EQ(free_map[1], 2);
  EXPECT_EQ(free_map[2], 4);
  EXPECT_EQ(free_map[3], 5);
}

TEST(Solver, EliminateDofsPreservesValues) {
  Eigen::MatrixXd K_dense(4, 4);
  K_dense << 10, 1, 2, 3, 1, 20, 4, 5, 2, 4, 30, 6, 3, 5, 6, 40;
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(4, 4);
  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  // Eliminate DOF 0 and 2
  std::vector<int> constrained = {0, 2};
  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(K, M, constrained);

  // Reduced K should be the submatrix for DOFs {1, 3}
  Eigen::MatrixXd K_red_dense(K_red);
  EXPECT_NEAR(K_red_dense(0, 0), 20.0, 1e-10); // K(1,1)
  EXPECT_NEAR(K_red_dense(0, 1), 5.0, 1e-10);  // K(1,3)
  EXPECT_NEAR(K_red_dense(1, 0), 5.0, 1e-10);  // K(3,1)
  EXPECT_NEAR(K_red_dense(1, 1), 40.0, 1e-10); // K(3,3)
}

TEST(Solver, ExpandModeShapes) {
  int full_ndof = 6;
  std::vector<int> free_map = {1, 2, 4, 5};

  Eigen::MatrixXcd reduced(4, 2);
  reduced << 1.0, 0.5, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5;

  Eigen::MatrixXcd full =
      ModalSolver::expand_mode_shapes(reduced, full_ndof, free_map);

  EXPECT_EQ(full.rows(), 6);
  EXPECT_EQ(full.cols(), 2);

  // Constrained DOFs (0, 3) should be zero
  EXPECT_NEAR(std::abs(full(0, 0)), 0.0, 1e-15);
  EXPECT_NEAR(std::abs(full(3, 0)), 0.0, 1e-15);

  // Free DOFs should match reduced
  EXPECT_NEAR(std::abs(full(1, 0) - 1.0), 0.0, 1e-15);
  EXPECT_NEAR(std::abs(full(2, 0) - 2.0), 0.0, 1e-15);
  EXPECT_NEAR(std::abs(full(4, 0) - 3.0), 0.0, 1e-15);
  EXPECT_NEAR(std::abs(full(5, 0) - 4.0), 0.0, 1e-15);
}

// ---- Solve with DOF Elimination ----

TEST(Solver, SolveWithConstrainedDofs) {
  // 6-DOF spring chain, fix DOF 0 -> 5 remaining DOFs, request 4
  int n = 6;
  Eigen::MatrixXd K_dense = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = 2.0;
    if (i > 0)
      K_dense(i, i - 1) = -1.0;
    if (i < n - 1)
      K_dense(i, i + 1) = -1.0;
  }
  K_dense(n - 1, n - 1) = 1.0; // last spring has stiffness 1
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(n, n);

  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  // Fix DOF 0
  std::vector<int> constrained = {0};
  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(K, M, constrained);

  SolverConfig config;
  config.nev = 4; // max n_red - 1 = 4

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K_red, M_red, config);

  ASSERT_TRUE(status.converged) << status.message;
  ASSERT_EQ(result.frequencies.size(), 4);

  // All frequencies should be positive (no rigid body modes)
  for (int i = 0; i < result.frequencies.size(); i++) {
    EXPECT_GT(result.frequencies(i), 0.0)
        << "Frequency " << i << " is not positive";
  }

  // Expand mode shapes
  Eigen::MatrixXcd full_modes =
      ModalSolver::expand_mode_shapes(result.mode_shapes, n, free_map);
  EXPECT_EQ(full_modes.rows(), n);

  // Constrained DOF should be zero in all mode shapes
  for (int m = 0; m < full_modes.cols(); m++) {
    EXPECT_NEAR(std::abs(full_modes(0, m)), 0.0, 1e-15)
        << "Constrained DOF not zero in mode " << m;
  }
}

// ---- Single Element Free Vibration ----

TEST(Solver, SingleElementConstrainedVibration) {
  // Build a single TET10 element, clamp one face, and solve
  Mesh mesh;
  mesh.nodes.resize(10, 3);
  mesh.nodes.row(0) = Eigen::Vector3d(0, 0, 0);
  mesh.nodes.row(1) = Eigen::Vector3d(1, 0, 0);
  mesh.nodes.row(2) = Eigen::Vector3d(0, 1, 0);
  mesh.nodes.row(3) = Eigen::Vector3d(0, 0, 1);
  mesh.nodes.row(4) = Eigen::Vector3d(0.5, 0, 0);   // mid 0-1
  mesh.nodes.row(5) = Eigen::Vector3d(0.5, 0.5, 0); // mid 1-2
  mesh.nodes.row(6) = Eigen::Vector3d(0, 0.5, 0);   // mid 0-2
  mesh.nodes.row(7) = Eigen::Vector3d(0, 0, 0.5);   // mid 0-3
  mesh.nodes.row(8) = Eigen::Vector3d(0.5, 0, 0.5); // mid 1-3
  mesh.nodes.row(9) = Eigen::Vector3d(0, 0.5, 0.5); // mid 2-3
  mesh.elements.resize(1, 10);
  mesh.elements.row(0) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Material mat(200e9, 0.3, 7850);
  GlobalAssembler assembler;
  assembler.assemble(mesh, mat);

  // Clamp the base face (nodes 0,1,2 and mid-edge nodes 4,5,6) — 6 nodes × 3
  // DOF = 18 constrained This leaves only nodes 3,7,8,9 free (12 DOFs), so
  // request up to 11 eigenvalues
  std::vector<int> constrained;
  for (int node : {0, 1, 2, 4, 5, 6}) {
    constrained.push_back(3 * node);
    constrained.push_back(3 * node + 1);
    constrained.push_back(3 * node + 2);
  }
  std::sort(constrained.begin(), constrained.end());

  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(assembler.K(), assembler.M(), constrained);

  EXPECT_EQ(K_red.rows(), 12); // 4 free nodes × 3 DOF

  SolverConfig config;
  config.nev = 5;

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K_red, M_red, config);

  ASSERT_TRUE(status.converged) << status.message;

  // All frequencies should be positive (fully constrained, no rigid body modes)
  for (int i = 0; i < result.frequencies.size(); i++) {
    EXPECT_GT(result.frequencies(i), 0.0)
        << "Frequency " << i << " is not positive";
  }
}

// ---- Wedge Sector with Hub Constraint ----

TEST(Solver, WedgeSectorWithHubConstraint) {
  Mesh mesh;
  mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
  Material mat(200e9, 0.3, 7850);

  GlobalAssembler assembler;
  assembler.assemble(mesh, mat);

  // Get hub constraint nodes
  const NodeSet *hub = mesh.find_node_set("hub_constraint");
  ASSERT_NE(hub, nullptr);

  // Constrain all DOFs on hub nodes
  std::vector<int> constrained;
  for (int node_id : hub->node_ids) {
    constrained.push_back(3 * node_id);
    constrained.push_back(3 * node_id + 1);
    constrained.push_back(3 * node_id + 2);
  }
  std::sort(constrained.begin(), constrained.end());

  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(assembler.K(), assembler.M(), constrained);

  SolverConfig config;
  config.nev = 10;

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K_red, M_red, config);

  ASSERT_TRUE(status.converged) << status.message;
  EXPECT_GE(result.frequencies.size(), 5);

  // All frequencies should be positive
  for (int i = 0; i < result.frequencies.size(); i++) {
    EXPECT_GT(result.frequencies(i), 1.0)
        << "Frequency " << i << " (" << result.frequencies(i) << " Hz) too low";
  }

  // Frequencies should be in ascending order
  for (int i = 1; i < result.frequencies.size(); i++) {
    EXPECT_GE(result.frequencies(i), result.frequencies(i - 1) - 1e-6);
  }
}

// ---- Complex Hermitian: Real Matrices (k=0 case) ----

TEST(Solver, ComplexHermitianRealCase) {
  // When K and M are purely real (k=0 cyclic case), complex solver should
  // give same results as real solver
  // Use 5x5 system so real solver can request 4, and doubled (10x10) has plenty
  // of room
  int n = 5;
  Eigen::MatrixXd K_dense = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = 2.0;
    if (i > 0)
      K_dense(i, i - 1) = -1.0;
    if (i < n - 1)
      K_dense(i, i + 1) = -1.0;
  }
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(n, n);

  SpMatd K_real = dense_to_sparse(K_dense);
  SpMatd M_real = dense_to_sparse(M_dense);

  // Build complex versions (purely real)
  SpMatcd K_complex(n, n), M_complex(n, n);
  {
    std::vector<TripletC> kc, mc;
    for (int k = 0; k < K_real.outerSize(); ++k)
      for (SpMatd::InnerIterator it(K_real, k); it; ++it)
        kc.emplace_back(it.row(), it.col(),
                        std::complex<double>(it.value(), 0.0));
    for (int k = 0; k < M_real.outerSize(); ++k)
      for (SpMatd::InnerIterator it(M_real, k); it; ++it)
        mc.emplace_back(it.row(), it.col(),
                        std::complex<double>(it.value(), 0.0));
    K_complex.setFromTriplets(kc.begin(), kc.end());
    M_complex.setFromTriplets(mc.begin(), mc.end());
  }

  SolverConfig config;
  config.nev = 4; // max n-1=4 for the real solver

  ModalSolver solver;
  auto [result_real, status_real] = solver.solve_real(K_real, M_real, config);
  auto [result_complex, status_complex] =
      solver.solve_complex_hermitian(K_complex, M_complex, config);

  ASSERT_TRUE(status_real.converged) << status_real.message;
  ASSERT_TRUE(status_complex.converged) << status_complex.message;

  int num_compare = std::min(result_real.frequencies.size(),
                             result_complex.frequencies.size());
  EXPECT_GE(num_compare, 4);

  for (int i = 0; i < num_compare; i++) {
    EXPECT_NEAR(result_real.frequencies(i), result_complex.frequencies(i), 1e-4)
        << "Frequency mismatch at mode " << i;
  }
}

// ---- M-Orthogonality of Mode Shapes ----

TEST(Solver, ModeShapesMOrthogonal) {
  // 6x6 system so we can request up to 5 eigenvalues
  int n = 6;
  Eigen::MatrixXd K_dense = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = 3.0;
    if (i > 0)
      K_dense(i, i - 1) = -1.0;
    if (i < n - 1)
      K_dense(i, i + 1) = -1.0;
  }
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(n, n);

  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  SolverConfig config;
  config.nev = 4;

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K, M, config);

  ASSERT_TRUE(status.converged);

  // Check M-orthogonality: phi_i^T * M * phi_j ≈ 0 for i != j
  for (int i = 0; i < result.mode_shapes.cols(); i++) {
    for (int j = i + 1; j < result.mode_shapes.cols(); j++) {
      Eigen::VectorXcd phi_i = result.mode_shapes.col(i);
      Eigen::VectorXcd phi_j = result.mode_shapes.col(j);
      double ortho = std::abs(phi_i.dot(phi_j));
      EXPECT_NEAR(ortho, 0.0, 1e-6)
          << "Modes " << i << " and " << j << " not M-orthogonal: " << ortho;
    }
  }
}

// ---- SolverConfig Defaults ----

TEST(Solver, SolverConfigDefaults) {
  SolverConfig config;
  EXPECT_EQ(config.nev, 20);
  EXPECT_EQ(config.ncv, 0);
  EXPECT_DOUBLE_EQ(config.shift, 0.0);
  EXPECT_DOUBLE_EQ(config.tolerance, 1e-8);
  EXPECT_EQ(config.max_iterations, 1000);
}

// ---- SolverStatus Defaults ----

TEST(Solver, SolverStatusDefaults) {
  SolverStatus status;
  EXPECT_FALSE(status.converged);
  EXPECT_EQ(status.num_converged, 0);
  EXPECT_EQ(status.iterations, 0);
}

// ---- Lanczos vs Dense Comparison ----

// Helper: build complex sparse matrix from dense complex matrix
static SpMatcd cdense_to_sparse(const Eigen::MatrixXcd &D) {
  int n = static_cast<int>(D.rows());
  std::vector<TripletC> trips;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (std::abs(D(i, j)) > 1e-15) {
        trips.emplace_back(i, j, D(i, j));
      }
    }
  }
  SpMatcd S(n, n);
  S.setFromTriplets(trips.begin(), trips.end());
  return S;
}

TEST(Solver, ComplexHermitianSmallSystem) {
  // Build a complex Hermitian system and solve via Hermitian Lanczos.
  // K = tridiagonal with complex off-diagonals, M = identity.
  const int n = 50;
  const int nev = 5;

  // Build K as a complex Hermitian tridiagonal matrix
  Eigen::MatrixXcd K_dense = Eigen::MatrixXcd::Zero(n, n);
  std::complex<double> off(-1.0, 0.2);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = std::complex<double>(3.0 + 0.01 * i, 0.0);
    if (i < n - 1) {
      K_dense(i, i + 1) = off;
      K_dense(i + 1, i) = std::conj(off);
    }
  }

  // M = identity (sparse)
  Eigen::MatrixXcd M_dense = Eigen::MatrixXcd::Identity(n, n);

  SpMatcd K = cdense_to_sparse(K_dense);
  SpMatcd M = cdense_to_sparse(M_dense);

  ModalSolver solver;
  SolverConfig config;
  config.nev = nev;
  config.shift = 0.0;
  auto [result, status] = solver.solve_complex_hermitian(K, M, config);
  ASSERT_TRUE(status.converged) << "Solver did not converge: " << status.message;
  ASSERT_EQ(result.frequencies.size(), nev);

  // Eigenvalues should be positive (K is positive definite)
  for (int i = 0; i < nev; i++) {
    EXPECT_GT(result.frequencies(i), 0.0)
        << "Frequency " << i << " should be positive";
  }

  // Frequencies should be in ascending order
  for (int i = 1; i < nev; i++) {
    EXPECT_GE(result.frequencies(i), result.frequencies(i - 1))
        << "Frequencies should be sorted ascending";
  }
}

// ---- Lanczos vs Dense Comparison ----

TEST(Solver, HermitianLanczosVsDense) {
  // Build a complex Hermitian system above the dense fallback threshold
  // (n > 200), solve with both Lanczos and dense, assert eigenvalues match.
  const int n = 300;
  const int nev = 10;

  // Build a banded complex Hermitian positive-definite matrix
  Eigen::MatrixXcd K_dense = Eigen::MatrixXcd::Zero(n, n);
  std::complex<double> off(-1.0, 0.3);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = std::complex<double>(4.0 + 0.005 * i, 0.0);
    if (i < n - 1) {
      K_dense(i, i + 1) = off;
      K_dense(i + 1, i) = std::conj(off);
    }
    if (i < n - 2) {
      std::complex<double> off2(0.3, -0.1);
      K_dense(i, i + 2) = off2;
      K_dense(i + 2, i) = std::conj(off2);
    }
  }

  Eigen::MatrixXcd M_dense = Eigen::MatrixXcd::Identity(n, n);

  SpMatcd K = cdense_to_sparse(K_dense);
  SpMatcd M = cdense_to_sparse(M_dense);

  // Solve with dense (reference)
  auto dense_result = HermitianLanczosEigenSolver::solve_dense(K, M, nev);
  ASSERT_TRUE(dense_result.converged) << "Dense solver failed: " << dense_result.message;
  ASSERT_EQ(dense_result.nconv, nev);

  // Solve with Lanczos (forces Lanczos path since n > 200)
  HermitianLanczosEigenSolver lanczos;
  HermitianLanczosEigenSolver::Config cfg;
  cfg.nev = nev;
  cfg.shift = 0.0;
  cfg.tolerance = 1e-8;
  cfg.max_iterations = 20;
  auto lanczos_result = lanczos.solve(K, M, cfg);
  ASSERT_TRUE(lanczos_result.converged) << "Lanczos did not converge: " << lanczos_result.message;
  ASSERT_EQ(lanczos_result.nconv, nev);

  // Compare eigenvalues (dense returns λ directly, Lanczos returns λ via shift-invert)
  for (int i = 0; i < nev; i++) {
    EXPECT_NEAR(dense_result.eigenvalues(i), lanczos_result.eigenvalues(i),
                1e-4 * std::abs(dense_result.eigenvalues(i)))
        << "Eigenvalue mismatch at mode " << i
        << ": dense=" << dense_result.eigenvalues(i)
        << " lanczos=" << lanczos_result.eigenvalues(i);
  }
}

// ---- Test 1: Large Tridiagonal — All Intermediate Modes ----

TEST(Solver, LargeTridiagonalAllModesAccurate) {
  // n=50 tridiagonal: verify EVERY eigenvalue matches analytical,
  // not just first and last.  This catches intermediate-mode bugs.
  const int n = 50;
  Eigen::MatrixXd K_dense = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = 2.0;
    if (i > 0)
      K_dense(i, i - 1) = -1.0;
    if (i < n - 1)
      K_dense(i, i + 1) = -1.0;
  }
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Identity(n, n);

  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  SolverConfig config;
  config.nev = n - 1; // Request all possible eigenvalues
  config.ncv = n;     // Full subspace

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K, M, config);

  ASSERT_TRUE(status.converged) << status.message;
  ASSERT_EQ(result.frequencies.size(), n - 1);

  // Analytical: lambda_k = 2 - 2*cos(k*pi/(n+1)), k = 1..n
  for (int i = 0; i < n - 1; i++) {
    double lambda = 2.0 - 2.0 * std::cos((i + 1) * PI / (n + 1));
    double f_expected = std::sqrt(lambda) / (2.0 * PI);
    EXPECT_NEAR(result.frequencies(i), f_expected, 1e-6)
        << "Mode " << i << " (of " << (n - 1) << "): got "
        << result.frequencies(i) << " expected " << f_expected;
  }
}

// ---- Test 2: Eigenvalue Residual on Real Mesh ----

TEST(Solver, EigenvalueResidualReal) {
  // Verify K*phi = lambda*M*phi for each converged eigenpair on the
  // wedge sector mesh.  This is the most fundamental correctness check.
  Mesh mesh;
  mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
  Material mat(200e9, 0.3, 7850);

  GlobalAssembler assembler;
  assembler.assemble(mesh, mat);

  const NodeSet *hub = mesh.find_node_set("hub_constraint");
  ASSERT_NE(hub, nullptr);

  std::vector<int> constrained;
  for (int node_id : hub->node_ids) {
    constrained.push_back(3 * node_id);
    constrained.push_back(3 * node_id + 1);
    constrained.push_back(3 * node_id + 2);
  }
  std::sort(constrained.begin(), constrained.end());

  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(assembler.K(), assembler.M(), constrained);

  SolverConfig config;
  config.nev = 10;

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K_red, M_red, config);
  ASSERT_TRUE(status.converged) << status.message;

  int num_modes = static_cast<int>(result.frequencies.size());
  ASSERT_GE(num_modes, 5);

  for (int i = 0; i < num_modes; i++) {
    double omega_sq = std::pow(2.0 * PI * result.frequencies(i), 2);
    // Mode shapes are stored as complex but should be real here
    Eigen::VectorXd phi = result.mode_shapes.col(i).real();

    // Residual: r = K*phi - omega^2 * M*phi
    Eigen::VectorXd K_phi = K_red * phi;
    Eigen::VectorXd M_phi = M_red * phi;
    Eigen::VectorXd residual = K_phi - omega_sq * M_phi;

    double rel_residual = residual.norm() / (K_phi.norm() + 1e-30);
    EXPECT_LT(rel_residual, 1e-6)
        << "Mode " << i << " (f=" << result.frequencies(i)
        << " Hz): relative residual = " << rel_residual;
  }
}

// ---- Test 3: Rayleigh Quotient Consistency ----

TEST(Solver, RayleighQuotientConsistency) {
  // For each mode the Rayleigh quotient phi^T*K*phi / phi^T*M*phi
  // must equal (2*pi*f)^2 — the returned eigenvalue.
  Mesh mesh;
  mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
  Material mat(200e9, 0.3, 7850);

  GlobalAssembler assembler;
  assembler.assemble(mesh, mat);

  const NodeSet *hub = mesh.find_node_set("hub_constraint");
  ASSERT_NE(hub, nullptr);

  std::vector<int> constrained;
  for (int node_id : hub->node_ids) {
    constrained.push_back(3 * node_id);
    constrained.push_back(3 * node_id + 1);
    constrained.push_back(3 * node_id + 2);
  }
  std::sort(constrained.begin(), constrained.end());

  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(assembler.K(), assembler.M(), constrained);

  SolverConfig config;
  config.nev = 10;

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K_red, M_red, config);
  ASSERT_TRUE(status.converged) << status.message;

  int num_modes = static_cast<int>(result.frequencies.size());
  for (int i = 0; i < num_modes; i++) {
    Eigen::VectorXd phi = result.mode_shapes.col(i).real();
    double KE = phi.dot(K_red * phi);
    double ME = phi.dot(M_red * phi);
    double rq = KE / ME;
    double omega_sq = std::pow(2.0 * PI * result.frequencies(i), 2);

    double rel_err = std::abs(rq - omega_sq) / (omega_sq + 1e-30);
    EXPECT_LT(rel_err, 1e-6)
        << "Mode " << i << " (f=" << result.frequencies(i)
        << " Hz): RQ=" << rq << " omega^2=" << omega_sq
        << " rel_err=" << rel_err;
  }
}

// ---- Test 4: Generalized Eigenvalue with Non-Identity Mass ----

TEST(Solver, GeneralizedEigNonIdentityMass) {
  // Verify solve_real handles K*x = lambda*M*x with non-trivial M.
  // Compare all eigenvalues against dense reference solver.
  const int n = 20;

  // Build tridiagonal K
  Eigen::MatrixXd K_dense = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = 4.0 + 0.2 * i;
    if (i > 0)
      K_dense(i, i - 1) = -1.0;
    if (i < n - 1)
      K_dense(i, i + 1) = -1.0;
  }

  // Build non-uniform SPD tridiagonal M
  Eigen::MatrixXd M_dense = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    M_dense(i, i) = 2.0 + 0.1 * i;
    if (i > 0) {
      M_dense(i, i - 1) = -0.3;
      M_dense(i - 1, i) = -0.3;
    }
  }

  // Dense reference solution
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges(K_dense,
                                                                 M_dense);
  ASSERT_EQ(ges.info(), Eigen::Success) << "Dense generalized eigensolver failed";
  Eigen::VectorXd ref_evals = ges.eigenvalues();

  // Sparse solve via Spectra
  SpMatd K = dense_to_sparse(K_dense);
  SpMatd M = dense_to_sparse(M_dense);

  SolverConfig config;
  config.nev = n - 1;
  config.ncv = n;

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K, M, config);
  ASSERT_TRUE(status.converged) << status.message;

  int num_modes = static_cast<int>(result.frequencies.size());
  ASSERT_GE(num_modes, n - 1);

  for (int i = 0; i < num_modes; i++) {
    double lambda = std::pow(2.0 * PI * result.frequencies(i), 2);
    double ref = ref_evals(i);
    double rel_err = std::abs(lambda - ref) / (std::abs(ref) + 1e-30);
    EXPECT_LT(rel_err, 1e-6)
        << "Mode " << i << ": lambda=" << lambda << " ref=" << ref
        << " rel_err=" << rel_err;
  }
}

// ---- Test 5: Complex Hermitian Eigenvalue Residual ----

TEST(Solver, ComplexHermitianResidual) {
  // Verify K*phi = lambda*M*phi for the complex Hermitian solver.
  const int n = 50;
  const int nev = 5;

  Eigen::MatrixXcd K_dense = Eigen::MatrixXcd::Zero(n, n);
  std::complex<double> off(-1.0, 0.2);
  for (int i = 0; i < n; i++) {
    K_dense(i, i) = std::complex<double>(3.0 + 0.01 * i, 0.0);
    if (i < n - 1) {
      K_dense(i, i + 1) = off;
      K_dense(i + 1, i) = std::conj(off);
    }
  }
  Eigen::MatrixXcd M_dense = Eigen::MatrixXcd::Identity(n, n);

  SpMatcd K = cdense_to_sparse(K_dense);
  SpMatcd M = cdense_to_sparse(M_dense);

  ModalSolver solver;
  SolverConfig config;
  config.nev = nev;
  auto [result, status] = solver.solve_complex_hermitian(K, M, config);
  ASSERT_TRUE(status.converged) << status.message;

  int num_modes = static_cast<int>(result.frequencies.size());
  ASSERT_GE(num_modes, nev);

  for (int i = 0; i < num_modes; i++) {
    double omega_sq = std::pow(2.0 * PI * result.frequencies(i), 2);
    Eigen::VectorXcd phi = result.mode_shapes.col(i);

    // Residual: r = K*phi - omega^2 * M*phi
    Eigen::VectorXcd K_phi = K * phi;
    Eigen::VectorXcd M_phi = M * phi;
    Eigen::VectorXcd residual = K_phi - omega_sq * M_phi;

    double rel_residual = residual.norm() / (K_phi.norm() + 1e-30);
    EXPECT_LT(rel_residual, 1e-5)
        << "Mode " << i << " (f=" << result.frequencies(i)
        << " Hz): relative residual = " << rel_residual;
  }
}

// ---- Test 6: Lanczos vs Dense — Varied Spectral Distributions ----

TEST(Solver, LanczosVsDenseVariedSpectra) {
  // Test the Hermitian Lanczos on systems with different eigenvalue spacing.
  const int n = 300;
  const int nev = 10;
  std::complex<double> off(-1.0, 0.3);

  struct TestCase {
    std::string name;
    double diag_base;
    double diag_slope;
  };
  TestCase cases[] = {
      {"well-separated", 4.0, 0.1},
      {"clustered", 4.0, 0.001},
      {"wide-spread", 1.0, 10.0 / n},
  };

  for (const auto &tc : cases) {
    Eigen::MatrixXcd K_dense = Eigen::MatrixXcd::Zero(n, n);
    for (int i = 0; i < n; i++) {
      K_dense(i, i) =
          std::complex<double>(tc.diag_base + tc.diag_slope * i, 0.0);
      if (i < n - 1) {
        K_dense(i, i + 1) = off;
        K_dense(i + 1, i) = std::conj(off);
      }
    }
    Eigen::MatrixXcd M_dense = Eigen::MatrixXcd::Identity(n, n);

    SpMatcd K = cdense_to_sparse(K_dense);
    SpMatcd M = cdense_to_sparse(M_dense);

    // Dense reference
    auto dense_result = HermitianLanczosEigenSolver::solve_dense(K, M, nev);
    ASSERT_TRUE(dense_result.converged)
        << tc.name << ": dense solver failed: " << dense_result.message;

    // Lanczos
    HermitianLanczosEigenSolver lanczos;
    HermitianLanczosEigenSolver::Config cfg;
    cfg.nev = nev;
    cfg.shift = 0.0;
    cfg.tolerance = 1e-8;
    cfg.max_iterations = 30;
    auto lanczos_result = lanczos.solve(K, M, cfg);
    ASSERT_TRUE(lanczos_result.converged)
        << tc.name << ": Lanczos did not converge: " << lanczos_result.message;
    ASSERT_EQ(lanczos_result.nconv, nev) << tc.name;

    for (int i = 0; i < nev; i++) {
      double rel_err =
          std::abs(dense_result.eigenvalues(i) - lanczos_result.eigenvalues(i)) /
          (std::abs(dense_result.eigenvalues(i)) + 1e-30);
      EXPECT_LT(rel_err, 1e-4)
          << tc.name << " mode " << i
          << ": dense=" << dense_result.eigenvalues(i)
          << " lanczos=" << lanczos_result.eigenvalues(i);
    }
  }
}

// ---- Test 7: M-Orthogonality on Real Mesh ----

TEST(Solver, WedgeSectorMOrthogonality) {
  // Verify mode shapes are M-orthogonal on the real mesh (not just identity-M).
  Mesh mesh;
  mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
  Material mat(200e9, 0.3, 7850);

  GlobalAssembler assembler;
  assembler.assemble(mesh, mat);

  const NodeSet *hub = mesh.find_node_set("hub_constraint");
  ASSERT_NE(hub, nullptr);

  std::vector<int> constrained;
  for (int node_id : hub->node_ids) {
    constrained.push_back(3 * node_id);
    constrained.push_back(3 * node_id + 1);
    constrained.push_back(3 * node_id + 2);
  }
  std::sort(constrained.begin(), constrained.end());

  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(assembler.K(), assembler.M(), constrained);

  SolverConfig config;
  config.nev = 10;

  ModalSolver solver;
  auto [result, status] = solver.solve_real(K_red, M_red, config);
  ASSERT_TRUE(status.converged) << status.message;

  int nm = static_cast<int>(result.mode_shapes.cols());
  ASSERT_GE(nm, 5);

  // Build Gram matrix G(i,j) = phi_i^T * M * phi_j
  for (int i = 0; i < nm; i++) {
    Eigen::VectorXd phi_i = result.mode_shapes.col(i).real();
    Eigen::VectorXd M_phi_i = M_red * phi_i;
    double diag = phi_i.dot(M_phi_i);
    EXPECT_GT(diag, 0.0) << "Mode " << i << " has non-positive M-norm";

    for (int j = i + 1; j < nm; j++) {
      Eigen::VectorXd phi_j = result.mode_shapes.col(j).real();
      double cross = std::abs(phi_j.dot(M_phi_i));
      double bound = 1e-6 * std::sqrt(diag * phi_j.dot(M_red * phi_j));
      EXPECT_LT(cross, bound)
          << "Modes " << i << " and " << j << " not M-orthogonal: "
          << cross << " > " << bound;
    }
  }
}

// ---- Test 8: Solver Determinism ----

TEST(Solver, SolverDeterministic) {
  // Two identical solves must produce identical results.
  Mesh mesh;
  mesh.load_from_gmsh(test_data_path("wedge_sector.msh"));
  Material mat(200e9, 0.3, 7850);

  GlobalAssembler assembler;
  assembler.assemble(mesh, mat);

  const NodeSet *hub = mesh.find_node_set("hub_constraint");
  ASSERT_NE(hub, nullptr);

  std::vector<int> constrained;
  for (int node_id : hub->node_ids) {
    constrained.push_back(3 * node_id);
    constrained.push_back(3 * node_id + 1);
    constrained.push_back(3 * node_id + 2);
  }
  std::sort(constrained.begin(), constrained.end());

  auto [K_red, M_red, free_map] =
      ModalSolver::eliminate_dofs(assembler.K(), assembler.M(), constrained);

  SolverConfig config;
  config.nev = 10;

  ModalSolver solver1, solver2;
  auto [r1, s1] = solver1.solve_real(K_red, M_red, config);
  auto [r2, s2] = solver2.solve_real(K_red, M_red, config);

  ASSERT_TRUE(s1.converged) << s1.message;
  ASSERT_TRUE(s2.converged) << s2.message;
  ASSERT_EQ(r1.frequencies.size(), r2.frequencies.size());

  for (int i = 0; i < r1.frequencies.size(); i++) {
    EXPECT_NEAR(r1.frequencies(i), r2.frequencies(i), 1e-15)
        << "Mode " << i << " frequencies differ between runs";
  }
}
