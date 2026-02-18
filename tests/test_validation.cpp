#include <gtest/gtest.h>
#include "turbomodal/cyclic_solver.hpp"
#include "turbomodal/element.hpp"
#include "turbomodal/forced_response.hpp"
#include "turbomodal/mistuning.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace turbomodal;

static std::string test_data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

static Mesh load_leissa_mesh() {
    Mesh mesh;
    mesh.load_from_gmsh(test_data_path("leissa_disk_sector.msh"));
    mesh.num_sectors = 24;
    mesh.identify_cyclic_boundaries();
    mesh.match_boundary_nodes();
    return mesh;
}

// Leissa analytical frequency for clamped-center, free-edge circular plate
// f = (lambda^2 / (2*pi*a^2)) * sqrt(D / (rho*h))
// D = E*h^3 / (12*(1-nu^2))
static double leissa_frequency(double lambda_sq, double E, double nu,
                                double rho, double h, double a) {
    double D = E * h * h * h / (12.0 * (1.0 - nu * nu));
    return (lambda_sq / (2.0 * PI * a * a)) * std::sqrt(D / (rho * h));
}

// Leissa lambda^2 values for free-outer, clamped-inner annular plate
// b/a = 0.1, nu = 1/3 (close to 0.33)
// lambda^2 = omega * a^2 * sqrt(rho*h / D)
// Reference: Leissa (1969), NASA SP-160, Table 2.29
// NC=0 (no nodal circles, fundamental mode family)
static const double LEISSA_LAMBDA_SQ[] = {
    4.235,   // ND=0  (Table 2.29, b/a=0.1)
    3.482,   // ND=1  (Table 2.29, b/a=0.1)
    5.499,   // ND=2  (Table 2.29, b/a=0.1)
    12.23,   // ND=3  (Table 2.5, free plate; for n>=2 center clamp has diminishing effect)
};

TEST(Validation, MatrixDiagnostics) {
    // Debug: check assembled matrices on Leissa mesh
    Mesh mesh = load_leissa_mesh();
    Material mat(200e9, 0.33, 7850.0);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);
    const SpMatd& K = assembler.K();
    const SpMatd& M = assembler.M();

    int ndof = mesh.num_dof();
    std::cout << "\n=== Matrix Diagnostics ===\n";
    std::cout << "  ndof=" << ndof << " K: " << K.rows() << "x" << K.cols()
              << " nnz=" << K.nonZeros()
              << " M: " << M.rows() << "x" << M.cols()
              << " nnz=" << M.nonZeros() << "\n";

    // Check for NaN/Inf in K
    int k_nan = 0, k_inf = 0, k_zero_diag = 0;
    double k_min_diag = 1e300, k_max_diag = 0;
    for (int i = 0; i < ndof; i++) {
        double d = K.coeff(i, i);
        if (std::isnan(d)) k_nan++;
        if (std::isinf(d)) k_inf++;
        if (std::abs(d) < 1e-30) k_zero_diag++;
        if (d > 0 && d < k_min_diag) k_min_diag = d;
        if (d > k_max_diag) k_max_diag = d;
    }
    std::cout << "  K diag: nan=" << k_nan << " inf=" << k_inf
              << " zero=" << k_zero_diag
              << " min=" << k_min_diag << " max=" << k_max_diag
              << " ratio=" << k_max_diag/k_min_diag << "\n";

    // Check for NaN/Inf in M
    int m_nan = 0, m_inf = 0, m_zero_diag = 0, m_neg_diag = 0;
    double m_min_diag = 1e300, m_max_diag = 0;
    for (int i = 0; i < ndof; i++) {
        double d = M.coeff(i, i);
        if (std::isnan(d)) m_nan++;
        if (std::isinf(d)) m_inf++;
        if (std::abs(d) < 1e-30) m_zero_diag++;
        if (d < 0) m_neg_diag++;
        if (d > 0 && d < m_min_diag) m_min_diag = d;
        if (d > m_max_diag) m_max_diag = d;
    }
    std::cout << "  M diag: nan=" << m_nan << " inf=" << m_inf
              << " zero=" << m_zero_diag << " neg=" << m_neg_diag
              << " min=" << m_min_diag << " max=" << m_max_diag
              << " ratio=" << m_max_diag/m_min_diag << "\n";

    // Check symmetry
    SpMatd K_t = K.transpose();
    SpMatd diff_K = K - K_t;
    double K_sym_err = 0;
    for (int col = 0; col < diff_K.outerSize(); ++col)
        for (SpMatd::InnerIterator it(diff_K, col); it; ++it)
            K_sym_err = std::max(K_sym_err, std::abs(it.value()));
    std::cout << "  K symmetry error: " << K_sym_err << "\n";

    SpMatd M_t = M.transpose();
    SpMatd diff_M = M - M_t;
    double M_sym_err = 0;
    for (int col = 0; col < diff_M.outerSize(); ++col)
        for (SpMatd::InnerIterator it(diff_M, col); it; ++it)
            M_sym_err = std::max(M_sym_err, std::abs(it.value()));
    std::cout << "  M symmetry error: " << M_sym_err << "\n";

    // Try direct SparseLU factorization of K
    Eigen::SparseLU<SpMatd> lu_solver;
    lu_solver.compute(K);
    std::cout << "  K SparseLU: " << (lu_solver.info() == Eigen::Success ? "OK" : "FAILED") << "\n";

    // Try a simple solve: K*x = e1
    if (lu_solver.info() == Eigen::Success) {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(ndof);
        rhs(0) = 1.0;
        Eigen::VectorXd x = lu_solver.solve(rhs);
        bool has_nan = false;
        for (int i = 0; i < ndof; i++) {
            if (std::isnan(x(i)) || std::isinf(x(i))) { has_nan = true; break; }
        }
        std::cout << "  K\\e1 has NaN/Inf: " << (has_nan ? "YES" : "NO")
                  << " norm=" << x.norm() << "\n";
    }

    // Print sample element and check mid-edge node positions
    for (int e = 0; e < std::min(2, mesh.num_elements()); e++) {
        std::cout << "  Element " << e << ":\n";
        // Verify mid-edge node assignments
        auto V = [&](int local) -> Eigen::Vector3d {
            return mesh.nodes.row(mesh.elements(e, local)).transpose();
        };
        auto mid = [](Eigen::Vector3d a, Eigen::Vector3d b) { return (a + b) / 2.0; };

        // Expected mid-edge positions per our code convention
        struct { int node_idx; int v1, v2; const char* label; } edges[] = {
            {4, 0, 1, "0-1"}, {5, 1, 2, "1-2"}, {6, 0, 2, "0-2"},
            {7, 0, 3, "0-3"}, {8, 1, 3, "1-3"}, {9, 2, 3, "2-3"}
        };

        for (auto& edge : edges) {
            Eigen::Vector3d expected = mid(V(edge.v1), V(edge.v2));
            Eigen::Vector3d actual = V(edge.node_idx);
            double dist = (actual - expected).norm();
            std::cout << "    Node " << edge.node_idx << " (edge " << edge.label
                      << "): dist from expected midpoint = " << dist;
            if (dist > 1e-6) {
                // Check if it matches a different edge
                for (auto& other : edges) {
                    if (other.node_idx == edge.node_idx) continue;
                    Eigen::Vector3d other_mid = mid(V(other.v1), V(other.v2));
                    double d2 = (actual - other_mid).norm();
                    if (d2 < 1e-6) {
                        std::cout << " ** MATCHES edge " << other.label << " instead!";
                    }
                }
            }
            std::cout << "\n";
        }
    }

    // Check element Jacobians
    int neg_jac = 0, zero_jac = 0;
    double min_det = 1e300, max_det = -1e300;
    for (int e = 0; e < mesh.num_elements(); e++) {
        Eigen::Matrix<double, 10, 3> coords;
        for (int n = 0; n < 10; n++) {
            int nid = mesh.elements(e, n);
            coords.row(n) = mesh.nodes.row(nid);
        }
        // Check Jacobian at element centroid (xi=eta=zeta=0.25)
        // Using isoparametric shape function derivatives
        double xi = 0.25, eta = 0.25, zeta = 0.25;
        double L1 = 1.0 - xi - eta - zeta;
        double L2 = xi, L3 = eta, L4 = zeta;

        // dN/dxi, dN/deta, dN/dzeta for TET10
        Eigen::Matrix<double, 10, 3> dN;
        // Corner nodes
        dN(0,0) = -(4*L1-1); dN(0,1) = -(4*L1-1); dN(0,2) = -(4*L1-1);
        dN(1,0) = 4*L2-1;    dN(1,1) = 0;          dN(1,2) = 0;
        dN(2,0) = 0;          dN(2,1) = 4*L3-1;     dN(2,2) = 0;
        dN(3,0) = 0;          dN(3,1) = 0;           dN(3,2) = 4*L4-1;
        // Mid-edge nodes
        dN(4,0) = 4*(L1-L2); dN(4,1) = -4*L2;      dN(4,2) = -4*L2;
        dN(5,0) = 4*L3;       dN(5,1) = 4*L2;       dN(5,2) = 0;
        dN(6,0) = -4*L3;      dN(6,1) = 4*(L1-L3);  dN(6,2) = -4*L3;
        dN(7,0) = -4*L4;      dN(7,1) = -4*L4;      dN(7,2) = 4*(L1-L4);
        dN(8,0) = 4*L4;       dN(8,1) = 0;           dN(8,2) = 4*L2;
        dN(9,0) = 0;          dN(9,1) = 4*L4;        dN(9,2) = 4*L3;

        Eigen::Matrix3d J = dN.transpose() * coords;
        double det = J.determinant();
        if (det < 0) neg_jac++;
        if (std::abs(det) < 1e-20) zero_jac++;
        min_det = std::min(min_det, det);
        max_det = std::max(max_det, det);
    }
    std::cout << "  Element Jacobians: neg=" << neg_jac << " zero=" << zero_jac
              << " min=" << min_det << " max=" << max_det
              << " total=" << mesh.num_elements() << "\n";

    // Check total mass: u^T * M * u for unit x-displacement
    {
        Eigen::VectorXd ux = Eigen::VectorXd::Zero(ndof);
        Eigen::VectorXd uz = Eigen::VectorXd::Zero(ndof);
        for (int i = 0; i < mesh.num_nodes(); i++) {
            ux(3*i) = 1.0;     // unit x displacement
            uz(3*i+2) = 1.0;   // unit z displacement
        }
        double mass_x = ux.dot(M * ux);
        double mass_z = uz.dot(M * uz);
        // Expected: sector volume * density
        double alpha_rad = 2.0 * PI / 24.0;
        double V_sector = alpha_rad / 2.0 * (0.3*0.3 - 0.03*0.03) * 0.01;
        double m_expected = 7850.0 * V_sector;
        std::cout << "  Total mass from M (x): " << mass_x << " kg\n";
        std::cout << "  Total mass from M (z): " << mass_z << " kg\n";
        std::cout << "  Expected mass: " << m_expected << " kg\n";
        std::cout << "  Mass ratio x: " << mass_x / m_expected << "\n";
    }

    // Verify stiffness matrix with known deformation fields
    {
        double V_sector = 0.000117768;  // sector volume
        double E_val = 200e9;
        double nu_val = 0.3;
        double c_coef = E_val / ((1.0+nu_val)*(1.0-2.0*nu_val));

        // Test 1: u_x=x (uniform epsilon_xx=1)
        Eigen::VectorXd u1 = Eigen::VectorXd::Zero(ndof);
        for (int i = 0; i < mesh.num_nodes(); i++) u1(3*i) = mesh.nodes(i, 0);
        double KE1 = u1.dot(K * u1);
        double KE1_exp = c_coef * (1.0-nu_val) * V_sector;
        std::cout << "  K check eps_xx=1: ratio=" << KE1/KE1_exp << "\n";

        // Test 2: u_z=z (uniform epsilon_zz=1)
        Eigen::VectorXd u2 = Eigen::VectorXd::Zero(ndof);
        for (int i = 0; i < mesh.num_nodes(); i++) u2(3*i+2) = mesh.nodes(i, 2);
        double KE2 = u2.dot(K * u2);
        double KE2_exp = c_coef * (1.0-nu_val) * V_sector;
        std::cout << "  K check eps_zz=1: ratio=" << KE2/KE2_exp << "\n";

        // Per-element strain check for first element
        {
            TET10Element elem;
            Eigen::Matrix<double, 30, 1> ue_x, ue_z;
            ue_x.setZero(); ue_z.setZero();
            for (int n = 0; n < 10; n++) {
                int nid = mesh.elements(0, n);
                elem.node_coords.row(n) = mesh.nodes.row(nid);
                ue_x(3*n)   = mesh.nodes(nid, 0);  // u_x = x
                ue_z(3*n+2) = mesh.nodes(nid, 2);  // u_z = z
            }
            Eigen::Matrix<double, 30, 30> ke = elem.stiffness(mat);
            double ke_ux = ue_x.dot(ke * ue_x);
            double ke_uz = ue_z.dot(ke * ue_z);

            // Compute element volume
            double Ve = 0;
            for (int gp = 0; gp < 4; gp++) {
                double xi = TET10Element::gauss_points[gp](0);
                double eta = TET10Element::gauss_points[gp](1);
                double zeta = TET10Element::gauss_points[gp](2);
                double w = TET10Element::gauss_weights[gp];
                Eigen::Matrix3d J = elem.jacobian(xi, eta, zeta);
                Ve += J.determinant() * w;
            }
            double ke_exp = c_coef * (1.0-nu_val) * Ve;
            std::cout << "  Elem 0: Ve=" << Ve << " ke_ux=" << ke_ux
                      << " ke_uz=" << ke_uz << " exp=" << ke_exp
                      << " ratio_x=" << ke_ux/ke_exp
                      << " ratio_z=" << ke_uz/ke_exp << "\n";

            // Print strain at each Gauss point for u_x=x
            for (int gp = 0; gp < 4; gp++) {
                double xi = TET10Element::gauss_points[gp](0);
                double eta = TET10Element::gauss_points[gp](1);
                double zeta = TET10Element::gauss_points[gp](2);
                auto B = elem.B_matrix(xi, eta, zeta);
                Eigen::Matrix<double,6,1> eps_x = B * ue_x;
                Eigen::Matrix<double,6,1> eps_z = B * ue_z;
                std::cout << "  GP" << gp << " eps_x=[" << eps_x.transpose()
                          << "] eps_z=[" << eps_z.transpose() << "]\n";
            }
        }
    }

    // Check Rayleigh quotient for trial bending mode (w = 1 - (r/a)^2)
    {
        double a = 0.3;
        double h_plate = 0.01;
        Eigen::VectorXd u_bend = Eigen::VectorXd::Zero(ndof);
        for (int i = 0; i < mesh.num_nodes(); i++) {
            double x = mesh.nodes(i, 0);
            double y = mesh.nodes(i, 1);
            double z = mesh.nodes(i, 2);
            double r = std::sqrt(x*x + y*y);
            double w = 1.0 - (r/a)*(r/a);
            double z_mid = z - h_plate/2.0;  // distance from mid-plane

            // Plate bending: u_x = -z*dw/dx, u_y = -z*dw/dy, u_z = w
            double dw_dr = -2.0 * r / (a*a);
            double dw_dx = (r > 1e-10) ? dw_dr * x/r : 0.0;
            double dw_dy = (r > 1e-10) ? dw_dr * y/r : 0.0;

            u_bend(3*i)     = -z_mid * dw_dx;
            u_bend(3*i+1)   = -z_mid * dw_dy;
            u_bend(3*i+2)   = w;
        }

        double KE = u_bend.dot(K * u_bend);
        double ME = u_bend.dot(M * u_bend);
        double omega_sq = KE / ME;
        double f_rayleigh = std::sqrt(omega_sq) / (2.0 * PI);
        std::cout << "  Rayleigh quotient bending: KE=" << KE << " ME=" << ME
                  << " f=" << f_rayleigh << " Hz\n";

        // Also try pure z-displacement mode
        Eigen::VectorXd u_z_only = Eigen::VectorXd::Zero(ndof);
        for (int i = 0; i < mesh.num_nodes(); i++) {
            double x = mesh.nodes(i, 0);
            double y = mesh.nodes(i, 1);
            double r = std::sqrt(x*x + y*y);
            u_z_only(3*i+2) = 1.0 - (r/a)*(r/a);
        }
        KE = u_z_only.dot(K * u_z_only);
        ME = u_z_only.dot(M * u_z_only);
        omega_sq = KE / ME;
        f_rayleigh = std::sqrt(omega_sq) / (2.0 * PI);
        std::cout << "  Rayleigh quotient z-only: KE=" << KE << " ME=" << ME
                  << " f=" << f_rayleigh << " Hz\n";
    }

    // Direct eigenvalue solve (no cyclic BCs) to check assembly
    {
        // Constrain hub DOFs
        const NodeSet* hub = mesh.find_node_set("hub_constraint");
        std::vector<int> constrained_dofs;
        if (hub) {
            for (int nid : hub->node_ids) {
                constrained_dofs.push_back(3*nid);
                constrained_dofs.push_back(3*nid+1);
                constrained_dofs.push_back(3*nid+2);
            }
            std::sort(constrained_dofs.begin(), constrained_dofs.end());
        }
        std::cout << "  Hub nodes: " << (hub ? hub->node_ids.size() : 0)
                  << " constrained DOFs: " << constrained_dofs.size() << "\n";

        ModalSolver modal;
        auto [K_red, M_red, free_map] = ModalSolver::eliminate_dofs(K, M, constrained_dofs);
        std::cout << "  Reduced system: " << K_red.rows() << " DOFs\n";

        SolverConfig cfg;
        cfg.nev = 10;
        cfg.ncv = 60;
        cfg.shift = 0.0;
        auto [result, status] = modal.solve_real(K_red, M_red, cfg);
        std::cout << "  Direct solve: " << status.message
                  << " (" << status.num_converged << " converged)\n";
        std::cout << "  First 10 frequencies (Hz):";
        for (int i = 0; i < std::min(10, (int)result.frequencies.size()); i++) {
            std::cout << " " << result.frequencies(i);
        }
        std::cout << "\n";
    }

    std::cout << "\n";
    SUCCEED();
}

TEST(Validation, LeissaFlatDisk) {
    // Load the Leissa validation mesh
    Mesh mesh = load_leissa_mesh();

    // Steel properties (nu=0.33 to match Leissa table values)
    double E = 200e9;
    double nu = 0.33;
    double rho = 7850.0;
    Material mat(E, nu, rho);

    // Disk geometry
    double h = 0.01;   // thickness
    double a = 0.3;    // outer radius

    // Compute analytical Leissa frequencies
    std::cout << "\n=== Leissa Flat Disk Validation ===\n";
    std::cout << "  E=" << E/1e9 << " GPa, nu=" << nu << ", rho=" << rho << " kg/m^3\n";
    std::cout << "  a=" << a << " m, h=" << h << " m\n";
    std::cout << "  Mesh: " << mesh.num_nodes() << " nodes, "
              << mesh.num_elements() << " elements, "
              << mesh.num_sectors << " sectors\n\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  ND  | Analytical (Hz) | FEA (Hz)   | Error (%)\n";
    std::cout << "  ----|-----------------|------------|----------\n";

    // Solve with cyclic symmetry at 0 RPM
    CyclicSymmetrySolver solver(mesh, mat);
    auto results = solver.solve_at_rpm(0.0, 5);

    // Compare first mode (NC=0) at each nodal diameter
    int max_nd = std::min(4, static_cast<int>(results.size()));

    for (int nd = 0; nd < max_nd; nd++) {
        double f_analytical = leissa_frequency(LEISSA_LAMBDA_SQ[nd], E, nu, rho, h, a);

        // Find result for this harmonic index
        const ModalResult* r = nullptr;
        for (const auto& res : results) {
            if (res.harmonic_index == nd) {
                r = &res;
                break;
            }
        }

        if (r && r->frequencies.size() > 0) {
            double f_fea = r->frequencies(0);  // First mode of this ND
            double error_pct = 100.0 * std::abs(f_fea - f_analytical) / f_analytical;

            std::cout << "  " << std::setw(3) << nd << " | "
                      << std::setw(14) << f_analytical << " | "
                      << std::setw(10) << f_fea << " | "
                      << std::setw(8) << error_pct << "\n";

            // Leissa validation: 3D solid TET10 vs Kirchhoff plate theory
            // Expected sources of error: 3D solid discretization of thin plate,
            // shear deformation (h/a=0.033), mesh density, hub finite radius.
            // ND>=2 should be very accurate; ND=0 may have larger mesh error.
            double tol = (nd == 3) ? 15.0 : 10.0;  // ND=3 uses approx value
            EXPECT_LT(error_pct, tol)
                << "ND=" << nd << " frequency error exceeds " << tol << "%";
        } else {
            std::cout << "  " << std::setw(3) << nd << " | "
                      << std::setw(14) << leissa_frequency(LEISSA_LAMBDA_SQ[nd], E, nu, rho, h, a)
                      << " |    MISSING  |        \n";
            ADD_FAILURE() << "No result for ND=" << nd;
        }
    }
    std::cout << "\n";
}

TEST(Validation, KwakVsBEMAddedMass) {
    // Compare Kwak (flat-disk analytical) vs BEM (geometry-based) for a flat disk.
    // Both should produce similar frequency reductions since the mesh IS a flat disk.
    // The Kwak model is based on Kwak (1991) gamma coefficients for clamped circular
    // plates. The BEM solves the actual potential flow problem from the mesh geometry.
    // For a flat disk, they should agree within ~30% (BEM uses the actual 3D surface
    // while Kwak assumes an idealized thin disk).
    Mesh mesh = load_leissa_mesh();

    double E = 200e9;
    double nu = 0.33;
    double rho_s = 7850.0;
    Material mat(E, nu, rho_s);

    double rho_f = 1000.0;  // water
    double h = 0.01;
    double a = 0.3;

    // Solve dry
    CyclicSymmetrySolver solver_dry(mesh, mat);
    auto results_dry = solver_dry.solve_at_rpm(0.0, 5);

    // Solve with Kwak analytical model
    FluidConfig fluid_kwak;
    fluid_kwak.type = FluidConfig::Type::KWAK_ANALYTICAL;
    fluid_kwak.fluid_density = rho_f;
    fluid_kwak.disk_radius = a;
    fluid_kwak.disk_thickness = h;

    CyclicSymmetrySolver solver_kwak(mesh, mat, fluid_kwak);
    auto results_kwak = solver_kwak.solve_at_rpm(0.0, 5);

    // Solve with BEM potential flow model
    FluidConfig fluid_bem;
    fluid_bem.type = FluidConfig::Type::POTENTIAL_FLOW_BEM;
    fluid_bem.fluid_density = rho_f;

    CyclicSymmetrySolver solver_bem(mesh, mat, fluid_bem);
    auto results_bem = solver_bem.solve_at_rpm(0.0, 5);

    std::cout << "\n=== Kwak vs BEM Added Mass ===\n";
    std::cout << "  rho_fluid=" << rho_f << " kg/m^3 (water), flat disk a="
              << a << " m, h=" << h << " m\n\n";
    std::cout << "  ND  | f_dry (Hz) | f_kwak (Hz) | f_bem (Hz) | Kwak ratio | BEM ratio  | Diff (%)\n";
    std::cout << "  ----|------------|-------------|------------|------------|------------|--------\n";

    for (int nd = 0; nd < std::min(4, static_cast<int>(results_dry.size())); nd++) {
        const ModalResult* r_dry = nullptr;
        const ModalResult* r_kwak = nullptr;
        const ModalResult* r_bem = nullptr;
        for (const auto& r : results_dry) {
            if (r.harmonic_index == nd) { r_dry = &r; break; }
        }
        for (const auto& r : results_kwak) {
            if (r.harmonic_index == nd) { r_kwak = &r; break; }
        }
        for (const auto& r : results_bem) {
            if (r.harmonic_index == nd) { r_bem = &r; break; }
        }

        if (r_dry && r_kwak && r_bem &&
            r_dry->frequencies.size() > 0 &&
            r_kwak->frequencies.size() > 0 &&
            r_bem->frequencies.size() > 0) {

            double f_dry = r_dry->frequencies(0);
            double f_kwak = r_kwak->frequencies(0);
            double f_bem = r_bem->frequencies(0);
            double ratio_kwak = f_kwak / f_dry;
            double ratio_bem = f_bem / f_dry;
            double diff_pct = 100.0 * std::abs(ratio_bem - ratio_kwak) / ratio_kwak;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  " << std::setw(3) << nd << " | "
                      << std::setw(10) << f_dry << " | "
                      << std::setw(11) << f_kwak << " | "
                      << std::setw(10) << f_bem << " | "
                      << std::setw(10) << std::setprecision(4) << ratio_kwak << " | "
                      << std::setw(10) << ratio_bem << " | "
                      << std::setw(6) << std::setprecision(1) << diff_pct << "\n";

            // Both wet frequencies should be lower than dry
            EXPECT_LT(f_kwak, f_dry)
                << "ND=" << nd << " Kwak wet freq should be less than dry";
            EXPECT_LT(f_bem, f_dry)
                << "ND=" << nd << " BEM wet freq should be less than dry";

            // Ratios should be physically reasonable (between 0.3 and 1.0)
            EXPECT_GT(ratio_kwak, 0.3) << "ND=" << nd << " Kwak ratio too low";
            EXPECT_LT(ratio_kwak, 1.0) << "ND=" << nd << " Kwak ratio >= 1";
            EXPECT_GT(ratio_bem, 0.3) << "ND=" << nd << " BEM ratio too low";
            EXPECT_LT(ratio_bem, 1.0) << "ND=" << nd << " BEM ratio >= 1";

            // Kwak and BEM should agree within 30% for a flat disk
            // (BEM solves the actual geometry, Kwak is for an idealized thin disk)
            EXPECT_LT(diff_pct, 30.0)
                << "ND=" << nd << " Kwak vs BEM ratio difference exceeds 30%";
        }
    }
    std::cout << "\n";
}

TEST(Validation, CentrifugalStiffeningEffect) {
    // Verify that rotating effects change frequencies
    Mesh mesh = load_leissa_mesh();
    Material mat(200e9, 0.33, 7850);

    CyclicSymmetrySolver solver(mesh, mat);

    auto results_0 = solver.solve_at_rpm(0.0, 3);
    auto results_5k = solver.solve_at_rpm(5000.0, 3);

    double omega_hz = 5000.0 / 60.0;  // rotation rate in Hz
    std::cout << "\n=== Centrifugal Stiffening Validation ===\n";
    std::cout << "  ND  | f@0rpm (Hz) | f_rot@5k (Hz) | Stiffening (%) | BW (Hz) | FW (Hz)\n";
    std::cout << "  ----|-------------|---------------|----------------|---------|--------\n";

    for (int nd = 0; nd < std::min(4, static_cast<int>(results_0.size())); nd++) {
        const ModalResult* r0 = nullptr;
        const ModalResult* r5k = nullptr;
        for (const auto& r : results_0) {
            if (r.harmonic_index == nd) { r0 = &r; break; }
        }
        for (const auto& r : results_5k) {
            if (r.harmonic_index == nd) { r5k = &r; break; }
        }

        if (r0 && r5k && r0->frequencies.size() > 0 && r5k->frequencies.size() > 0) {
            double f0 = r0->frequencies(0);
            std::cout << std::fixed << std::setprecision(2);

            if (nd == 0) {
                // k=0: no FW/BW splitting
                double f5k = r5k->frequencies(0);
                double stiff_pct = 100.0 * (f5k - f0) / f0;
                std::cout << "  " << std::setw(3) << nd << " | "
                          << std::setw(11) << f0 << " | "
                          << std::setw(13) << f5k << " | "
                          << std::setw(14) << stiff_pct << " |     N/A |    N/A\n";
            } else {
                // k>0: recover rotating-frame frequency from FW mode
                // FW = f_rot + k*omega_hz, so f_rot = FW - k*omega_hz
                double k_omega_hz = nd * omega_hz;
                int n_modes = static_cast<int>(r5k->frequencies.size());
                // Find first FW frequency
                double f_fw = -1, f_bw = -1;
                for (int m = 0; m < n_modes; m++) {
                    if (r5k->whirl_direction(m) > 0 && f_fw < 0) f_fw = r5k->frequencies(m);
                    if (r5k->whirl_direction(m) < 0 && f_bw < 0) f_bw = r5k->frequencies(m);
                }
                double f_rot = (f_fw >= 0) ? f_fw - k_omega_hz : r5k->frequencies(0);
                double stiff_pct = 100.0 * (f_rot - f0) / f0;
                std::cout << "  " << std::setw(3) << nd << " | "
                          << std::setw(11) << f0 << " | "
                          << std::setw(13) << f_rot << " | "
                          << std::setw(14) << stiff_pct << " | "
                          << std::setw(7) << (f_bw >= 0 ? f_bw : 0.0) << " | "
                          << std::setw(6) << (f_fw >= 0 ? f_fw : 0.0) << "\n";
            }
        }
    }
    std::cout << "\n";

    // Verify the solver produces results and frequencies are positive
    ASSERT_FALSE(results_0.empty());
    ASSERT_FALSE(results_5k.empty());

    // Centrifugal stiffening check:
    // - For k=0 (no Coriolis splitting): frequency should increase with RPM
    // - For k>0: Coriolis splitting creates FW/BW pairs. The BW mode can be
    //   LOWER than 0 RPM (because k*omega subtracted exceeds stiffening).
    //   Check that the MAX frequency (FW) exceeds the 0 RPM value.
    for (int nd = 0; nd < std::min(4, static_cast<int>(results_0.size())); nd++) {
        const ModalResult* r0 = nullptr;
        const ModalResult* r5k = nullptr;
        for (const auto& r : results_0)
            if (r.harmonic_index == nd) { r0 = &r; break; }
        for (const auto& r : results_5k)
            if (r.harmonic_index == nd) { r5k = &r; break; }

        if (r0 && r5k && r0->frequencies.size() > 0 && r5k->frequencies.size() > 0) {
            double f0_first = r0->frequencies(0);
            if (nd == 0) {
                // k=0: no splitting, frequency should increase
                double f5k_first = r5k->frequencies(0);
                EXPECT_GT(f5k_first, f0_first * 0.95)
                    << "ND=0 frequency should not decrease at 5000 RPM";
            } else {
                // k>0: FW mode (highest of first pair) should exceed 0 RPM value
                // The FW mode gains k*omega/(2*pi) from Coriolis + centrifugal stiffening
                int n_modes = static_cast<int>(r5k->frequencies.size());
                double f5k_max = r5k->frequencies(std::min(1, n_modes - 1));
                EXPECT_GT(f5k_max, f0_first)
                    << "ND=" << nd << " FW frequency should exceed 0 RPM value";
            }
        }
    }
}

TEST(Validation, RayleighQuotientBound) {
    // Rayleigh quotient with trial w = 1 - (r/a)^2 gives upper bound on ND=0 fundamental
    Mesh mesh = load_leissa_mesh();
    double E = 200e9, nu = 0.33, rho = 7850.0;
    Material mat(E, nu, rho);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);
    const SpMatd& K = assembler.K();
    const SpMatd& M = assembler.M();
    int ndof = mesh.num_dof();

    double a = 0.3, h_plate = 0.01;
    Eigen::VectorXd u_z = Eigen::VectorXd::Zero(ndof);
    for (int i = 0; i < mesh.num_nodes(); i++) {
        double x = mesh.nodes(i, 0);
        double y = mesh.nodes(i, 1);
        double r = std::sqrt(x * x + y * y);
        u_z(3 * i + 2) = 1.0 - (r / a) * (r / a);
    }
    double KE = u_z.dot(K * u_z);
    double ME = u_z.dot(M * u_z);
    double f_rayleigh = std::sqrt(KE / ME) / (2.0 * PI);

    // FEA ND=0 fundamental should be <= Rayleigh quotient (upper bound)
    CyclicSymmetrySolver solver(mesh, mat);
    auto results = solver.solve_at_rpm(0.0, 3);
    const ModalResult* r0 = nullptr;
    for (const auto& r : results) {
        if (r.harmonic_index == 0) { r0 = &r; break; }
    }
    ASSERT_NE(r0, nullptr);
    ASSERT_GT(r0->frequencies.size(), 0);

    double f_fea = r0->frequencies(0);
    EXPECT_LT(f_fea, f_rayleigh * 1.01)
        << "FEA frequency should not exceed Rayleigh quotient upper bound";
}

TEST(Validation, MassConservation) {
    // u^T * M * u for unit translation = rho * V_sector within 2%
    Mesh mesh = load_leissa_mesh();
    double rho = 7850.0;
    Material mat(200e9, 0.33, rho);

    GlobalAssembler assembler;
    assembler.assemble(mesh, mat);
    const SpMatd& M = assembler.M();
    int ndof = mesh.num_dof();

    Eigen::VectorXd ux = Eigen::VectorXd::Zero(ndof);
    for (int i = 0; i < mesh.num_nodes(); i++) ux(3 * i) = 1.0;
    double mass_fea = ux.dot(M * ux);

    // Analytical sector volume: (alpha/2) * (R_outer^2 - R_inner^2) * h
    double alpha_rad = 2.0 * PI / 24.0;
    double V_sector = alpha_rad / 2.0 * (0.3 * 0.3 - 0.03 * 0.03) * 0.01;
    double mass_expected = rho * V_sector;

    EXPECT_NEAR(mass_fea, mass_expected, mass_expected * 0.02)
        << "Total mass from M should match rho*V_sector within 2%";
}

TEST(Validation, ModalFRFAnalytical) {
    // ForcedResponseSolver::modal_frf at 200 points matches exact formula < 1e-10
    double omega_r = 2.0 * PI * 800.0;
    double zeta = 0.015;
    std::complex<double> Q(3.0, -2.0);

    for (int i = 0; i < 200; i++) {
        double f = 200.0 + i * 5.0;
        double omega = 2.0 * PI * f;
        auto H = ForcedResponseSolver::modal_frf(omega, omega_r, Q, zeta);

        std::complex<double> denom(omega_r * omega_r - omega * omega,
                                    2.0 * zeta * omega_r * omega);
        std::complex<double> expected = Q / denom;

        EXPECT_NEAR(H.real(), expected.real(), std::abs(expected) * 1e-10)
            << "FRF real mismatch at f=" << f << " Hz";
        EXPECT_NEAR(H.imag(), expected.imag(), std::abs(expected) * 1e-10)
            << "FRF imag mismatch at f=" << f << " Hz";
    }
}

TEST(Validation, FMMTunedIdentity) {
    // FMM with zero deviations: frequencies preserved, bounded magnification/IPR.
    // Degenerate eigenvalue pairs (harmonics k=1..N/2-1) allow eigenvector
    // arbitrariness, so magnification can reach sqrt(2) and IPR up to 1.5.
    int N = 24;
    Eigen::VectorXd tuned(13);
    for (int i = 0; i < 13; i++) tuned(i) = 500.0 + i * 100.0;
    Eigen::VectorXd dev = Eigen::VectorXd::Zero(N);

    auto result = FMMSolver::solve(N, tuned, dev);
    EXPECT_LE(result.peak_magnification, std::sqrt(2.0) + 0.01);
    EXPECT_GE(result.peak_magnification, 1.0 - 1e-10);
    for (int m = 0; m < N; m++) {
        EXPECT_LE(result.localization_ipr(m), 2.0)
            << "IPR should be bounded for tuned system at mode " << m;
        EXPECT_GE(result.localization_ipr(m), 0.5)
            << "IPR should be at least 0.5 for tuned system at mode " << m;
    }
}

TEST(Validation, FMMWhiteheadBound) {
    // 100 random seeds: mag <= 1 + sqrt((N-1)/2) ≈ 4.39 for N=24
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

TEST(Validation, CoriolisFWBWSplitting) {
    // T3: For k>0 at non-zero RPM, modes should split into FW and BW pairs
    // Forward whirl has higher frequency, backward whirl has lower (in stationary frame)
    Mesh mesh = load_leissa_mesh();
    Material mat(200e9, 0.33, 7850);

    CyclicSymmetrySolver solver(mesh, mat);

    double rpm = 3000.0;
    auto results = solver.solve_at_rpm(rpm, 3);

    std::cout << "\n=== Coriolis FW/BW Splitting Validation ===\n";
    std::cout << "  RPM=" << rpm << "\n\n";
    std::cout << "  ND  | Mode | Freq (Hz) | Whirl | Expected\n";
    std::cout << "  ----|------|-----------|-------|----------\n";

    ASSERT_FALSE(results.empty());

    for (const auto& r : results) {
        int k = r.harmonic_index;
        if (k == 0) {
            // k=0: standing waves, no splitting
            for (int m = 0; m < std::min(2, (int)r.frequencies.size()); m++) {
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "  " << std::setw(3) << k << " | "
                          << std::setw(4) << m << " | "
                          << std::setw(9) << r.frequencies(m) << " | "
                          << std::setw(5) << r.whirl_direction(m) << " | standing\n";
                EXPECT_EQ(r.whirl_direction(m), 0)
                    << "k=0 modes should be standing (whirl=0)";
            }
        } else if (k > 0 && k < mesh.num_sectors / 2) {
            // k>0: FW/BW splitting
            int n_modes = static_cast<int>(r.frequencies.size());
            int fw_count = 0, bw_count = 0;
            for (int m = 0; m < std::min(4, n_modes); m++) {
                const char* label = r.whirl_direction(m) > 0 ? "FW" :
                                   (r.whirl_direction(m) < 0 ? "BW" : "??");
                std::cout << "  " << std::setw(3) << k << " | "
                          << std::setw(4) << m << " | "
                          << std::setw(9) << r.frequencies(m) << " | "
                          << std::setw(5) << r.whirl_direction(m) << " | " << label << "\n";
                if (r.whirl_direction(m) > 0) fw_count++;
                if (r.whirl_direction(m) < 0) bw_count++;
            }
            // Should have both FW and BW modes
            EXPECT_GT(fw_count, 0)
                << "k=" << k << " should have forward whirl modes";
            EXPECT_GT(bw_count, 0)
                << "k=" << k << " should have backward whirl modes";
        }
        if (k > 2) break;  // Only check first few NDs
    }
    std::cout << "\n";
}
