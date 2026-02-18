#include "turbomodal/added_mass.hpp"

namespace turbomodal {

// Kwak (1991) Gamma_n coefficients for clamped-center free-edge circular disk
const std::array<double, 11> AddedMassModel::gamma_coefficients_ = {{
    0.6526, 0.3910, 0.2737, 0.2078, 0.1660,
    0.1372, 0.1163, 0.1003, 0.0878, 0.0777,
    0.0695
}};

double AddedMassModel::gamma_n(int n) {
    if (n < 0)
        throw std::invalid_argument("Nodal diameter must be non-negative");
    if (n < 11)
        return gamma_coefficients_[n];
    // Approximate formula for n > 10
    return 0.65 / std::pow(n + 0.05, 0.85);
}

double AddedMassModel::kwak_avmi(int nodal_diameter, double rho_fluid,
                                  double rho_structure, double thickness,
                                  double radius) {
    return gamma_n(nodal_diameter) * (rho_fluid / rho_structure) * (radius / thickness);
}

double AddedMassModel::frequency_ratio(int nodal_diameter, double rho_fluid,
                                        double rho_structure, double thickness,
                                        double radius) {
    double avmi = kwak_avmi(nodal_diameter, rho_fluid, rho_structure, thickness, radius);
    return 1.0 / std::sqrt(1.0 + avmi);
}

SpMatd AddedMassModel::corrected_mass_matrix(
    const SpMatd& M_dry,
    int harmonic_index, double rho_fluid,
    double rho_structure, double thickness, double radius) {
    double avmi = kwak_avmi(harmonic_index, rho_fluid, rho_structure, thickness, radius);
    return M_dry * (1.0 + avmi);
}

// ========================================================================
// PotentialFlowAddedMass
// ========================================================================

PotentialFlowAddedMass::PotentialFlowAddedMass(const Mesh& mesh, double rho_fluid)
    : mesh_(mesh), rho_fluid_(rho_fluid) {}

void PotentialFlowAddedMass::build_projection_matrix() {
    int np = static_cast<int>(meridional_mesh_.panels.size());
    int nw = static_cast<int>(wetted_node_ids_.size());
    if (np == 0 || nw == 0) return;

    N_proj_ = Eigen::MatrixXd::Zero(np, 3 * nw);

    // Determine coordinate axes for cylindrical conversion
    int c1, c2, c_axial;
    if (mesh_.rotation_axis == 0) {
        c1 = 1; c2 = 2; c_axial = 0;
    } else if (mesh_.rotation_axis == 1) {
        c1 = 0; c2 = 2; c_axial = 1;
    } else {
        c1 = 0; c2 = 1; c_axial = 2;
    }

    // For each panel, find the closest wetted node and set up the projection.
    // The projection maps structural displacement (ux, uy, uz) to the panel
    // normal velocity: w = nr*(cos(theta)*u_c1 + sin(theta)*u_c2) + nz*u_axial
    for (int p = 0; p < np; p++) {
        double panel_r = meridional_mesh_.panels[p].r_mid;
        double panel_z = meridional_mesh_.panels[p].z_mid;
        double nr = meridional_mesh_.panels[p].nr;
        double nz = meridional_mesh_.panels[p].nz;

        // Find closest wetted node in (r, z) space
        double best_dist = std::numeric_limits<double>::max();
        int best_w = -1;
        for (int w = 0; w < nw; w++) {
            int nid = wetted_node_ids_[w];
            double x1 = mesh_.nodes(nid, c1);
            double x2 = mesh_.nodes(nid, c2);
            double r = std::sqrt(x1 * x1 + x2 * x2);
            double z = mesh_.nodes(nid, c_axial);
            double d = std::sqrt((r - panel_r) * (r - panel_r) + (z - panel_z) * (z - panel_z));
            if (d < best_dist) {
                best_dist = d;
                best_w = w;
            }
        }
        if (best_w < 0) continue;

        int nid = wetted_node_ids_[best_w];
        double x1 = mesh_.nodes(nid, c1);
        double x2 = mesh_.nodes(nid, c2);
        double r_node = std::sqrt(x1 * x1 + x2 * x2);

        double cos_theta = (r_node > 1e-15) ? x1 / r_node : 1.0;
        double sin_theta = (r_node > 1e-15) ? x2 / r_node : 0.0;

        // w = nr*(cos(theta)*u_c1 + sin(theta)*u_c2) + nz*u_axial
        // DOF ordering in wetted-local: [u_0, u_1, u_2] for 3D Cartesian
        // c1, c2, c_axial map to global DOF indices 0,1,2
        N_proj_(p, 3 * best_w + c1) = nr * cos_theta;
        N_proj_(p, 3 * best_w + c2) = nr * sin_theta;
        N_proj_(p, 3 * best_w + c_axial) = nz;
    }
}

void PotentialFlowAddedMass::precompute(int max_nd) {
    // Step 1: Build meridional mesh from wetted surface
    Eigen::MatrixXd profile = mesh_.get_meridional_profile();
    if (profile.rows() < 2) {
        throw std::runtime_error("PotentialFlowAddedMass: could not extract meridional profile "
                                 "(no wetted surface found)");
    }
    meridional_mesh_ = MeridionalMesh::from_points(profile);

    // Step 2: Get wetted nodes for DOF projection
    wetted_node_ids_ = mesh_.get_wetted_nodes();

    // Step 3: Build projection matrix N
    build_projection_matrix();

    // Step 4: Solve BEM for each ND
    AxiBEMSolver bem;
    auto bem_results = bem.solve_all_nd(max_nd, rho_fluid_, meridional_mesh_);

    // Step 5: Build M_added in sector DOF space for each ND
    int ndof = mesh_.num_dof();
    int nw = static_cast<int>(wetted_node_ids_.size());
    double sector_fraction = 1.0 / mesh_.num_sectors;

    M_added_sector_.resize(max_nd + 1);

    for (int nd = 0; nd <= max_nd; nd++) {
        const auto& M_panel = bem_results[nd].M_panel;
        if (M_panel.rows() == 0) {
            M_added_sector_[nd] = SpMatcd(ndof, ndof);
            continue;
        }

        // M_added_wetted = N^T * M_panel * N  (3*nw x 3*nw, dense-ish)
        Eigen::MatrixXd M_wetted = N_proj_.transpose() * M_panel * N_proj_;

        // Scale by sector fraction
        M_wetted *= sector_fraction;

        // Map to full sector DOF space as sparse complex matrix
        std::vector<TripletC> trips;
        for (int i = 0; i < 3 * nw; i++) {
            int gi = 3 * wetted_node_ids_[i / 3] + (i % 3);
            for (int j = 0; j < 3 * nw; j++) {
                double val = M_wetted(i, j);
                if (std::abs(val) > 1e-20) {
                    int gj = 3 * wetted_node_ids_[j / 3] + (j % 3);
                    trips.emplace_back(gi, gj, std::complex<double>(val, 0.0));
                }
            }
        }

        SpMatcd M(ndof, ndof);
        M.setFromTriplets(trips.begin(), trips.end());
        M_added_sector_[nd] = std::move(M);
    }

    max_nd_ = max_nd;
    precomputed_ = true;
}

SpMatcd PotentialFlowAddedMass::get_sector_added_mass(int nd) const {
    if (!precomputed_ || nd < 0 || nd > max_nd_) {
        return SpMatcd(mesh_.num_dof(), mesh_.num_dof());
    }
    return M_added_sector_[nd];
}

}  // namespace turbomodal
