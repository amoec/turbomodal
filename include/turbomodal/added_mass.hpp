#pragma once

#include "turbomodal/common.hpp"
#include "turbomodal/mesh.hpp"
#include "turbomodal/potential_flow.hpp"
#include <memory>

namespace turbomodal {

// Legacy Kwak (1991) analytical model — hardcoded gamma coefficients for flat disk
class AddedMassModel {
public:
    static double kwak_avmi(int nodal_diameter, double rho_fluid,
                            double rho_structure, double thickness,
                            double radius);

    static double frequency_ratio(int nodal_diameter, double rho_fluid,
                                  double rho_structure, double thickness,
                                  double radius);

    static SpMatd corrected_mass_matrix(
        const SpMatd& M_dry,
        int harmonic_index, double rho_fluid,
        double rho_structure, double thickness, double radius);

private:
    static const std::array<double, 11> gamma_coefficients_;
    static double gamma_n(int n);
};

// Potential flow BEM added mass — geometry-based, replaces Kwak for real problems
class PotentialFlowAddedMass {
public:
    PotentialFlowAddedMass(const Mesh& mesh, double rho_fluid);

    // Precompute BEM added mass for harmonics 0..max_nd.
    // Builds meridional mesh, solves BEM per ND, computes projection to sector DOFs.
    void precompute(int max_nd);

    // Get added mass matrix in sector DOF space for a given ND.
    // Returns sparse complex matrix (same size as sector mass matrix).
    // Scaled by 1/num_sectors (one sector's share of the full-ring added mass).
    SpMatcd get_sector_added_mass(int nd) const;

    bool is_precomputed() const { return precomputed_; }
    int max_precomputed_nd() const { return max_nd_; }

    // Get the meridional mesh (for testing/debugging)
    const MeridionalMesh& meridional_mesh() const { return meridional_mesh_; }

private:
    const Mesh& mesh_;
    double rho_fluid_;
    bool precomputed_ = false;
    int max_nd_ = -1;

    MeridionalMesh meridional_mesh_;
    std::vector<int> wetted_nodes_;

    // Projection matrix: maps sector DOF displacement to panel normal velocities
    // N_proj: N_panel x (3 * N_wetted_nodes)
    Eigen::MatrixXd N_proj_;

    // Mapping from wetted node index to global node ID
    std::vector<int> wetted_node_ids_;

    // Per-ND added mass in sector DOF space (sparse, complex)
    std::vector<SpMatcd> M_added_sector_;

    void build_projection_matrix();
};

}  // namespace turbomodal
