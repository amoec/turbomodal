#pragma once

#include "turbomodal/common.hpp"
#include "turbomodal/mesh.hpp"
#include "turbomodal/material.hpp"
#include "turbomodal/assembler.hpp"
#include "turbomodal/modal_solver.hpp"
#include "turbomodal/added_mass.hpp"

namespace turbomodal {

struct FluidConfig {
    enum class Type {
        NONE,
        GAS_AIC,
        KWAK_ANALYTICAL,       // Legacy Kwak (1991) flat-disk formula
        LIQUID_ANALYTICAL = KWAK_ANALYTICAL,  // Backward-compatible alias
        POTENTIAL_FLOW_BEM,    // Axisymmetric BEM potential flow
        LIQUID_ACOUSTIC_FEM
    };

    Type type = Type::NONE;

    double fluid_density = 0.0;    // kg/m^3
    double disk_radius = 0.0;      // m (Kwak formula only)
    double disk_thickness = 0.0;   // m (Kwak formula only)
    double speed_of_sound = 0.0;   // m/s
};

// Stationary-frame FW/BW split result (computed from rotating-frame eigenvalues)
struct StationaryFrameResult {
    Eigen::VectorXd frequencies;     // Hz, sorted ascending
    Eigen::VectorXi whirl_direction; // +1 FW, -1 BW, 0 standing
};

class CyclicSymmetrySolver {
public:
    CyclicSymmetrySolver(const Mesh& mesh, const Material& mat,
                         const FluidConfig& fluid = FluidConfig(),
                         bool apply_hub_constraint = true);

    std::vector<ModalResult> solve_at_rpm(
        double rpm, int num_modes_per_harmonic,
        const std::vector<int>& harmonic_indices = {},
        int max_threads = 0);

    std::vector<std::vector<ModalResult>> solve_rpm_sweep(
        const Eigen::VectorXd& rpm_values, int num_modes_per_harmonic,
        const std::vector<int>& harmonic_indices = {},
        int max_threads = 0);

    void export_campbell_csv(const std::string& filename,
                             const std::vector<std::vector<ModalResult>>& results);
    void export_zzenf_csv(const std::string& filename,
                          const std::vector<ModalResult>& results_at_rpm);
    void export_mode_shape_vtk(const std::string& filename,
                                const ModalResult& result, int mode_index);

    // Convert rotating-frame modal result to stationary-frame FW/BW frequencies.
    // For k=0 and k=N/2: no split, whirl=0 (standing waves).
    // For 0 < k < N/2: each mode splits into FW and BW, returning 2*N modes.
    static StationaryFrameResult compute_stationary_frame(
        const ModalResult& rotating_result, int num_sectors);

    // Apply cyclic symmetry boundary conditions for harmonic index k
    // Returns reduced (complex) K_k and M_k matrices
    std::pair<SpMatcd, SpMatcd> apply_cyclic_bc_public(
        int harmonic_index,
        const SpMatd& K_full,
        const SpMatd& M_full) const {
        return apply_cyclic_bc(harmonic_index, K_full, M_full);
    }

    // Build the cyclic transformation matrix T for harmonic index k
    SpMatcd get_transformation(int harmonic_index) const {
        return build_cyclic_transformation(harmonic_index);
    }

private:
    const Mesh& mesh_;
    Material mat_;
    FluidConfig fluid_;
    bool apply_hub_constraint_;
    GlobalAssembler assembler_;

    // DOF classification for cyclic symmetry
    std::vector<int> interior_dofs_;
    std::vector<int> left_dofs_;
    std::vector<int> right_dofs_;

    // Cached base K,M (geometry + material only, no RPM dependence)
    SpMatd K_base_;
    SpMatd M_base_;
    bool base_assembled_ = false;

    // BEM added mass cache (precomputed once per geometry, reused across RPM points)
    std::unique_ptr<PotentialFlowAddedMass> bem_added_mass_;
    std::vector<SpMatcd> M_added_free_cache_;  // per-harmonic, in free-DOF space
    bool bem_precomputed_ = false;

    void classify_dofs();

    SpMatcd build_cyclic_transformation(int harmonic_index) const;

    std::pair<SpMatcd, SpMatcd> apply_cyclic_bc(
        int harmonic_index,
        const SpMatd& K_full,
        const SpMatd& M_full) const;

    double added_mass_factor(int harmonic_index) const;

    Eigen::VectorXd static_centrifugal(double omega);
};

}  // namespace turbomodal
