#pragma once

#include <functional>

#include "turbomodal/common.hpp"
#include "turbomodal/mesh.hpp"
#include "turbomodal/material.hpp"
#include "turbomodal/assembler.hpp"
#include "turbomodal/modal_solver.hpp"
#include "turbomodal/added_mass.hpp"

namespace turbomodal {

// Callback invoked from worker threads as each harmonic finishes.
// Args: completed, total, harmonic_index k, did the eigensolver converge.
using ProgressCallback = std::function<void(int completed, int total, int harmonic_k, bool converged)>;

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

// Condition for parametric sweep
struct ParametricCondition {
    double rpm = 0.0;
    double temperature = 293.15;   // K (default room temp)
    std::vector<double> mistuning; // per-blade frequency deviations (empty = tuned)
};

class CyclicSymmetrySolver {
public:
    CyclicSymmetrySolver(const Mesh& mesh, const Material& mat,
                         const FluidConfig& fluid = FluidConfig(),
                         bool apply_hub_constraint = true);

    // Constructor with arbitrary constraint groups (replaces hub_constraint boolean)
    CyclicSymmetrySolver(const Mesh& mesh, const Material& mat,
                         const std::vector<ConstraintGroup>& constraints,
                         const FluidConfig& fluid = FluidConfig());

    std::vector<ModalResult> solve_at_rpm(
        double rpm, int num_modes_per_harmonic,
        const std::vector<int>& harmonic_indices = {},
        int max_threads = 0,
        bool include_coriolis = false,
        double min_frequency = 0.0,
        ProgressCallback progress_cb = nullptr,
        bool allow_condensation = false,
        double memory_reserve_fraction = 0.2);

    std::vector<std::vector<ModalResult>> solve_rpm_sweep(
        const Eigen::VectorXd& rpm_values, int num_modes_per_harmonic,
        const std::vector<int>& harmonic_indices = {},
        int max_threads = 0,
        bool include_coriolis = false,
        double min_frequency = 0.0);

    // Fast parametric sweep: precompute once, then scale for each condition.
    // Returns one vector<ModalResult> (all harmonics) per condition.
    std::vector<std::vector<ModalResult>> solve_parametric(
        const std::vector<ParametricCondition>& conditions,
        int num_modes_per_harmonic,
        const std::vector<int>& harmonic_indices = {},
        int max_threads = 0,
        bool include_coriolis = false,
        double min_frequency = 0.0,
        bool allow_condensation = false,
        double memory_reserve_fraction = 0.2,
        ProgressCallback progress_cb = nullptr);

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
    std::vector<ConstraintGroup> constraints_;
    bool has_constraints_ = false;
    GlobalAssembler assembler_;

    // Build the full list of constrained DOF indices from all constraint groups
    std::vector<int> build_constrained_dofs() const;

    // DOF classification for cyclic symmetry
    std::vector<int> interior_dofs_;
    std::vector<int> left_dofs_;
    std::vector<int> right_dofs_;

    // Cached base K,M (geometry + material only, no RPM dependence)
    SpMatd K_base_;
    SpMatd M_base_;
    bool base_assembled_ = false;

    // --- Precomputed cyclic projections (Feature 3 Phase 1) ---
    // Cached after first call to precompute_cyclic_projections().
    SpMatcd Kcf_, Kpf_, Kpf_H_;     // Stiffness projections (free DOF space)
    SpMatcd Mcf_, Mpf_, Mpf_H_;     // Mass projections (free DOF space)
    SpMatcd T0_;                      // T(0) transformation
    std::set<int> right_dof_set_;
    std::set<int> hub_red_set_;
    std::vector<int> free_reduced_map_;
    int n_reduced_ = 0;
    int n_free_ = 0;
    bool projections_precomputed_ = false;

    // Precompute real K/M for k=0 and k=N/2 (P8 optimization)
    SpMatd K0_real_, M0_real_;
    SpMatd Khalf_real_, Mhalf_real_;
    bool real_matrices_precomputed_ = false;

    void precompute_real_matrices();

    // Precompute cyclic projections from K_eff and M_sector.
    // Caches Kcf_, Kpf_, Mcf_, Mpf_, etc. for fast per-harmonic assembly.
    void precompute_cyclic_projections(const SpMatd& K_eff, const SpMatd& M_sector);

    // --- Unit-speed rotation effects for parametric scaling ---
    SpMatd K_omega_unit_;     // K_omega at omega=1 rad/s (scales as omega^2)
    bool K_omega_unit_computed_ = false;

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
    Eigen::VectorXd static_centrifugal_scaled(double omega, double E_scale);
};

}  // namespace turbomodal
