#pragma once

#include "turbomodal/common.hpp"

#include <vector>
#include <set>

namespace turbomodal {

struct CondensationResult {
    SpMatd K_reduced;
    SpMatd M_reduced;
    SpMatd T_c;            // Transformation: u_full = T_c * u_master
    std::vector<int> master_dofs;
    int original_size;
};

// Compute the target DOF count that fits within the given memory budget.
// Uses the same cost model as estimate_per_harmonic_bytes().
int compute_target_dofs(int n_free, int nev, size_t available_bytes);

// Select master DOFs for Guyan reduction.
// Mandatory: all DOFs belonging to boundary nodes (left, right, hub) are retained.
// Interior DOFs are ranked by diagonal mass participation and retained until
// n_target is reached.
std::vector<int> select_master_dofs(
    const SpMatd& M,
    const std::set<int>& boundary_dofs,
    int n_target);

// Perform Guyan (static) condensation: K_mm - K_ms * K_ss^{-1} * K_sm.
// Returns reduced K, M, and the transformation matrix T_c that maps
// master DOFs back to full DOF space (u = T_c * u_master).
CondensationResult condense(
    const SpMatd& K,
    const SpMatd& M,
    const std::vector<int>& master_dofs);

// Expand condensed mode shapes back to the full DOF space.
// full_modes = T_c * reduced_modes
Eigen::MatrixXcd expand_modes(
    const SpMatd& T_c,
    const Eigen::MatrixXcd& reduced_modes);

}  // namespace turbomodal
