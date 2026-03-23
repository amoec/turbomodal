#include "turbomodal/cyclic_solver.hpp"
#include "turbomodal/static_condensation.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <algorithm>
#include <numeric>
#include <optional>
#include <Eigen/SparseLU>
#include <atomic>
#include <mutex>
#include <thread>
#include <cmath>

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace turbomodal {

// Total physical memory (bytes).  Returns 0 on unsupported platforms.
static size_t total_system_memory() {
#if defined(__APPLE__)
    int64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0)
        return static_cast<size_t>(mem);
    return 0;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0)
        return static_cast<size_t>(si.totalram) * si.mem_unit;
    return 0;
#elif defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return static_cast<size_t>(ms.ullTotalPhys);
    return 0;
#else
    return 0;
#endif
}

// Estimate peak memory (bytes) for one shift-invert eigensolver call.
//
// Two estimation modes:
//   1. nnz-based (preferred): when the actual nnz of the projected operator is
//      known, use empirical fill ratio for SimplicialLDLT with AMD ordering.
//      For 3D FE meshes this is typically 8-15x; we use 12x as conservative default.
//   2. dimension-based (fallback): C * n^(4/3) heuristic for when nnz is unknown.
//
// Entry size: 16 bytes (complex double) for complex Hermitian,
//             8 bytes (double) for k=0 / k=N/2 real-valued harmonics.
//
// Lanczos workspace: V and MV matrices, each n × (ncv+1).
static size_t estimate_per_harmonic_bytes(int n_dofs, int nev = 20,
                                          int64_t operator_nnz = 0,
                                          int entry_bytes = 16) {
    double n = static_cast<double>(n_dofs);

    size_t factorization;
    if (operator_nnz > 0) {
        // LDLT fill estimate: fill_ratio * nnz * entry_size.
        // Eigen's SimplicialLDLT uses AMD ordering; empirical fill ratio
        // for 3D FE sparsity patterns is typically 8-15x.  Use 12x.
        constexpr double fill_ratio = 12.0;
        factorization = static_cast<size_t>(
            fill_ratio * static_cast<double>(operator_nnz) * entry_bytes);
    } else {
        // Fallback: 800 * n^(4/3) calibrated so n=40000 → ~1 GB
        factorization = static_cast<size_t>(800.0 * std::pow(n, 4.0 / 3.0));
    }

    // Lanczos V and MV: 2 matrices × n × (ncv+1) × entry_bytes
    int ncv = std::min(4 * nev + 1, n_dofs);
    size_t lanczos = static_cast<size_t>(2.0 * n * (ncv + 1) * entry_bytes);
    return factorization + lanczos;
}

CyclicSymmetrySolver::CyclicSymmetrySolver(
    const Mesh& mesh, const Material& mat, const FluidConfig& fluid,
    bool apply_hub_constraint)
    : mesh_(mesh), mat_(mat), fluid_(fluid) {
    // Map legacy boolean to constraint groups
    if (apply_hub_constraint) {
        const NodeSet* hub = mesh_.find_node_set("hub_constraint");
        if (hub) {
            ConstraintGroup cg;
            cg.name = "hub_constraint";
            cg.node_ids = hub->node_ids;
            cg.type = BCType::FIXED;
            constraints_.push_back(std::move(cg));
            has_constraints_ = true;
        }
    }
    classify_dofs();
}

CyclicSymmetrySolver::CyclicSymmetrySolver(
    const Mesh& mesh, const Material& mat,
    const std::vector<ConstraintGroup>& constraints,
    const FluidConfig& fluid)
    : mesh_(mesh), mat_(mat), fluid_(fluid),
      constraints_(constraints),
      has_constraints_(!constraints.empty()) {
    classify_dofs();
}

std::vector<int> CyclicSymmetrySolver::build_constrained_dofs() const {
    std::set<int> dof_set;
    for (const auto& cg : constraints_) {
        for (int node_id : cg.node_ids) {
            switch (cg.type) {
                case BCType::FIXED:
                    dof_set.insert(3 * node_id);
                    dof_set.insert(3 * node_id + 1);
                    dof_set.insert(3 * node_id + 2);
                    break;
                case BCType::DISPLACEMENT:
                    for (int k = 0; k < 3; k++) {
                        if (cg.constrained_components[k]) {
                            dof_set.insert(3 * node_id + k);
                        }
                    }
                    break;
                case BCType::FRICTIONLESS: {
                    // Constrain the DOF component most aligned with the surface normal.
                    // This is exact for axis-aligned normals (common case).
                    Eigen::Vector3d n = cg.surface_normal.normalized();
                    int max_comp = 0;
                    double max_val = std::abs(n(0));
                    for (int k = 1; k < 3; k++) {
                        if (std::abs(n(k)) > max_val) {
                            max_val = std::abs(n(k));
                            max_comp = k;
                        }
                    }
                    dof_set.insert(3 * node_id + max_comp);
                    break;
                }
                case BCType::ELASTIC_SUPPORT:
                    // Springs add to K matrix, they don't eliminate DOFs
                    break;
                case BCType::CYLINDRICAL: {
                    // Constrain radial/tangential/axial DOFs in cylindrical coordinates.
                    // constrained_components[0]=radial, [1]=tangential, [2]=axial.
                    Eigen::Vector3d axis = cg.cylinder_axis.normalized();
                    Eigen::Vector3d pos = mesh_.nodes.row(node_id).transpose() - cg.cylinder_origin;
                    Eigen::Vector3d axial_proj = pos.dot(axis) * axis;
                    Eigen::Vector3d radial = pos - axial_proj;
                    double r = radial.norm();
                    if (r < 1e-12) {
                        // Node is on the cylinder axis — constrain all requested DOFs
                        for (int k = 0; k < 3; k++) {
                            if (cg.constrained_components[k])
                                dof_set.insert(3 * node_id + k);
                        }
                        break;
                    }
                    radial /= r;
                    Eigen::Vector3d tangential = axis.cross(radial).normalized();

                    // Map cylindrical directions to most-aligned Cartesian DOF
                    Eigen::Vector3d dirs[3] = {radial, tangential, axis};
                    for (int c = 0; c < 3; c++) {
                        if (!cg.constrained_components[c]) continue;
                        int best = 0;
                        double best_val = std::abs(dirs[c](0));
                        for (int k = 1; k < 3; k++) {
                            if (std::abs(dirs[c](k)) > best_val) {
                                best_val = std::abs(dirs[c](k));
                                best = k;
                            }
                        }
                        dof_set.insert(3 * node_id + best);
                    }
                    break;
                }
            }
        }
    }
    return std::vector<int>(dof_set.begin(), dof_set.end());
}

SpMatd CyclicSymmetrySolver::build_spring_stiffness() const {
    int ndof = mesh_.num_dof();
    std::vector<Eigen::Triplet<double>> trips;
    for (const auto& cg : constraints_) {
        if (cg.type != BCType::ELASTIC_SUPPORT) continue;
        for (int node_id : cg.node_ids) {
            for (int k = 0; k < 3; k++) {
                if (cg.spring_stiffness(k) > 0.0) {
                    trips.emplace_back(3 * node_id + k, 3 * node_id + k,
                                       cg.spring_stiffness(k));
                }
            }
        }
    }
    SpMatd K_spring(ndof, ndof);
    if (!trips.empty()) {
        K_spring.setFromTriplets(trips.begin(), trips.end());
    }
    return K_spring;
}

void CyclicSymmetrySolver::classify_dofs() {
    interior_dofs_.clear();
    left_dofs_.clear();
    right_dofs_.clear();

    // Build sets of boundary node IDs for fast lookup
    std::set<int> left_nodes(mesh_.left_boundary.begin(), mesh_.left_boundary.end());
    std::set<int> right_nodes(mesh_.right_boundary.begin(), mesh_.right_boundary.end());

    for (int i = 0; i < mesh_.num_nodes(); i++) {
        bool is_left = left_nodes.count(i) > 0;
        bool is_right = right_nodes.count(i) > 0;

        for (int d = 0; d < 3; d++) {
            int dof = 3 * i + d;
            if (is_right) {
                right_dofs_.push_back(dof);
            } else if (is_left) {
                left_dofs_.push_back(dof);
            } else {
                interior_dofs_.push_back(dof);
            }
        }
    }
}

SpMatcd CyclicSymmetrySolver::build_cyclic_transformation(int harmonic_index) const {
    int full_ndof = mesh_.num_dof();
    int reduced_ndof = static_cast<int>(interior_dofs_.size() + left_dofs_.size());

    double alpha = 2.0 * PI / mesh_.num_sectors;
    std::complex<double> phase(std::cos(harmonic_index * alpha),
                                std::sin(harmonic_index * alpha));

    // Build mapping from full DOF index to reduced DOF index for interior+left DOFs
    std::map<int, int> full_to_reduced;
    int idx = 0;
    for (int dof : interior_dofs_) {
        full_to_reduced[dof] = idx++;
    }
    for (int dof : left_dofs_) {
        full_to_reduced[dof] = idx++;
    }

    // Build mapping from right boundary node to left partner
    std::map<int, int> right_node_to_left_node;
    for (const auto& [left_node, right_node] : mesh_.matched_pairs) {
        right_node_to_left_node[right_node] = left_node;
    }

    // Assemble T
    std::vector<TripletC> trips;
    trips.reserve(full_ndof);

    for (int dof : interior_dofs_) {
        trips.emplace_back(dof, full_to_reduced[dof], std::complex<double>(1.0, 0.0));
    }
    for (int dof : left_dofs_) {
        trips.emplace_back(dof, full_to_reduced[dof], std::complex<double>(1.0, 0.0));
    }
    // Right boundary DOFs: apply phase factor AND coordinate rotation.
    // In cyclic symmetry with Cartesian DOFs, the displacement at the right
    // boundary (at angle alpha from left) relates to the left boundary as:
    //   u_right = e^{ik*alpha} * R(alpha) * u_left
    // where R(alpha) rotates the two in-plane displacement components by the
    // sector angle.  The axial component (along the rotation axis) is unchanged.
    double cos_a = std::cos(alpha);
    double sin_a = std::sin(alpha);

    // Determine DOF indices for the rotation plane and the axial direction.
    // rotation_axis=0 (X): rotate DOFs 1,2 (Y,Z); axial DOF 0 (X)
    // rotation_axis=1 (Y): rotate DOFs 0,2 (X,Z); axial DOF 1 (Y)
    // rotation_axis=2 (Z): rotate DOFs 0,1 (X,Y); axial DOF 2 (Z)
    int dof_c1, dof_c2, dof_ax;
    if (mesh_.rotation_axis == 0) {
        dof_c1 = 1; dof_c2 = 2; dof_ax = 0;
    } else if (mesh_.rotation_axis == 1) {
        dof_c1 = 0; dof_c2 = 2; dof_ax = 1;
    } else {
        dof_c1 = 0; dof_c2 = 1; dof_ax = 2;
    }

    // Process right boundary nodes (all 3 DOFs per node together)
    std::set<int> right_nodes_processed;
    for (int right_dof : right_dofs_) {
        int right_node = right_dof / 3;
        if (right_nodes_processed.count(right_node)) continue;
        right_nodes_processed.insert(right_node);

        auto it = right_node_to_left_node.find(right_node);
        if (it == right_node_to_left_node.end()) {
            throw std::runtime_error("Right boundary node " + std::to_string(right_node) +
                                     " has no matched left partner");
        }
        int left_node = it->second;

        int right_base = 3 * right_node;
        int left_base_red_c1 = full_to_reduced.at(3 * left_node + dof_c1);
        int left_base_red_c2 = full_to_reduced.at(3 * left_node + dof_c2);
        int left_base_red_ax = full_to_reduced.at(3 * left_node + dof_ax);

        // u_c1_right = phase * (cos_a * u_c1_left - sin_a * u_c2_left)
        trips.emplace_back(right_base + dof_c1, left_base_red_c1, phase * cos_a);
        trips.emplace_back(right_base + dof_c1, left_base_red_c2, phase * (-sin_a));

        // u_c2_right = phase * (sin_a * u_c1_left + cos_a * u_c2_left)
        trips.emplace_back(right_base + dof_c2, left_base_red_c1, phase * sin_a);
        trips.emplace_back(right_base + dof_c2, left_base_red_c2, phase * cos_a);

        // u_axial_right = phase * u_axial_left (no rotation for axial component)
        trips.emplace_back(right_base + dof_ax, left_base_red_ax, phase);
    }

    SpMatcd T(full_ndof, reduced_ndof);
    T.setFromTriplets(trips.begin(), trips.end());
    return T;
}

std::pair<SpMatcd, SpMatcd> CyclicSymmetrySolver::apply_cyclic_bc(
    int harmonic_index,
    const SpMatd& K_full,
    const SpMatd& M_full) const {

    SpMatcd T = build_cyclic_transformation(harmonic_index);

    // Convert K, M to complex sparse
    int n = static_cast<int>(K_full.rows());
    SpMatcd K_complex(n, n), M_complex(n, n);
    {
        std::vector<TripletC> kc, mc;
        kc.reserve(K_full.nonZeros());
        mc.reserve(M_full.nonZeros());
        for (int k = 0; k < K_full.outerSize(); ++k) {
            for (SpMatd::InnerIterator it(K_full, k); it; ++it)
                kc.emplace_back(it.row(), it.col(), std::complex<double>(it.value(), 0.0));
        }
        for (int k = 0; k < M_full.outerSize(); ++k) {
            for (SpMatd::InnerIterator it(M_full, k); it; ++it)
                mc.emplace_back(it.row(), it.col(), std::complex<double>(it.value(), 0.0));
        }
        K_complex.setFromTriplets(kc.begin(), kc.end());
        M_complex.setFromTriplets(mc.begin(), mc.end());
    }

    SpMatcd T_H = T.adjoint();
    SpMatcd K_k = T_H * K_complex * T;
    SpMatcd M_k = T_H * M_complex * T;

    return {K_k, M_k};
}

double CyclicSymmetrySolver::added_mass_factor(int harmonic_index) const {
    if (fluid_.type != FluidConfig::Type::KWAK_ANALYTICAL || fluid_.fluid_density <= 0.0) {
        return 1.0;
    }
    return AddedMassModel::frequency_ratio(
        harmonic_index, fluid_.fluid_density, mat_.rho,
        fluid_.disk_thickness, fluid_.disk_radius);
}

StationaryFrameResult CyclicSymmetrySolver::compute_stationary_frame(
    const ModalResult& rotating_result, int num_sectors) {

    StationaryFrameResult sf;
    int k = rotating_result.harmonic_index;
    int max_k = num_sectors / 2;
    double omega = rotating_result.rpm * 2.0 * PI / 60.0;
    int num_modes = static_cast<int>(rotating_result.frequencies.size());

    // Check if Coriolis splitting is already present (whirl_direction != 0)
    bool has_coriolis = false;
    for (int m = 0; m < num_modes; m++) {
        if (rotating_result.whirl_direction(m) != 0) {
            has_coriolis = true;
            break;
        }
    }

    if (has_coriolis && std::abs(omega) > 0.0 && k > 0) {
        // Coriolis modes: each mode already has FW/BW designation.
        // Convert to stationary frame: FW adds k*Omega/(2pi), BW subtracts.
        double k_omega_hz = k * std::abs(omega) / (2.0 * PI);
        sf.frequencies.resize(num_modes);
        sf.whirl_direction.resize(num_modes);

        for (int m = 0; m < num_modes; m++) {
            int whirl = rotating_result.whirl_direction(m);
            sf.frequencies(m) = std::abs(
                rotating_result.frequencies(m) + whirl * k_omega_hz);
            sf.whirl_direction(m) = whirl;
        }

        // Sort by ascending frequency
        std::vector<int> sort_idx(num_modes);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) {
            return sf.frequencies(a) < sf.frequencies(b);
        });
        Eigen::VectorXd sorted_f(num_modes);
        Eigen::VectorXi sorted_w(num_modes);
        for (int i = 0; i < num_modes; i++) {
            sorted_f(i) = sf.frequencies(sort_idx[i]);
            sorted_w(i) = sf.whirl_direction(sort_idx[i]);
        }
        sf.frequencies = sorted_f;
        sf.whirl_direction = sorted_w;
    } else if (std::abs(omega) > 0.0 && k > 0 && k < max_k) {
        // Kinematic splitting: each rotating-frame mode produces FW + BW pair
        double k_omega_hz = k * std::abs(omega) / (2.0 * PI);

        Eigen::VectorXd all_freqs(2 * num_modes);
        Eigen::VectorXi all_whirl(2 * num_modes);

        for (int m = 0; m < num_modes; m++) {
            all_freqs(2 * m) = rotating_result.frequencies(m) + k_omega_hz;
            all_whirl(2 * m) = 1;   // FW
            all_freqs(2 * m + 1) = std::abs(rotating_result.frequencies(m) - k_omega_hz);
            all_whirl(2 * m + 1) = -1;  // BW
        }

        // Sort by frequency
        std::vector<int> sort_idx(2 * num_modes);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) {
            return all_freqs(a) < all_freqs(b);
        });

        sf.frequencies.resize(2 * num_modes);
        sf.whirl_direction.resize(2 * num_modes);
        for (int i = 0; i < 2 * num_modes; i++) {
            sf.frequencies(i) = all_freqs(sort_idx[i]);
            sf.whirl_direction(i) = all_whirl(sort_idx[i]);
        }
    } else {
        // k=0, k=N/2, or omega=0: no splitting
        sf.frequencies = rotating_result.frequencies;
        sf.whirl_direction = Eigen::VectorXi::Zero(num_modes);
    }

    return sf;
}

Eigen::VectorXd CyclicSymmetrySolver::static_centrifugal(double omega) {
    // Solve K * u = F_centrifugal with k=0 cyclic BCs and hub constraints.
    // Cyclic BCs are essential: without them the sector boundaries are free,
    // hoop stress doesn't develop, and K_sigma is far too small.
    Eigen::Vector3d rot_axis = Eigen::Vector3d::Unit(mesh_.rotation_axis);
    Eigen::VectorXd F = assembler_.assemble_centrifugal_load(mesh_, mat_, omega, rot_axis);

    SpMatd K = assembler_.K();
    int ndof = mesh_.num_dof();

    // Step 1: Apply k=0 cyclic BCs (axisymmetric static deformation)
    // For k=0, phase=1 so T is real: u_right = R(alpha) * u_left
    SpMatcd T_complex = build_cyclic_transformation(0);
    int reduced_ndof = static_cast<int>(T_complex.cols());

    SpMatd T_real(ndof, reduced_ndof);
    {
        std::vector<Triplet> trips;
        trips.reserve(T_complex.nonZeros());
        for (int col = 0; col < T_complex.outerSize(); ++col) {
            for (SpMatcd::InnerIterator it(T_complex, col); it; ++it) {
                double rv = it.value().real();
                if (std::abs(rv) > 1e-15)
                    trips.emplace_back(static_cast<int>(it.row()),
                                       static_cast<int>(it.col()), rv);
            }
        }
        T_real.setFromTriplets(trips.begin(), trips.end());
    }

    // Project K and F to cyclic-reduced space
    SpMatd K_cyc = (T_real.transpose() * K * T_real).pruned(1e-15);
    Eigen::VectorXd F_cyc = T_real.transpose() * F;

    // Step 2: Find constrained DOFs in the cyclic-reduced space
    std::vector<int> reduced_to_full;
    reduced_to_full.reserve(interior_dofs_.size() + left_dofs_.size());
    for (int dof : interior_dofs_) reduced_to_full.push_back(dof);
    for (int dof : left_dofs_) reduced_to_full.push_back(dof);

    std::vector<int> constrained_dofs = build_constrained_dofs();
    std::set<int> constrained_full_set(constrained_dofs.begin(), constrained_dofs.end());

    std::set<int> hub_red_set;
    for (int r = 0; r < reduced_ndof; r++) {
        if (constrained_full_set.count(reduced_to_full[r]) > 0) {
            hub_red_set.insert(r);
        }
    }

    // Step 3: Eliminate constrained DOFs from cyclic-reduced system
    std::vector<int> free_cyc;
    std::vector<int> cyc_to_free(reduced_ndof, -1);
    for (int i = 0; i < reduced_ndof; i++) {
        if (hub_red_set.count(i) == 0) {
            cyc_to_free[i] = static_cast<int>(free_cyc.size());
            free_cyc.push_back(i);
        }
    }

    int n_free = static_cast<int>(free_cyc.size());
    if (n_free == 0) {
        return Eigen::VectorXd::Zero(ndof);
    }

    SpMatd K_free(n_free, n_free);
    {
        std::vector<Triplet> trips;
        trips.reserve(K_cyc.nonZeros());
        for (int col = 0; col < K_cyc.outerSize(); ++col) {
            for (SpMatd::InnerIterator it(K_cyc, col); it; ++it) {
                int ri = cyc_to_free[it.row()];
                int rj = cyc_to_free[it.col()];
                if (ri >= 0 && rj >= 0) {
                    trips.emplace_back(ri, rj, it.value());
                }
            }
        }
        K_free.setFromTriplets(trips.begin(), trips.end());
    }

    Eigen::VectorXd F_free(n_free);
    for (int i = 0; i < n_free; i++) {
        F_free(i) = F_cyc(free_cyc[i]);
    }

    // Step 4: Solve K_free * u_free = F_free
    Eigen::SparseLU<SpMatd> solver;
    solver.compute(K_free);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("SparseLU decomposition failed in static_centrifugal");
    }
    Eigen::VectorXd u_free = solver.solve(F_free);

    // Step 5: Expand to cyclic-reduced, then to full DOF space via T
    Eigen::VectorXd u_cyc = Eigen::VectorXd::Zero(reduced_ndof);
    for (int i = 0; i < n_free; i++) {
        u_cyc(free_cyc[i]) = u_free(i);
    }

    Eigen::VectorXd u = T_real * u_cyc;
    return u;
}

void CyclicSymmetrySolver::precompute_cyclic_projections(
    const SpMatd& K_eff, const SpMatd& M_sector) {

    int ndof_full = mesh_.num_dof();

    // Build constrained DOFs and reduced-to-full map
    std::vector<int> constrained_dofs = build_constrained_dofs();
    std::vector<int> reduced_to_full;
    reduced_to_full.reserve(interior_dofs_.size() + left_dofs_.size());
    for (int dof : interior_dofs_) reduced_to_full.push_back(dof);
    for (int dof : left_dofs_) reduced_to_full.push_back(dof);

    std::set<int> constrained_set(constrained_dofs.begin(), constrained_dofs.end());
    hub_red_set_.clear();
    for (int r = 0; r < static_cast<int>(reduced_to_full.size()); r++) {
        if (constrained_set.count(reduced_to_full[r]) > 0) {
            hub_red_set_.insert(r);
        }
    }

    // Convert K and M to complex
    SpMatcd K_complex(ndof_full, ndof_full), M_complex(ndof_full, ndof_full);
    {
        std::vector<TripletC> kc, mc;
        kc.reserve(K_eff.nonZeros());
        mc.reserve(M_sector.nonZeros());
        for (int col = 0; col < K_eff.outerSize(); ++col)
            for (SpMatd::InnerIterator it(K_eff, col); it; ++it)
                kc.emplace_back(it.row(), it.col(), std::complex<double>(it.value(), 0.0));
        for (int col = 0; col < M_sector.outerSize(); ++col)
            for (SpMatd::InnerIterator it(M_sector, col); it; ++it)
                mc.emplace_back(it.row(), it.col(), std::complex<double>(it.value(), 0.0));
        K_complex.setFromTriplets(kc.begin(), kc.end());
        M_complex.setFromTriplets(mc.begin(), mc.end());
    }

    // Precompute T0 and fast-path projections
    double alpha = 2.0 * PI / mesh_.num_sectors;
    right_dof_set_.clear();
    right_dof_set_.insert(right_dofs_.begin(), right_dofs_.end());

    T0_ = build_cyclic_transformation(0);
    SpMatcd T0_H = T0_.adjoint();
    n_reduced_ = static_cast<int>(T0_.cols());

    // Partition by right-DOF membership
    auto partition = [&](const SpMatcd& A) {
        std::vector<TripletC> ts, tnr;
        for (int col = 0; col < A.outerSize(); ++col) {
            bool cr = right_dof_set_.count(col) > 0;
            for (SpMatcd::InnerIterator it(A, col); it; ++it) {
                bool rr = right_dof_set_.count(static_cast<int>(it.row())) > 0;
                if (rr == cr)
                    ts.emplace_back(it.row(), it.col(), it.value());
                else if (!rr && cr)
                    tnr.emplace_back(it.row(), it.col(), it.value());
            }
        }
        int sz = static_cast<int>(A.rows());
        SpMatcd As(sz, sz), Anr(sz, sz);
        As.setFromTriplets(ts.begin(), ts.end());
        Anr.setFromTriplets(tnr.begin(), tnr.end());
        return std::make_pair(std::move(As), std::move(Anr));
    };

    auto [K_same, K_nr] = partition(K_complex);
    auto [M_same, M_nr] = partition(M_complex);

    SpMatcd K_const_proj = (T0_H * K_same * T0_).pruned(1e-15);
    SpMatcd K_phase_proj = (T0_H * K_nr * T0_).pruned(1e-15);
    SpMatcd M_const_proj = (T0_H * M_same * T0_).pruned(1e-15);
    SpMatcd M_phase_proj = (T0_H * M_nr * T0_).pruned(1e-15);

    // Pre-eliminate constrained DOFs
    free_reduced_map_.clear();
    std::vector<int> reduced_to_free_vec(n_reduced_, -1);
    for (int i = 0; i < n_reduced_; i++) {
        if (hub_red_set_.count(i) == 0) {
            reduced_to_free_vec[i] = static_cast<int>(free_reduced_map_.size());
            free_reduced_map_.push_back(i);
        }
    }
    n_free_ = static_cast<int>(free_reduced_map_.size());

    auto extract_free = [&](const SpMatcd& A) -> SpMatcd {
        std::vector<TripletC> trips;
        for (int col = 0; col < A.outerSize(); ++col) {
            int rj = reduced_to_free_vec[col];
            if (rj < 0) continue;
            for (SpMatcd::InnerIterator it(A, col); it; ++it) {
                int ri = reduced_to_free_vec[static_cast<int>(it.row())];
                if (ri >= 0)
                    trips.emplace_back(ri, rj, it.value());
            }
        }
        SpMatcd R(n_free_, n_free_);
        R.setFromTriplets(trips.begin(), trips.end());
        return R;
    };

    Kcf_ = extract_free(K_const_proj);
    Kpf_ = extract_free(K_phase_proj);
    Mcf_ = extract_free(M_const_proj);
    Mpf_ = extract_free(M_phase_proj);
    Kpf_H_ = Kpf_.adjoint();
    Mpf_H_ = Mpf_.adjoint();

    projections_precomputed_ = true;

    // Precompute real K/M for k=0 and k=N/2
    precompute_real_matrices();
}

void CyclicSymmetrySolver::precompute_real_matrices() {
    // P8: Precompute real K/M for k=0 (P=1) and k=N/2 (P=-1).
    // These harmonics produce real matrices, so we can avoid complex arithmetic.

    // k=0: K_free = Kcf + Kpf + Kpf_H, all coefficients are real
    {
        SpMatcd K0_complex = Kcf_ + Kpf_ + Kpf_H_;
        SpMatcd M0_complex = Mcf_ + Mpf_ + Mpf_H_;

        std::vector<Triplet> kr, mr;
        kr.reserve(K0_complex.nonZeros());
        mr.reserve(M0_complex.nonZeros());
        for (int col = 0; col < K0_complex.outerSize(); ++col)
            for (SpMatcd::InnerIterator it(K0_complex, col); it; ++it)
                if (std::abs(it.value().real()) > 1e-15)
                    kr.emplace_back(it.row(), it.col(), it.value().real());
        for (int col = 0; col < M0_complex.outerSize(); ++col)
            for (SpMatcd::InnerIterator it(M0_complex, col); it; ++it)
                if (std::abs(it.value().real()) > 1e-15)
                    mr.emplace_back(it.row(), it.col(), it.value().real());

        K0_real_.resize(n_free_, n_free_);
        M0_real_.resize(n_free_, n_free_);
        K0_real_.setFromTriplets(kr.begin(), kr.end());
        M0_real_.setFromTriplets(mr.begin(), mr.end());
    }

    // k=N/2 (only exists for even num_sectors): P = e^{i*pi} = -1
    int max_k = mesh_.num_sectors / 2;
    if (mesh_.num_sectors % 2 == 0 && max_k > 0) {
        SpMatcd Khalf_complex = Kcf_ - Kpf_ - Kpf_H_;
        SpMatcd Mhalf_complex = Mcf_ - Mpf_ - Mpf_H_;

        std::vector<Triplet> kr, mr;
        kr.reserve(Khalf_complex.nonZeros());
        mr.reserve(Mhalf_complex.nonZeros());
        for (int col = 0; col < Khalf_complex.outerSize(); ++col)
            for (SpMatcd::InnerIterator it(Khalf_complex, col); it; ++it)
                if (std::abs(it.value().real()) > 1e-15)
                    kr.emplace_back(it.row(), it.col(), it.value().real());
        for (int col = 0; col < Mhalf_complex.outerSize(); ++col)
            for (SpMatcd::InnerIterator it(Mhalf_complex, col); it; ++it)
                if (std::abs(it.value().real()) > 1e-15)
                    mr.emplace_back(it.row(), it.col(), it.value().real());

        Khalf_real_.resize(n_free_, n_free_);
        Mhalf_real_.resize(n_free_, n_free_);
        Khalf_real_.setFromTriplets(kr.begin(), kr.end());
        Mhalf_real_.setFromTriplets(mr.begin(), mr.end());
    }
}

std::vector<ModalResult> CyclicSymmetrySolver::solve_at_rpm(
    double rpm, int num_modes_per_harmonic,
    const std::vector<int>& harmonic_indices,
    int max_threads,
    bool include_coriolis,
    double min_frequency,
    const ProgressCallback& progress_cb,
    bool allow_condensation,
    double memory_reserve_fraction) {

    double omega = rpm * 2.0 * PI / 60.0;

    // Step 1: Assemble base sector K, M (cached — geometry/material only)
    if (!base_assembled_) {
        assembler_.assemble(mesh_, mat_);
        K_base_ = assembler_.K();
        M_base_ = assembler_.M();
        base_assembled_ = true;
    }
    SpMatd K_sector = K_base_;
    SpMatd M_sector = M_base_;

    // Step 1b: Add grounded spring stiffness (ELASTIC_SUPPORT BCs)
    SpMatd K_spring = build_spring_stiffness();
    if (K_spring.nonZeros() > 0) {
        K_sector += K_spring;
    }

    // Step 2: Rotating effects
    SpMatd K_eff = K_sector;

    if (std::abs(omega) > 0.0) {
        // 2a: Spin softening — pure kinematic effect of rotating frame,
        //     K_omega = rho * omega^2 * N^T * (I - a*a^T) * N.
        //     Always applied when spinning, regardless of hub constraint.
        assembler_.assemble_rotating_effects(mesh_, mat_, omega);
        SpMatd K_omega = assembler_.K_omega();
        K_eff = K_sector - K_omega;

        // 2b: Stress stiffening from centrifugal prestress.
        //     Requires solving static K*u = F_centrifugal, which needs
        //     constraints to make K non-singular.  Skipped if unconstrained.
        if (has_constraints_) {
            Eigen::VectorXd u_static = static_centrifugal(omega);
            assembler_.assemble_stress_stiffening(mesh_, mat_, u_static, omega);
            SpMatd K_sigma = assembler_.K_sigma();
            K_eff += K_sigma;
        }
    }

    // Step 2.5: Optional static condensation for large meshes
    SpMatd T_condensation;  // identity if no condensation
    bool condensation_active = false;
    if (allow_condensation) {
        int ndof_reduced = static_cast<int>(interior_dofs_.size() + left_dofs_.size());
        size_t per_harmonic = estimate_per_harmonic_bytes(ndof_reduced, num_modes_per_harmonic);
        size_t total_mem = total_system_memory();
        size_t budget = static_cast<size_t>(total_mem * (1.0 - memory_reserve_fraction));

        if (total_mem > 0 && per_harmonic > budget) {
            int n_target = compute_target_dofs(ndof_reduced, num_modes_per_harmonic, budget);

            // Collect boundary DOFs (mandatory for cyclic correctness)
            std::set<int> boundary_dof_indices;
            // In reduced space, left DOFs start after interior DOFs
            int interior_count = static_cast<int>(interior_dofs_.size());
            for (int i = interior_count; i < ndof_reduced; i++) {
                boundary_dof_indices.insert(i);
            }
            // Also add constrained DOFs
            std::vector<int> constrained = build_constrained_dofs();
            std::set<int> constrained_full(constrained.begin(), constrained.end());
            std::vector<int> reduced_to_full_map;
            for (int dof : interior_dofs_) reduced_to_full_map.push_back(dof);
            for (int dof : left_dofs_) reduced_to_full_map.push_back(dof);
            for (int r = 0; r < ndof_reduced; r++) {
                if (constrained_full.count(reduced_to_full_map[r]) > 0)
                    boundary_dof_indices.insert(r);
            }

            if (n_target < ndof_reduced) {
                auto master = select_master_dofs(K_eff, boundary_dof_indices, n_target);
                auto cond = condense(K_eff, M_sector, master);
                std::cerr << "[CyclicSolver] Condensation active: "
                          << ndof_reduced << " → " << static_cast<int>(master.size())
                          << " DOFs (target " << n_target << ")" << std::endl;
                K_eff = std::move(cond.K_reduced);
                M_sector = std::move(cond.M_reduced);
                T_condensation = std::move(cond.T_c);
                condensation_active = true;
            }
        }
    }

    // Step 3: Precompute cyclic projections
    precompute_cyclic_projections(K_eff, M_sector);

    // Use cached projections
    const SpMatcd& T0 = T0_;
    const std::set<int>& right_dof_set = right_dof_set_;
    const std::set<int>& hub_red_set = hub_red_set_;
    const std::vector<int>& free_reduced_map = free_reduced_map_;
    int n_reduced = n_reduced_;
    int n_free = n_free_;
    double alpha = 2.0 * PI / mesh_.num_sectors;

    int max_k = mesh_.num_sectors / 2;

    // Helper to extract free DOFs from a projected matrix
    std::vector<int> reduced_to_free_vec(n_reduced, -1);
    for (int i = 0; i < n_reduced; i++) {
        if (hub_red_set.count(i) == 0) {
            reduced_to_free_vec[i] = static_cast<int>(
                std::find(free_reduced_map.begin(), free_reduced_map.end(), i)
                - free_reduced_map.begin());
        }
    }
    auto extract_free = [&](const SpMatcd& A) -> SpMatcd {
        std::vector<TripletC> trips;
        for (int col = 0; col < A.outerSize(); ++col) {
            int rj = reduced_to_free_vec[col];
            if (rj < 0) continue;
            for (SpMatcd::InnerIterator it(A, col); it; ++it) {
                int ri = reduced_to_free_vec[static_cast<int>(it.row())];
                if (ri >= 0)
                    trips.emplace_back(ri, rj, it.value());
            }
        }
        SpMatcd R(n_free, n_free);
        R.setFromTriplets(trips.begin(), trips.end());
        return R;
    };

    // Gyroscopic matrix projection (same pipeline as K/M)
    SpMatcd Gcf, Gpf, Gpf_H;
    bool coriolis_active = include_coriolis && std::abs(omega) > 0.0;
    if (coriolis_active) {
        int ndof_full = mesh_.num_dof();
        SpMatd G_global = assembler_.G();
        SpMatcd G_complex(ndof_full, ndof_full);
        {
            std::vector<TripletC> gc;
            gc.reserve(G_global.nonZeros());
            for (int col = 0; col < G_global.outerSize(); ++col)
                for (SpMatd::InnerIterator it(G_global, col); it; ++it)
                    gc.emplace_back(it.row(), it.col(), std::complex<double>(it.value(), 0.0));
            G_complex.setFromTriplets(gc.begin(), gc.end());
        }
        // Partition by right-DOF membership
        auto partition_g = [&](const SpMatcd& A) {
            std::vector<TripletC> ts, tnr;
            for (int col = 0; col < A.outerSize(); ++col) {
                bool cr = right_dof_set.count(col) > 0;
                for (SpMatcd::InnerIterator it(A, col); it; ++it) {
                    bool rr = right_dof_set.count(static_cast<int>(it.row())) > 0;
                    if (rr == cr)
                        ts.emplace_back(it.row(), it.col(), it.value());
                    else if (!rr && cr)
                        tnr.emplace_back(it.row(), it.col(), it.value());
                }
            }
            int sz = static_cast<int>(A.rows());
            SpMatcd As(sz, sz), Anr(sz, sz);
            As.setFromTriplets(ts.begin(), ts.end());
            Anr.setFromTriplets(tnr.begin(), tnr.end());
            return std::make_pair(std::move(As), std::move(Anr));
        };
        SpMatcd T0_H_local = T0.adjoint();
        auto [G_same, G_nr] = partition_g(G_complex);
        SpMatcd G_const_proj = (T0_H_local * G_same * T0).pruned(1e-15);
        SpMatcd G_phase_proj = (T0_H_local * G_nr * T0).pruned(1e-15);
        Gcf = extract_free(G_const_proj);
        Gpf = extract_free(G_phase_proj);
        Gpf_H = Gpf.adjoint();
    }

    // BEM added mass precomputation (once per geometry, cached)
    if (fluid_.type == FluidConfig::Type::POTENTIAL_FLOW_BEM &&
        fluid_.fluid_density > 0.0 && !bem_precomputed_) {
        bem_added_mass_ = std::make_unique<PotentialFlowAddedMass>(mesh_, fluid_.fluid_density);
        bem_added_mass_->precompute(max_k);

        M_added_free_cache_.resize(max_k + 1);
        for (int k = 0; k <= max_k; k++) {
            // Get sector-level added mass for this ND
            SpMatcd M_added_sector = bem_added_mass_->get_sector_added_mass(k);

            // Project through cyclic transformation T(k)
            SpMatcd T_k = build_cyclic_transformation(k);
            SpMatcd T_k_H = T_k.adjoint();
            SpMatcd M_added_cyclic = (T_k_H * M_added_sector * T_k).pruned(1e-15);

            // Extract free DOFs (same elimination as structural M)
            M_added_free_cache_[k] = extract_free(M_added_cyclic);
        }
        bem_precomputed_ = true;
    }

    // Determine which harmonics to solve
    std::vector<int> harmonics_to_solve;
    if (harmonic_indices.empty()) {
        for (int k = 0; k <= max_k; k++) harmonics_to_solve.push_back(k);
    } else {
        for (int k : harmonic_indices) {
            if (k >= 0 && k <= max_k) harmonics_to_solve.push_back(k);
        }
    }
    int n_harmonics = static_cast<int>(harmonics_to_solve.size());

    // Per-harmonic solve using precomputed projections
    // Returns (ModalResult, SolverStatus) or nullopt on failure.
    using HarmonicResult = std::optional<std::pair<ModalResult, SolverStatus>>;
    auto solve_harmonic = [&](int k, ModalSolver& modal_solver) -> HarmonicResult {
        try {
            if (n_free <= 1) return std::nullopt;

            // Phase factor needed by both paths (also used by Coriolis G below)
            std::complex<double> P(std::cos(k * alpha), std::sin(k * alpha));
            std::complex<double> Pc = std::conj(P);

            // Fast path: K_k = K_const + P*K_phase + conj(P)*K_phase^H
            SpMatcd K_free = Kcf_ + P * Kpf_ + Pc * Kpf_H_;
            SpMatcd M_free = Mcf_ + P * Mpf_ + Pc * Mpf_H_;

            // Add BEM potential flow added mass (precomputed, one sparse addition)
            if (bem_precomputed_ && k < static_cast<int>(M_added_free_cache_.size()) &&
                M_added_free_cache_[k].nonZeros() > 0) {
                const auto& Ma = M_added_free_cache_[k];
                if (Ma.rows() != M_free.rows() || Ma.cols() != M_free.cols()) {
                    throw std::runtime_error(
                        "BEM added mass dimension mismatch at harmonic " +
                        std::to_string(k) + ": (" + std::to_string(Ma.rows()) +
                        "x" + std::to_string(Ma.cols()) + ") vs (" +
                        std::to_string(M_free.rows()) + "x" +
                        std::to_string(M_free.cols()) + ")");
                }
                M_free += Ma;
            }

            ModalResult result;
            SolverStatus status;
            bool used_qep = false;

            // Degenerate harmonics (0 < k < N/2) have multiplicity 2: each
            // eigenvalue of the complex Hermitian problem represents a pair
            // of standing-wave modes (cosine + sine, or equivalently forward
            // + backward traveling waves).  Solve for ceil(M/2) eigenvalues
            // and duplicate them afterwards to match the full-model mode count.
            //
            // Exception: when Coriolis is active, the Lancaster QEP explicitly
            // breaks the degeneracy into distinct FW/BW eigenvalues, so we
            // request the full mode count and skip the duplication step.
            bool is_degenerate = (k > 0) &&
                !(mesh_.num_sectors % 2 == 0 && k == max_k);
            bool will_use_qep = coriolis_active && k > 0;
            int nev_target = (is_degenerate && !will_use_qep)
                ? (num_modes_per_harmonic + 1) / 2
                : num_modes_per_harmonic;

            // Coriolis QEP branch: for k > 0 with Coriolis enabled, solve the
            // Lancaster-linearized QEP (K + omega*D - omega^2*M)*phi = 0.
            if (will_use_qep) {
                SpMatcd G_free = Gcf + P * Gpf + std::conj(P) * Gpf_H;
                // D = i * 2 * omega * G_k. Since G_k is skew-Hermitian, D is Hermitian.
                std::complex<double> coriolis_factor(0.0, 2.0 * omega);
                SpMatcd D_free = coriolis_factor * G_free;

                SolverConfig qep_config;
                int n_extra_qep = (min_frequency > 0.0) ? 6 : 0;
                qep_config.nev = nev_target + n_extra_qep;
                // Adaptive shift from K/M norms: ω² ≈ ||K|| / ||M||
                // This targets the middle of the spectrum for the given model,
                // rather than a hardcoded frequency that may miss everything.
                double k_norm = K_free.norm();
                double m_norm = M_free.norm();
                qep_config.shift = (m_norm > 0) ? k_norm / m_norm : 1.0;
                qep_config.tolerance = 1e-8;

                std::tie(result, status) = modal_solver.solve_lancaster_qep(
                    K_free, M_free, D_free, nev_target, qep_config);
                used_qep = true;
            } else {
                // Standard n x n Hermitian GEP: K*x = lambda*M*x
                SolverConfig config;
                // Request extra eigenvalues when filtering is active (free hub
                // produces rigid body modes that will be discarded).
                int n_extra = (min_frequency > 0.0) ? 6 : 0;
                config.nev = std::min(nev_target + n_extra, n_free - 1);
                if (config.nev <= 0) return std::nullopt;
                config.ncv = std::min(std::max(4 * config.nev + 1, 40), n_free);
                // Adaptive shift for shift-invert: σ ≈ ||K||/||M|| targets the
                // middle of the eigenvalue spectrum regardless of unit system.
                bool is_real_case = (k == 0) || (mesh_.num_sectors % 2 == 0 && k == max_k);

                if (is_real_case) {
                    // Use precomputed real matrices (P8 optimization).
                    // When BEM added mass is active, M_free already includes it
                    // but precomputed M0_real_ does not, so extract from M_free.
                    const SpMatd& K_real = (k == 0) ? K0_real_ : Khalf_real_;
                    bool bem_active = bem_precomputed_ &&
                        k < static_cast<int>(M_added_free_cache_.size()) &&
                        M_added_free_cache_[k].nonZeros() > 0;
                    if (bem_active) {
                        // Extract real parts from M_free (which has BEM mass added)
                        SpMatd M_real(n_free, n_free);
                        std::vector<Triplet> mr;
                        mr.reserve(M_free.nonZeros());
                        for (int col = 0; col < M_free.outerSize(); ++col)
                            for (SpMatcd::InnerIterator it(M_free, col); it; ++it)
                                if (std::abs(it.value().real()) > 1e-15)
                                    mr.emplace_back(it.row(), it.col(), it.value().real());
                        M_real.setFromTriplets(mr.begin(), mr.end());
                        double k_n = K_real.norm();
                        double m_n = M_real.norm();
                        config.shift = (m_n > 0) ? k_n / m_n : 1.0;
                        std::tie(result, status) = modal_solver.solve_real(K_real, M_real, config);
                    } else {
                        const SpMatd& M_real = (k == 0) ? M0_real_ : Mhalf_real_;
                        double k_n = K_real.norm();
                        double m_n = M_real.norm();
                        config.shift = (m_n > 0) ? k_n / m_n : 1.0;
                        std::tie(result, status) = modal_solver.solve_real(K_real, M_real, config);
                    }
                } else {
                    double k_n = K_free.norm();
                    double m_n = M_free.norm();
                    double base_shift = (m_n > 0) ? k_n / m_n : 1.0;
                    config.shift = base_shift;
                    std::cerr << "[DEBUG] k=" << k << " shift=" << config.shift
                              << " n=" << n_free << std::endl;
                    std::tie(result, status) = modal_solver.solve_complex_hermitian(
                        K_free, M_free, config);
                    std::cerr << "[DEBUG] k=" << k << " result: nconv="
                              << status.num_converged << " msg=" << status.message
                              << std::endl;
                }
            }

            if (!status.converged && status.num_converged == 0) return std::nullopt;

            if (!status.converged) {
                std::cerr << "[CyclicSolver] WARNING: k=" << k
                          << " eigenvalue solve did not fully converge: "
                          << status.message << "\n";
            }

            // Filter out rigid body / near-zero modes when min_frequency is set.
            // Free hub constraint produces RBMs (lambda ~ 0) that contaminate results.
            if (min_frequency > 0.0) {
                int n_orig = static_cast<int>(result.frequencies.size());
                std::vector<int> keep;
                keep.reserve(n_orig);
                for (int m = 0; m < n_orig; m++) {
                    if (result.frequencies(m) > min_frequency)
                        keep.push_back(m);
                }
                if (static_cast<int>(keep.size()) < n_orig) {
                    int nk = static_cast<int>(keep.size());
                    if (nk == 0) return std::nullopt;
                    Eigen::VectorXd new_f(nk);
                    Eigen::MatrixXcd new_s(result.mode_shapes.rows(), nk);
                    Eigen::VectorXi new_w(nk);
                    for (int i = 0; i < nk; i++) {
                        new_f(i) = result.frequencies(keep[i]);
                        new_s.col(i) = result.mode_shapes.col(keep[i]);
                        new_w(i) = (result.whirl_direction.size() > keep[i])
                            ? result.whirl_direction(keep[i]) : 0;
                    }
                    result.frequencies = new_f;
                    result.mode_shapes = new_s;
                    result.whirl_direction = new_w;
                }
            }

            // Truncate to nev_target (extra were requested above
            // to compensate for filtered-out rigid body modes).
            {
                int n_have = static_cast<int>(result.frequencies.size());
                int n_want = nev_target;
                if (n_have > n_want) {
                    result.frequencies.conservativeResize(n_want);
                    result.mode_shapes.conservativeResize(Eigen::NoChange, n_want);
                    result.whirl_direction.conservativeResize(n_want);
                }
            }

            // Eigenvalue residual diagnostics (skip for QEP — different eigenvalue semantics)
            if (!used_qep) {
                int n_modes = static_cast<int>(result.mode_shapes.cols());
                double max_res = 0.0;
                for (int m = 0; m < n_modes; m++) {
                    // Skip residual check for sub-threshold modes (shouldn't
                    // exist after filter, but guard against edge cases)
                    if (min_frequency > 0.0 && result.frequencies(m) < min_frequency)
                        continue;
                    double lambda = std::pow(2.0 * PI * result.frequencies(m), 2);
                    Eigen::VectorXcd x = result.mode_shapes.col(m);
                    Eigen::VectorXcd Kx = K_free * x;
                    Eigen::VectorXcd r = Kx - lambda * (M_free * x);
                    double Kx_norm = Kx.norm();
                    double res = (Kx_norm > 0) ? r.norm() / Kx_norm : r.norm();
                    max_res = std::max(max_res, res);
                }
                if (max_res > 1e-4) {
                    SpMatcd M_diff = M_free - SpMatcd(M_free.adjoint());
                    double herm_err = 0.0;
                    for (int col = 0; col < M_diff.outerSize(); ++col)
                        for (SpMatcd::InnerIterator it(M_diff, col); it; ++it)
                            herm_err = std::max(herm_err, std::abs(it.value()));

                    double bem_norm = 0.0;
                    if (bem_precomputed_ && k < static_cast<int>(M_added_free_cache_.size()))
                        bem_norm = M_added_free_cache_[k].norm();
                    SpMatcd M_struct = Mcf_ + P * Mpf_ + std::conj(P) * Mpf_H_;
                    double M_struct_norm = M_struct.norm();

                    std::cerr << "  [residual] k=" << k
                              << " max_res=" << std::scientific << max_res
                              << " M_herm_err=" << herm_err
                              << " M_struct=" << M_struct_norm
                              << " BEM=" << bem_norm
                              << " ratio=" << std::defaultfloat
                              << (M_struct_norm > 0 ? bem_norm / M_struct_norm : 0.0)
                              << " n=" << n_free
                              << std::endl;
                }
            }

            // Added mass correction
            double freq_ratio = added_mass_factor(k);
            if (freq_ratio < 1.0 && freq_ratio > 0.0) {
                result.frequencies *= freq_ratio;
            }

            // For standard solver: set whirl to zero (rotating frame, no Coriolis).
            // For QEP solver: whirl_direction is already set by solve_lancaster_qep.
            if (!used_qep) {
                int num_modes = static_cast<int>(result.frequencies.size());
                result.whirl_direction = Eigen::VectorXi::Zero(num_modes);
            }

            // Expand degenerate modes: duplicate each eigenvalue/eigenvector
            // pair for 0 < k < N/2.  The conjugate eigenvector represents
            // the opposite traveling wave (same frequency, opposite phase).
            // The original eigenvector has circumferential phase exp(+ikθ),
            // which is forward whirl (+1); the conjugate has exp(-ikθ),
            // which is backward whirl (-1).  This labelling is always
            // applied for degenerate harmonics, regardless of whether the
            // Coriolis QEP was used — the physics of FW/BW travelling waves
            // exists even when the rotating-frame frequencies are degenerate.
            if (is_degenerate && !used_qep) {
                int n_unique = static_cast<int>(result.frequencies.size());
                int n_expanded = std::min(2 * n_unique, num_modes_per_harmonic);
                Eigen::VectorXd exp_f(n_expanded);
                Eigen::MatrixXcd exp_s(result.mode_shapes.rows(), n_expanded);
                Eigen::VectorXi exp_w(n_expanded);
                for (int m = 0; m < n_expanded; m++) {
                    int src = m / 2;
                    exp_f(m) = result.frequencies(src);
                    if (m % 2 == 0) {
                        exp_s.col(m) = result.mode_shapes.col(src);
                        exp_w(m) = 1;   // Forward whirl: exp(+ikθ)
                    } else {
                        // Conjugate eigenvector: opposite traveling wave
                        exp_s.col(m) = result.mode_shapes.col(src).conjugate();
                        exp_w(m) = -1;  // Backward whirl: exp(-ikθ)
                    }
                }
                result.frequencies = exp_f;
                result.mode_shapes = exp_s;
                result.whirl_direction = exp_w;
            }

            // Mode shape expansion: T(k) = S_k * T0
            {
                int n_result_modes = static_cast<int>(result.mode_shapes.cols());

                // If condensation was used, expand through T_condensation first
                Eigen::MatrixXcd modes_to_expand = result.mode_shapes;
                if (condensation_active) {
                    modes_to_expand = expand_modes(T_condensation, result.mode_shapes);
                }

                // Expand from n_free to n_reduced (insert zeros at hub DOFs)
                Eigen::MatrixXcd modes_reduced = Eigen::MatrixXcd::Zero(
                    n_reduced, n_result_modes);
                for (Eigen::Index i = 0; i < modes_to_expand.rows()
                     && i < static_cast<Eigen::Index>(free_reduced_map.size()); i++) {
                    modes_reduced.row(free_reduced_map[i]) = modes_to_expand.row(i);
                }

                // u_full = T0 * u_reduced, then scale right DOF rows by phase
                Eigen::MatrixXcd modes_full = T0 * modes_reduced;
                for (auto r : right_dofs_) {
                    if (r < static_cast<int>(modes_full.rows())) {
                        modes_full.row(r) *= P;
                    }
                }
                result.mode_shapes = modes_full;
            }

            result.harmonic_index = k;
            result.rpm = rpm;
            result.converged = status.converged;
            return std::make_pair(std::move(result), status);
        } catch (const std::bad_alloc&) {
            std::cerr << "[CyclicSolver] k=" << k << " out of memory, will retry" << std::endl;
            return std::nullopt;
        } catch (const std::exception& e) {
            std::cerr << "[CyclicSolver] k=" << k << " FAILED: " << e.what() << std::endl;
            return std::nullopt;
        }
    };

    // Dynamic work-stealing pool with memory-aware concurrency cap.
    // Each worker gets a thread-local ModalSolver that persists across harmonics,
    // allowing symbolic factorization reuse (same sparsity pattern for all k).
    int max_concurrent = (max_threads > 0) ? max_threads
        : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    max_concurrent = std::min(max_concurrent, n_harmonics);

    // Memory-aware cap using actual nnz of the projected operator.
    //
    // The shifted operator (K_free - σ*M_free) has the union sparsity of
    // Kcf_, Kpf_, Mcf_, Mpf_.  Use Kcf_ nnz (the dominant part) as the
    // base; Kpf_ adds relatively few new entries due to identical mesh
    // topology on left/right boundaries.
    //
    // Differentiate:
    //   k=0 / k=N/2 : real-valued (8 bytes/entry)
    //   k>0          : complex Hermitian (16 bytes/entry)
    //   k>0+Coriolis : Lancaster QEP doubles system to 2n (16 bytes/entry)
    int64_t op_nnz = 0;
    if (projections_precomputed_) {
        // Upper bound on shifted operator nnz: sum of constituent nnz
        // (actual is less due to overlapping patterns, but safe for estimate)
        op_nnz = static_cast<int64_t>(Kcf_.nonZeros() + Kpf_.nonZeros()
                                      + Mcf_.nonZeros() + Mpf_.nonZeros());
    }

    // Most harmonics are k>0 (complex).  Use worst-case for cap.
    int n_est = coriolis_active ? 2 * n_free : n_free;
    int entry_bytes = 16;  // complex double
    int64_t op_nnz_est = coriolis_active ? 4 * op_nnz : op_nnz;

    size_t per_harmonic = estimate_per_harmonic_bytes(
        n_est, num_modes_per_harmonic, op_nnz_est, entry_bytes);
    size_t total_mem = total_system_memory();
    if (total_mem > 0 && per_harmonic > 0) {
        size_t budget = static_cast<size_t>(
            total_mem * (1.0 - memory_reserve_fraction));
        int mem_limit = std::max(1, static_cast<int>(budget / per_harmonic));
        if (mem_limit < max_concurrent) {
            std::cerr << "[CyclicSolver] Memory cap: " << mem_limit
                      << " concurrent harmonics (est. "
                      << (per_harmonic / (1024*1024)) << " MB each, "
                      << (total_mem / (1024*1024)) << " MB total RAM, "
                      << static_cast<int>((1.0 - memory_reserve_fraction) * 100)
                      << "% budget)" << std::endl;
            max_concurrent = mem_limit;
        }
    }

    std::atomic<int> next_idx{0};
    std::atomic<int> completed{0};
    std::vector<HarmonicResult> results_vec(n_harmonics);
    // Track OOM failures for sequential retry
    std::vector<int> oom_indices;
    std::mutex oom_mutex;

    auto wall_start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(max_concurrent);
    for (int w = 0; w < max_concurrent; w++) {
        workers.emplace_back([&]() {
            ModalSolver thread_solver;  // persists across harmonics for this thread
            while (true) {
                int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n_harmonics) break;
                int k = harmonics_to_solve[idx];
                try {
                    results_vec[idx] = solve_harmonic(k, thread_solver);
                } catch (...) {
                    // solve_harmonic has its own try/catch; this is a safety net.
                    results_vec[idx] = std::nullopt;
                }
                // Check if this was an OOM failure (solve_harmonic printed the message)
                if (!results_vec[idx].has_value()) {
                    std::lock_guard<std::mutex> lock(oom_mutex);
                    oom_indices.push_back(idx);
                }
                // Report progress with rich SolverProgress struct
                int done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
                if (progress_cb) {
                    SolverProgress prog;
                    prog.completed = done;
                    prog.total = n_harmonics;
                    prog.harmonic_k = k;
                    auto now = std::chrono::steady_clock::now();
                    prog.elapsed_s = std::chrono::duration<double>(now - wall_start).count();
                    if (results_vec[idx].has_value()) {
                        auto& [res, st] = results_vec[idx].value();
                        prog.converged = res.converged;
                        prog.iterations = st.iterations;
                        prog.num_converged = st.num_converged;
                        prog.max_residual = st.max_residual;
                        prog.num_modes = static_cast<int>(res.frequencies.size());
                        if (prog.num_modes > 0) {
                            prog.min_freq_hz = res.frequencies.minCoeff();
                            prog.max_freq_hz = res.frequencies.maxCoeff();
                        }
                    } else {
                        prog.converged = false;
                    }
                    progress_cb(prog);
                }
            }
        });
    }
    for (auto& t : workers) t.join();

    // Retry failed harmonics sequentially (1 thread = minimal peak memory)
    if (!oom_indices.empty()) {
        std::cerr << "[CyclicSolver] Retrying " << oom_indices.size()
                  << " failed harmonics sequentially" << std::endl;
        ModalSolver retry_solver;
        for (int idx : oom_indices) {
            results_vec[idx] = solve_harmonic(harmonics_to_solve[idx], retry_solver);
        }
    }

    std::vector<ModalResult> results;
    for (auto& opt : results_vec) {
        if (opt.has_value()) {
            results.push_back(std::move(opt->first));
        }
    }

    // Sort by harmonic index for deterministic output
    std::sort(results.begin(), results.end(), [](const ModalResult& a, const ModalResult& b) {
        return a.harmonic_index < b.harmonic_index;
    });

    return results;
}

std::vector<std::vector<ModalResult>> CyclicSymmetrySolver::solve_rpm_sweep(
    const Eigen::VectorXd& rpm_values, int num_modes_per_harmonic,
    const std::vector<int>& harmonic_indices,
    int max_threads,
    bool include_coriolis,
    double min_frequency) {
    std::vector<std::vector<ModalResult>> all_results;
    all_results.reserve(rpm_values.size());
    for (Eigen::Index i = 0; i < rpm_values.size(); i++) {
        all_results.push_back(solve_at_rpm(rpm_values(i), num_modes_per_harmonic,
                                            harmonic_indices, max_threads,
                                            include_coriolis, min_frequency));
    }
    return all_results;
}

std::vector<std::vector<ModalResult>> CyclicSymmetrySolver::solve_parametric(
    const std::vector<ParametricCondition>& conditions,
    int num_modes_per_harmonic,
    const std::vector<int>& harmonic_indices,
    int max_threads,
    bool include_coriolis,
    double min_frequency,
    bool allow_condensation,
    double memory_reserve_fraction,
    const ProgressCallback& progress_cb) {

    if (conditions.empty()) return {};

    // Step 1: Assemble base K, M at reference temperature (cached)
    if (!base_assembled_) {
        assembler_.assemble(mesh_, mat_);
        K_base_ = assembler_.K();
        M_base_ = assembler_.M();
        base_assembled_ = true;
    }

    // Step 2: Precompute K_omega at unit speed (omega=1) for fast RPM scaling.
    // K_omega ∝ omega², so K_omega(omega) = omega² * K_omega_unit.
    if (!K_omega_unit_computed_) {
        assembler_.assemble_rotating_effects(mesh_, mat_, 1.0);
        K_omega_unit_ = assembler_.K_omega();
        K_omega_unit_computed_ = true;
    }

    // Step 3: Find unique temperatures for K_sigma precomputation
    double E_ref = mat_.E;
    double T_ref = mat_.T_ref;
    double E_slope = mat_.E_slope;

    // Collect unique RPM values for stress stiffening precomputation
    std::map<double, Eigen::VectorXd> rpm_to_u_static;  // rpm -> static displacement

    std::vector<std::vector<ModalResult>> all_results(conditions.size());

    int total_conditions = static_cast<int>(conditions.size());
    std::atomic<int> completed{0};

    for (int ci = 0; ci < total_conditions; ci++) {
        const auto& cond = conditions[ci];
        double omega = cond.rpm * 2.0 * PI / 60.0;

        // Temperature scaling: E(T) = E_ref + E_slope * (T - T_ref)
        double E_T = E_ref + E_slope * (cond.temperature - T_ref);
        double E_scale = (E_ref > 0) ? E_T / E_ref : 1.0;

        // Build K_eff for this condition
        SpMatd K_eff = E_scale * K_base_;

        // Add grounded spring stiffness (not scaled by temperature)
        SpMatd K_spring = build_spring_stiffness();
        if (K_spring.nonZeros() > 0) {
            K_eff += K_spring;
        }

        if (std::abs(omega) > 0.0) {
            // Spin softening: K_omega = omega^2 * K_omega_unit
            SpMatd K_omega = omega * omega * K_omega_unit_;
            K_eff -= K_omega;

            // Stress stiffening
            if (has_constraints_) {
                // Check if we've already computed u_static for this RPM
                auto it = rpm_to_u_static.find(cond.rpm);
                if (it == rpm_to_u_static.end()) {
                    Eigen::VectorXd u_static = static_centrifugal(omega);
                    rpm_to_u_static[cond.rpm] = u_static;
                    it = rpm_to_u_static.find(cond.rpm);
                }
                // K_sigma also scales with E(T): stress = E * strain, so
                // K_sigma(T) = E_scale * K_sigma_ref.
                // But K_sigma depends on the stress field which depends on E,
                // and the static displacement u = K^{-1}*F ∝ 1/E.
                // So stress = E * B * u = E * B * (F/E) = B*F, independent of E!
                // This means K_sigma is independent of temperature (for uniform E scaling).
                assembler_.assemble_stress_stiffening(
                    mesh_, mat_, it->second, omega);
                K_eff += assembler_.K_sigma();
            }
        }

        SpMatd M_sector = M_base_;  // M is independent of temperature

        // Solve at this condition using the existing infrastructure
        // Precompute projections for this K_eff
        precompute_cyclic_projections(K_eff, M_sector);

        // Delegate to solve_at_rpm-like logic with the precomputed projections
        all_results[ci] = solve_at_rpm(cond.rpm, num_modes_per_harmonic,
                                        harmonic_indices, max_threads,
                                        include_coriolis, min_frequency,
                                        nullptr, allow_condensation,
                                        memory_reserve_fraction);

        int done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
        if (progress_cb) {
            SolverProgress prog;
            prog.completed = done;
            prog.total = total_conditions;
            prog.harmonic_k = -1;
            prog.converged = true;
            progress_cb(prog);
        }
    }

    return all_results;
}

void CyclicSymmetrySolver::export_campbell_csv(
    const std::string& filename,
    const std::vector<std::vector<ModalResult>>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    file << "rpm,harmonic_index,mode_family,frequency_hz,whirl_direction\n";
    file << std::fixed << std::setprecision(6);

    for (const auto& rpm_results : results) {
        for (const auto& r : rpm_results) {
            auto sf = compute_stationary_frame(r, mesh_.num_sectors);
            for (Eigen::Index m = 0; m < sf.frequencies.size(); m++) {
                int whirl = (sf.whirl_direction.size() > m) ? sf.whirl_direction(m) : 0;
                file << r.rpm << "," << r.harmonic_index << ","
                     << (m + 1) << "," << sf.frequencies(m) << ","
                     << whirl << "\n";
            }
        }
    }
}

void CyclicSymmetrySolver::export_zzenf_csv(
    const std::string& filename,
    const std::vector<ModalResult>& results_at_rpm) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    file << "nodal_diameter,mode_family,frequency_hz,rpm\n";
    file << std::fixed << std::setprecision(6);

    for (const auto& r : results_at_rpm) {
        for (Eigen::Index m = 0; m < r.frequencies.size(); m++) {
            file << r.harmonic_index << "," << (m + 1) << ","
                 << r.frequencies(m) << "," << r.rpm << "\n";
        }
    }
}

void CyclicSymmetrySolver::export_mode_shape_vtk(
    const std::string& filename,
    const ModalResult& result, int mode_index) {
    if (mode_index < 0 || mode_index >= result.mode_shapes.cols()) {
        throw std::runtime_error("Mode index out of range");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    int n_nodes = mesh_.num_nodes();
    int n_elements = mesh_.num_elements();
    Eigen::VectorXcd mode = result.mode_shapes.col(mode_index);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\">\n";
    file << "  <UnstructuredGrid>\n";
    file << "    <Piece NumberOfPoints=\"" << n_nodes
         << "\" NumberOfCells=\"" << n_elements << "\">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < n_nodes; i++) {
        file << "          " << mesh_.nodes(i, 0) << " "
             << mesh_.nodes(i, 1) << " " << mesh_.nodes(i, 2) << "\n";
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Cells
    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int e = 0; e < n_elements; e++) {
        file << "         ";
        for (int n = 0; n < 10; n++) {
            file << " " << mesh_.elements(e, n);
        }
        file << "\n";
    }
    file << "        </DataArray>\n";
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int e = 0; e < n_elements; e++) {
        file << "          " << (e + 1) * 10 << "\n";
    }
    file << "        </DataArray>\n";
    file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int e = 0; e < n_elements; e++) {
        file << "          24\n";
    }
    file << "        </DataArray>\n";
    file << "      </Cells>\n";

    // Point data
    file << "      <PointData>\n";

    file << "        <DataArray type=\"Float64\" Name=\"displacement_real\" "
         << "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < n_nodes; i++) {
        double ux = 0, uy = 0, uz = 0;
        if (3 * i + 2 < static_cast<int>(mode.size())) {
            ux = mode(3 * i).real();
            uy = mode(3 * i + 1).real();
            uz = mode(3 * i + 2).real();
        }
        file << "          " << ux << " " << uy << " " << uz << "\n";
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Float64\" Name=\"displacement_imag\" "
         << "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < n_nodes; i++) {
        double ux = 0, uy = 0, uz = 0;
        if (3 * i + 2 < static_cast<int>(mode.size())) {
            ux = mode(3 * i).imag();
            uy = mode(3 * i + 1).imag();
            uz = mode(3 * i + 2).imag();
        }
        file << "          " << ux << " " << uy << " " << uz << "\n";
    }
    file << "        </DataArray>\n";

    file << "        <DataArray type=\"Float64\" Name=\"displacement_magnitude\" "
         << "format=\"ascii\">\n";
    for (int i = 0; i < n_nodes; i++) {
        double mag = 0;
        if (3 * i + 2 < static_cast<int>(mode.size())) {
            double ux = std::abs(mode(3 * i));
            double uy = std::abs(mode(3 * i + 1));
            double uz = std::abs(mode(3 * i + 2));
            mag = std::sqrt(ux * ux + uy * uy + uz * uz);
        }
        file << "          " << mag << "\n";
    }
    file << "        </DataArray>\n";

    file << "      </PointData>\n";
    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";
}

}  // namespace turbomodal
