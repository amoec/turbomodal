#include "turbomodal/cyclic_solver.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <algorithm>
#include <numeric>
#include <future>
#include <optional>
#include <Eigen/SparseLU>

namespace turbomodal {

CyclicSymmetrySolver::CyclicSymmetrySolver(
    const Mesh& mesh, const Material& mat, const FluidConfig& fluid)
    : mesh_(mesh), mat_(mat), fluid_(fluid) {
    classify_dofs();
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
    if (fluid_.type != FluidConfig::Type::LIQUID_ANALYTICAL || fluid_.fluid_density <= 0.0) {
        return 1.0;
    }
    return AddedMassModel::frequency_ratio(
        harmonic_index, fluid_.fluid_density, mat_.rho,
        fluid_.disk_thickness, fluid_.disk_radius);
}

Eigen::VectorXd CyclicSymmetrySolver::static_centrifugal(double omega) {
    // Solve K * u = F_centrifugal with k=0 cyclic BCs and hub constraints.
    // Cyclic BCs are essential: without them the sector boundaries are free,
    // hoop stress doesn't develop, and K_sigma is far too small.
    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::VectorXd F = assembler_.assemble_centrifugal_load(mesh_, mat_, omega, z_axis);

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

    // Step 2: Find hub DOFs in the cyclic-reduced space
    std::vector<int> reduced_to_full;
    reduced_to_full.reserve(interior_dofs_.size() + left_dofs_.size());
    for (int dof : interior_dofs_) reduced_to_full.push_back(dof);
    for (int dof : left_dofs_) reduced_to_full.push_back(dof);

    const NodeSet* hub = mesh_.find_node_set("hub_constraint");
    std::set<int> hub_full_set;
    if (hub) {
        for (int node_id : hub->node_ids) {
            hub_full_set.insert(3 * node_id);
            hub_full_set.insert(3 * node_id + 1);
            hub_full_set.insert(3 * node_id + 2);
        }
    }

    std::set<int> hub_red_set;
    for (int r = 0; r < reduced_ndof; r++) {
        if (hub_full_set.count(reduced_to_full[r]) > 0) {
            hub_red_set.insert(r);
        }
    }

    // Step 3: Eliminate hub DOFs from cyclic-reduced system
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

std::vector<ModalResult> CyclicSymmetrySolver::solve_at_rpm(
    double rpm, int num_modes_per_harmonic) {

    double omega = rpm * 2.0 * PI / 60.0;

    // Step 1: Assemble base sector K, M
    assembler_.assemble(mesh_, mat_);
    SpMatd K_sector = assembler_.K();
    SpMatd M_sector = assembler_.M();

    // Step 2: Rotating effects (if rpm > 0)
    SpMatd K_eff = K_sector;

    if (omega > 0.0) {
        // 2a: Static prestress analysis
        Eigen::VectorXd u_static = static_centrifugal(omega);

        // 2b: Assemble stress stiffening from prestress
        assembler_.assemble_stress_stiffening(mesh_, mat_, u_static, omega);
        SpMatd K_sigma = assembler_.K_sigma();

        // 2c: Assemble spin softening and gyroscopic matrices
        assembler_.assemble_rotating_effects(mesh_, mat_, omega);
        SpMatd K_omega = assembler_.K_omega();

        // 2d: Form effective stiffness
        // K_sigma already contains omega^2 dependence (from prestress proportional to omega^2)
        // K_omega already contains omega^2 (from spin softening formula)
        K_eff = K_sector + K_sigma - K_omega;
    }

    // Step 3: Apply hub boundary conditions
    const NodeSet* hub = mesh_.find_node_set("hub_constraint");
    std::vector<int> hub_constrained_dofs;
    if (hub) {
        for (int node_id : hub->node_ids) {
            hub_constrained_dofs.push_back(3 * node_id);
            hub_constrained_dofs.push_back(3 * node_id + 1);
            hub_constrained_dofs.push_back(3 * node_id + 2);
        }
        std::sort(hub_constrained_dofs.begin(), hub_constrained_dofs.end());
    }

    // Build the reduced-to-full DOF mapping (interior + left DOFs)
    std::vector<int> reduced_to_full;
    reduced_to_full.reserve(interior_dofs_.size() + left_dofs_.size());
    for (int dof : interior_dofs_) {
        reduced_to_full.push_back(dof);
    }
    for (int dof : left_dofs_) {
        reduced_to_full.push_back(dof);
    }

    // Find which reduced DOFs are hub-constrained
    std::set<int> hub_set(hub_constrained_dofs.begin(), hub_constrained_dofs.end());
    std::vector<int> hub_reduced_dofs;
    for (int r = 0; r < static_cast<int>(reduced_to_full.size()); r++) {
        if (hub_set.count(reduced_to_full[r]) > 0) {
            hub_reduced_dofs.push_back(r);
        }
    }

    int max_k = mesh_.num_sectors / 2;

    // Build hub reduced DOF set once (shared across all k)
    std::set<int> hub_red_set(hub_reduced_dofs.begin(), hub_reduced_dofs.end());

    // Lambda to solve a single harmonic index k
    // All captured references are read-only (K_eff, M_sector, mesh_ etc.)
    auto solve_harmonic = [&](int k) -> std::optional<ModalResult> {
        try {
            // Step 4: Apply cyclic symmetry BCs
            auto [K_k, M_k] = apply_cyclic_bc(k, K_eff, M_sector);

            // Step 5: Eliminate hub DOFs from the reduced system
            int n_reduced = static_cast<int>(K_k.rows());

            std::vector<int> free_reduced_map;
            for (int i = 0; i < n_reduced; i++) {
                if (hub_red_set.count(i) == 0) {
                    free_reduced_map.push_back(i);
                }
            }

            int n_free = static_cast<int>(free_reduced_map.size());
            if (n_free <= 1) return std::nullopt;

            // Extract submatrices for free DOFs
            SpMatcd K_free(n_free, n_free), M_free(n_free, n_free);
            {
                std::vector<TripletC> kf, mf;
                std::vector<int> reduced_to_free(n_reduced, -1);
                for (int i = 0; i < n_free; i++) {
                    reduced_to_free[free_reduced_map[i]] = i;
                }

                for (int col = 0; col < K_k.outerSize(); ++col) {
                    for (SpMatcd::InnerIterator it(K_k, col); it; ++it) {
                        int ri = reduced_to_free[it.row()];
                        int rj = reduced_to_free[it.col()];
                        if (ri >= 0 && rj >= 0) {
                            kf.emplace_back(ri, rj, it.value());
                        }
                    }
                }
                for (int col = 0; col < M_k.outerSize(); ++col) {
                    for (SpMatcd::InnerIterator it(M_k, col); it; ++it) {
                        int ri = reduced_to_free[it.row()];
                        int rj = reduced_to_free[it.col()];
                        if (ri >= 0 && rj >= 0) {
                            mf.emplace_back(ri, rj, it.value());
                        }
                    }
                }
                K_free.setFromTriplets(kf.begin(), kf.end());
                M_free.setFromTriplets(mf.begin(), mf.end());
            }

            // Step 6: Solve the eigenvalue problem
            SolverConfig config;
            config.nev = std::min(num_modes_per_harmonic, n_free - 1);
            if (config.nev <= 0) return std::nullopt;
            config.ncv = std::min(std::max(4 * config.nev + 1, 40), n_free);
            config.shift = 1.0;

            ModalSolver modal_solver;
            ModalResult result;
            SolverStatus status;

            bool is_real_case = (k == 0) || (mesh_.num_sectors % 2 == 0 && k == max_k);

            if (is_real_case) {
                // k=0 or k=N/2: matrices are real symmetric
                SpMatd K_real(n_free, n_free), M_real(n_free, n_free);
                {
                    std::vector<Triplet> kr, mr;
                    for (int col = 0; col < K_free.outerSize(); ++col) {
                        for (SpMatcd::InnerIterator it(K_free, col); it; ++it)
                            if (std::abs(it.value().real()) > 1e-15)
                                kr.emplace_back(it.row(), it.col(), it.value().real());
                    }
                    for (int col = 0; col < M_free.outerSize(); ++col) {
                        for (SpMatcd::InnerIterator it(M_free, col); it; ++it)
                            if (std::abs(it.value().real()) > 1e-15)
                                mr.emplace_back(it.row(), it.col(), it.value().real());
                    }
                    K_real.setFromTriplets(kr.begin(), kr.end());
                    M_real.setFromTriplets(mr.begin(), mr.end());
                }
                std::tie(result, status) = modal_solver.solve_real(K_real, M_real, config);
            } else {
                // General k: complex Hermitian matrices
                std::tie(result, status) = modal_solver.solve_complex_hermitian(
                    K_free, M_free, config);
            }

            if (!status.converged && status.num_converged == 0) return std::nullopt;

            // Step 7: Apply added mass correction
            double freq_ratio = added_mass_factor(k);
            if (freq_ratio < 1.0 && freq_ratio > 0.0) {
                result.frequencies *= freq_ratio;
            }

            // Step 8: Compute FW/BW whirl direction for rotating case
            int num_modes = static_cast<int>(result.frequencies.size());
            result.whirl_direction = Eigen::VectorXi::Zero(num_modes);

            if (omega > 0.0 && k > 0 && k < max_k) {
                double k_omega_hz = k * omega / (2.0 * PI);

                Eigen::VectorXd fw_freqs(num_modes);
                Eigen::VectorXd bw_freqs(num_modes);

                for (int m = 0; m < num_modes; m++) {
                    fw_freqs(m) = result.frequencies(m) + k_omega_hz;
                    bw_freqs(m) = std::abs(result.frequencies(m) - k_omega_hz);
                }

                Eigen::VectorXd all_freqs(2 * num_modes);
                Eigen::VectorXi all_whirl(2 * num_modes);
                Eigen::MatrixXcd all_modes(result.mode_shapes.rows(), 2 * num_modes);

                for (int m = 0; m < num_modes; m++) {
                    all_freqs(2 * m) = fw_freqs(m);
                    all_whirl(2 * m) = 1;
                    all_modes.col(2 * m) = result.mode_shapes.col(m);

                    all_freqs(2 * m + 1) = bw_freqs(m);
                    all_whirl(2 * m + 1) = -1;
                    all_modes.col(2 * m + 1) = result.mode_shapes.col(m);
                }

                std::vector<int> sort_idx(2 * num_modes);
                std::iota(sort_idx.begin(), sort_idx.end(), 0);
                std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) {
                    return all_freqs(a) < all_freqs(b);
                });

                result.frequencies.resize(2 * num_modes);
                result.whirl_direction.resize(2 * num_modes);
                result.mode_shapes.resize(result.mode_shapes.rows(), 2 * num_modes);

                for (int i = 0; i < 2 * num_modes; i++) {
                    result.frequencies(i) = all_freqs(sort_idx[i]);
                    result.whirl_direction(i) = all_whirl(sort_idx[i]);
                    result.mode_shapes.col(i) = all_modes.col(sort_idx[i]);
                }
            } else if (omega > 0.0 && (k == 0 || k == max_k)) {
                result.whirl_direction = Eigen::VectorXi::Zero(num_modes);
            }

            // Step 9: Expand mode shapes from free DOFs to full DOF space
            {
                SpMatcd T = build_cyclic_transformation(k);
                int full_ndof = mesh_.num_dof();
                int n_reduced_T = static_cast<int>(T.cols());
                int n_result_modes = static_cast<int>(result.mode_shapes.cols());

                // 9a: Expand from n_free to n_reduced (insert zeros at hub DOFs)
                Eigen::MatrixXcd modes_reduced = Eigen::MatrixXcd::Zero(
                    n_reduced_T, n_result_modes);
                for (int i = 0; i < static_cast<int>(free_reduced_map.size()); i++) {
                    if (i < result.mode_shapes.rows()) {
                        modes_reduced.row(free_reduced_map[i]) = result.mode_shapes.row(i);
                    }
                }

                // 9b: Apply T to get full DOF mode shapes: u_full = T * u_reduced
                result.mode_shapes = T * modes_reduced;
            }

            result.harmonic_index = k;
            result.rpm = rpm;
            return result;
        } catch (const std::exception& e) {
            std::cerr << "[CyclicSolver] k=" << k << " FAILED: " << e.what() << std::endl;
            return std::nullopt;
        }
    };

    // Launch all harmonic index solves in parallel
    std::vector<std::future<std::optional<ModalResult>>> futures;
    futures.reserve(max_k + 1);
    for (int k = 0; k <= max_k; k++) {
        futures.push_back(std::async(std::launch::async, solve_harmonic, k));
    }

    // Collect results
    std::vector<ModalResult> results;
    for (auto& fut : futures) {
        auto opt = fut.get();
        if (opt.has_value()) {
            results.push_back(std::move(*opt));
        }
    }

    // Sort by harmonic index for deterministic output
    std::sort(results.begin(), results.end(), [](const ModalResult& a, const ModalResult& b) {
        return a.harmonic_index < b.harmonic_index;
    });

    return results;
}

std::vector<std::vector<ModalResult>> CyclicSymmetrySolver::solve_rpm_sweep(
    const Eigen::VectorXd& rpm_values, int num_modes_per_harmonic) {
    std::vector<std::vector<ModalResult>> all_results;
    all_results.reserve(rpm_values.size());
    for (int i = 0; i < rpm_values.size(); i++) {
        all_results.push_back(solve_at_rpm(rpm_values(i), num_modes_per_harmonic));
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
            for (int m = 0; m < r.frequencies.size(); m++) {
                int whirl = (r.whirl_direction.size() > m) ? r.whirl_direction(m) : 0;
                file << r.rpm << "," << r.harmonic_index << ","
                     << (m + 1) << "," << r.frequencies(m) << ","
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
        for (int m = 0; m < r.frequencies.size(); m++) {
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
        if (3 * i + 2 < mode.size()) {
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
        if (3 * i + 2 < mode.size()) {
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
        if (3 * i + 2 < mode.size()) {
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
