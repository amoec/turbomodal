#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include "turbomodal/common.hpp"
#include "turbomodal/material.hpp"
#include "turbomodal/element.hpp"
#include "turbomodal/mesh.hpp"
#include "turbomodal/assembler.hpp"
#include "turbomodal/modal_solver.hpp"
#include "turbomodal/cyclic_solver.hpp"
#include "turbomodal/added_mass.hpp"
#include "turbomodal/rotating_effects.hpp"
#include "turbomodal/damping.hpp"
#include "turbomodal/forced_response.hpp"
#include "turbomodal/mistuning.hpp"
#include "turbomodal/mode_identification.hpp"

namespace py = pybind11;
using namespace turbomodal;

// Helper: convert Eigen sparse matrix to scipy.sparse.csc_matrix
// Returns a Python object (scipy.sparse.csc_matrix)
template <typename Scalar>
py::object eigen_sparse_to_scipy(const Eigen::SparseMatrix<Scalar>& mat) {
    // Ensure compressed format
    Eigen::SparseMatrix<Scalar> compressed = mat;
    compressed.makeCompressed();

    int nnz = static_cast<int>(compressed.nonZeros());
    int rows = static_cast<int>(compressed.rows());
    int cols = static_cast<int>(compressed.cols());
    int outer_size = static_cast<int>(compressed.outerSize());

    // Copy data, indices, indptr to numpy arrays via raw pointers
    py::array_t<Scalar> data(nnz);
    py::array_t<int> indices(nnz);
    py::array_t<int> indptr(outer_size + 1);

    std::memcpy(data.mutable_data(), compressed.valuePtr(), nnz * sizeof(Scalar));
    std::memcpy(indices.mutable_data(), compressed.innerIndexPtr(), nnz * sizeof(int));
    std::memcpy(indptr.mutable_data(), compressed.outerIndexPtr(), (outer_size + 1) * sizeof(int));

    py::module_ scipy_sparse = py::module_::import("scipy.sparse");
    return scipy_sparse.attr("csc_matrix")(
        py::make_tuple(data, indices, indptr),
        py::make_tuple(rows, cols)
    );
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Cyclic symmetry FEA solver for turbomachinery modal analysis";

    // --- Material ---
    py::class_<Material>(m, "Material")
        .def(py::init<double, double, double>(),
             py::arg("E"), py::arg("nu"), py::arg("rho"),
             "Create material with Young's modulus (Pa), Poisson's ratio, and density (kg/m^3)")
        .def(py::init<double, double, double, double, double>(),
             py::arg("E"), py::arg("nu"), py::arg("rho"),
             py::arg("T_ref"), py::arg("E_slope"),
             "Create temperature-dependent material")
        .def_readwrite("E", &Material::E, "Young's modulus (Pa)")
        .def_readwrite("nu", &Material::nu, "Poisson's ratio")
        .def_readwrite("rho", &Material::rho, "Density (kg/m^3)")
        .def_readwrite("T_ref", &Material::T_ref, "Reference temperature (K)")
        .def_readwrite("E_slope", &Material::E_slope, "dE/dT (Pa/K)")
        .def("at_temperature", &Material::at_temperature, py::arg("T"),
             "Return material with modulus adjusted to temperature T")
        .def("validate", &Material::validate, "Validate material properties")
        .def("__repr__", [](const Material& m) {
            return "<Material E=" + std::to_string(m.E) +
                   " nu=" + std::to_string(m.nu) +
                   " rho=" + std::to_string(m.rho) + ">";
        });

    // --- NodeSet ---
    py::class_<NodeSet>(m, "NodeSet")
        .def(py::init<>())
        .def_readwrite("name", &NodeSet::name)
        .def_readwrite("node_ids", &NodeSet::node_ids)
        .def("__repr__", [](const NodeSet& ns) {
            return "<NodeSet '" + ns.name + "' with " +
                   std::to_string(ns.node_ids.size()) + " nodes>";
        });

    // --- Mesh ---
    py::class_<Mesh>(m, "Mesh")
        .def(py::init<>())
        .def_readwrite("nodes", &Mesh::nodes, "Node coordinates (N x 3)")
        .def_readwrite("elements", &Mesh::elements, "Element connectivity (E x 10)")
        .def_readwrite("node_sets", &Mesh::node_sets)
        .def_readwrite("left_boundary", &Mesh::left_boundary)
        .def_readwrite("right_boundary", &Mesh::right_boundary)
        .def_readwrite("matched_pairs", &Mesh::matched_pairs)
        .def_readwrite("free_boundary", &Mesh::free_boundary)
        .def_readwrite("num_sectors", &Mesh::num_sectors)
        .def_readwrite("rotation_axis", &Mesh::rotation_axis)
        .def("load_from_gmsh", &Mesh::load_from_gmsh, py::arg("filename"),
             "Load mesh from Gmsh MSH 2.x file",
             py::call_guard<py::gil_scoped_release>())
        .def("load_from_arrays", &Mesh::load_from_arrays,
             py::arg("node_coords"), py::arg("element_connectivity"),
             py::arg("node_sets"), py::arg("num_sectors"),
             py::arg("rotation_axis") = 2,
             "Load mesh from arrays (auto-identifies cyclic boundaries)")
        .def("identify_cyclic_boundaries", &Mesh::identify_cyclic_boundaries,
             py::arg("tolerance") = 1e-6)
        .def("match_boundary_nodes", &Mesh::match_boundary_nodes)
        .def("find_node_set", &Mesh::find_node_set, py::arg("name"),
             py::return_value_policy::reference_internal)
        .def("num_nodes", &Mesh::num_nodes)
        .def("num_elements", &Mesh::num_elements)
        .def("num_dof", &Mesh::num_dof)
        .def("__repr__", [](const Mesh& m) {
            return "<Mesh " + std::to_string(m.num_nodes()) + " nodes, " +
                   std::to_string(m.num_elements()) + " elements, " +
                   std::to_string(m.num_sectors) + " sectors>";
        });

    // --- GlobalAssembler ---
    py::class_<GlobalAssembler>(m, "GlobalAssembler")
        .def(py::init<>())
        .def("assemble", &GlobalAssembler::assemble,
             py::arg("mesh"), py::arg("material"),
             "Assemble global stiffness and mass matrices",
             py::call_guard<py::gil_scoped_release>())
        .def("assemble_stress_stiffening", &GlobalAssembler::assemble_stress_stiffening,
             py::arg("mesh"), py::arg("material"),
             py::arg("displacement"), py::arg("omega"),
             "Assemble stress stiffening matrix from prestress",
             py::call_guard<py::gil_scoped_release>())
        .def("assemble_rotating_effects", &GlobalAssembler::assemble_rotating_effects,
             py::arg("mesh"), py::arg("material"), py::arg("omega"),
             "Assemble spin softening and gyroscopic matrices",
             py::call_guard<py::gil_scoped_release>())
        .def("assemble_centrifugal_load", &GlobalAssembler::assemble_centrifugal_load,
             py::arg("mesh"), py::arg("material"),
             py::arg("omega"), py::arg("axis"),
             "Assemble centrifugal load vector",
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("K", [](const GlobalAssembler& a) {
            return eigen_sparse_to_scipy(a.K());
        }, "Global stiffness matrix (scipy.sparse.csc_matrix)")
        .def_property_readonly("M", [](const GlobalAssembler& a) {
            return eigen_sparse_to_scipy(a.M());
        }, "Global mass matrix (scipy.sparse.csc_matrix)")
        .def_property_readonly("K_sigma", [](const GlobalAssembler& a) {
            return eigen_sparse_to_scipy(a.K_sigma());
        }, "Stress stiffening matrix (scipy.sparse.csc_matrix)")
        .def_property_readonly("G", [](const GlobalAssembler& a) {
            return eigen_sparse_to_scipy(a.G());
        }, "Gyroscopic matrix (scipy.sparse.csc_matrix)")
        .def_property_readonly("K_omega", [](const GlobalAssembler& a) {
            return eigen_sparse_to_scipy(a.K_omega());
        }, "Spin softening matrix (scipy.sparse.csc_matrix)");

    // --- SolverConfig ---
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("nev", &SolverConfig::nev, "Number of eigenvalues")
        .def_readwrite("ncv", &SolverConfig::ncv, "Lanczos vectors (0=auto)")
        .def_readwrite("shift", &SolverConfig::shift, "Shift-invert sigma")
        .def_readwrite("tolerance", &SolverConfig::tolerance)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations);

    // --- SolverStatus ---
    py::class_<SolverStatus>(m, "SolverStatus")
        .def(py::init<>())
        .def_readonly("converged", &SolverStatus::converged)
        .def_readonly("num_converged", &SolverStatus::num_converged)
        .def_readonly("iterations", &SolverStatus::iterations)
        .def_readonly("max_residual", &SolverStatus::max_residual)
        .def_readonly("message", &SolverStatus::message)
        .def("__repr__", [](const SolverStatus& s) {
            return "<SolverStatus converged=" + std::string(s.converged ? "True" : "False") +
                   " num_converged=" + std::to_string(s.num_converged) + ">";
        });

    // --- ModalResult ---
    py::class_<ModalResult>(m, "ModalResult")
        .def(py::init<>())
        .def_readwrite("harmonic_index", &ModalResult::harmonic_index)
        .def_readwrite("rpm", &ModalResult::rpm)
        .def_readwrite("frequencies", &ModalResult::frequencies)
        .def_readwrite("mode_shapes", &ModalResult::mode_shapes)
        .def_readwrite("whirl_direction", &ModalResult::whirl_direction)
        .def("wave_propagation_velocity", &ModalResult::wave_propagation_velocity,
             py::arg("radius"),
             "Compute wave propagation velocity (m/s) for each mode. v = 2*pi*f*R/ND")
        .def("__repr__", [](const ModalResult& r) {
            return "<ModalResult ND=" + std::to_string(r.harmonic_index) +
                   " rpm=" + std::to_string(r.rpm) +
                   " n_modes=" + std::to_string(r.frequencies.size()) + ">";
        });

    // --- FluidConfig ---
    py::enum_<FluidConfig::Type>(m, "FluidType")
        .value("NONE", FluidConfig::Type::NONE)
        .value("GAS_AIC", FluidConfig::Type::GAS_AIC)
        .value("LIQUID_ANALYTICAL", FluidConfig::Type::LIQUID_ANALYTICAL)
        .value("KWAK_ANALYTICAL", FluidConfig::Type::KWAK_ANALYTICAL)
        .value("POTENTIAL_FLOW_BEM", FluidConfig::Type::POTENTIAL_FLOW_BEM)
        .value("LIQUID_ACOUSTIC_FEM", FluidConfig::Type::LIQUID_ACOUSTIC_FEM);

    py::class_<FluidConfig>(m, "FluidConfig")
        .def(py::init<>())
        .def_readwrite("type", &FluidConfig::type)
        .def_readwrite("fluid_density", &FluidConfig::fluid_density)
        .def_readwrite("disk_radius", &FluidConfig::disk_radius)
        .def_readwrite("disk_thickness", &FluidConfig::disk_thickness)
        .def_readwrite("speed_of_sound", &FluidConfig::speed_of_sound);

    // --- CyclicSymmetrySolver ---
    py::class_<CyclicSymmetrySolver>(m, "CyclicSymmetrySolver")
        .def(py::init<const Mesh&, const Material&, const FluidConfig&, bool>(),
             py::arg("mesh"), py::arg("material"),
             py::arg("fluid") = FluidConfig(),
             py::arg("apply_hub_constraint") = true,
             py::keep_alive<1, 2>())  // solver keeps mesh alive
        .def("solve_at_rpm", &CyclicSymmetrySolver::solve_at_rpm,
             py::arg("rpm"), py::arg("num_modes_per_harmonic"),
             py::arg("harmonic_indices") = std::vector<int>{},
             py::arg("max_threads") = 0,
             "Solve modal analysis at a given RPM",
             py::call_guard<py::gil_scoped_release>())
        .def("solve_rpm_sweep", &CyclicSymmetrySolver::solve_rpm_sweep,
             py::arg("rpm_values"), py::arg("num_modes_per_harmonic"),
             py::arg("harmonic_indices") = std::vector<int>{},
             py::arg("max_threads") = 0,
             "Solve over a range of RPM values",
             py::call_guard<py::gil_scoped_release>())
        .def("export_campbell_csv", &CyclicSymmetrySolver::export_campbell_csv,
             py::arg("filename"), py::arg("results"))
        .def("export_zzenf_csv", &CyclicSymmetrySolver::export_zzenf_csv,
             py::arg("filename"), py::arg("results"))
        .def("export_mode_shape_vtk", &CyclicSymmetrySolver::export_mode_shape_vtk,
             py::arg("filename"), py::arg("result"), py::arg("mode_index"));

    // --- AddedMassModel ---
    py::class_<AddedMassModel>(m, "AddedMassModel")
        .def_static("kwak_avmi", &AddedMassModel::kwak_avmi,
             py::arg("nodal_diameter"), py::arg("rho_fluid"),
             py::arg("rho_structure"), py::arg("thickness"), py::arg("radius"))
        .def_static("frequency_ratio", &AddedMassModel::frequency_ratio,
             py::arg("nodal_diameter"), py::arg("rho_fluid"),
             py::arg("rho_structure"), py::arg("thickness"), py::arg("radius"));

    // --- ModalSolver ---
    py::class_<ModalSolver>(m, "ModalSolver")
        .def(py::init<>());

    // --- DampingConfig ---
    py::enum_<DampingConfig::Type>(m, "DampingType")
        .value("NONE", DampingConfig::Type::NONE)
        .value("MODAL", DampingConfig::Type::MODAL)
        .value("RAYLEIGH", DampingConfig::Type::RAYLEIGH);

    py::class_<DampingConfig>(m, "DampingConfig")
        .def(py::init<>())
        .def_readwrite("type", &DampingConfig::type)
        .def_readwrite("modal_damping_ratios", &DampingConfig::modal_damping_ratios)
        .def_readwrite("rayleigh_alpha", &DampingConfig::rayleigh_alpha)
        .def_readwrite("rayleigh_beta", &DampingConfig::rayleigh_beta)
        .def_readwrite("aero_damping_ratios", &DampingConfig::aero_damping_ratios)
        .def("effective_damping", &DampingConfig::effective_damping,
             py::arg("mode_index"), py::arg("omega_r"),
             "Compute effective damping ratio for mode");

    // --- ForcedResponseConfig ---
    py::enum_<ForcedResponseConfig::ExcitationType>(m, "ExcitationType")
        .value("UNIFORM_PRESSURE", ForcedResponseConfig::ExcitationType::UNIFORM_PRESSURE)
        .value("POINT_FORCE", ForcedResponseConfig::ExcitationType::POINT_FORCE)
        .value("SPATIAL_DISTRIBUTION", ForcedResponseConfig::ExcitationType::SPATIAL_DISTRIBUTION);

    py::class_<ForcedResponseConfig>(m, "ForcedResponseConfig")
        .def(py::init<>())
        .def_readwrite("engine_order", &ForcedResponseConfig::engine_order)
        .def_readwrite("force_amplitude", &ForcedResponseConfig::force_amplitude)
        .def_readwrite("excitation_type", &ForcedResponseConfig::excitation_type)
        .def_readwrite("force_node_id", &ForcedResponseConfig::force_node_id)
        .def_readwrite("force_direction", &ForcedResponseConfig::force_direction)
        .def_readwrite("force_vector", &ForcedResponseConfig::force_vector)
        .def_readwrite("freq_min", &ForcedResponseConfig::freq_min)
        .def_readwrite("freq_max", &ForcedResponseConfig::freq_max)
        .def_readwrite("num_freq_points", &ForcedResponseConfig::num_freq_points);

    // --- ForcedResponseResult ---
    py::class_<ForcedResponseResult>(m, "ForcedResponseResult")
        .def(py::init<>())
        .def_readwrite("engine_order", &ForcedResponseResult::engine_order)
        .def_readwrite("rpm", &ForcedResponseResult::rpm)
        .def_readwrite("natural_frequencies", &ForcedResponseResult::natural_frequencies)
        .def_readwrite("modal_forces", &ForcedResponseResult::modal_forces)
        .def_readwrite("modal_damping_ratios", &ForcedResponseResult::modal_damping_ratios)
        .def_readwrite("participation_factors", &ForcedResponseResult::participation_factors)
        .def_readwrite("effective_modal_mass", &ForcedResponseResult::effective_modal_mass)
        .def_readwrite("sweep_frequencies", &ForcedResponseResult::sweep_frequencies)
        .def_readwrite("modal_amplitudes", &ForcedResponseResult::modal_amplitudes)
        .def_readwrite("max_response_amplitude", &ForcedResponseResult::max_response_amplitude)
        .def_readwrite("resonance_frequencies", &ForcedResponseResult::resonance_frequencies);

    // --- ForcedResponseSolver ---
    py::class_<ForcedResponseSolver>(m, "ForcedResponseSolver")
        .def(py::init<const Mesh&, const DampingConfig&>(),
             py::arg("mesh"), py::arg("damping") = DampingConfig(),
             py::keep_alive<1, 2>())
        .def("solve", &ForcedResponseSolver::solve,
             py::arg("modal_results"), py::arg("rpm"), py::arg("config"),
             "Compute forced response from modal results",
             py::call_guard<py::gil_scoped_release>())
        .def_static("compute_modal_forces", &ForcedResponseSolver::compute_modal_forces,
             py::arg("mode_shapes"), py::arg("force_vector"))
        .def_static("modal_frf", &ForcedResponseSolver::modal_frf,
             py::arg("omega"), py::arg("omega_r"), py::arg("Q_r"), py::arg("zeta_r"))
        .def_static("compute_participation_factors",
             &ForcedResponseSolver::compute_participation_factors,
             py::arg("mode_shapes"), py::arg("M"), py::arg("direction"))
        .def_static("compute_effective_modal_mass",
             &ForcedResponseSolver::compute_effective_modal_mass,
             py::arg("mode_shapes"), py::arg("M"), py::arg("direction"))
        .def("build_eo_excitation", &ForcedResponseSolver::build_eo_excitation,
             py::arg("engine_order"), py::arg("amplitude"),
             py::arg("type") = ForcedResponseConfig::ExcitationType::UNIFORM_PRESSURE,
             py::arg("force_node_id") = -1,
             py::arg("force_dir") = Eigen::Vector3d(0, 0, 1));

    // --- MistuningConfig ---
    py::class_<MistuningConfig>(m, "MistuningConfig")
        .def(py::init<>())
        .def_readwrite("blade_frequency_deviations", &MistuningConfig::blade_frequency_deviations)
        .def_readwrite("tuned_frequencies", &MistuningConfig::tuned_frequencies)
        .def_readwrite("mode_family_index", &MistuningConfig::mode_family_index);

    // --- MistuningResult ---
    py::class_<MistuningResult>(m, "MistuningResult")
        .def(py::init<>())
        .def_readwrite("frequencies", &MistuningResult::frequencies)
        .def_readwrite("blade_amplitudes", &MistuningResult::blade_amplitudes)
        .def_readwrite("amplitude_magnification", &MistuningResult::amplitude_magnification)
        .def_readwrite("peak_magnification", &MistuningResult::peak_magnification)
        .def_readwrite("localization_ipr", &MistuningResult::localization_ipr);

    // --- FMMSolver ---
    py::class_<FMMSolver>(m, "FMMSolver")
        .def_static("solve", &FMMSolver::solve,
             py::arg("num_sectors"), py::arg("tuned_frequencies"),
             py::arg("blade_frequency_deviations"))
        .def_static("random_mistuning", &FMMSolver::random_mistuning,
             py::arg("num_sectors"), py::arg("sigma"),
             py::arg("seed") = 42u);

    // --- ModeIdentification ---
    py::class_<ModeIdentification>(m, "ModeIdentification")
        .def(py::init<>())
        .def_readwrite("nodal_diameter", &ModeIdentification::nodal_diameter)
        .def_readwrite("nodal_circle", &ModeIdentification::nodal_circle)
        .def_readwrite("whirl_direction", &ModeIdentification::whirl_direction)
        .def_readwrite("frequency", &ModeIdentification::frequency)
        .def_readwrite("wave_velocity", &ModeIdentification::wave_velocity)
        .def_readwrite("participation_factor", &ModeIdentification::participation_factor)
        .def_readwrite("family_label", &ModeIdentification::family_label)
        .def("__repr__", [](const ModeIdentification& id) {
            return "<ModeIdentification ND=" + std::to_string(id.nodal_diameter) +
                   " NC=" + std::to_string(id.nodal_circle) +
                   " " + id.family_label +
                   " f=" + std::to_string(id.frequency) + " Hz>";
        });

    // --- GroundTruthLabel ---
    py::class_<GroundTruthLabel>(m, "GroundTruthLabel")
        .def(py::init<>())
        .def_readwrite("rpm", &GroundTruthLabel::rpm)
        .def_readwrite("temperature", &GroundTruthLabel::temperature)
        .def_readwrite("pressure_ratio", &GroundTruthLabel::pressure_ratio)
        .def_readwrite("inlet_distortion", &GroundTruthLabel::inlet_distortion)
        .def_readwrite("tip_clearance", &GroundTruthLabel::tip_clearance)
        .def_readwrite("nodal_diameter", &GroundTruthLabel::nodal_diameter)
        .def_readwrite("nodal_circle", &GroundTruthLabel::nodal_circle)
        .def_readwrite("whirl_direction", &GroundTruthLabel::whirl_direction)
        .def_readwrite("frequency", &GroundTruthLabel::frequency)
        .def_readwrite("wave_velocity", &GroundTruthLabel::wave_velocity)
        .def_readwrite("family_label", &GroundTruthLabel::family_label)
        .def_readwrite("participation_factor", &GroundTruthLabel::participation_factor)
        .def_readwrite("effective_modal_mass", &GroundTruthLabel::effective_modal_mass)
        .def_readwrite("amplitude", &GroundTruthLabel::amplitude)
        .def_readwrite("damping_ratio", &GroundTruthLabel::damping_ratio)
        .def_readwrite("amplitude_magnification", &GroundTruthLabel::amplitude_magnification)
        .def_readwrite("is_localized", &GroundTruthLabel::is_localized)
        .def_readwrite("condition_id", &GroundTruthLabel::condition_id)
        .def_readwrite("mode_index", &GroundTruthLabel::mode_index);

    // --- Free functions ---
    m.def("identify_nodal_circles", &identify_nodal_circles,
          py::arg("mode_shape"), py::arg("mesh"),
          "Identify nodal circle count from mode shape");
    m.def("classify_mode_family", &classify_mode_family,
          py::arg("mode_shape"), py::arg("mesh"),
          "Classify mode as B(ending), T(orsion), or A(xial)");
    m.def("identify_modes", &identify_modes,
          py::arg("result"), py::arg("mesh"),
          py::arg("characteristic_radius") = 0.0,
          "Identify all modes in a ModalResult");
}
