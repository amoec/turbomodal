#pragma once

#include "turbomodal/common.hpp"
#include "turbomodal/modal_solver.hpp"
#include "turbomodal/mesh.hpp"

namespace turbomodal {

struct ModeIdentification {
    int nodal_diameter = 0;
    int nodal_circle = 0;
    int whirl_direction = 0;       // +1 FW, -1 BW, 0 standing
    double frequency = 0.0;        // Hz
    double wave_velocity = 0.0;    // m/s
    double participation_factor = 0.0;
    std::string family_label;      // "1B", "2B", "1T", etc.
};

struct GroundTruthLabel {
    // Operating condition
    double rpm = 0.0;
    double temperature = 293.15;     // K
    double pressure_ratio = 1.0;
    double inlet_distortion = 0.0;   // fraction 0-1
    double tip_clearance = 0.0;      // mm

    // Mode identification
    int nodal_diameter = 0;
    int nodal_circle = 0;
    int whirl_direction = 0;
    double frequency = 0.0;          // Hz
    double wave_velocity = 0.0;      // m/s
    std::string family_label;

    // Response quantities
    double participation_factor = 0.0;
    double effective_modal_mass = 0.0;
    double amplitude = 0.0;          // from forced response (0 if not computed)
    double damping_ratio = 0.0;

    // Mistuning state
    double amplitude_magnification = 1.0;  // 1.0 for tuned
    bool is_localized = false;

    // Metadata
    int condition_id = 0;
    int mode_index = 0;
};

// Identify nodal circle count for a mode by counting radial zero-crossings
// of the dominant displacement component along radial lines.
int identify_nodal_circles(
    const Eigen::VectorXcd& mode_shape,
    const Mesh& mesh);

// Classify mode family based on dominant displacement component.
// Returns "B" for bending, "T" for torsion, "A" for axial.
std::string classify_mode_family(
    const Eigen::VectorXcd& mode_shape,
    const Mesh& mesh);

// Build full identification for all modes in a ModalResult.
// characteristic_radius: 0 = auto-compute from mesh extents.
std::vector<ModeIdentification> identify_modes(
    const ModalResult& result,
    const Mesh& mesh,
    double characteristic_radius = 0.0);

}  // namespace turbomodal
