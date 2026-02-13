#include "turbomodal/mode_identification.hpp"
#include <algorithm>
#include <numeric>

namespace turbomodal {

int identify_nodal_circles(
    const Eigen::VectorXcd& mode_shape,
    const Mesh& mesh)
{
    int n_nodes = mesh.num_nodes();
    if (n_nodes == 0 || mode_shape.size() < 3) return 0;

    // Compute radial distance and z-displacement for each node
    // Use nodes near the middle circumferential angle of the sector
    double sector_angle = 2.0 * PI / mesh.num_sectors;
    double mid_angle = sector_angle / 2.0;

    // Collect (radius, displacement_z) pairs for nodes near mid_angle
    struct RadialSample {
        double radius;
        double displacement;
    };
    std::vector<RadialSample> samples;
    samples.reserve(n_nodes);

    double angle_tolerance = sector_angle * 0.15;  // 15% of sector width

    for (int i = 0; i < n_nodes; i++) {
        double x = mesh.nodes(i, 0);
        double y = mesh.nodes(i, 1);
        double r = std::sqrt(x * x + y * y);
        double theta = std::atan2(y, x);
        if (theta < 0) theta += 2.0 * PI;

        if (std::abs(theta - mid_angle) < angle_tolerance && r > 1e-10) {
            // Use the z-component of displacement (axial/bending)
            double uz = std::abs(mode_shape(3 * i + 2));
            // Also check radial displacement
            double ux = std::abs(mode_shape(3 * i));
            double uy = std::abs(mode_shape(3 * i + 1));
            double ur = (x * ux + y * uy) / r;

            // Use dominant component
            double disp = (uz > std::abs(ur)) ?
                mode_shape(3 * i + 2).real() : ur;
            samples.push_back({r, disp});
        }
    }

    if (samples.size() < 3) return 0;

    // Sort by radius
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) { return a.radius < b.radius; });

    // Normalize displacement
    double max_disp = 0.0;
    for (const auto& s : samples) {
        max_disp = std::max(max_disp, std::abs(s.displacement));
    }
    if (max_disp < 1e-15) return 0;

    // Count zero crossings (sign changes)
    int crossings = 0;
    for (size_t i = 1; i < samples.size(); i++) {
        double d_prev = samples[i - 1].displacement / max_disp;
        double d_curr = samples[i].displacement / max_disp;
        // Only count if both values are significant
        if (std::abs(d_prev) > 0.05 && std::abs(d_curr) > 0.05) {
            if ((d_prev > 0 && d_curr < 0) || (d_prev < 0 && d_curr > 0)) {
                crossings++;
            }
        }
    }

    return crossings;
}

std::string classify_mode_family(
    const Eigen::VectorXcd& mode_shape,
    const Mesh& mesh)
{
    int n_nodes = mesh.num_nodes();
    if (n_nodes == 0) return "B";

    // Compute RMS of each displacement component across all nodes
    double rms_radial = 0.0;
    double rms_tangential = 0.0;
    double rms_axial = 0.0;

    for (int i = 0; i < n_nodes; i++) {
        double x = mesh.nodes(i, 0);
        double y = mesh.nodes(i, 1);
        double r = std::sqrt(x * x + y * y);

        std::complex<double> ux = mode_shape(3 * i);
        std::complex<double> uy = mode_shape(3 * i + 1);
        std::complex<double> uz = mode_shape(3 * i + 2);

        if (r > 1e-10) {
            // Radial and tangential components
            std::complex<double> ur = (x * ux + y * uy) / r;
            std::complex<double> ut = (-y * ux + x * uy) / r;
            rms_radial += std::norm(ur);
            rms_tangential += std::norm(ut);
        } else {
            rms_radial += std::norm(ux) + std::norm(uy);
        }
        rms_axial += std::norm(uz);
    }

    rms_radial = std::sqrt(rms_radial / n_nodes);
    rms_tangential = std::sqrt(rms_tangential / n_nodes);
    rms_axial = std::sqrt(rms_axial / n_nodes);

    double total = rms_radial + rms_tangential + rms_axial;
    if (total < 1e-15) return "B";

    double frac_axial = rms_axial / total;
    double frac_tangential = rms_tangential / total;

    // Bending: axial-dominated
    // Torsion: tangential-dominated
    // Axial/radial: radial-dominated
    if (frac_axial > 0.5) return "B";
    if (frac_tangential > 0.5) return "T";
    return "A";
}

std::vector<ModeIdentification> identify_modes(
    const ModalResult& result,
    const Mesh& mesh,
    double characteristic_radius)
{
    int n_modes = static_cast<int>(result.frequencies.size());
    std::vector<ModeIdentification> ids(n_modes);

    // Auto-compute characteristic radius from mesh extents
    double radius = characteristic_radius;
    if (radius <= 0.0) {
        double max_r = 0.0;
        double min_r = 1e30;
        for (int i = 0; i < mesh.num_nodes(); i++) {
            double x = mesh.nodes(i, 0);
            double y = mesh.nodes(i, 1);
            double r = std::sqrt(x * x + y * y);
            if (r > 1e-10) {
                max_r = std::max(max_r, r);
                min_r = std::min(min_r, r);
            }
        }
        radius = (max_r + min_r) / 2.0;
    }

    Eigen::VectorXd wave_vel = result.wave_propagation_velocity(radius);

    for (int m = 0; m < n_modes; m++) {
        ids[m].nodal_diameter = result.harmonic_index;
        ids[m].whirl_direction = (m < result.whirl_direction.size()) ?
            result.whirl_direction(m) : 0;
        ids[m].frequency = result.frequencies(m);
        ids[m].wave_velocity = wave_vel(m);

        // NC identification from mode shape
        if (m < result.mode_shapes.cols()) {
            Eigen::VectorXcd mode = result.mode_shapes.col(m);
            ids[m].nodal_circle = identify_nodal_circles(mode, mesh);
            std::string family_type = classify_mode_family(mode, mesh);
            // Label: NC+1 followed by type, e.g. "1B", "2B", "1T"
            ids[m].family_label = std::to_string(ids[m].nodal_circle + 1) + family_type;
        }
    }

    return ids;
}

}  // namespace turbomodal
