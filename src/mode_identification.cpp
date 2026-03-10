#include "turbomodal/mode_identification.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace turbomodal {

int identify_nodal_circles(
    const Eigen::VectorXcd& mode_shape,
    const Mesh& mesh)
{
    int n_nodes = mesh.num_nodes();
    if (n_nodes == 0 || mode_shape.size() < 3) return 0;
    if (mode_shape.size() < 3 * n_nodes) {
        throw std::invalid_argument(
            "identify_nodal_circles: mode_shape size (" +
            std::to_string(mode_shape.size()) + ") < 3 * n_nodes (" +
            std::to_string(3 * n_nodes) + ")");
    }

    double sector_angle = 2.0 * PI / mesh.num_sectors;
    double mid_angle = sector_angle / 2.0;
    double angle_tolerance = sector_angle * 0.15;  // 15% of sector width

    // --- Step 1: collect 3D complex displacement along radial strip ---
    struct RadialSample {
        double radius;
        std::complex<double> dx, dy, dz;  // Cartesian displacement
    };
    std::vector<RadialSample> samples;
    samples.reserve(n_nodes);

    for (int i = 0; i < n_nodes; i++) {
        double x = mesh.nodes(i, 0);
        double y = mesh.nodes(i, 1);
        double r = std::sqrt(x * x + y * y);
        double theta = std::atan2(y, x);
        if (theta < 0) theta += 2.0 * PI;

        if (std::abs(theta - mid_angle) < angle_tolerance && r > 1e-10) {
            samples.push_back({r,
                mode_shape(3 * i), mode_shape(3 * i + 1), mode_shape(3 * i + 2)});
        }
    }

    if (samples.size() < 3) return 0;

    // Sort by radius
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) { return a.radius < b.radius; });

    // --- Step 2: find reference displacement vector (max amplitude node) ---
    double max_mag2 = 0.0;
    size_t ref_idx = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        double m2 = std::norm(samples[i].dx)
                   + std::norm(samples[i].dy)
                   + std::norm(samples[i].dz);
        if (m2 > max_mag2) {
            max_mag2 = m2;
            ref_idx = i;
        }
    }
    if (max_mag2 < 1e-30) return 0;
    double ref_mag = std::sqrt(max_mag2);
    const auto& ref = samples[ref_idx];

    // --- Step 3: project each sample onto reference direction+phase ---
    // p_i = Re(d_i · conj(d_ref)) / |d_ref|
    // This gives a signed scalar: positive when displacement aligns with
    // the reference, negative when it opposes.  Handles mixed-direction
    // modes (staggered blades) and complex phase automatically.

    // --- Step 4: bin radial profile to suppress mesh noise ---
    double r_min = samples.front().radius;
    double r_max = samples.back().radius;
    double r_span = r_max - r_min;
    if (r_span < 1e-12) return 0;

    int n_bins = std::max(10, std::min(60,
        static_cast<int>(samples.size() / 5)));
    std::vector<double> bin_sum(n_bins, 0.0);
    std::vector<int> bin_count(n_bins, 0);

    for (size_t i = 0; i < samples.size(); i++) {
        // 3D complex dot product with conjugate of reference
        double proj = (samples[i].dx * std::conj(ref.dx)
                     + samples[i].dy * std::conj(ref.dy)
                     + samples[i].dz * std::conj(ref.dz)).real() / ref_mag;
        double frac = (samples[i].radius - r_min) / r_span;
        int b = static_cast<int>(frac * n_bins);
        if (b >= n_bins) b = n_bins - 1;
        bin_sum[b] += proj;
        bin_count[b]++;
    }

    // Build smoothed profile from non-empty bins
    std::vector<double> profile;
    profile.reserve(n_bins);
    for (int b = 0; b < n_bins; b++) {
        if (bin_count[b] > 0) {
            profile.push_back(bin_sum[b] / bin_count[b]);
        }
    }

    if (profile.size() < 3) return 0;

    // Normalize
    double max_val = 0.0;
    for (double v : profile) max_val = std::max(max_val, std::abs(v));
    if (max_val < 1e-15) return 0;
    for (double& v : profile) v /= max_val;

    // --- Step 5: count zero crossings on smoothed profile ---
    // Track the last bin with significant amplitude and detect sign
    // changes against it.  This avoids missing crossings where the
    // profile passes through near-zero values at the nodal circle.
    int crossings = 0;
    double last_sig = 0.0;
    bool has_last = false;
    for (size_t i = 0; i < profile.size(); i++) {
        if (std::abs(profile[i]) > 0.02) {
            if (has_last &&
                ((last_sig > 0.0 && profile[i] < 0.0) ||
                 (last_sig < 0.0 && profile[i] > 0.0))) {
                crossings++;
            }
            last_sig = profile[i];
            has_last = true;
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
    if (mode_shape.size() < 3 * n_nodes) {
        throw std::invalid_argument(
            "classify_mode_family: mode_shape size (" +
            std::to_string(mode_shape.size()) + ") < 3 * n_nodes (" +
            std::to_string(3 * n_nodes) + ")");
    }

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
