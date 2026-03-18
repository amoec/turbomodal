#include "turbomodal/mode_identification.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace turbomodal {

// ---------------------------------------------------------------------------
// Cylindrical coordinate helpers (anonymous namespace)
// ---------------------------------------------------------------------------

namespace {

struct CylindricalAxes {
    int c1, c2, c_axial;
};

// Map rotation_axis (0=X, 1=Y, 2=Z) to coordinate indices.
// c1,c2 span the rotation plane, c_axial is the rotation axis.
// Same convention used in mesh.cpp for boundary matching.
CylindricalAxes get_cylindrical_axes(int rotation_axis) {
    if (rotation_axis == 0) return {1, 2, 0};  // X-axis: rotate in YZ
    if (rotation_axis == 1) return {0, 2, 1};  // Y-axis: rotate in XZ
    return {0, 1, 2};                           // Z-axis: rotate in XY
}

struct NodeCylData {
    double r;       // radial distance from axis
    double theta;   // circumferential angle [0, 2*PI)
    std::complex<double> ur, ut, ua;  // radial, tangential, axial
};

enum class DominantComponent { RADIAL, TANGENTIAL, AXIAL };

// Count zero crossings in a normalised profile where both sides
// of the crossing exceed a noise threshold.
int count_zero_crossings(const std::vector<double>& profile, double threshold) {
    int crossings = 0;
    double last_sig = 0.0;
    bool has_last = false;
    for (size_t i = 0; i < profile.size(); i++) {
        if (std::abs(profile[i]) > threshold) {
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

// Gaussian-smooth a profile in-place with the given sigma (in bin units).
void gaussian_smooth(std::vector<double>& profile, double sigma) {
    int n = static_cast<int>(profile.size());
    if (n < 3) return;
    int half_w = static_cast<int>(std::ceil(3.0 * sigma));
    std::vector<double> smoothed(n, 0.0);
    for (int b = 0; b < n; b++) {
        double sum_w = 0.0, sum_wv = 0.0;
        for (int k = -half_w; k <= half_w; k++) {
            int idx = b + k;
            if (idx < 0 || idx >= n) continue;
            double w = std::exp(-0.5 * (k * k) / (sigma * sigma));
            sum_w += w;
            sum_wv += w * profile[idx];
        }
        smoothed[b] = sum_wv / sum_w;
    }
    profile = std::move(smoothed);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// identify_nodal_circles — multi-line, cylindrical-decomposition approach
// ---------------------------------------------------------------------------

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

    CylindricalAxes axes = get_cylindrical_axes(mesh.rotation_axis);
    double sector_angle = 2.0 * PI / std::max(mesh.num_sectors, 1);

    // --- Step 1: Precompute cylindrical decomposition for all nodes ---
    std::vector<NodeCylData> all_nodes;
    all_nodes.reserve(n_nodes);
    double r_min_global = 1e30, r_max_global = 0.0;

    for (int i = 0; i < n_nodes; i++) {
        double p1 = mesh.nodes(i, axes.c1);
        double p2 = mesh.nodes(i, axes.c2);
        double r = std::sqrt(p1 * p1 + p2 * p2);
        if (r < 1e-10) continue;  // skip on-axis nodes

        double theta = std::atan2(p2, p1);
        if (theta < 0) theta += 2.0 * PI;

        // Displacement components in the rotation-plane and axial directions
        std::complex<double> d1 = mode_shape(3 * i + axes.c1);
        std::complex<double> d2 = mode_shape(3 * i + axes.c2);
        std::complex<double> da = mode_shape(3 * i + axes.c_axial);

        // Cylindrical decomposition
        std::complex<double> ur = (p1 * d1 + p2 * d2) / r;
        std::complex<double> ut = (-p2 * d1 + p1 * d2) / r;

        all_nodes.push_back({r, theta, ur, ut, da});

        r_min_global = std::min(r_min_global, r);
        r_max_global = std::max(r_max_global, r);
    }

    if (all_nodes.size() < 10) return 0;
    double r_span = r_max_global - r_min_global;
    if (r_span < 1e-12) return 0;

    // --- Step 2: Determine dominant displacement component ---
    double sum_ur2 = 0.0, sum_ut2 = 0.0, sum_ua2 = 0.0;
    for (const auto& nd : all_nodes) {
        sum_ur2 += std::norm(nd.ur);
        sum_ut2 += std::norm(nd.ut);
        sum_ua2 += std::norm(nd.ua);
    }

    DominantComponent dominant;
    if (sum_ua2 >= sum_ur2 && sum_ua2 >= sum_ut2) dominant = DominantComponent::AXIAL;
    else if (sum_ut2 >= sum_ur2) dominant = DominantComponent::TANGENTIAL;
    else dominant = DominantComponent::RADIAL;

    auto get_dominant = [dominant](const NodeCylData& nd) -> std::complex<double> {
        switch (dominant) {
            case DominantComponent::RADIAL:      return nd.ur;
            case DominantComponent::TANGENTIAL:   return nd.ut;
            case DominantComponent::AXIAL:        return nd.ua;
        }
        return nd.ua;
    };

    // --- Step 3: Multi-line radial sampling ---
    double sa_deg = sector_angle * 180.0 / PI;
    int n_lines = std::max(3, std::min(9, static_cast<int>(sa_deg)));
    double margin = sector_angle * 0.08;

    // For single-line fallback (very narrow sectors)
    if (n_lines < 2) n_lines = 1;

    double usable = sector_angle - 2.0 * margin;
    if (usable <= 0) {
        // Sector too narrow for margins — use mid-angle only
        n_lines = 1;
        margin = 0.0;
        usable = sector_angle;
    }

    double line_spacing = (n_lines > 1) ? usable / (n_lines - 1) : 0.0;

    // Adaptive angular tolerance per line
    double angle_tol;
    if (n_lines > 1) {
        angle_tol = std::min(line_spacing * 0.45, sector_angle * 0.06);
        angle_tol = std::max(angle_tol, sector_angle * 0.02);
    } else {
        angle_tol = sector_angle * 0.15;  // single-line fallback
    }

    // --- Steps 4-7: Per-line processing ---
    std::vector<int> crossing_counts;
    crossing_counts.reserve(n_lines);

    for (int li = 0; li < n_lines; li++) {
        double theta_line = margin + li * line_spacing;
        if (n_lines == 1) theta_line = sector_angle / 2.0;

        // Collect nodes near this angular line
        struct LineSample {
            double r;
            std::complex<double> value;  // dominant component
        };
        std::vector<LineSample> line_nodes;

        for (const auto& nd : all_nodes) {
            double dtheta = std::abs(nd.theta - theta_line);
            // Handle wrap-around at 2*PI
            if (dtheta > PI) dtheta = 2.0 * PI - dtheta;
            if (dtheta < angle_tol) {
                line_nodes.push_back({nd.r, get_dominant(nd)});
            }
        }

        if (line_nodes.size() < 5) continue;  // too sparse

        // Sort by radius
        std::sort(line_nodes.begin(), line_nodes.end(),
                  [](const auto& a, const auto& b) { return a.r < b.r; });

        // Phase-align to the max-amplitude node on this line
        double max_mag2 = 0.0;
        size_t ref_idx = 0;
        for (size_t i = 0; i < line_nodes.size(); i++) {
            double m2 = std::norm(line_nodes[i].value);
            if (m2 > max_mag2) {
                max_mag2 = m2;
                ref_idx = i;
            }
        }
        if (max_mag2 < 1e-30) continue;

        double ref_phase = std::arg(line_nodes[ref_idx].value);
        std::complex<double> phase_factor = std::exp(
            std::complex<double>(0.0, -ref_phase));

        // Bin radially
        int n_bins = std::max(10, std::min(40,
            static_cast<int>(2.0 * std::sqrt(static_cast<double>(line_nodes.size())))));
        std::vector<double> bin_sum(n_bins, 0.0);
        std::vector<int> bin_count(n_bins, 0);

        for (const auto& ln : line_nodes) {
            double signed_amp = (ln.value * phase_factor).real();
            double frac = (ln.r - r_min_global) / r_span;
            int b = static_cast<int>(frac * n_bins);
            if (b >= n_bins) b = n_bins - 1;
            if (b < 0) b = 0;
            bin_sum[b] += signed_amp;
            bin_count[b]++;
        }

        // Build profile with linear interpolation for empty bins
        std::vector<double> profile(n_bins, 0.0);
        for (int b = 0; b < n_bins; b++) {
            if (bin_count[b] > 0) {
                profile[b] = bin_sum[b] / bin_count[b];
            }
        }

        // Interpolate empty bins from nearest non-empty neighbours
        for (int b = 0; b < n_bins; b++) {
            if (bin_count[b] > 0) continue;
            int left = b - 1, right = b + 1;
            while (left >= 0 && bin_count[left] == 0) left--;
            while (right < n_bins && bin_count[right] == 0) right++;
            if (left >= 0 && right < n_bins) {
                double t = static_cast<double>(b - left) / (right - left);
                profile[b] = (1.0 - t) * profile[left] + t * profile[right];
            } else if (left >= 0) {
                profile[b] = profile[left];
            } else if (right < n_bins) {
                profile[b] = profile[right];
            }
        }

        // Gaussian smooth (sigma = 1.5 bins)
        gaussian_smooth(profile, 1.5);

        // Normalise to [-1, 1]
        double max_abs = 0.0;
        for (double v : profile) max_abs = std::max(max_abs, std::abs(v));
        if (max_abs < 1e-15) continue;
        for (double& v : profile) v /= max_abs;

        // Count zero crossings (5% noise threshold)
        int nc = count_zero_crossings(profile, 0.05);
        crossing_counts.push_back(nc);
    }

    if (crossing_counts.empty()) return 0;

    // --- Step 8: Consensus vote — median ---
    std::sort(crossing_counts.begin(), crossing_counts.end());
    return crossing_counts[crossing_counts.size() / 2];
}

// ---------------------------------------------------------------------------
// classify_mode_family — rotation-axis-aware cylindrical decomposition
// ---------------------------------------------------------------------------

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

    CylindricalAxes axes = get_cylindrical_axes(mesh.rotation_axis);

    // Compute RMS of each displacement component across all nodes
    double rms_radial = 0.0;
    double rms_tangential = 0.0;
    double rms_axial = 0.0;

    for (int i = 0; i < n_nodes; i++) {
        double p1 = mesh.nodes(i, axes.c1);
        double p2 = mesh.nodes(i, axes.c2);
        double r = std::sqrt(p1 * p1 + p2 * p2);

        std::complex<double> d1 = mode_shape(3 * i + axes.c1);
        std::complex<double> d2 = mode_shape(3 * i + axes.c2);
        std::complex<double> da = mode_shape(3 * i + axes.c_axial);

        if (r > 1e-10) {
            std::complex<double> ur = (p1 * d1 + p2 * d2) / r;
            std::complex<double> ut = (-p2 * d1 + p1 * d2) / r;
            rms_radial += std::norm(ur);
            rms_tangential += std::norm(ut);
        } else {
            rms_radial += std::norm(d1) + std::norm(d2);
        }
        rms_axial += std::norm(da);
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

// ---------------------------------------------------------------------------
// identify_modes — wrapper that builds ModeIdentification structs
// ---------------------------------------------------------------------------

std::vector<ModeIdentification> identify_modes(
    const ModalResult& result,
    const Mesh& mesh,
    double characteristic_radius)
{
    int n_modes = static_cast<int>(result.frequencies.size());
    std::vector<ModeIdentification> ids(n_modes);

    // Auto-compute characteristic radius from mesh extents
    CylindricalAxes axes = get_cylindrical_axes(mesh.rotation_axis);
    double radius = characteristic_radius;
    if (radius <= 0.0) {
        double max_r = 0.0;
        double min_r = 1e30;
        for (int i = 0; i < mesh.num_nodes(); i++) {
            double p1 = mesh.nodes(i, axes.c1);
            double p2 = mesh.nodes(i, axes.c2);
            double r = std::sqrt(p1 * p1 + p2 * p2);
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
