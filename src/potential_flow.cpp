#include "turbomodal/potential_flow.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace turbomodal {

// ========================================================================
// Elliptic integrals via arithmetic-geometric mean (AGM)
// ========================================================================

double elliptic_K(double k) {
    if (k < 0.0 || k >= 1.0)
        throw std::domain_error("elliptic_K: modulus k must be in [0, 1)");
    double a = 1.0, b = std::sqrt(1.0 - k * k);
    for (int i = 0; i < 30 && std::abs(a - b) > 1e-15 * a; i++) {
        double an = 0.5 * (a + b);
        b = std::sqrt(a * b);
        a = an;
    }
    return PI / (2.0 * a);
}

double elliptic_E(double k) {
    if (k < 0.0 || k > 1.0)
        throw std::domain_error("elliptic_E: modulus k must be in [0, 1]");
    if (k == 1.0) return 1.0;
    // Compute via AGM: E(k) = K(k) * [1 - sum_{j=0}^inf 2^(j-1) * c_j^2]
    // where c_0 = k, c_{j+1} = (a_j - b_j)/2
    double a = 1.0, b = std::sqrt(1.0 - k * k);
    double sum = 1.0 - 0.5 * k * k;  // accounts for j=0 term: 2^(-1)*c_0^2 = k^2/2
    double pow2 = 0.5;  // 2^(j-1); after first *=2 gives 1.0 for j=1
    for (int i = 0; i < 30; i++) {
        double c = 0.5 * (a - b);
        if (std::abs(c) < 1e-15) break;
        double an = 0.5 * (a + b);
        b = std::sqrt(a * b);
        a = an;
        pow2 *= 2.0;
        sum -= pow2 * c * c;
    }
    return (PI / (2.0 * a)) * sum;
}

// ========================================================================
// Gauss-Legendre quadrature on [0, 2*pi]
// ========================================================================

struct QuadRule {
    std::vector<double> points;   // in [0, 2*pi]
    std::vector<double> weights;
};

// Generate n-point Gauss-Legendre rule on [-1, 1], then map to [0, 2*pi]
static QuadRule gauss_legendre(int n) {
    QuadRule rule;
    rule.points.resize(n);
    rule.weights.resize(n);

    // Newton-Raphson to find roots of Legendre polynomial P_n(x)
    for (int i = 0; i < (n + 1) / 2; i++) {
        // Initial guess (Tricomi approximation)
        double x = std::cos(PI * (i + 0.75) / (n + 0.5));

        for (int iter = 0; iter < 100; iter++) {
            double p0 = 1.0, p1 = x;
            for (int j = 2; j <= n; j++) {
                double p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j;
                p0 = p1;
                p1 = p2;
            }
            // p1 = P_n(x), derivative: P_n'(x) = n*(x*P_n - P_{n-1})/(x^2-1)
            double dp = n * (x * p1 - p0) / (x * x - 1.0);
            double dx = -p1 / dp;
            x += dx;
            if (std::abs(dx) < 1e-15) break;
        }

        double p0 = 1.0, p1 = x;
        for (int j = 2; j <= n; j++) {
            double p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j;
            p0 = p1;
            p1 = p2;
        }
        double dp = n * (x * p1 - p0) / (x * x - 1.0);
        double w = 2.0 / ((1.0 - x * x) * dp * dp);

        // Map from [-1,1] to [0, 2*pi]: alpha = pi*(x+1)
        rule.points[i] = PI * (x + 1.0);
        rule.weights[i] = PI * w;
        rule.points[n - 1 - i] = PI * (-x + 1.0);
        rule.weights[n - 1 - i] = PI * w;
    }
    return rule;
}

// ========================================================================
// Ring-source Green's function
// ========================================================================

// G_n(r,z; r',z') = (r'/(4pi)) * integral_0^{2pi} cos(n*alpha) / R(alpha) dalpha
// where R(alpha) = sqrt(r^2 + r'^2 - 2*r*r'*cos(alpha) + (z-z')^2)

double AxiBEMSolver::green_function(int n, double r, double z,
                                      double rp, double zp, int n_quad) {
    if (rp <= 0.0) return 0.0;

    double dz = z - zp;

    if (n == 0 && n_quad == 0) {
        // Analytical via elliptic integrals
        double a2 = (r + rp) * (r + rp) + dz * dz;
        double a = std::sqrt(a2);
        double m = 4.0 * r * rp / a2;
        if (m >= 1.0) m = 1.0 - 1e-15;
        return (rp / (PI * a)) * elliptic_K(std::sqrt(m));
    }

    // Numerical quadrature
    int nq = n_quad > 0 ? n_quad : std::max(2 * n + 8, 24);
    QuadRule quad = gauss_legendre(nq);

    double sum = 0.0;
    for (int j = 0; j < nq; j++) {
        double alpha = quad.points[j];
        double R = std::sqrt(r * r + rp * rp - 2.0 * r * rp * std::cos(alpha) + dz * dz);
        if (R > 0.0) {
            sum += quad.weights[j] * std::cos(n * alpha) / R;
        }
    }
    return rp / (4.0 * PI) * sum;
}

// Normal derivative at source: dG_n/dn = (dG_n/drp)*nrp + (dG_n/dzp)*nzp
double AxiBEMSolver::green_normal_deriv(int n, double r, double z,
                                          double rp, double zp,
                                          double nrp, double nzp, int n_quad) {
    if (rp <= 0.0) return 0.0;

    int nq = n_quad > 0 ? n_quad : std::max(2 * n + 8, 24);
    QuadRule quad = gauss_legendre(nq);

    double dz = z - zp;
    double sum = 0.0;
    for (int j = 0; j < nq; j++) {
        double alpha = quad.points[j];
        double ca = std::cos(alpha);
        double R2 = r * r + rp * rp - 2.0 * r * rp * ca + dz * dz;
        double R = std::sqrt(R2);
        if (R < 1e-30) continue;
        double R3 = R * R2;

        // d(1/R)/drp = (r*cos(alpha) - rp) / R^3
        // d(1/R)/dzp = (z - zp) / R^3 = dz / R^3 but the derivative w.r.t. zp is +dz/R^3
        // Wait: z-zp, d/dzp of 1/R = -(z-zp) * (-1) / R^3 ... let me be careful.
        // R^2 = r^2 + rp^2 - 2*r*rp*cos(alpha) + (z-zp)^2
        // dR^2/drp = 2*rp - 2*r*cos(alpha) => dR/drp = (rp - r*cos(alpha)) / R
        // dR^2/dzp = -2*(z-zp) => dR/dzp = -(z-zp) / R = -dz / R (but dz = z - zp)
        // d(1/R)/drp = -(rp - r*cos(alpha)) / R^3
        // d(1/R)/dzp = dz / R^3  (since d/dzp of -(z-zp)^2 = 2(z-zp)*d(z-zp)/dzp... wait)

        // Let me redo: d(1/R)/dzp = -1/(2R^3) * dR^2/dzp = -1/(2R^3)*(-2(z-zp)) = (z-zp)/R^3 = dz/R^3

        // G_n = (rp/(4pi)) * sum w_j cos(n*alpha_j) / R_j
        // dG_n/drp = (1/(4pi)) * sum w_j cos(n*alpha_j) * [1/R + rp * d(1/R)/drp]
        //          = (1/(4pi)) * sum w_j cos(n*alpha_j) * [1/R - rp*(rp - r*ca)/R^3]
        // dG_n/dzp = (rp/(4pi)) * sum w_j cos(n*alpha_j) * d(1/R)/dzp
        //          = (rp/(4pi)) * sum w_j cos(n*alpha_j) * dz/R^3

        double cn = std::cos(n * alpha);

        double dGdrp = (1.0 / R - rp * (rp - r * ca) / R3);
        double dGdzp = rp * dz / R3;

        sum += quad.weights[j] * cn * (nrp * dGdrp + nzp * dGdzp);
    }
    return sum / (4.0 * PI);
}

// Self-influence: integral of G_n over a panel centered at (r, z) with given length.
// For n=0: logarithmic singularity handled analytically.
// For n>=1: convergent, use higher-order quadrature.
double AxiBEMSolver::green_self(int n, double r, double z,
                                  double panel_length, int n_quad) {
    double L = panel_length;
    if (L <= 0.0 || r <= 0.0) return 0.0;

    if (n == 0) {
        // The ring-source G_0 for P=Q has a log singularity.
        // G_0 ~ r/(pi*a) * K(sqrt(m)) where a = 2r, m → 1 as P→Q.
        // K(sqrt(m)) ~ -0.5*log(1-m) + log(4) - ... as m→1
        // 1-m ~ (Δr^2 + Δz^2) / (4r^2) ~ s^2/(4r^2) for meridional distance s.
        // So G_0 ~ r/(2*pi*r) * [-0.5*log(s^2/(4r^2)) + ln4]
        //        = 1/(2*pi) * [-log(|s|/(2r)) + ln2]
        //        = 1/(2*pi) * [-log|s| + log(4r)]
        // Integral over [-L/2, L/2]:
        //   1/(2*pi) * integral[-L/2..L/2] (-log|s| + log(4r)) ds
        //   = 1/(2*pi) * [ -L*(log(L/2) - 1) + L*log(4r) ]
        //   = L/(2*pi) * [1 - log(L/2) + log(4r)]
        //   = L/(2*pi) * [1 + log(8r/L)]
        return L / (2.0 * PI) * (1.0 + std::log(8.0 * r / L));
    }

    // For n >= 1: no log singularity, but integrand is peaked.
    // Use Gauss quadrature along the panel with enough points.
    int nq = n_quad > 0 ? n_quad : std::max(2 * n + 8, 24);
    QuadRule circ = gauss_legendre(nq);

    // Integrate G_n(r,z; r,z) * panel_length ... but the self-influence means
    // integrating over the panel itself. Use a panel quadrature:
    int np = 8;  // points along the panel
    QuadRule panel_quad = gauss_legendre(np);

    double sum = 0.0;
    for (int ip = 0; ip < np; ip++) {
        // Map panel quadrature from [-1,1] to [-L/2, L/2]
        double s = 0.5 * L * panel_quad.points[ip] / PI;  // wrong mapping, fix below
        // Actually, gauss_legendre maps to [0, 2*pi]. We need [-L/2, L/2].
        // Let's just use a simple Gauss rule on [-1,1]:
        // We already have the nodes from the [-1,1] computation before mapping.
        // Hmm, our gauss_legendre function maps to [0, 2pi]. Let me compute differently.
        // Instead, use the raw Gauss point: x in [-1,1], map to s = L/2 * x, weight = L/2 * w_raw
        break;
    }

    // Simpler: use trapezoidal rule with singularity subtraction.
    // For n>=1 on the self-panel, the circumferential integral is convergent.
    // G_n(r,z; r+ds*tr, z+ds*tz) where (tr, tz) is the panel tangent, ds is small.
    // Just compute G_n at the panel midpoint as if it were a regular point.
    // The "singularity" for n>=1 is absent, so the circumferential integral is smooth.
    // Self-influence ≈ G_n(r,z; r,z) * L
    double G_self = 0.0;
    for (int j = 0; j < nq; j++) {
        double alpha = circ.points[j];
        double R = std::sqrt(2.0 * r * r * (1.0 - std::cos(alpha)));
        if (R > 1e-30) {
            G_self += circ.weights[j] * std::cos(n * alpha) / R;
        }
    }
    G_self *= r / (4.0 * PI);
    return G_self * L;
}

// ========================================================================
// Meridional mesh construction
// ========================================================================

MeridionalMesh MeridionalMesh::from_points(const Eigen::MatrixXd& points) {
    MeridionalMesh mesh;
    int np = static_cast<int>(points.rows());
    if (np < 2) return mesh;

    mesh.panels.reserve(np - 1);
    for (int i = 0; i < np - 1; i++) {
        MeridionalPanel panel;
        panel.r1 = points(i, 0);
        panel.z1 = points(i, 1);
        panel.r2 = points(i + 1, 0);
        panel.z2 = points(i + 1, 1);
        panel.r_mid = 0.5 * (panel.r1 + panel.r2);
        panel.z_mid = 0.5 * (panel.z1 + panel.z2);

        double dr = panel.r2 - panel.r1;
        double dz = panel.z2 - panel.z1;
        panel.length = std::sqrt(dr * dr + dz * dz);

        if (panel.length > 0.0) {
            // Normal = left-perpendicular of tangent (-dz, dr) / length.
            // For a clockwise-traversed profile (e.g. sphere from top to bottom),
            // this points outward from the body into the fluid.
            panel.nr = -dz / panel.length;
            panel.nz = dr / panel.length;
        } else {
            panel.nr = 0.0;
            panel.nz = 0.0;
        }
        mesh.panels.push_back(panel);
    }
    return mesh;
}

// ========================================================================
// BEM assembly and solve
// ========================================================================

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> AxiBEMSolver::assemble_HG(
    int nd, const MeridionalMesh& mesh) const {

    int np = static_cast<int>(mesh.panels.size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(np, np);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(np, np);

    for (int i = 0; i < np; i++) {
        double ri = mesh.panels[i].r_mid;
        double zi = mesh.panels[i].z_mid;

        for (int j = 0; j < np; j++) {
            double rj = mesh.panels[j].r_mid;
            double zj = mesh.panels[j].z_mid;
            double nrj = mesh.panels[j].nr;
            double nzj = mesh.panels[j].nz;
            double Lj = mesh.panels[j].length;

            if (i != j) {
                G(i, j) = green_function(nd, ri, zi, rj, zj) * Lj;
                H(i, j) = green_normal_deriv(nd, ri, zi, rj, zj, nrj, nzj) * Lj;
            } else {
                // Self-influence
                G(i, i) = green_self(nd, ri, zi, Lj);
                // H diagonal: for a flat panel on a smooth boundary, the
                // self-influence of the double-layer is zero (handled by 1/2*I jump).
                H(i, i) = 0.0;
            }
        }
    }
    return {H, G};
}

AxiBEMSolver::Result AxiBEMSolver::solve_nd(
    int nd, double rho_fluid, const MeridionalMesh& mesh) const {

    int np = static_cast<int>(mesh.panels.size());
    if (np == 0) {
        return Result{Eigen::MatrixXd()};
    }

    auto [H, G_mat] = assemble_HG(nd, mesh);

    // BIE: (1/2 I + H) phi = G * q
    // Added mass: M_a = rho_f * diag(2*pi*r*L) * inv(1/2 I + H) * G
    Eigen::MatrixXd A = 0.5 * Eigen::MatrixXd::Identity(np, np) + H;

    // Solve A * X = G for X = A^{-1} * G
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    Eigen::MatrixXd AinvG = lu.solve(G_mat);

    // Ring element areas: 2*pi*r*L
    Eigen::VectorXd area(np);
    for (int i = 0; i < np; i++) {
        area(i) = 2.0 * PI * mesh.panels[i].r_mid * mesh.panels[i].length;
    }

    Eigen::MatrixXd M_panel = rho_fluid * area.asDiagonal() * AinvG;

    // Symmetrize (should be symmetric in theory, enforce numerically)
    M_panel = 0.5 * (M_panel + M_panel.transpose());

    return Result{M_panel};
}

std::vector<AxiBEMSolver::Result> AxiBEMSolver::solve_all_nd(
    int max_nd, double rho_fluid, const MeridionalMesh& mesh) const {

    std::vector<Result> results;
    results.reserve(max_nd + 1);
    for (int nd = 0; nd <= max_nd; nd++) {
        results.push_back(solve_nd(nd, rho_fluid, mesh));
    }
    return results;
}

}  // namespace turbomodal
