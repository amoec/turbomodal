#pragma once

#include "turbomodal/common.hpp"
#include <Eigen/Dense>

namespace turbomodal {

// Complete elliptic integrals via AGM (portable, Apple Clang lacks std::comp_ellint_*)
double elliptic_K(double k);  // First kind K(k), k = modulus (not parameter m=k^2)
double elliptic_E(double k);  // Second kind E(k)

// Meridional panel for axisymmetric BEM
struct MeridionalPanel {
    double r1, z1;         // start endpoint (r, z)
    double r2, z2;         // end endpoint (r, z)
    double r_mid, z_mid;   // midpoint
    double nr, nz;         // outward unit normal (into fluid)
    double length;         // panel arc length
};

// Ordered collection of meridional panels forming the wetted boundary
struct MeridionalMesh {
    std::vector<MeridionalPanel> panels;

    // Build from ordered boundary points. Normal convention: outward from body.
    // points: Nx2 matrix (r, z) forming an ordered polyline
    static MeridionalMesh from_points(const Eigen::MatrixXd& points);
};

// Axisymmetric BEM solver for potential flow added mass
class AxiBEMSolver {
public:
    struct Result {
        Eigen::MatrixXd M_panel;  // N_panel x N_panel added mass matrix (meridional)
    };

    // Compute added mass matrix for a single nodal diameter
    Result solve_nd(int nd, double rho_fluid,
                    const MeridionalMesh& mesh) const;

    // Compute for all NDs from 0 to max_nd
    std::vector<Result> solve_all_nd(int max_nd, double rho_fluid,
                                      const MeridionalMesh& mesh) const;

    // --- Low-level Green's function API (exposed for testing) ---

    // Ring-source Green's function G_n(r,z; r',z')
    // = (r'/(4pi)) * integral_0^{2pi} cos(n*alpha) / R(alpha) dalpha
    static double green_function(int n, double r, double z,
                                  double rp, double zp, int n_quad = 0);

    // Normal derivative of Green's function at source (rp, zp) with normal (nrp, nzp)
    static double green_normal_deriv(int n, double r, double z,
                                      double rp, double zp,
                                      double nrp, double nzp, int n_quad = 0);

    // Self-influence (singular) G_n integral over a panel
    static double green_self(int n, double r, double z,
                              double panel_length, int n_quad = 0);

private:
    // Assemble H and G influence matrices for a given ND
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> assemble_HG(
        int nd, const MeridionalMesh& mesh) const;
};

}  // namespace turbomodal
