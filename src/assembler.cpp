#include "turbomodal/assembler.hpp"
#include "turbomodal/element.hpp"
#include "turbomodal/rotating_effects.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turbomodal {

void GlobalAssembler::assemble(const Mesh& mesh, const Material& mat) {
    int ndof = mesh.num_dof();
    int n_elem = mesh.num_elements();

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    std::vector<std::vector<Triplet>> k_per_thread(n_threads);
    std::vector<std::vector<Triplet>> m_per_thread(n_threads);
    for (auto& v : k_per_thread) v.reserve(n_elem * 900 / n_threads + 900);
    for (auto& v : m_per_thread) v.reserve(n_elem * 900 / n_threads + 900);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int e = 0; e < n_elem; e++) {
        int tid = omp_get_thread_num();
        TET10Element elem;
        Eigen::VectorXi dof_map(30);
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
            dof_map(3 * n)     = 3 * node_id;
            dof_map(3 * n + 1) = 3 * node_id + 1;
            dof_map(3 * n + 2) = 3 * node_id + 2;
        }
        Matrix30d Ke = elem.stiffness(mat);
        Matrix30d Me = elem.mass(mat);
        add_element_matrix(k_per_thread[tid], Ke, dof_map);
        add_element_matrix(m_per_thread[tid], Me, dof_map);
    }

    std::vector<Triplet> k_triplets, m_triplets;
    for (auto& v : k_per_thread) k_triplets.insert(k_triplets.end(), v.begin(), v.end());
    for (auto& v : m_per_thread) m_triplets.insert(m_triplets.end(), v.begin(), v.end());
#else
    std::vector<Triplet> k_triplets, m_triplets;
    k_triplets.reserve(n_elem * 900);
    m_triplets.reserve(n_elem * 900);

    for (int e = 0; e < n_elem; e++) {
        TET10Element elem;
        Eigen::VectorXi dof_map(30);
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
            dof_map(3 * n)     = 3 * node_id;
            dof_map(3 * n + 1) = 3 * node_id + 1;
            dof_map(3 * n + 2) = 3 * node_id + 2;
        }
        Matrix30d Ke = elem.stiffness(mat);
        Matrix30d Me = elem.mass(mat);
        add_element_matrix(k_triplets, Ke, dof_map);
        add_element_matrix(m_triplets, Me, dof_map);
    }
#endif

    K_global_.resize(ndof, ndof);
    M_global_.resize(ndof, ndof);
    K_global_.setFromTriplets(k_triplets.begin(), k_triplets.end());
    M_global_.setFromTriplets(m_triplets.begin(), m_triplets.end());
}

void GlobalAssembler::assemble_stress_stiffening(
    const Mesh& mesh, const Material& mat,
    const Eigen::VectorXd& displacement, double omega) {

    int ndof = mesh.num_dof();
    int n_elem = mesh.num_elements();
    Matrix6d D = mat.constitutive_matrix();

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    std::vector<std::vector<Triplet>> ks_per_thread(n_threads);
    for (auto& v : ks_per_thread) v.reserve(n_elem * 900 / n_threads + 900);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int e = 0; e < n_elem; e++) {
        int tid = omp_get_thread_num();
        TET10Element elem;
        Eigen::VectorXi dof_map(30);
        Vector30d u_e;

        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
            dof_map(3 * n)     = 3 * node_id;
            dof_map(3 * n + 1) = 3 * node_id + 1;
            dof_map(3 * n + 2) = 3 * node_id + 2;
            u_e(3 * n)     = displacement(3 * node_id);
            u_e(3 * n + 1) = displacement(3 * node_id + 1);
            u_e(3 * n + 2) = displacement(3 * node_id + 2);
        }

        std::array<Vector6d, 4> prestress;
        for (int gp = 0; gp < 4; gp++) {
            double xi   = TET10Element::gauss_points[gp](0);
            double eta  = TET10Element::gauss_points[gp](1);
            double zeta = TET10Element::gauss_points[gp](2);
            Matrix6x30d B = elem.B_matrix(xi, eta, zeta);
            Vector6d strain = B * u_e;
            prestress[gp] = D * strain;
        }

        Matrix30d Ks_e = RotatingEffects::stress_stiffening(elem, prestress);
        add_element_matrix(ks_per_thread[tid], Ks_e, dof_map);
    }

    std::vector<Triplet> ks_triplets;
    for (auto& v : ks_per_thread) ks_triplets.insert(ks_triplets.end(), v.begin(), v.end());
#else
    std::vector<Triplet> ks_triplets;
    ks_triplets.reserve(n_elem * 900);

    for (int e = 0; e < n_elem; e++) {
        TET10Element elem;
        Eigen::VectorXi dof_map(30);
        Vector30d u_e;

        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
            dof_map(3 * n)     = 3 * node_id;
            dof_map(3 * n + 1) = 3 * node_id + 1;
            dof_map(3 * n + 2) = 3 * node_id + 2;
            u_e(3 * n)     = displacement(3 * node_id);
            u_e(3 * n + 1) = displacement(3 * node_id + 1);
            u_e(3 * n + 2) = displacement(3 * node_id + 2);
        }

        std::array<Vector6d, 4> prestress;
        for (int gp = 0; gp < 4; gp++) {
            double xi   = TET10Element::gauss_points[gp](0);
            double eta  = TET10Element::gauss_points[gp](1);
            double zeta = TET10Element::gauss_points[gp](2);
            Matrix6x30d B = elem.B_matrix(xi, eta, zeta);
            Vector6d strain = B * u_e;
            prestress[gp] = D * strain;
        }

        Matrix30d Ks_e = RotatingEffects::stress_stiffening(elem, prestress);
        add_element_matrix(ks_triplets, Ks_e, dof_map);
    }
#endif

    K_sigma_global_.resize(ndof, ndof);
    K_sigma_global_.setFromTriplets(ks_triplets.begin(), ks_triplets.end());
}

void GlobalAssembler::assemble_rotating_effects(
    const Mesh& mesh, const Material& mat, double omega) {

    int ndof = mesh.num_dof();
    int n_elem = mesh.num_elements();
    Eigen::Vector3d axis = Eigen::Vector3d::Unit(mesh.rotation_axis);

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    std::vector<std::vector<Triplet>> kw_per_thread(n_threads);
    std::vector<std::vector<Triplet>> g_per_thread(n_threads);
    for (auto& v : kw_per_thread) v.reserve(n_elem * 900 / n_threads + 900);
    for (auto& v : g_per_thread)  v.reserve(n_elem * 900 / n_threads + 900);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int e = 0; e < n_elem; e++) {
        int tid = omp_get_thread_num();
        TET10Element elem;
        Eigen::VectorXi dof_map(30);
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
            dof_map(3 * n)     = 3 * node_id;
            dof_map(3 * n + 1) = 3 * node_id + 1;
            dof_map(3 * n + 2) = 3 * node_id + 2;
        }
        Matrix30d Kw_e = RotatingEffects::spin_softening(elem, mat, omega, axis);
        Matrix30d Ge = RotatingEffects::gyroscopic(elem, mat, axis);
        add_element_matrix(kw_per_thread[tid], Kw_e, dof_map);
        add_element_matrix(g_per_thread[tid], Ge, dof_map);
    }

    std::vector<Triplet> kw_triplets, g_triplets;
    for (auto& v : kw_per_thread) kw_triplets.insert(kw_triplets.end(), v.begin(), v.end());
    for (auto& v : g_per_thread)  g_triplets.insert(g_triplets.end(), v.begin(), v.end());
#else
    std::vector<Triplet> kw_triplets, g_triplets;
    kw_triplets.reserve(n_elem * 900);
    g_triplets.reserve(n_elem * 900);

    for (int e = 0; e < n_elem; e++) {
        TET10Element elem;
        Eigen::VectorXi dof_map(30);
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
            dof_map(3 * n)     = 3 * node_id;
            dof_map(3 * n + 1) = 3 * node_id + 1;
            dof_map(3 * n + 2) = 3 * node_id + 2;
        }
        Matrix30d Kw_e = RotatingEffects::spin_softening(elem, mat, omega, axis);
        Matrix30d Ge = RotatingEffects::gyroscopic(elem, mat, axis);
        add_element_matrix(kw_triplets, Kw_e, dof_map);
        add_element_matrix(g_triplets, Ge, dof_map);
    }
#endif

    K_omega_global_.resize(ndof, ndof);
    G_global_.resize(ndof, ndof);
    K_omega_global_.setFromTriplets(kw_triplets.begin(), kw_triplets.end());
    G_global_.setFromTriplets(g_triplets.begin(), g_triplets.end());
}

Eigen::VectorXd GlobalAssembler::assemble_centrifugal_load(
    const Mesh& mesh, const Material& mat,
    double omega, const Eigen::Vector3d& axis) {

    int ndof = mesh.num_dof();
    int n_elem = mesh.num_elements();

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    std::vector<Eigen::VectorXd> F_per_thread(n_threads, Eigen::VectorXd::Zero(ndof));

    #pragma omp parallel for schedule(dynamic, 64)
    for (int e = 0; e < n_elem; e++) {
        int tid = omp_get_thread_num();
        TET10Element elem;
        Eigen::VectorXi dof_map(30);
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
            dof_map(3 * n)     = 3 * node_id;
            dof_map(3 * n + 1) = 3 * node_id + 1;
            dof_map(3 * n + 2) = 3 * node_id + 2;
        }
        Vector30d Fe = RotatingEffects::centrifugal_load(elem, mat, omega, axis);
        for (int i = 0; i < 30; i++) {
            F_per_thread[tid](dof_map(i)) += Fe(i);
        }
    }

    Eigen::VectorXd F = Eigen::VectorXd::Zero(ndof);
    for (auto& Ft : F_per_thread) F += Ft;
#else
    Eigen::VectorXd F = Eigen::VectorXd::Zero(ndof);

    for (int e = 0; e < n_elem; e++) {
        TET10Element elem;
        Eigen::VectorXi dof_map(30);
        for (int n = 0; n < 10; n++) {
            int node_id = mesh.elements(e, n);
            elem.node_coords.row(n) = mesh.nodes.row(node_id);
            dof_map(3 * n)     = 3 * node_id;
            dof_map(3 * n + 1) = 3 * node_id + 1;
            dof_map(3 * n + 2) = 3 * node_id + 2;
        }
        Vector30d Fe = RotatingEffects::centrifugal_load(elem, mat, omega, axis);
        for (int i = 0; i < 30; i++) {
            F(dof_map(i)) += Fe(i);
        }
    }
#endif

    return F;
}

void GlobalAssembler::add_element_matrix(
    std::vector<Triplet>& triplets,
    const Eigen::MatrixXd& Ke,
    const Eigen::VectorXi& dof_map) {
    int n = static_cast<int>(dof_map.size());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = Ke(i, j);
            if (val != 0.0) {
                triplets.emplace_back(dof_map(i), dof_map(j), val);
            }
        }
    }
}

}  // namespace turbomodal
