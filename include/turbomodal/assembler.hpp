#pragma once

#include "turbomodal/common.hpp"
#include "turbomodal/mesh.hpp"
#include "turbomodal/material.hpp"

namespace turbomodal {

class GlobalAssembler {
public:
    void assemble(const Mesh& mesh, const Material& mat);

    void assemble_stress_stiffening(
        const Mesh& mesh, const Material& mat,
        const Eigen::VectorXd& displacement, double omega);

    void assemble_rotating_effects(
        const Mesh& mesh, const Material& mat, double omega);

    Eigen::VectorXd assemble_centrifugal_load(
        const Mesh& mesh, const Material& mat,
        double omega, const Eigen::Vector3d& axis);

    const SpMatd& K() const { return K_global_; }
    const SpMatd& M() const { return M_global_; }
    const SpMatd& K_sigma() const { return K_sigma_global_; }
    const SpMatd& G() const { return G_global_; }
    const SpMatd& K_omega() const { return K_omega_global_; }

private:
    SpMatd K_global_;
    SpMatd M_global_;
    SpMatd K_sigma_global_;
    SpMatd G_global_;
    SpMatd K_omega_global_;

    static void add_element_matrix(
        std::vector<Triplet>& triplets,
        const Eigen::MatrixXd& Ke,
        const Eigen::VectorXi& dof_map);
};

}  // namespace turbomodal
