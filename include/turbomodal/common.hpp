#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <stdexcept>

namespace turbomodal {

// Fixed-size matrix types for TET10 element computation (30 DOF = 10 nodes x 3 DOF)
using Matrix30d = Eigen::Matrix<double, 30, 30>;
using Vector30d = Eigen::Matrix<double, 30, 1>;
using Matrix6x30d = Eigen::Matrix<double, 6, 30>;
using Matrix9x30d = Eigen::Matrix<double, 9, 30>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix9d = Eigen::Matrix<double, 9, 9>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix10x3d = Eigen::Matrix<double, 10, 3>;
using Vector10d = Eigen::Matrix<double, 10, 1>;
using Matrix3x30d = Eigen::Matrix<double, 3, 30>;

// Sparse matrix types
using SpMatd = Eigen::SparseMatrix<double>;
using SpMatcd = Eigen::SparseMatrix<std::complex<double>>;
using Triplet = Eigen::Triplet<double>;
using TripletC = Eigen::Triplet<std::complex<double>>;

// Constants
constexpr double PI = 3.14159265358979323846;

}  // namespace turbomodal
