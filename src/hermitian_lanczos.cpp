#include "turbomodal/hermitian_lanczos.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace turbomodal {

void HermitianLanczosEigenSolver::factorize(const SpMatcd &K, const SpMatcd &M,
                                            double sigma) {

  SpMatcd mat = K - std::complex<double>(sigma, 0.0) * M;

  if (!m_pattern_set) {
    // First call: full symbolic + numeric factorization
    m_ldlt = std::make_unique<LDLTSolver>();
    m_ldlt->analyzePattern(mat);
    m_ldlt->factorize(mat);
    if (m_ldlt->info() == Eigen::Success) {
      m_using_ldlt = true;
      m_lu.reset();
      m_pattern_set = true;
      return;
    }

    // Fallback to SparseLU
    m_ldlt.reset();
    m_lu = std::make_unique<LUSolver>();
    m_lu->analyzePattern(mat);
    m_lu->factorize(mat);
    if (m_lu->info() != Eigen::Success) {
      throw std::runtime_error(
          "HermitianLanczos: both LDLT and LU factorization failed");
    }
    m_using_ldlt = false;
    m_pattern_set = true;
  } else {
    // Reuse symbolic, only do numeric factorization
    if (m_using_ldlt) {
      m_ldlt->factorize(mat);
      if (m_ldlt->info() != Eigen::Success) {
        m_pattern_set = false;
        factorize(K, M, sigma);
        return;
      }
    } else {
      m_lu->factorize(mat);
      if (m_lu->info() != Eigen::Success) {
        m_pattern_set = false;
        factorize(K, M, sigma);
        return;
      }
    }
  }
}

Eigen::VectorXcd
HermitianLanczosEigenSolver::solve_shift(const Eigen::VectorXcd &rhs) const {
  if (m_using_ldlt)
    return m_ldlt->solve(rhs);
  else
    return m_lu->solve(rhs);
}

HermitianLanczosResult
HermitianLanczosEigenSolver::solve_dense(const SpMatcd &K, const SpMatcd &M,
                                         int nev, double sigma) {
  HermitianLanczosResult result;
  int n = static_cast<int>(K.rows());
  nev = std::min(nev, n);

  // Convert to dense
  Eigen::MatrixXcd K_dense(K);
  Eigen::MatrixXcd M_dense(M);

  // Cholesky decomposition: M = L * L^H
  Eigen::LLT<Eigen::MatrixXcd> llt(M_dense);
  if (llt.info() != Eigen::Success) {
    // M not positive-definite — shouldn't happen for a valid mass matrix
    return result;
  }

  // Transform to standard eigenproblem: L^{-1}*K*L^{-H}*y = λ*y
  Eigen::MatrixXcd L(llt.matrixL());
  Eigen::MatrixXcd L_inv =
      L.triangularView<Eigen::Lower>().solve(
          Eigen::MatrixXcd::Identity(n, n));
  Eigen::MatrixXcd A = L_inv * K_dense * L_inv.adjoint();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eig(A);

  // All eigenvalues, sorted ascending by Eigen
  Eigen::VectorXd all_evals = eig.eigenvalues();
  Eigen::MatrixXcd all_evecs = L_inv.adjoint() * eig.eigenvectors();

  // Pick nev eigenvalues closest to sigma
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return std::abs(all_evals(a) - sigma) < std::abs(all_evals(b) - sigma);
  });

  // Sort the selected nev by ascending eigenvalue
  std::vector<int> sel(idx.begin(), idx.begin() + nev);
  std::sort(sel.begin(), sel.end(), [&](int a, int b) {
    return all_evals(a) < all_evals(b);
  });

  result.eigenvalues.resize(nev);
  result.eigenvectors.resize(n, nev);
  for (int i = 0; i < nev; i++) {
    result.eigenvalues(i) = all_evals(sel[i]);
    result.eigenvectors.col(i) = all_evecs.col(sel[i]);
  }
  result.num_converged = nev;
  result.converged = true;
  return result;
}

HermitianLanczosResult
HermitianLanczosEigenSolver::solve(const SpMatcd &K, const SpMatcd &M, int nev,
                                   int ncv, double sigma, double tol,
                                   int max_iter) {

  HermitianLanczosResult result;
  int n = static_cast<int>(K.rows());

  if (n == 0 || nev <= 0)
    return result;

  nev = std::min(nev, n - 1);
  ncv = std::max(ncv, 2 * nev + 1);
  ncv = std::min(ncv, n);

  // Dense fallback for small systems where Lanczos has insufficient
  // Krylov subspace room, or where dense is simply faster.
  if (n <= 200) {
    return solve_dense(K, M, nev, sigma);
  }

  // Factorize (K - sigma*M)
  m_M = &M;
  factorize(K, M, sigma);

  // Storage for Lanczos vectors and cached M*V products
  Eigen::MatrixXcd V(n, ncv);   // M-orthonormal basis vectors
  Eigen::MatrixXcd MV(n, ncv);  // Cached M * V columns
  Eigen::VectorXd alpha(ncv);   // Diagonal of tridiagonal T
  Eigen::VectorXd beta(ncv);    // Sub-diagonal of tridiagonal T
  alpha.setZero();
  beta.setZero();

  // Initial random vector
  std::mt19937 rng(42);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXcd v0(n);
  for (int i = 0; i < n; i++) {
    v0(i) = std::complex<double>(dist(rng), dist(rng));
  }

  // M-normalize: ||v0||_M = sqrt(v0^H * M * v0) should be real and positive
  Eigen::VectorXcd Mv0 = M * v0;
  double norm_M = std::sqrt(std::abs(v0.dot(Mv0)));
  V.col(0) = v0 / norm_M;
  MV.col(0) = Mv0 / norm_M;

  // Implicitly restarted Lanczos
  for (int restart = 0; restart < max_iter; restart++) {

    // Lanczos factorization: extend from current basis
    int j_start = (restart == 0) ? 0 : nev;
    for (int j = j_start; j < ncv; j++) {
      // w = (K - sigma*M)^{-1} * M * V(:,j)
      Eigen::VectorXcd w = solve_shift(MV.col(j));

      // alpha_j = Re(V(:,j)^H * M * w) — real for Hermitian problems
      Eigen::VectorXcd Mw = M * w;
      alpha(j) = (V.col(j).dot(Mw)).real();

      // Orthogonalize: w = w - alpha_j * V(:,j) - beta_{j-1} * V(:,j-1)
      w -= alpha(j) * V.col(j);
      if (j > 0) {
        w -= beta(j - 1) * V.col(j - 1);
      }

      // Full reorthogonalization using cached MV
      // Identity: v^H * M * w = (M*v)^H * w  (M is Hermitian)
      for (int k = 0; k <= j; k++) {
        std::complex<double> h = MV.col(k).dot(w);
        w -= h * V.col(k);
      }

      // beta_j = ||w||_M
      Eigen::VectorXcd Mw2 = M * w;
      double beta_j = std::sqrt(std::abs(w.dot(Mw2)));

      if (j < ncv - 1) {
        beta(j) = beta_j;
        if (beta_j > 1e-14) {
          V.col(j + 1) = w / beta_j;
          MV.col(j + 1) = Mw2 / beta_j;
        } else {
          // Invariant subspace found — generate new random vector
          Eigen::VectorXcd r(n);
          for (int i = 0; i < n; i++)
            r(i) = std::complex<double>(dist(rng), dist(rng));
          // Orthogonalize against existing basis using cached MV
          for (int k = 0; k <= j; k++) {
            r -= MV.col(k).dot(r) * V.col(k);
          }
          Eigen::VectorXcd Mr = M * r;
          double nr = std::sqrt(std::abs(r.dot(Mr)));
          V.col(j + 1) = r / nr;
          MV.col(j + 1) = Mr / nr;
          beta(j) = 0.0;
        }
      }
    }

    // Build tridiagonal matrix T (ncv x ncv)
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(ncv, ncv);
    for (int j = 0; j < ncv; j++) {
      T(j, j) = alpha(j);
      if (j < ncv - 1) {
        T(j, j + 1) = beta(j);
        T(j + 1, j) = beta(j);
      }
    }

    // Eigendecomposition of T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(T);
    Eigen::VectorXd theta = eig.eigenvalues(); // Ritz values (shift-inverted)
    Eigen::MatrixXd S = eig.eigenvectors();    // Ritz vectors in T-space

    // Sort by largest magnitude (closest to sigma in original space)
    std::vector<int> idx(ncv);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
      return std::abs(theta(a)) > std::abs(theta(b));
    });

    // Check convergence of the nev wanted Ritz pairs
    // Residual bound: |beta_{ncv-1} * S(ncv-1, i)|
    double beta_last = (ncv > 1) ? beta(ncv - 2) : 0.0;
    int nconv = 0;
    for (int i = 0; i < nev; i++) {
      double res = std::abs(beta_last * S(ncv - 1, idx[i]));
      if (res < tol * std::max(1.0, std::abs(theta(idx[i])))) {
        nconv++;
      }
    }

    if (nconv >= nev || restart == max_iter - 1) {
      // Extract converged eigenvalues and eigenvectors
      int num_extract = std::min(nev, ncv);
      result.eigenvalues.resize(num_extract);
      result.eigenvectors.resize(n, num_extract);

      for (int i = 0; i < num_extract; i++) {
        // Transform shift-inverted eigenvalue back:
        // theta = 1/(lambda - sigma) → lambda = 1/theta + sigma
        double nu = theta(idx[i]);
        double lambda = (std::abs(nu) > 1e-15) ? (1.0 / nu + sigma) : 0.0;
        result.eigenvalues(i) = lambda;

        // Ritz vector: V * S(:, idx[i])
        result.eigenvectors.col(i) =
            V.leftCols(ncv) * S.col(idx[i]).cast<std::complex<double>>();
      }

      // Sort by ascending eigenvalue
      std::vector<int> sort_idx(num_extract);
      std::iota(sort_idx.begin(), sort_idx.end(), 0);
      std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) {
        return result.eigenvalues(a) < result.eigenvalues(b);
      });
      Eigen::VectorXd sorted_evals(num_extract);
      Eigen::MatrixXcd sorted_evecs(n, num_extract);
      for (int i = 0; i < num_extract; i++) {
        sorted_evals(i) = result.eigenvalues(sort_idx[i]);
        sorted_evecs.col(i) = result.eigenvectors.col(sort_idx[i]);
      }
      result.eigenvalues = sorted_evals;
      result.eigenvectors = sorted_evecs;
      result.num_converged = nconv;
      result.converged = (nconv >= nev);
      return result;
    }

    // Implicit restart: apply QR shifts to compress from ncv to nev vectors
    // Use the unwanted Ritz values as shifts
    Eigen::MatrixXd Q_total = Eigen::MatrixXd::Identity(ncv, ncv);

    for (int i = nev; i < ncv; i++) {
      double mu = theta(idx[i]); // Unwanted Ritz value as shift

      // QR factorization of (T - mu*I)
      Eigen::MatrixXd Tshift = T - mu * Eigen::MatrixXd::Identity(ncv, ncv);
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(Tshift);
      Eigen::MatrixXd Qi = qr.householderQ();

      T = Qi.transpose() * T * Qi;
      Q_total = Q_total * Qi;
    }

    // Update Lanczos vectors: V_new = V * Q_total(:, 0:nev-1)
    Eigen::MatrixXcd V_new =
        V.leftCols(ncv) * Q_total.leftCols(nev).cast<std::complex<double>>();

    // Update tridiagonal entries
    for (int j = 0; j < nev; j++) {
      alpha(j) = T(j, j);
      if (j < nev - 1) {
        beta(j) = T(j, j + 1);
      }
    }

    // Set up continuation vector for extending Lanczos
    V.leftCols(nev) = V_new;

    // Recompute MV cache for updated basis vectors
    for (int j = 0; j < nev; j++) {
      MV.col(j) = M * V.col(j);
    }

    // The residual vector for restarting
    if (nev < ncv) {
      beta(nev - 1) = T(nev, nev - 1);
      // V(:, nev) is the residual direction from the QR compression
      if (std::abs(beta(nev - 1)) > 1e-14) {
        V.col(nev) =
            V.leftCols(ncv) * Q_total.col(nev).cast<std::complex<double>>();
        // Re-normalize
        Eigen::VectorXcd Mv = M * V.col(nev);
        double nm = std::sqrt(std::abs(V.col(nev).dot(Mv)));
        if (nm > 1e-14) {
          V.col(nev) /= nm;
          MV.col(nev) = Mv / nm;
          beta(nev - 1) *= nm;
        }
      }
    }
  }

  return result;
}

} // namespace turbomodal
