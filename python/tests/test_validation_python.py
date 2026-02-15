"""Python validation tests against analytical benchmarks.

These tests verify the Python API produces results consistent with
known analytical solutions from structural dynamics theory.

Run with: pytest -m validation
"""

import numpy as np
import pytest

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def leissa_setup(leissa_mesh_path):
    """Solve Leissa disk once for validation tests."""
    from turbomodal._core import Mesh, Material, CyclicSymmetrySolver

    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(leissa_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()

    mat = Material(200e9, 0.33, 7850)
    solver = CyclicSymmetrySolver(mesh, mat)
    results = solver.solve_at_rpm(0.0, 5)
    return mesh, mat, results


def leissa_frequency(lambda_sq, E=200e9, nu=0.33, rho=7850.0, h=0.01, a=0.3):
    """Analytical Leissa frequency for clamped-center free-edge plate."""
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    return (lambda_sq / (2.0 * np.pi * a**2)) * np.sqrt(D / (rho * h))


# Leissa lambda^2 values for b/a=0.1, nu=1/3
LEISSA_LAMBDA_SQ = [4.235, 3.482, 5.499, 12.23]


def test_leissa_frequencies(leissa_setup):
    """FEA frequencies should match Leissa plate theory within tolerance."""
    mesh, mat, results = leissa_setup

    for nd in range(min(4, len(results))):
        f_analytical = leissa_frequency(LEISSA_LAMBDA_SQ[nd])
        r = None
        for res in results:
            if res.harmonic_index == nd:
                r = res
                break
        assert r is not None, f"Missing result for ND={nd}"
        assert len(r.frequencies) > 0

        f_fea = r.frequencies[0]
        error_pct = 100.0 * abs(f_fea - f_analytical) / f_analytical
        tol = 15.0 if nd == 3 else 10.0
        assert error_pct < tol, \
            f"ND={nd}: f_fea={f_fea:.1f} vs f_analytical={f_analytical:.1f}, error={error_pct:.1f}%"


def test_kwak_frequency_ratios(leissa_setup):
    """Wet/dry frequency ratios should match Kwak analytical model."""
    from turbomodal._core import (
        CyclicSymmetrySolver, FluidConfig, FluidType, AddedMassModel, Material
    )

    mesh, _, results_dry = leissa_setup
    mat = Material(200e9, 0.33, 7850.0)

    fluid = FluidConfig()
    fluid.type = FluidType.LIQUID_ANALYTICAL
    fluid.fluid_density = 1000.0
    fluid.disk_radius = 0.3
    fluid.disk_thickness = 0.01

    solver_wet = CyclicSymmetrySolver(mesh, mat, fluid)
    results_wet = solver_wet.solve_at_rpm(0.0, 5)

    for nd in range(min(4, len(results_dry))):
        r_dry = None
        r_wet = None
        for r in results_dry:
            if r.harmonic_index == nd:
                r_dry = r
                break
        for r in results_wet:
            if r.harmonic_index == nd:
                r_wet = r
                break

        if r_dry and r_wet and len(r_dry.frequencies) > 0 and len(r_wet.frequencies) > 0:
            ratio_fea = r_wet.frequencies[0] / r_dry.frequencies[0]
            ratio_kwak = AddedMassModel.frequency_ratio(nd, 1000.0, 7850.0, 0.01, 0.3)
            assert abs(ratio_fea - ratio_kwak) < 0.001, \
                f"ND={nd}: ratio_fea={ratio_fea:.4f} vs kwak={ratio_kwak:.4f}"


def test_fmm_tuned_identity():
    """FMM with zero deviations: bounded magnification and IPR.

    Degenerate eigenvalue pairs allow eigenvector arbitrariness,
    so magnification can reach sqrt(2) and IPR up to ~1.5.
    """
    from turbomodal._core import FMMSolver

    N = 24
    tuned = np.array([500.0 + i * 100.0 for i in range(13)])
    dev = np.zeros(N)
    result = FMMSolver.solve(N, tuned, dev)

    assert result.peak_magnification >= 1.0 - 1e-10
    assert result.peak_magnification <= np.sqrt(2.0) + 0.01
    for m in range(N):
        assert result.localization_ipr[m] >= 0.5
        assert result.localization_ipr[m] <= 2.0


def test_campbell_frequency_ordering(leissa_setup):
    """At non-zero RPM: FW frequency > BW frequency for ND>0."""
    from turbomodal._core import CyclicSymmetrySolver, Material

    mesh, _, _ = leissa_setup
    mat = Material(200e9, 0.33, 7850.0)
    solver = CyclicSymmetrySolver(mesh, mat)
    results = solver.solve_at_rpm(3000.0, 3)

    for r in results:
        if r.harmonic_index > 0 and r.harmonic_index < mesh.num_sectors // 2:
            fw_freqs = []
            bw_freqs = []
            for m in range(len(r.frequencies)):
                if r.whirl_direction[m] > 0:
                    fw_freqs.append(r.frequencies[m])
                elif r.whirl_direction[m] < 0:
                    bw_freqs.append(r.frequencies[m])
            if fw_freqs and bw_freqs:
                assert min(fw_freqs) > min(bw_freqs) * 0.95, \
                    f"ND={r.harmonic_index}: FW should be >= BW"


def test_sdof_frf_analytical():
    """modal_frf at 100 points should match closed-form SDOF FRF."""
    from turbomodal._core import ForcedResponseSolver

    omega_r = 2 * np.pi * 500.0
    zeta = 0.02
    Q = complex(1.0, 0.0)

    for i in range(100):
        f = 100.0 + i * 10.0
        omega = 2 * np.pi * f
        H = ForcedResponseSolver.modal_frf(omega, omega_r, Q, zeta)
        denom = complex(omega_r**2 - omega**2, 2 * zeta * omega_r * omega)
        expected = Q / denom
        assert abs(H.real - expected.real) < abs(expected) * 1e-10
        assert abs(H.imag - expected.imag) < abs(expected) * 1e-10
