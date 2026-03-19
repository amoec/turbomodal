"""Tests for turbomodal.mac — Modal Assurance Criterion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from turbomodal.mac import compute_auto_mac, compute_mac


class TestComputeMAC:
    """compute_mac correctness and edge-case tests."""

    def test_identity(self):
        """MAC(phi, phi) diagonal should be 1.0."""
        rng = np.random.default_rng(0)
        phi = rng.standard_normal((4, 30)) + 1j * rng.standard_normal((4, 30))
        mac = compute_mac(phi, phi)
        np.testing.assert_allclose(np.diag(mac), 1.0, atol=1e-12)

    def test_orthogonal_modes(self):
        """Two orthogonal vectors should produce MAC = 0."""
        a = np.array([[1, 0, 0]], dtype=np.complex128)
        b = np.array([[0, 1, 0]], dtype=np.complex128)
        mac = compute_mac(a, b)
        assert mac.shape == (1, 1)
        assert mac[0, 0] == pytest.approx(0.0, abs=1e-15)

    def test_identical_modes(self):
        """Two identical vectors should produce MAC = 1."""
        v = np.array([[1, 2, 3]], dtype=np.complex128)
        mac = compute_mac(v, v)
        assert mac[0, 0] == pytest.approx(1.0, abs=1e-12)

    def test_shape_2d(self):
        """(3,20) vs (4,20) should produce (3,4) output."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((3, 20))
        b = rng.standard_normal((4, 20))
        mac = compute_mac(a, b)
        assert mac.shape == (3, 4)

    def test_1d_input(self):
        """Single 1-D vector should be promoted and return (1,1)."""
        v = np.array([1.0, 2.0, 3.0])
        mac = compute_mac(v, v)
        assert mac.shape == (1, 1)
        assert mac[0, 0] == pytest.approx(1.0, abs=1e-12)

    def test_complex_modes(self):
        """Complex mode shapes: result should be real and in [0, 1]."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal((5, 40)) + 1j * rng.standard_normal((5, 40))
        b = rng.standard_normal((3, 40)) + 1j * rng.standard_normal((3, 40))
        mac = compute_mac(a, b)
        assert mac.shape == (5, 3)
        assert np.all(np.isreal(mac))
        assert np.all(mac >= -1e-15)
        assert np.all(mac <= 1.0 + 1e-12)

    def test_scaled_invariance(self):
        """MAC should be invariant to scalar scaling: MAC(2*phi, phi) == 1."""
        rng = np.random.default_rng(3)
        phi = rng.standard_normal((1, 20))
        mac = compute_mac(2.0 * phi, phi)
        assert mac[0, 0] == pytest.approx(1.0, abs=1e-12)

    def test_near_zero_norm(self):
        """Near-zero vector should not produce NaN or inf."""
        a = np.array([[1e-200, 0, 0]], dtype=np.complex128)
        b = np.array([[0, 1e-200, 0]], dtype=np.complex128)
        mac = compute_mac(a, b)
        assert np.all(np.isfinite(mac))


class TestComputeAutoMAC:
    """compute_auto_mac tests."""

    def test_diagonal(self):
        """Auto-MAC diagonal should be all 1.0."""
        rng = np.random.default_rng(4)
        phi = rng.standard_normal((6, 50))
        mac = compute_auto_mac(phi)
        np.testing.assert_allclose(np.diag(mac), 1.0, atol=1e-12)

    def test_well_separated(self):
        """Orthogonal modes: off-diagonal auto-MAC should be near 0."""
        # Use identity-like structure for well-separated modes
        phi = np.eye(5, dtype=np.complex128)
        mac = compute_auto_mac(phi)
        off_diag = mac - np.diag(np.diag(mac))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)

    def test_output_shape(self):
        """(5,30) input should produce (5,5) output."""
        rng = np.random.default_rng(5)
        phi = rng.standard_normal((5, 30))
        mac = compute_auto_mac(phi)
        assert mac.shape == (5, 5)
