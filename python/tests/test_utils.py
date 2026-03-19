"""Tests for turbomodal._utils — shared utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from turbomodal._utils import (
    ensure_rng,
    progress_bar,
    rotation_matrix_3x3,
    rotation_matrix_4x4_flat,
)


# ---------------------------------------------------------------------------
# ensure_rng
# ---------------------------------------------------------------------------

class TestEnsureRng:

    def test_passthrough(self):
        """Passing an existing Generator returns it unchanged."""
        gen = np.random.default_rng(42)
        assert ensure_rng(gen) is gen

    def test_none_creates_generator(self):
        """Passing None returns a valid Generator instance."""
        result = ensure_rng(None)
        assert isinstance(result, np.random.Generator)


# ---------------------------------------------------------------------------
# progress_bar
# ---------------------------------------------------------------------------

class TestProgressBar:

    def test_format(self):
        """Output contains brackets, percentage, and counts."""
        s = progress_bar(5, 10)
        assert "[" in s
        assert "]" in s
        assert "%" in s
        assert "5/10" in s

    def test_complete(self):
        """At current==total, bar shows 100%."""
        s = progress_bar(10, 10)
        assert "100.0%" in s
        assert "10/10" in s

    def test_eta_seconds(self):
        """With elapsed > 0 and current < total, ETA is shown in seconds."""
        s = progress_bar(5, 10, elapsed=10.0)
        assert "ETA" in s
        assert "s" in s

    def test_eta_minutes(self):
        """Large ETA displays minutes."""
        # 1 of 100 done in 100s -> ETA = 9900s = 165 min
        s = progress_bar(1, 100, elapsed=100.0)
        assert "ETA" in s
        assert "m" in s

    def test_zero_total(self):
        """total=0 should not divide by zero."""
        s = progress_bar(0, 0)
        assert isinstance(s, str)

    def test_prefix_suffix(self):
        """Custom prefix and suffix appear in output."""
        s = progress_bar(3, 10, prefix="Train: ", suffix=" done")
        assert "Train: " in s
        assert " done" in s


# ---------------------------------------------------------------------------
# rotation_matrix_3x3
# ---------------------------------------------------------------------------

class TestRotationMatrix3x3:

    def test_identity_at_zero(self):
        """theta=0 for all axes should produce the identity matrix."""
        for axis in (0, 1, 2):
            R = rotation_matrix_3x3(0.0, axis)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_orthogonal(self):
        """R^T R = I for a random angle and all 3 axes."""
        theta = 1.234
        for axis in (0, 1, 2):
            R = rotation_matrix_3x3(theta, axis)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)

    def test_x_axis_90deg(self):
        """90-degree X rotation: [0,1,0] -> [0,0,1]."""
        R = rotation_matrix_3x3(np.pi / 2, axis=0)
        v = np.array([0, 1, 0])
        result = R @ v
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-14)

    def test_y_axis_90deg(self):
        """90-degree Y rotation: [0,0,1] -> [1,0,0]."""
        R = rotation_matrix_3x3(np.pi / 2, axis=1)
        v = np.array([0, 0, 1])
        result = R @ v
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-14)

    def test_z_axis_90deg(self):
        """90-degree Z rotation: [1,0,0] -> [0,1,0]."""
        R = rotation_matrix_3x3(np.pi / 2, axis=2)
        v = np.array([1, 0, 0])
        result = R @ v
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-14)

    def test_determinant(self):
        """det(R) should be 1 for all axes (proper rotation)."""
        theta = 0.7
        for axis in (0, 1, 2):
            R = rotation_matrix_3x3(theta, axis)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)


# ---------------------------------------------------------------------------
# rotation_matrix_4x4_flat
# ---------------------------------------------------------------------------

class TestRotationMatrix4x4Flat:

    def test_length(self):
        """Should return exactly 16 floats."""
        flat = rotation_matrix_4x4_flat(axis=2, angle=0.5)
        assert len(flat) == 16
        assert all(isinstance(x, float) for x in flat)

    def test_affine_structure(self):
        """Last row is [0,0,0,1]; translation column is [0,0,0]."""
        flat = rotation_matrix_4x4_flat(axis=0, angle=1.0)
        M = np.array(flat).reshape(4, 4)
        np.testing.assert_allclose(M[3, :], [0, 0, 0, 1], atol=1e-15)
        np.testing.assert_allclose(M[:3, 3], [0, 0, 0], atol=1e-15)

    def test_matches_3x3(self):
        """Upper-left 3x3 should match rotation_matrix_3x3."""
        theta = 0.8
        for axis in (0, 1, 2):
            R3 = rotation_matrix_3x3(theta, axis)
            flat = rotation_matrix_4x4_flat(axis, theta)
            M4 = np.array(flat).reshape(4, 4)
            np.testing.assert_allclose(M4[:3, :3], R3, atol=1e-14)
