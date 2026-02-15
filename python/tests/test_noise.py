"""Tests for turbomodal.noise module — pure numpy, no C++ extension needed."""

import numpy as np
import pytest

from turbomodal.noise import (
    NoiseConfig,
    add_gaussian_noise,
    apply_noise,
    apply_bandwidth_limit,
    apply_drift,
    apply_quantization,
    apply_dropout,
)


# ---- NoiseConfig defaults ----

def test_noise_config_defaults():
    cfg = NoiseConfig()
    assert cfg.gaussian_snr_db == 40.0
    assert cfg.harmonic_interference == []
    assert cfg.drift_rate == 0.0
    assert cfg.drift_type == "none"
    assert cfg.bandwidth_hz == 0.0
    assert cfg.filter_order == 4
    assert cfg.adc_bits == 0
    assert cfg.adc_range == 10.0
    assert cfg.dropout_probability == 0.0


# ---- add_gaussian_noise ----

def test_gaussian_noise_snr():
    """Output SNR should be approximately the requested SNR (within 3 dB)."""
    rng = np.random.default_rng(42)
    signal = np.sin(2 * np.pi * 100 * np.arange(100000) / 10000)
    snr_target = 20.0

    noisy = add_gaussian_noise(signal, snr_target, rng=rng)
    noise = noisy - signal
    rms_signal = np.sqrt(np.mean(signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    snr_actual = 20 * np.log10(rms_signal / rms_noise)

    assert abs(snr_actual - snr_target) < 3.0


def test_gaussian_noise_shape_preserved():
    signal = np.random.default_rng(0).standard_normal((3, 1000))
    noisy = add_gaussian_noise(signal, 30.0, rng=np.random.default_rng(1))
    assert noisy.shape == signal.shape


def test_gaussian_noise_reproducible():
    signal = np.sin(np.linspace(0, 10, 1000))
    n1 = add_gaussian_noise(signal, 30.0, rng=np.random.default_rng(42))
    n2 = add_gaussian_noise(signal, 30.0, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(n1, n2)


def test_gaussian_noise_different_seeds():
    signal = np.sin(np.linspace(0, 10, 1000))
    n1 = add_gaussian_noise(signal, 30.0, rng=np.random.default_rng(42))
    n2 = add_gaussian_noise(signal, 30.0, rng=np.random.default_rng(99))
    assert not np.allclose(n1, n2)


def test_gaussian_noise_disabled():
    """SNR <= 0 or inf should return unchanged signal."""
    signal = np.ones(100)
    result = add_gaussian_noise(signal, 0.0)
    np.testing.assert_array_equal(result, signal)

    result_inf = add_gaussian_noise(signal, np.inf)
    np.testing.assert_array_equal(result_inf, signal)


# ---- apply_noise combined pipeline ----

def test_apply_noise_no_effects():
    """NoiseConfig with all defaults off returns close to original."""
    signal = np.sin(np.linspace(0, 10, 10000))
    cfg = NoiseConfig(gaussian_snr_db=np.inf)  # disable Gaussian
    result = apply_noise(signal, cfg, sample_rate=10000.0)
    np.testing.assert_allclose(result, signal, atol=1e-10)


def test_apply_noise_1d_input():
    signal = np.sin(np.linspace(0, 10, 1000))
    cfg = NoiseConfig(gaussian_snr_db=30.0)
    result = apply_noise(signal, cfg, sample_rate=1000.0,
                         rng=np.random.default_rng(42))
    assert result.ndim == 1
    assert result.shape == signal.shape


def test_apply_noise_2d_input():
    signal = np.random.default_rng(0).standard_normal((4, 2000))
    cfg = NoiseConfig(gaussian_snr_db=30.0)
    result = apply_noise(signal, cfg, sample_rate=1000.0,
                         rng=np.random.default_rng(42))
    assert result.ndim == 2
    assert result.shape == signal.shape


def test_apply_noise_high_snr():
    """SNR=100 dB: output should be very close to input."""
    signal = np.sin(np.linspace(0, 10, 10000))
    cfg = NoiseConfig(gaussian_snr_db=100.0)
    result = apply_noise(signal, cfg, sample_rate=10000.0,
                         rng=np.random.default_rng(42))
    np.testing.assert_allclose(result, signal, atol=1e-3)


# ---- apply_bandwidth_limit ----

def test_bandwidth_limit_removes_high_freq():
    """Low-pass filter should attenuate signal above cutoff."""
    sr = 10000.0
    t = np.arange(10000) / sr
    low_freq = np.sin(2 * np.pi * 100 * t)
    high_freq = np.sin(2 * np.pi * 4000 * t)
    signal = low_freq + high_freq

    filtered = apply_bandwidth_limit(signal, 500.0, sr)
    # High frequency content should be attenuated
    fft_orig = np.abs(np.fft.rfft(signal))
    fft_filt = np.abs(np.fft.rfft(filtered))
    # Energy above 500 Hz bin
    freq_bins = np.fft.rfftfreq(len(signal), 1.0 / sr)
    high_mask = freq_bins > 500
    assert np.sum(fft_filt[high_mask]**2) < 0.01 * np.sum(fft_orig[high_mask]**2)


# ---- apply_quantization ----

def test_quantization_discrete_levels():
    signal = np.linspace(-5, 5, 10000)
    quantized = apply_quantization(signal, 8, 10.0)
    # Should have at most 2^8 + 1 unique values (boundary endpoint can add one)
    assert len(np.unique(quantized)) <= 257


# ---- apply_dropout ----

def test_dropout_zeros_some_samples():
    signal = np.ones(10000)
    dropped = apply_dropout(signal, 0.1, rng=np.random.default_rng(42))
    n_zeros = np.sum(dropped == 0.0)
    # Should be roughly 10% zeros (within 2%)
    assert abs(n_zeros / len(signal) - 0.1) < 0.02
