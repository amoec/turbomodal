"""Noise models for synthetic sensor signal generation.

Provides configurable noise injection including Gaussian noise, harmonic
interference, bandwidth limiting, sensor drift, ADC quantization, and
signal dropout. All functions handle both 1-D (n_samples,) and
2-D (n_channels, n_samples) signal arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt


@dataclass
class NoiseConfig:
    """Configuration for all noise effects applied to a synthetic signal.

    Parameters
    ----------
    gaussian_snr_db : Signal-to-noise ratio in dB for additive white
        Gaussian noise.  Set to ``0.0`` or ``inf`` to disable.
    harmonic_interference : List of interference tones.  Each entry is a
        dict with keys ``frequency_hz``, ``amplitude_ratio``
        (relative to signal RMS), and ``phase_deg``.
    drift_rate : Magnitude of the drift per second.  Units match the
        signal amplitude.
    drift_type : ``"none"``, ``"linear"``, or ``"random_walk"``.
    bandwidth_hz : If positive, low-pass filter the signal with a
        Butterworth filter at this cutoff frequency.  ``0.0`` disables.
    filter_order : Order of the Butterworth bandwidth-limit filter.
    adc_bits : If positive, quantize the signal to this many bits over
        ``[-adc_range/2, adc_range/2]``.  ``0`` disables.
    adc_range : Full-scale voltage range of the ADC.
    dropout_probability : Per-sample probability of replacing the sample
        with zero (simulating intermittent sensor dropouts).
    """

    gaussian_snr_db: float = 40.0
    harmonic_interference: list[dict] = field(default_factory=list)
    drift_rate: float = 0.0
    drift_type: str = "none"
    bandwidth_hz: float = 0.0
    filter_order: int = 4
    adc_bits: int = 0
    adc_range: float = 10.0
    dropout_probability: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ensure_2d(signal: NDArray) -> tuple[NDArray, bool]:
    """Return a 2-D view of *signal* and whether it was originally 1-D."""
    if signal.ndim == 1:
        return signal[np.newaxis, :], True
    return signal, False


def _restore_shape(signal: NDArray, was_1d: bool) -> NDArray:
    """Squeeze back to 1-D if the input was originally 1-D."""
    if was_1d:
        return signal[0]
    return signal


def add_gaussian_noise(
    signal: NDArray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Add white Gaussian noise at a specified SNR (dB).

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples) real array.
    snr_db : Desired signal-to-noise ratio in dB.  If ``<= 0`` or
        ``inf`` the signal is returned unchanged.
    rng : Optional NumPy random generator for reproducibility.

    Returns
    -------
    Noisy signal with the same shape as *signal*.
    """
    if snr_db <= 0 or not np.isfinite(snr_db):
        return signal.copy()

    if rng is None:
        rng = np.random.default_rng()

    sig, was_1d = _ensure_2d(signal)
    out = sig.copy()

    for ch in range(out.shape[0]):
        rms_signal = np.sqrt(np.mean(out[ch] ** 2))
        if rms_signal < 1e-30:
            continue
        rms_noise = rms_signal / (10.0 ** (snr_db / 20.0))
        out[ch] += rng.normal(0.0, rms_noise, size=out.shape[1])

    return _restore_shape(out, was_1d)


def add_harmonic_interference(
    signal: NDArray,
    harmonics: Sequence[dict],
    sample_rate: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Add sinusoidal interference tones to the signal.

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples) real array.
    harmonics : Sequence of dicts, each containing:
        - ``frequency_hz`` : tone frequency in Hz
        - ``amplitude_ratio`` : amplitude relative to the per-channel
          signal RMS
        - ``phase_deg`` : initial phase in degrees
    sample_rate : Sampling rate in Hz.
    rng : Unused; accepted for API consistency.

    Returns
    -------
    Signal with interference added, same shape as input.
    """
    if not harmonics:
        return signal.copy()

    sig, was_1d = _ensure_2d(signal)
    out = sig.copy()
    n_samples = out.shape[1]
    t = np.arange(n_samples) / sample_rate

    for ch in range(out.shape[0]):
        rms_signal = np.sqrt(np.mean(out[ch] ** 2))
        if rms_signal < 1e-30:
            rms_signal = 1.0  # fallback so interference is still audible

        for h in harmonics:
            freq = float(h["frequency_hz"])
            amp = float(h["amplitude_ratio"]) * rms_signal
            phase = np.deg2rad(float(h.get("phase_deg", 0.0)))
            out[ch] += amp * np.sin(2.0 * np.pi * freq * t + phase)

    return _restore_shape(out, was_1d)


def apply_bandwidth_limit(
    signal: NDArray,
    bandwidth_hz: float,
    sample_rate: float,
    order: int = 4,
) -> NDArray:
    """Low-pass filter the signal with a Butterworth filter.

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples) real array.
    bandwidth_hz : Cutoff frequency in Hz.  If ``<= 0`` or above the
        Nyquist frequency the signal is returned unchanged.
    sample_rate : Sampling rate in Hz.
    order : Filter order (passed to :func:`scipy.signal.butter`).

    Returns
    -------
    Filtered signal with the same shape as input.
    """
    nyquist = sample_rate / 2.0
    if bandwidth_hz <= 0 or bandwidth_hz >= nyquist:
        return signal.copy()

    sos = butter(order, bandwidth_hz / nyquist, btype="low", output="sos")

    sig, was_1d = _ensure_2d(signal)
    out = np.empty_like(sig)
    for ch in range(sig.shape[0]):
        out[ch] = sosfiltfilt(sos, sig[ch])

    return _restore_shape(out, was_1d)


def apply_drift(
    signal: NDArray,
    drift_rate: float,
    drift_type: str,
    sample_rate: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Add sensor drift to the signal.

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples) real array.
    drift_rate : Drift magnitude per second.
    drift_type : ``"none"``, ``"linear"``, or ``"random_walk"``.
    sample_rate : Sampling rate in Hz.
    rng : Optional NumPy random generator (used for ``"random_walk"``).

    Returns
    -------
    Signal with drift added, same shape as input.
    """
    if drift_type == "none" or drift_rate == 0.0:
        return signal.copy()

    if rng is None:
        rng = np.random.default_rng()

    sig, was_1d = _ensure_2d(signal)
    out = sig.copy()
    n_samples = out.shape[1]
    dt = 1.0 / sample_rate

    if drift_type == "linear":
        ramp = drift_rate * np.arange(n_samples) * dt
        out += ramp[np.newaxis, :]

    elif drift_type == "random_walk":
        for ch in range(out.shape[0]):
            steps = rng.normal(0.0, drift_rate * np.sqrt(dt), size=n_samples)
            out[ch] += np.cumsum(steps)
    else:
        raise ValueError(
            f"Unknown drift_type '{drift_type}'. "
            f"Expected 'none', 'linear', or 'random_walk'."
        )

    return _restore_shape(out, was_1d)


def apply_quantization(
    signal: NDArray,
    adc_bits: int,
    adc_range: float = 10.0,
) -> NDArray:
    """Simulate ADC quantization.

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples) real array.
    adc_bits : Resolution in bits.  If ``<= 0`` the signal is returned
        unchanged.
    adc_range : Full-scale voltage range (signal is clipped to
        ``[-adc_range/2, adc_range/2]``).

    Returns
    -------
    Quantized signal with the same shape as input.
    """
    if adc_bits <= 0:
        return signal.copy()

    n_levels = 2 ** adc_bits
    half_range = adc_range / 2.0
    step = adc_range / n_levels

    out = np.clip(signal, -half_range, half_range)
    out = np.round((out + half_range) / step) * step - half_range
    return out


def apply_dropout(
    signal: NDArray,
    probability: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Simulate intermittent sensor dropouts by zeroing random samples.

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples) real array.
    probability : Per-sample dropout probability in ``[0, 1]``.
    rng : Optional NumPy random generator.

    Returns
    -------
    Signal with dropouts applied, same shape as input.
    """
    if probability <= 0.0:
        return signal.copy()

    if rng is None:
        rng = np.random.default_rng()

    out = signal.copy()
    mask = rng.random(out.shape) < probability
    out[mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def apply_noise(
    signal: NDArray,
    config: NoiseConfig,
    sample_rate: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Apply all configured noise effects to *signal* in sequence.

    The processing order is:

    1. Harmonic interference
    2. Additive Gaussian noise
    3. Bandwidth limiting (Butterworth low-pass)
    4. Sensor drift
    5. ADC quantization
    6. Signal dropout

    Parameters
    ----------
    signal : (n_samples,) or (n_channels, n_samples) real array.
    config : :class:`NoiseConfig` controlling which effects are active.
    sample_rate : Sampling rate in Hz.
    rng : Optional NumPy random generator for reproducibility.

    Returns
    -------
    Corrupted signal with the same shape as *signal*.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = signal.copy().astype(np.float64)

    # 1. Harmonic interference
    if config.harmonic_interference:
        out = add_harmonic_interference(
            out, config.harmonic_interference, sample_rate, rng=rng,
        )

    # 2. Gaussian noise
    if config.gaussian_snr_db > 0 and np.isfinite(config.gaussian_snr_db):
        out = add_gaussian_noise(out, config.gaussian_snr_db, rng=rng)

    # 3. Bandwidth limit
    if config.bandwidth_hz > 0:
        out = apply_bandwidth_limit(
            out, config.bandwidth_hz, sample_rate, order=config.filter_order,
        )

    # 4. Drift
    if config.drift_type != "none" and config.drift_rate != 0.0:
        out = apply_drift(
            out, config.drift_rate, config.drift_type, sample_rate, rng=rng,
        )

    # 5. ADC quantization
    if config.adc_bits > 0:
        out = apply_quantization(out, config.adc_bits, config.adc_range)

    # 6. Dropout
    if config.dropout_probability > 0.0:
        out = apply_dropout(out, config.dropout_probability, rng=rng)

    return out
