"""Feature extraction for ML pipeline.

Transforms raw time-domain sensor signals into ML-ready features:
1. FFT magnitude and phase spectra per sensor
2. Traveling Wave Decomposition (spatial DFT across circumferential sensors)
3. Cross-spectral density matrix between sensor pairs
4. Order tracking features at integer engine orders
5. Physics-informed features (frequency ratios, centrifugal correction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FeatureConfig:
    """Configuration for feature extraction from sensor signals."""

    # FFT parameters
    fft_size: int = 2048
    hop_size: int = 512
    window: str = "hann"

    # Feature type
    feature_type: str = "spectrogram"  # "spectrogram", "mel", "order_tracking", "twd"

    # Mel spectrogram (if feature_type == "mel")
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 0.0  # 0 = Nyquist

    # Order tracking (if feature_type == "order_tracking")
    max_engine_order: int = 48
    rpm: float = 0.0

    # Traveling wave decomposition
    sensor_angles: Optional[np.ndarray] = None  # circumferential positions (rad)

    # Cross-spectral features
    include_cross_spectra: bool = False
    coherence_threshold: float = 0.5


def _hz_to_mel(f: float) -> float:
    """Convert frequency in Hz to mel scale.

    Parameters
    ----------
    f : float
        Frequency in Hz.

    Returns
    -------
    float
        Frequency in mels.
    """
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: float) -> float:
    """Convert mel scale value to frequency in Hz.

    Parameters
    ----------
    m : float
        Frequency in mels.

    Returns
    -------
    float
        Frequency in Hz.
    """
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _build_mel_filterbank(
    n_mels: int,
    n_fft_bins: int,
    sample_rate: float,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Build a mel-scale triangular filterbank matrix.

    Parameters
    ----------
    n_mels : int
        Number of mel bands.
    n_fft_bins : int
        Number of FFT frequency bins (from one-sided spectrum).
    sample_rate : float
        Sampling rate in Hz.
    f_min : float
        Minimum frequency in Hz.
    f_max : float
        Maximum frequency in Hz.

    Returns
    -------
    filterbank : ndarray, shape (n_mels, n_fft_bins)
        Triangular mel filterbank weights.
    """
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)

    # n_mels + 2 points: left edge, center of each band, right edge
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    # FFT bin frequencies
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_fft_bins)

    filterbank = np.zeros((n_mels, n_fft_bins))

    for i in range(n_mels):
        f_left = hz_points[i]
        f_center = hz_points[i + 1]
        f_right = hz_points[i + 2]

        # Rising slope: f_left to f_center
        if f_center > f_left:
            rising = (fft_freqs - f_left) / (f_center - f_left)
        else:
            rising = np.zeros_like(fft_freqs)

        # Falling slope: f_center to f_right
        if f_right > f_center:
            falling = (f_right - fft_freqs) / (f_right - f_center)
        else:
            falling = np.zeros_like(fft_freqs)

        filterbank[i] = np.maximum(0.0, np.minimum(rising, falling))

    return filterbank


def extract_features(
    signals: np.ndarray,
    sample_rate: float,
    config: FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """Extract features from time-domain sensor signals.

    Parameters
    ----------
    signals : ndarray, shape (n_sensors, n_samples) or (n_samples,)
        Time-domain signals from sensor array.
    sample_rate : float
        Sampling rate in Hz.
    config : FeatureConfig
        Feature extraction configuration.

    Returns
    -------
    features : ndarray, shape (n_features,)
        1-D feature vector whose length depends on ``config.feature_type``
        and the signal dimensions.

    Raises
    ------
    ValueError
        If ``config.feature_type`` is not one of the supported types, or if
        required parameters are missing (e.g. ``rpm`` for order tracking).
    """
    from scipy.signal import stft, csd, coherence as _coherence

    signals = np.asarray(signals, dtype=np.float64)

    # Handle 1-D input
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]  # (1, n_samples)

    if signals.ndim != 2:
        raise ValueError(
            f"signals must be 1-D or 2-D, got ndim={signals.ndim}"
        )

    n_sensors, n_samples = signals.shape

    if n_samples == 0:
        return np.array([], dtype=np.float64)

    feature_type = config.feature_type

    # ------------------------------------------------------------------
    # Spectrogram features
    # ------------------------------------------------------------------
    if feature_type == "spectrogram":
        parts = []
        for ch in range(n_sensors):
            _, _, Zxx = stft(
                signals[ch],
                fs=sample_rate,
                window=config.window,
                nperseg=config.fft_size,
                noverlap=config.fft_size - config.hop_size,
            )
            mag = np.abs(Zxx)  # (n_freq_bins, n_time_frames)
            mean_mag = mag.mean(axis=1)  # (n_freq_bins,)
            parts.append(mean_mag)
        feature_vec = np.concatenate(parts)  # (n_sensors * n_freq_bins,)

    # ------------------------------------------------------------------
    # Mel spectrogram features
    # ------------------------------------------------------------------
    elif feature_type == "mel":
        f_max = config.f_max if config.f_max > 0.0 else sample_rate / 2.0
        f_min = config.f_min

        parts = []
        for ch in range(n_sensors):
            _, _, Zxx = stft(
                signals[ch],
                fs=sample_rate,
                window=config.window,
                nperseg=config.fft_size,
                noverlap=config.fft_size - config.hop_size,
            )
            mag = np.abs(Zxx)  # (n_freq_bins, n_time_frames)
            mean_mag = mag.mean(axis=1)  # (n_freq_bins,)

            n_fft_bins = mean_mag.shape[0]
            mel_fb = _build_mel_filterbank(
                config.n_mels, n_fft_bins, sample_rate, f_min, f_max
            )  # (n_mels, n_fft_bins)
            mel_spec = mel_fb @ mean_mag  # (n_mels,)
            parts.append(mel_spec)
        feature_vec = np.concatenate(parts)  # (n_sensors * n_mels,)

    # ------------------------------------------------------------------
    # Order tracking features
    # ------------------------------------------------------------------
    elif feature_type == "order_tracking":
        if config.rpm <= 0.0:
            raise ValueError(
                "Order tracking requires config.rpm > 0, "
                f"got rpm={config.rpm}"
            )
        parts = []
        for ch in range(n_sensors):
            _, amplitudes = compute_order_spectrum(
                signals[ch], sample_rate, config.rpm, config.max_engine_order
            )
            # Split complex amplitudes into real and imaginary parts
            parts.append(amplitudes.real)
            parts.append(amplitudes.imag)
        # (n_sensors * max_engine_order * 2,)
        feature_vec = np.concatenate(parts)

    # ------------------------------------------------------------------
    # Traveling wave decomposition features
    # ------------------------------------------------------------------
    elif feature_type == "twd":
        if config.sensor_angles is None:
            raise ValueError(
                "TWD feature type requires config.sensor_angles to be set."
            )
        sensor_angles = np.asarray(config.sensor_angles)

        # Build frequency list from the STFT grid
        n_freq_bins = config.fft_size // 2 + 1
        frequencies = np.linspace(0, sample_rate / 2.0, n_freq_bins)

        forward, backward = traveling_wave_decomposition(
            signals, sensor_angles, frequencies, sample_rate
        )
        # forward and backward are (n_freq, max_nd) complex
        feature_vec = np.concatenate([
            np.abs(forward).ravel(),
            np.abs(backward).ravel(),
        ])

    else:
        raise ValueError(
            f"Unknown feature_type '{feature_type}'. "
            "Supported: 'spectrogram', 'mel', 'order_tracking', 'twd'."
        )

    # ------------------------------------------------------------------
    # Optional: cross-spectral density features
    # ------------------------------------------------------------------
    if config.include_cross_spectra and n_sensors >= 2:
        cross_parts = []
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                f_csd, Pxy = csd(
                    signals[i],
                    signals[j],
                    fs=sample_rate,
                    window=config.window,
                    nperseg=config.fft_size,
                    noverlap=config.fft_size - config.hop_size,
                )
                f_coh, Cxy = _coherence(
                    signals[i],
                    signals[j],
                    fs=sample_rate,
                    window=config.window,
                    nperseg=config.fft_size,
                    noverlap=config.fft_size - config.hop_size,
                )
                mag_csd = np.abs(Pxy)
                # Zero out bins below coherence threshold
                mag_csd[Cxy < config.coherence_threshold] = 0.0
                cross_parts.append(mag_csd)
        if cross_parts:
            feature_vec = np.concatenate([feature_vec] + cross_parts)

    return feature_vec


def compute_order_spectrum(
    signal: np.ndarray,
    sample_rate: float,
    rpm: float,
    max_order: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute order spectrum from a time-domain signal.

    Extracts the complex amplitude at each integer engine order by locating
    the nearest FFT bin to the target frequency ``f_n = n * rpm / 60``.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Single-channel time-domain signal.
    sample_rate : float
        Sampling rate in Hz.
    rpm : float
        Rotational speed in revolutions per minute. Must be positive.
    max_order : int
        Maximum engine order to extract (1 through ``max_order``).

    Returns
    -------
    orders : ndarray, shape (max_order,)
        Integer engine orders ``1, 2, ..., max_order``.
    amplitudes : ndarray, shape (max_order,), dtype complex128
        Complex amplitude at each order, scaled by ``2 / N``.

    Raises
    ------
    ValueError
        If ``rpm <= 0`` or the signal is empty.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    N = len(signal)

    if N == 0:
        raise ValueError("Signal is empty; cannot compute order spectrum.")
    if rpm <= 0.0:
        raise ValueError(f"rpm must be positive, got {rpm}")

    X = np.fft.rfft(signal)  # (N//2 + 1,) complex
    df = sample_rate / N  # frequency resolution

    orders = np.arange(1, max_order + 1)
    amplitudes = np.zeros(max_order, dtype=np.complex128)

    n_bins = len(X)

    for idx, n in enumerate(orders):
        f_n = n * rpm / 60.0
        bin_idx = int(round(f_n / df))
        if 0 <= bin_idx < n_bins:
            amplitudes[idx] = X[bin_idx] * (2.0 / N)
        # else: leave as zero (order frequency beyond Nyquist)

    return orders, amplitudes


def traveling_wave_decomposition(
    signals: np.ndarray,
    sensor_angles: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Traveling wave decomposition via spatial DFT.

    Decomposes circumferentially distributed sensor signals into forward
    and backward traveling wave components for each spatial harmonic
    (nodal diameter).

    Parameters
    ----------
    signals : ndarray, shape (n_sensors, n_samples)
        Time-domain signals from circumferentially placed sensors.
    sensor_angles : ndarray, shape (n_sensors,)
        Angular positions of sensors in radians.
    frequencies : ndarray, shape (n_freq,)
        Target frequencies (Hz) at which to evaluate the decomposition.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    forward : ndarray, shape (n_freq, max_nd), dtype complex128
        Forward (co-rotating) traveling wave amplitudes for spatial
        harmonics ``n = 0, 1, ..., max_nd - 1``.
    backward : ndarray, shape (n_freq, max_nd), dtype complex128
        Backward (counter-rotating) traveling wave amplitudes.

    Notes
    -----
    The spatial DFT coefficients are defined as:

    .. math::

        C_n^+(f) = \\frac{1}{K} \\sum_{k=0}^{K-1}
                   x_k(f) \\, e^{-j n \\theta_k}  \\quad (\\text{forward})

        C_n^-(f) = \\frac{1}{K} \\sum_{k=0}^{K-1}
                   x_k(f) \\, e^{+j n \\theta_k}  \\quad (\\text{backward})

    where *K* is the number of sensors, :math:`\\theta_k` is the angular
    position of sensor *k*, and :math:`x_k(f)` is the complex FFT
    coefficient of sensor *k* at frequency *f*.
    """
    signals = np.asarray(signals, dtype=np.float64)
    sensor_angles = np.asarray(sensor_angles, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)

    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    n_sensors, n_samples = signals.shape

    if n_samples == 0:
        max_nd = n_sensors // 2 + 1
        n_freq = len(frequencies)
        return (
            np.zeros((n_freq, max_nd), dtype=np.complex128),
            np.zeros((n_freq, max_nd), dtype=np.complex128),
        )

    # FFT of each sensor channel
    X = np.fft.rfft(signals, axis=1)  # (n_sensors, n_fft_bins)
    freq_axis = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    max_nd = n_sensors // 2 + 1
    n_freq = len(frequencies)

    forward = np.zeros((n_freq, max_nd), dtype=np.complex128)
    backward = np.zeros((n_freq, max_nd), dtype=np.complex128)

    K = n_sensors

    for f_idx, f_target in enumerate(frequencies):
        # Find nearest FFT bin
        bin_idx = int(np.argmin(np.abs(freq_axis - f_target)))
        x_f = X[:, bin_idx]  # (n_sensors,) complex

        for n in range(max_nd):
            # Forward: C_n = (1/K) * sum_k x_k * exp(-j*n*theta_k)
            exp_neg = np.exp(-1j * n * sensor_angles)
            forward[f_idx, n] = np.sum(x_f * exp_neg) / K

            # Backward: C_{-n} = (1/K) * sum_k x_k * exp(+j*n*theta_k)
            exp_pos = np.exp(1j * n * sensor_angles)
            backward[f_idx, n] = np.sum(x_f * exp_pos) / K

    return forward, backward


def build_feature_matrix(
    dataset_path: str,
    sensor_array,  # VirtualSensorArray
    signal_config: "SignalGenerationConfig",
    feature_config: FeatureConfig = FeatureConfig(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a feature matrix and label arrays from an HDF5 modal dataset.

    Loads modal results, generates synthetic sensor signals for each
    operating condition, extracts features, and collects ground-truth
    labels suitable for supervised ML training.

    Parameters
    ----------
    dataset_path : str
        Path to the HDF5 file produced by
        ``turbomodal.dataset.export_modal_results``.
    sensor_array : VirtualSensorArray
        Virtual sensor array used to synthesize time-domain signals.
    signal_config : SignalGenerationConfig
        Configuration controlling signal synthesis (duration, sample rate,
        amplitude mode, etc.).
    feature_config : FeatureConfig
        Configuration controlling feature extraction.

    Returns
    -------
    X : ndarray, shape (n_total_samples, n_features)
        Feature matrix where each row corresponds to one (condition, mode)
        pair.
    y : dict of ndarray
        Ground-truth labels with keys:

        - ``"nodal_diameter"`` : int array, shape (n_total_samples,)
        - ``"nodal_circle"`` : int array, shape (n_total_samples,) --
          always 0 (NC identification requires C++ mode shape analysis).
        - ``"whirl_direction"`` : int array, shape (n_total_samples,)
        - ``"frequency"`` : float array, shape (n_total_samples,)
        - ``"wave_velocity"`` : float array, shape (n_total_samples,)

    Notes
    -----
    Imports ``turbomodal.dataset`` and ``turbomodal.signal_gen`` inside the
    function body to avoid circular imports.

    Wave velocity is computed as ``v = 2 * pi * f / nd`` for ``nd > 0``;
    for ``nd == 0`` (axisymmetric modes) it is set to 0.
    """
    # Deferred imports to avoid circular dependencies
    import turbomodal.dataset as ds
    import turbomodal.signal_gen as sg

    mesh_data, conditions, results_dict = ds.load_modal_results(dataset_path)

    all_features: list[np.ndarray] = []
    all_nd: list[int] = []
    all_nc: list[int] = []
    all_whirl: list[int] = []
    all_freq: list[float] = []
    all_velocity: list[float] = []

    for cond in conditions:
        cid = cond.condition_id
        if cid not in results_dict:
            continue

        entry = results_dict[cid]
        eigenvalues = entry["eigenvalues"]        # (n_harmonics, n_modes)
        harmonic_index = entry["harmonic_index"]   # (n_harmonics,)
        whirl_direction = entry["whirl_direction"] # (n_harmonics, n_modes)

        # Build lightweight proxy objects for signal generation.
        # ``generate_signals_for_condition`` expects objects with
        # ``.frequencies``, ``.mode_shapes``, and ``.harmonic_index``.

        class _ModalProxy:
            """Minimal stand-in for a ModalResult."""

            def __init__(
                self, freqs: np.ndarray, shapes: np.ndarray, h_idx: int
            ) -> None:
                self.frequencies = freqs
                self.mode_shapes = shapes
                self.harmonic_index = h_idx
                self.whirl_direction = np.zeros(len(freqs), dtype=np.int32)

        modal_result_proxies: list[object] = []

        has_shapes = "mode_shapes" in entry
        n_harmonics, n_modes = eigenvalues.shape

        for h in range(n_harmonics):
            freqs_h = eigenvalues[h]
            # Filter out NaN-padded trailing modes
            valid = ~np.isnan(freqs_h)
            freqs_valid = freqs_h[valid]

            if has_shapes:
                shapes_h = entry["mode_shapes"][h]  # (n_modes, n_dof)
                shapes_valid = shapes_h[valid]       # (n_valid, n_dof)
                # ModalResult stores mode_shapes as (n_dof, n_modes)
                shapes_valid = shapes_valid.T
            else:
                # No mode shapes available -- create zeros so signal gen
                # still works (produces zero signals).
                n_dof = 3  # minimal placeholder
                shapes_valid = np.zeros(
                    (n_dof, int(valid.sum())), dtype=np.complex128
                )

            proxy = _ModalProxy(
                freqs=freqs_valid,
                shapes=shapes_valid,
                h_idx=int(harmonic_index[h]),
            )
            proxy.whirl_direction = whirl_direction[h, valid].astype(np.int32)
            modal_result_proxies.append(proxy)

        # Generate signals for this condition
        sig_result = sg.generate_signals_for_condition(
            sensor_array,
            modal_result_proxies,
            cond.rpm,
            signal_config,
        )
        signals = sig_result["signals"]  # (n_sensors, n_samples)

        # Extract features (one feature vector for the whole condition)
        feat = extract_features(signals, signal_config.sample_rate, feature_config)

        # Create one sample per (harmonic, mode) pair, all sharing the
        # same feature vector (the condition-level features).
        for h in range(n_harmonics):
            freqs_h = eigenvalues[h]
            valid = ~np.isnan(freqs_h)
            nd = int(harmonic_index[h])

            for m_global in np.where(valid)[0]:
                freq = float(eigenvalues[h, m_global])
                whirl = int(whirl_direction[h, m_global])

                # Wave velocity: v = 2*pi*f / nd  (nd > 0)
                if nd > 0:
                    velocity = 2.0 * np.pi * freq / nd
                else:
                    velocity = 0.0

                all_features.append(feat)
                all_nd.append(nd)
                all_nc.append(0)
                all_whirl.append(whirl)
                all_freq.append(freq)
                all_velocity.append(velocity)

    if len(all_features) == 0:
        n_feat = 0
        X = np.empty((0, n_feat), dtype=np.float64)
    else:
        X = np.vstack(all_features)  # (n_total_samples, n_features)

    y = {
        "nodal_diameter": np.array(all_nd, dtype=np.int32),
        "nodal_circle": np.array(all_nc, dtype=np.int32),
        "whirl_direction": np.array(all_whirl, dtype=np.int32),
        "frequency": np.array(all_freq, dtype=np.float64),
        "wave_velocity": np.array(all_velocity, dtype=np.float64),
    }

    return X, y
