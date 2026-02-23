"""End-to-end signal generation pipeline (Subsystem B orchestrator).

Combines virtual sensor array, noise models, and modal results
to generate synthetic time-domain signals for ML training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import sys

import numpy as np

from turbomodal._utils import progress_bar as _progress_bar


@dataclass
class SignalGenerationConfig:
    """Configuration for the complete signal generation pipeline."""

    sample_rate: float = 100000.0      # Hz
    duration: float = 1.0              # seconds
    num_revolutions: int = 0           # If > 0, overrides duration
    seed: int = 42

    # Amplitude mode: how to set modal amplitudes
    amplitude_mode: str = "unit"       # "unit", "forced_response", "random"
    amplitude_scale: float = 1e-6      # meters (base amplitude for "unit" mode)

    # Which modes to include
    max_frequency: float = 0.0         # Hz (0 = all modes)
    max_modes_per_harmonic: int = 0    # 0 = all modes


def generate_signals_for_condition(
    sensor_array,
    modal_results: list,
    rpm: float,
    config: SignalGenerationConfig,
    noise_config=None,
    forced_response_result=None,
) -> dict:
    """Generate synthetic sensor signals for a single operating condition.

    Parameters
    ----------
    sensor_array : VirtualSensorArray instance
    modal_results : list of ModalResult (one per harmonic)
    rpm : rotational speed
    config : signal generation configuration
    noise_config : NoiseConfig (optional, for adding noise)
    forced_response_result : ForcedResponseResult (optional, for amplitudes)

    Returns
    -------
    dict with keys:
        'signals': (n_sensors, n_samples) float64
        'time': (n_samples,) float64
        'clean_signals': (n_sensors, n_samples) float64 (before noise)
        'modal_contributions': list of per-mode signal arrays
    """
    from turbomodal.noise import apply_noise, NoiseConfig

    n_sensors = len(sensor_array.config.sensors)
    duration = config.duration
    if config.num_revolutions > 0 and rpm > 0:
        duration = config.num_revolutions * 60.0 / rpm

    n_samples = int(duration * config.sample_rate)
    t = np.arange(n_samples) / config.sample_rate

    signals = np.zeros((n_sensors, n_samples))
    rng = np.random.default_rng(config.seed)

    # Rotation rate in Hz for stationary-frame frequency conversion
    omega_hz = rpm / 60.0

    mode_count = 0
    for result in modal_results:
        n_modes = len(result.frequencies)
        k = result.harmonic_index
        whirl_arr = np.asarray(result.whirl_direction) if hasattr(result, 'whirl_direction') else np.zeros(n_modes, dtype=np.int32)

        for m in range(n_modes):
            f_rot = result.frequencies[m]

            if config.max_modes_per_harmonic > 0 and m >= config.max_modes_per_harmonic:
                break

            # Get mode shape at sensor locations
            mode_shape = result.mode_shapes[:, m]
            sensor_response = sensor_array.sample_mode_shape(mode_shape)

            # Determine amplitude
            if config.amplitude_mode == "forced_response" and forced_response_result is not None:
                if mode_count < len(forced_response_result.max_response_amplitude):
                    amp = forced_response_result.max_response_amplitude[mode_count]
                else:
                    amp = config.amplitude_scale
            elif config.amplitude_mode == "random":
                amp = config.amplitude_scale * rng.exponential(1.0)
            else:
                amp = config.amplitude_scale

            # Convert rotating-frame frequency to stationary-frame.
            # For stator-mounted sensors observing a spinning disk:
            #   FW: f_stat = f_rot + k * Ω    (whirl_direction = +1)
            #   BW: f_stat = |f_rot - k * Ω|  (whirl_direction = -1)
            # When Coriolis is off (whirl=0) and k > 0, both FW and BW
            # components are generated from the degenerate mode pair.
            k_omega = k * omega_hz
            w = int(whirl_arr[m]) if m < len(whirl_arr) else 0

            if w != 0 or k == 0:
                # Single component: either Coriolis-split or k=0 standing
                freq = abs(f_rot + w * k_omega) if w != 0 else f_rot
                if config.max_frequency > 0 and freq > config.max_frequency:
                    mode_count += 1
                    continue
                phase = rng.uniform(0, 2 * np.pi)
                omega_t = 2 * np.pi * freq
                for s in range(n_sensors):
                    sensor_amp = np.abs(sensor_response[s])
                    sensor_phase = np.angle(sensor_response[s])
                    signals[s, :] += amp * sensor_amp * np.cos(
                        omega_t * t + phase + sensor_phase
                    )
            else:
                # No Coriolis, k > 0: generate both FW and BW from the
                # degenerate mode pair, each at half amplitude.
                f_fw = f_rot + k_omega
                f_bw = abs(f_rot - k_omega)
                phase = rng.uniform(0, 2 * np.pi)
                half_amp = amp * 0.5
                for f_stat in (f_fw, f_bw):
                    if config.max_frequency > 0 and f_stat > config.max_frequency:
                        continue
                    omega_t = 2 * np.pi * f_stat
                    for s in range(n_sensors):
                        sensor_amp = np.abs(sensor_response[s])
                        sensor_phase = np.angle(sensor_response[s])
                        signals[s, :] += half_amp * sensor_amp * np.cos(
                            omega_t * t + phase + sensor_phase
                        )

            mode_count += 1

    clean_signals = signals.copy()

    # Apply noise
    if noise_config is not None:
        signals = apply_noise(signals, noise_config, config.sample_rate, rng)

    return {
        "signals": signals,
        "time": t,
        "clean_signals": clean_signals,
    }


def generate_dataset_signals(
    mesh,
    modal_results_per_condition: list[list],
    conditions: list,
    sensor_array,
    config: SignalGenerationConfig = SignalGenerationConfig(),
    noise_config=None,
    forced_response_results: Optional[list] = None,
    verbose: int = 1,
) -> dict:
    """Generate synthetic sensor signals for a full parametric dataset.

    Parameters
    ----------
    mesh : Mesh
    modal_results_per_condition : results[cond_idx] = list of ModalResult
    conditions : list of OperatingCondition
    sensor_array : VirtualSensorArray instance
    config : signal generation config
    noise_config : NoiseConfig (optional)
    forced_response_results : list of ForcedResponseResult (optional)
    verbose : 0=silent, 1=progress

    Returns
    -------
    dict with keys:
        'signals': (n_conditions, n_sensors, n_samples) float64
        'clean_signals': (n_conditions, n_sensors, n_samples) float64
        'conditions': list of OperatingCondition
        'sample_rate': float
        'time': (n_samples,) float64
    """
    import time

    n_cond = len(modal_results_per_condition)
    n_sensors = len(sensor_array.config.sensors)

    duration = config.duration
    if config.num_revolutions > 0 and len(conditions) > 0:
        rpm = conditions[0].rpm if conditions[0].rpm > 0 else 3000
        duration = config.num_revolutions * 60.0 / rpm

    n_samples = int(duration * config.sample_rate)

    all_signals = np.zeros((n_cond, n_sensors, n_samples))
    all_clean = np.zeros((n_cond, n_sensors, n_samples))
    t_start = time.perf_counter()

    for i in range(n_cond):
        fr = forced_response_results[i] if forced_response_results else None
        rpm = conditions[i].rpm if i < len(conditions) else 0.0

        result = generate_signals_for_condition(
            sensor_array, modal_results_per_condition[i],
            rpm, config, noise_config, fr
        )

        all_signals[i] = result["signals"]
        all_clean[i] = result["clean_signals"]

        if verbose >= 1:
            elapsed = time.perf_counter() - t_start
            bar = _progress_bar(i + 1, n_cond, prefix="  Signal gen: ", elapsed=elapsed)
            sys.stdout.write(bar)
            sys.stdout.flush()
            if i == n_cond - 1:
                sys.stdout.write("\n")

    t = np.arange(n_samples) / config.sample_rate

    return {
        "signals": all_signals,
        "clean_signals": all_clean,
        "conditions": conditions,
        "sample_rate": config.sample_rate,
        "time": t,
    }
