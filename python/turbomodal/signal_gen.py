"""End-to-end signal generation pipeline (Subsystem B orchestrator).

Combines virtual sensor array, noise models, and modal results
to generate synthetic time-domain signals for ML training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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

    mode_count = 0
    for result in modal_results:
        n_modes = len(result.frequencies)
        for m in range(n_modes):
            freq = result.frequencies[m]

            if config.max_frequency > 0 and freq > config.max_frequency:
                continue
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

            # Random phase
            phase = rng.uniform(0, 2 * np.pi)

            # Add this mode's contribution to all sensors
            omega = 2 * np.pi * freq
            for s in range(n_sensors):
                sensor_amp = np.abs(sensor_response[s])
                sensor_phase = np.angle(sensor_response[s])
                signals[s, :] += amp * sensor_amp * np.cos(
                    omega * t + phase + sensor_phase
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
    import sys

    n_cond = len(modal_results_per_condition)
    n_sensors = len(sensor_array.config.sensors)

    duration = config.duration
    if config.num_revolutions > 0 and len(conditions) > 0:
        rpm = conditions[0].rpm if conditions[0].rpm > 0 else 3000
        duration = config.num_revolutions * 60.0 / rpm

    n_samples = int(duration * config.sample_rate)

    all_signals = np.zeros((n_cond, n_sensors, n_samples))
    all_clean = np.zeros((n_cond, n_sensors, n_samples))

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
            pct = 100 * (i + 1) / n_cond
            sys.stdout.write(f"\r  Signal generation: {i+1}/{n_cond} ({pct:.0f}%)")
            sys.stdout.flush()

    if verbose >= 1:
        sys.stdout.write("\n")

    t = np.arange(n_samples) / config.sample_rate

    return {
        "signals": all_signals,
        "clean_signals": all_clean,
        "conditions": conditions,
        "sample_rate": config.sample_rate,
        "time": t,
    }
