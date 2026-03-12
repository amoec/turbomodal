"""End-to-end signal generation pipeline (Subsystem B orchestrator).

Combines virtual sensor array, noise models, and modal results
to generate synthetic time-domain signals for ML training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import sys

import numpy as np

from turbomodal._utils import progress_bar as _progress_bar
from turbomodal.noise import apply_noise, NoiseConfig


# ---------------------------------------------------------------------------
# ExcitationModel — physics-based excitation configuration
# ---------------------------------------------------------------------------

@dataclass
class ExcitationModel:
    """Physics-based excitation model for realistic signal generation.

    Implements the simplified analytical model from turbomachinery modal
    excitation research: engine order selection rules, Campbell crossing
    detection, mode family roll-off, ND-dependent damping, mistuning,
    and colored broadband noise.

    Parameters
    ----------
    stator_vane_counts : Upstream/downstream stator vane counts.  Engine
        orders are ``EO = h * V`` for each vane count *V* and harmonic
        *h* = 1 .. *max_eo_harmonic*.
    max_eo_harmonic : Number of harmonics per primary EO to include.
        ``None`` auto-computes from the modal frequency range: enough
        harmonics so that ``h * V * RPM / 60`` covers the highest modal
        frequency in the results.
    leo_orders : Low engine orders from inlet distortion / geometric
        imperfections.  Excited at amplitudes *leo_amplitude_db* below
        the primary EO.
    leo_amplitude_db : LEO amplitude relative to primary EO (dB).
    base_amplitude : Reference amplitude A0 in metres.
    nc_rolloff_alpha : Exponent for nodal circle amplitude roll-off:
        ``A *= 1 / (1 + NC) ** alpha``.
    eo_harmonic_decay_beta : Exponent for EO harmonic amplitude decay:
        ``A *= 1 / h ** beta``.
    rpm_scaling_exponent : RPM-dependent scaling exponent:
        ``A *= (RPM / rpm_ref) ** exponent``.
    rpm_ref : Reference RPM for RPM-dependent scaling.
    random_amplitude_sigma_db : Log-normal random amplitude variation
        standard deviation in dB.
    structural_damping_ratio : Baseline structural damping ratio
        (material + friction).
    aero_damping_mean : Mean aerodynamic damping ratio.
    aero_damping_variation : Amplitude of sinusoidal ND-dependent aero
        damping: ``zeta_aero(k) = mean * [1 + A * sin(2*pi*k/N)]``.
    campbell_crossing_tolerance : Fractional frequency tolerance for
        detecting Campbell crossings: ``|f_mode - f_eo| / f_mode < tol``.
    multi_mode_sampling : If True, randomly select how many modes are
        active per example using *mode_count_weights*.
    mode_count_weights : Probability weights for (1, 2, 3+) active
        modes.  Only used when *multi_mode_sampling* is True.
    mistuning_sigma : Std of per-blade frequency deviation.  0 = tuned.
    mistuning_method : ``"jitter"`` for simplified per-blade jitter,
        ``"fmm"`` for Fundamental Mistuning Model.
    blade_amplitude_jitter_std : Std of per-blade amplitude multiplier
        (fraction of 1.0).  Used with ``mistuning_method="jitter"``.
    blade_phase_jitter_deg : Max per-blade phase jitter in degrees.
    broadband_snr_db : Colored broadband noise level relative to peak
        deterministic signal (dB).  0 = no broadband noise.
    broadband_spectral_exponent : PSD spectral roll-off exponent gamma
        in ``PSD ~ f^(-gamma)``.  Default 5/3 (Kolmogorov).
    """

    stator_vane_counts: list[int] = field(default_factory=lambda: [24])
    max_eo_harmonic: int | None = None
    leo_orders: list[int] = field(default_factory=lambda: [1, 2, 3])
    leo_amplitude_db: float = -20.0

    base_amplitude: float = 1e-6
    nc_rolloff_alpha: float = 1.5
    eo_harmonic_decay_beta: float = 1.0
    rpm_scaling_exponent: float = 2.0
    rpm_ref: float = 5000.0
    random_amplitude_sigma_db: float = 3.0

    structural_damping_ratio: float = 0.003
    aero_damping_mean: float = 0.002
    aero_damping_variation: float = 0.5

    campbell_crossing_tolerance: float = 0.05

    multi_mode_sampling: bool = False
    mode_count_weights: tuple[float, float, float] = (0.6, 0.3, 0.1)

    mistuning_sigma: float = 0.0
    mistuning_method: str = "jitter"
    blade_amplitude_jitter_std: float = 0.15
    blade_phase_jitter_deg: float = 10.0

    broadband_snr_db: float = 20.0
    broadband_spectral_exponent: float = 5.0 / 3.0


# ---------------------------------------------------------------------------
# ActiveMode — internal descriptor for physics-selected modes
# ---------------------------------------------------------------------------

@dataclass
class ActiveMode:
    """Descriptor for a single active mode in the physics model."""

    harmonic_index: int
    mode_index: int
    frequency: float
    amplitude: float
    whirl_direction: int
    engine_order: int
    eo_harmonic: int
    nodal_circle: int
    phase: float
    blade_amplitudes: np.ndarray | None = None
    blade_phases: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Physics helper functions
# ---------------------------------------------------------------------------

def eo_excites_harmonic(eo: int, k: int, num_sectors: int) -> bool:
    """Check if engine order *eo* excites harmonic index *k*.

    The aliasing rule: EO excites k if ``k == EO mod N`` or
    ``k == (-EO) mod N`` (backward alias), where N = *num_sectors*.
    """
    if num_sectors <= 0:
        return False
    eo_mod = eo % num_sectors
    if eo_mod < 0:
        eo_mod += num_sectors
    k_pos = k % num_sectors
    if k_pos < 0:
        k_pos += num_sectors
    k_neg = (-k) % num_sectors
    if k_neg < 0:
        k_neg += num_sectors
    return eo_mod == k_pos or eo_mod == k_neg


def eo_whirl_direction(eo: int, k: int, num_sectors: int) -> int:
    """Determine FW/BW direction from the EO aliasing rule.

    Returns +1 (forward wave) if ``EO mod N == k``, -1 (backward wave)
    if ``EO mod N == N - k``, or 0 for k=0 / k=N/2 (standing wave).
    """
    if k == 0:
        return 0
    N = num_sectors
    if N > 0 and N % 2 == 0 and k == N // 2:
        return 0  # k = N/2 is a standing wave
    eo_mod = ((eo % N) + N) % N
    k_pos = ((k % N) + N) % N
    k_neg = (((-k) % N) + N) % N
    if eo_mod == k_pos:
        return 1
    if eo_mod == k_neg:
        return -1
    return 0


def total_damping_ratio(
    k: int, num_sectors: int, model: ExcitationModel,
) -> float:
    """Compute total damping ratio (structural + aero) for harmonic *k*.

    ``zeta_aero(k) = mean * [1 + A * sin(2*pi*k/N)]``
    ``zeta_total  = zeta_struct + zeta_aero(k)``
    """
    N = max(num_sectors, 1)
    zeta_aero = model.aero_damping_mean * (
        1.0 + model.aero_damping_variation * math.sin(2.0 * math.pi * k / N)
    )
    return model.structural_damping_ratio + max(zeta_aero, 0.0)


def frf_detuning_factor(f_mode: float, f_eo: float, zeta: float) -> float:
    """Compute normalised single-DOF FRF magnitude.

    ``H(r) = 1 / sqrt((1 - r^2)^2 + (2*zeta*r)^2)``

    The result is normalised by the on-resonance peak ``1/(2*zeta)``
    so that an exactly resonant mode returns ~1.0.
    """
    if f_mode <= 0:
        return 0.0
    r = f_eo / f_mode
    denom_sq = (1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2
    if denom_sq <= 0:
        return 1.0
    H = 1.0 / math.sqrt(denom_sq)
    H_peak = 1.0 / (2.0 * zeta) if zeta > 0 else 1.0
    return H / H_peak if H_peak > 0 else H


def _resolve_max_eo_harmonic(
    model: ExcitationModel,
    modal_results: list,
    f_rev: float,
) -> int:
    """Resolve *max_eo_harmonic*: ``None`` → auto from modal frequency range."""
    if model.max_eo_harmonic is not None:
        return model.max_eo_harmonic
    f_max = 0.0
    for mr in modal_results:
        freqs = np.asarray(mr.frequencies)
        if len(freqs) > 0:
            f_max = max(f_max, float(freqs.max()))
    min_V = min(model.stator_vane_counts) if model.stator_vane_counts else 1
    if f_rev > 0 and min_V > 0:
        return max(1, int(math.ceil(f_max / (min_V * f_rev))))
    return 20  # fallback


def _build_eo_list(
    model: ExcitationModel,
    max_eo_h: int,
) -> list[tuple[int, int, bool]]:
    """Build list of ``(EO, harmonic_number, is_leo)`` tuples."""
    eo_list: list[tuple[int, int, bool]] = []
    for V in model.stator_vane_counts:
        for h in range(1, max_eo_h + 1):
            eo_list.append((h * V, h, False))
    for leo in model.leo_orders:
        eo_list.append((leo, 1, True))
    return eo_list


def find_resonance_crossings(
    modal_results: list,
    rpm: float,
    num_sectors: int,
    stator_vane_counts: list[int] | None = None,
    max_eo_harmonic: int | None = None,
    max_freq: float | None = None,
    tolerance: float = 0.05,
    mode_ids: list | None = None,
    mesh=None,
    eo_whitelist: list[int] | None = None,
) -> list[dict]:
    """Find resonance crossings at discrete modal frequencies.

    For each engine order, computes the excitation frequency
    ``f_eo = EO * RPM / 60``, determines which harmonic index *k* it
    excites via the aliasing rule (``EO mod N``), and checks whether any
    mode at that *k* has a frequency within *tolerance* of *f_eo*.

    This is the single source of truth for resonance detection, used by
    both the ZZENF/Campbell visualizations and the physics-based signal
    generation model.

    Parameters
    ----------
    modal_results : list of ModalResult at a single RPM.
    rpm : rotational speed in RPM.
    num_sectors : number of sectors in the full annulus (N).
    stator_vane_counts : vane counts to restrict crossings to NPF engine
        orders (``h * V``).  ``None`` means check all integer EOs up to
        the frequency ceiling.
    max_eo_harmonic : number of harmonics per vane count.  ``None``
        auto-computes from the modal frequency range.  Ignored when
        *stator_vane_counts* is ``None``.
    max_freq : upper frequency limit.  ``None`` = use highest modal
        frequency + 10 %%.
    tolerance : fractional frequency tolerance for crossing detection:
        ``|f_mode - f_eo| / f_mode < tolerance``.
    mode_ids : unused (kept for API compatibility).
    mesh : optional Mesh for nodal circle identification.
    eo_whitelist : if given, only check these specific EO values instead of
        building the full EO list from stator vane counts or frequency ceiling.

    Returns
    -------
    list of dict
        Each dict has keys:

        - ``nd`` : folded nodal diameter (int)
        - ``frequency`` : modal frequency in Hz (float)
        - ``eo`` : engine order (int)
        - ``harmonic_index`` : harmonic index *k* (int)
        - ``mode_index`` : mode index within that *k* (int)
        - ``whirl_direction`` : +1 (FW), -1 (BW), 0 (standing)
        - ``is_npf`` : True if the EO is an NPF harmonic
    """
    if rpm == 0 or num_sectors <= 0:
        return []

    N = num_sectors
    half_N = N // 2
    f_rev = abs(rpm) / 60.0

    # -- Determine frequency ceiling --
    if max_freq is None:
        max_freq = 0.0
        for mr in modal_results:
            freqs = np.asarray(mr.frequencies)
            if len(freqs) > 0:
                max_freq = max(max_freq, float(freqs.max()))
        max_freq *= 1.1

    # -- Build EO list --
    if eo_whitelist is not None:
        # Caller provided an explicit list of EOs to check
        eo_list = sorted(eo_whitelist)
        is_npf = False
    elif stator_vane_counts is not None:
        # NPF engine orders only
        if max_eo_harmonic is None:
            min_V = min(stator_vane_counts) if stator_vane_counts else 1
            if f_rev > 0 and min_V > 0:
                max_eo_h = max(1, int(math.ceil(max_freq / (min_V * f_rev))))
            else:
                max_eo_h = 20
        else:
            max_eo_h = max_eo_harmonic
        eo_set: set[int] = set()
        for V in stator_vane_counts:
            for h in range(1, max_eo_h + 1):
                eo_set.add(h * V)
        eo_list = sorted(eo_set)
        is_npf = True
    else:
        # All integer EOs up to frequency ceiling
        max_eo = int(max_freq / f_rev) + 1 if f_rev > 0 else 0
        eo_list = list(range(1, max_eo + 1))
        is_npf = False

    # -- Find crossings: for each EO, check aliased harmonic indices --
    crossings: list[dict] = []
    seen: set[tuple[int, int, int]] = set()  # (eo, k, mode_index)

    for eo in eo_list:
        f_eo = eo * f_rev
        if f_eo <= 0 or f_eo > max_freq:
            continue

        # Which harmonic indices does this EO excite?
        for mr in modal_results:
            k = mr.harmonic_index
            if not eo_excites_harmonic(eo, k, N):
                continue

            nd_folded = k if k <= half_N else N - k
            freqs = np.asarray(mr.frequencies)
            whirl_arr = np.asarray(mr.whirl_direction)

            for m_idx in range(len(freqs)):
                f_mode = float(freqs[m_idx])
                if f_mode <= 0:
                    continue

                detuning = abs(f_mode - f_eo) / f_mode
                if detuning > tolerance:
                    continue

                key = (eo, k, m_idx)
                if key in seen:
                    continue
                seen.add(key)

                w = int(whirl_arr[m_idx]) if m_idx < len(whirl_arr) else 0
                whirl = eo_whirl_direction(eo, k, N)

                crossings.append({
                    "nd": nd_folded,
                    "frequency": f_mode,
                    "eo": eo,
                    "harmonic_index": k,
                    "mode_index": m_idx,
                    "whirl_direction": whirl if whirl != 0 else w,
                    "is_npf": is_npf,
                })

    return crossings


def find_campbell_crossings(
    modal_results: list,
    rpm: float,
    num_sectors: int,
    model: ExcitationModel,
    mesh=None,
) -> list[dict]:
    """Find active Campbell crossings at the given RPM.

    Delegates to :func:`find_resonance_crossings` for geometric zig-zag
    intersection, then enriches the results with EO harmonic number,
    nodal circle, detuning ratio, and LEO classification.

    Returns a list of dicts with keys: ``eo``, ``eo_harmonic``,
    ``harmonic_index``, ``mode_index``, ``frequency``, ``nodal_circle``,
    ``whirl_direction``, ``detuning_ratio``, ``is_leo``.
    """
    if rpm == 0:
        return []

    f_rev = abs(rpm) / 60.0
    max_eo_h = _resolve_max_eo_harmonic(model, modal_results, f_rev)

    tol = model.campbell_crossing_tolerance

    # Find NPF crossings (stator-vane engine orders)
    geo_crossings = find_resonance_crossings(
        modal_results, rpm, num_sectors,
        stator_vane_counts=model.stator_vane_counts,
        max_eo_harmonic=max_eo_h,
        tolerance=tol,
        mesh=mesh,
    )

    # Also find LEO crossings (low engine orders not tied to stator vanes)
    # Only check the specific LEO EOs, not all integer EOs
    leo_eos = set(model.leo_orders)

    leo_crossings = []
    if leo_eos:
        leo_crossings = find_resonance_crossings(
            modal_results, rpm, num_sectors,
            stator_vane_counts=None,
            max_eo_harmonic=None,
            tolerance=tol,
            mode_ids=None,
            mesh=mesh,
            eo_whitelist=sorted(leo_eos),
        )

    # Build NPF EO set for classifying
    npf_eos: dict[int, int] = {}  # eo -> harmonic number h
    for V in model.stator_vane_counts:
        for h in range(1, max_eo_h + 1):
            npf_eos[h * V] = h

    # Optionally identify nodal circles
    nc_map: dict[tuple[int, int], int] = {}
    if mesh is not None:
        try:
            from turbomodal._core import identify_modes
            for mr in modal_results:
                ids = identify_modes(mr, mesh)
                for m_idx, mid in enumerate(ids):
                    nc_map[(mr.harmonic_index, m_idx)] = mid.nodal_circle
        except Exception:
            pass

    # Merge and enrich
    seen: set[tuple[int, int]] = set()
    result: list[dict] = []

    for cx in geo_crossings:
        key = (cx["harmonic_index"], cx["mode_index"])
        if key in seen:
            continue
        seen.add(key)

        eo = cx["eo"]
        h = npf_eos.get(eo, 1)
        # Compute detuning from nearest mode
        f_eo = eo * f_rev
        f_mode = cx["frequency"]
        detuning = abs(f_mode - f_eo) / f_mode if f_mode > 0 else 0.0

        result.append({
            "eo": eo,
            "eo_harmonic": h,
            "harmonic_index": cx["harmonic_index"],
            "mode_index": cx["mode_index"],
            "frequency": f_mode,
            "nodal_circle": nc_map.get(key, 0),
            "whirl_direction": cx["whirl_direction"],
            "detuning_ratio": detuning,
            "is_leo": False,
        })

    for cx in leo_crossings:
        key = (cx["harmonic_index"], cx["mode_index"])
        if key in seen:
            continue
        seen.add(key)

        eo = cx["eo"]
        f_eo = eo * f_rev
        f_mode = cx["frequency"]
        detuning = abs(f_mode - f_eo) / f_mode if f_mode > 0 else 0.0

        result.append({
            "eo": eo,
            "eo_harmonic": 1,
            "harmonic_index": cx["harmonic_index"],
            "mode_index": cx["mode_index"],
            "frequency": f_mode,
            "nodal_circle": nc_map.get(key, 0),
            "whirl_direction": cx["whirl_direction"],
            "detuning_ratio": detuning,
            "is_leo": True,
        })

    return result


def compute_physics_amplitudes(
    modal_results: list,
    rpm: float,
    num_sectors: int,
    model: ExcitationModel,
    mesh=None,
    rng: np.random.Generator | None = None,
) -> list[ActiveMode]:
    """Compute physically realistic amplitudes for active modes.

    Implements the simplified analytical excitation model (Appendix B):

    1. Find Campbell crossings
    2. Base amplitude: ``A = A0 / ((1+NC)^alpha * zeta_total(k))``
    3. EO harmonic decay: ``A *= 1 / h^beta``
    4. LEO scaling: ``A *= 10^(leo_db/20)``
    5. Off-resonance detuning via FRF transfer function
    6. RPM scaling: ``A *= (RPM / rpm_ref)^exp``
    7. Log-normal random variation
    8. FW/BW from aliasing rule
    9. Optional multi-mode sampling
    10. Optional mistuning (per-blade jitter or FMM)
    """
    if rng is None:
        rng = np.random.default_rng()

    crossings = find_campbell_crossings(
        modal_results, rpm, num_sectors, model, mesh=mesh,
    )
    if not crossings:
        return []

    f_rev = abs(rpm) / 60.0
    active: list[ActiveMode] = []

    for cx in crossings:
        k = cx["harmonic_index"]
        nc = cx["nodal_circle"]
        h = cx["eo_harmonic"]
        f_eo = cx["eo"] * f_rev
        f_mode = cx["frequency"]

        # 2. Base amplitude with NC roll-off and damping
        zeta = total_damping_ratio(k, num_sectors, model)
        A = model.base_amplitude / ((1.0 + nc) ** model.nc_rolloff_alpha * zeta)

        # 3. EO harmonic decay
        if h > 1:
            A /= h ** model.eo_harmonic_decay_beta

        # 4. LEO scaling
        if cx["is_leo"]:
            A *= 10.0 ** (model.leo_amplitude_db / 20.0)

        # 5. Off-resonance detuning
        A *= frf_detuning_factor(f_mode, f_eo, zeta)

        # 6. RPM scaling
        if model.rpm_ref > 0:
            A *= (abs(rpm) / model.rpm_ref) ** model.rpm_scaling_exponent

        # 7. Log-normal random variation
        if model.random_amplitude_sigma_db > 0:
            sigma_log10 = model.random_amplitude_sigma_db / 20.0
            A *= 10.0 ** rng.normal(0.0, sigma_log10)

        # 8. FW/BW from aliasing
        whirl = cx["whirl_direction"]

        phase = rng.uniform(0, 2 * np.pi)

        active.append(ActiveMode(
            harmonic_index=k,
            mode_index=cx["mode_index"],
            frequency=f_mode,
            amplitude=A,
            whirl_direction=whirl,
            engine_order=cx["eo"],
            eo_harmonic=h,
            nodal_circle=nc,
            phase=phase,
        ))

    # 9a. Amplitude thresholding — drop negligible components
    # Keep modes whose amplitude is within 60 dB of the strongest
    if active:
        max_amp = max(am.amplitude for am in active)
        if max_amp > 0:
            threshold = max_amp * 1e-3  # -60 dB
            active = [am for am in active if am.amplitude >= threshold]

    # 9b. Multi-mode sampling
    if model.multi_mode_sampling and len(active) > 1:
        w = np.array(model.mode_count_weights, dtype=np.float64)
        w /= w.sum()
        n_modes_choice = rng.choice([1, 2, 3], p=w)
        n_keep = min(n_modes_choice, len(active))
        # Keep the strongest modes
        active.sort(key=lambda m: m.amplitude, reverse=True)
        active = active[:n_keep]

    # 10. Mistuning
    if model.mistuning_sigma > 0 and num_sectors > 0:
        if model.mistuning_method == "fmm":
            _apply_fmm_mistuning(active, num_sectors, model, modal_results, rng)
        else:
            _apply_jitter_mistuning(active, num_sectors, model, rng)

    return active


def _apply_jitter_mistuning(
    active: list[ActiveMode],
    num_sectors: int,
    model: ExcitationModel,
    rng: np.random.Generator,
) -> None:
    """Apply simplified per-blade amplitude/phase jitter."""
    for am in active:
        am.blade_amplitudes = 1.0 + rng.normal(
            0.0, model.blade_amplitude_jitter_std, size=num_sectors,
        )
        am.blade_amplitudes = np.clip(am.blade_amplitudes, 0.1, None)
        am.blade_phases = np.deg2rad(
            rng.uniform(
                -model.blade_phase_jitter_deg,
                model.blade_phase_jitter_deg,
                size=num_sectors,
            )
        )


def _apply_fmm_mistuning(
    active: list[ActiveMode],
    num_sectors: int,
    model: ExcitationModel,
    modal_results: list,
    rng: np.random.Generator,
) -> None:
    """Apply Fundamental Mistuning Model for per-blade amplitudes."""
    try:
        from turbomodal._core import FMMSolver
    except ImportError:
        _apply_jitter_mistuning(active, num_sectors, model, rng)
        return

    blade_devs = FMMSolver.random_mistuning(
        num_sectors, model.mistuning_sigma,
        seed=int(rng.integers(0, 2**31)),
    )

    # Group active modes by mode family (NC)
    for am in active:
        # Get tuned frequencies for this mode family across NDs
        tuned_freqs = []
        for mr in modal_results:
            freqs = np.asarray(mr.frequencies)
            if am.mode_index < len(freqs):
                tuned_freqs.append(freqs[am.mode_index])
        if len(tuned_freqs) < 2:
            _apply_jitter_mistuning([am], num_sectors, model, rng)
            continue

        tuned_freqs_arr = np.array(tuned_freqs[:num_sectors // 2 + 1])
        try:
            result = FMMSolver.solve(num_sectors, tuned_freqs_arr, blade_devs)
            # Use blade amplitudes from FMM
            blade_amps = np.abs(result.blade_amplitudes[:, 0])
            blade_amps /= np.mean(blade_amps) if np.mean(blade_amps) > 0 else 1.0
            am.blade_amplitudes = blade_amps
            am.blade_phases = np.angle(result.blade_amplitudes[:, 0])
        except Exception:
            _apply_jitter_mistuning([am], num_sectors, model, rng)


# ---------------------------------------------------------------------------
# SignalGenerationConfig
# ---------------------------------------------------------------------------

@dataclass
class SignalGenerationConfig:
    """Configuration for the complete signal generation pipeline.

    Parameters
    ----------
    sample_rate : Sampling rate in Hz.
    duration : Total acquisition duration in seconds.
    num_revolutions : If > 0, overrides *duration* based on RPM.
    seed : Random seed for reproducibility.
    amplitude_mode : ``"unit"``, ``"forced_response"``, ``"random"``,
        or ``"physics"``.
    amplitude_scale : Base amplitude in metres (for ``"unit"`` mode).
    max_frequency : Upper frequency cutoff in Hz (0 = no limit).
    max_modes_per_harmonic : Maximum modes per harmonic index (0 = all).
    time : Custom time array; overrides all other time parameters.
    t_start : Start time in seconds.
    t_end : End time in seconds (0 = use *duration* or *num_revolutions*).
    damping_ratio : Modal damping ratio zeta (0 = undamped).
    excitation_model : Physics-based excitation model.  Required when
        *amplitude_mode* is ``"physics"``.
    """

    sample_rate: float = 100000.0      # Hz
    duration: float = 1.0              # seconds
    num_revolutions: int = 0           # If > 0, overrides duration
    seed: int = 42

    # Amplitude mode: how to set modal amplitudes
    amplitude_mode: str = "unit"       # "unit", "forced_response", "random", "physics"
    amplitude_scale: float = 1e-6      # meters (base amplitude for "unit" mode)

    # Which modes to include
    max_frequency: float = 0.0         # Hz (0 = all modes)
    max_modes_per_harmonic: int = 0    # 0 = all modes

    # Time vector control
    time: np.ndarray | None = None     # Custom time array (overrides all below)
    t_start: float = 0.0              # Start time (s)
    t_end: float = 0.0                # End time (0 = use duration)
    damping_ratio: float = 0.0        # Modal damping ζ (0 = undamped)

    # Physics-based excitation model (used when amplitude_mode="physics")
    excitation_model: ExcitationModel | None = None


@dataclass
class RayHitGeometry:
    """Pre-computed ray-surface intersection geometry for one sensor.

    Stores everything needed to evaluate mode shape displacement at each
    angular bin during one sector sweep, without per-time-step ray tracing.

    Attributes
    ----------
    hit_mask : (n_steps,) bool — True where the ray hits blade surface.
    hit_points : (n_steps, 3) float64 — intersection coordinates in
        disk rest frame.  NaN where no hit.
    cell_ids : (n_steps,) int64 — surface cell (triangle) index at each
        hit.  -1 where no hit.
    local_node_ids : (n_steps, 3) int64 — sector-0 node IDs of the three
        triangle vertices.  -1 where no hit.
    bary_coords : (n_steps, 3) float64 — barycentric coordinates of the
        hit point within its triangle.  0 where no hit.
    sector_ids : (n_steps,) int — which sector (0..N-1) each hit
        triangle belongs to.  -1 where no hit.
    """

    hit_mask: np.ndarray
    hit_points: np.ndarray
    cell_ids: np.ndarray
    local_node_ids: np.ndarray
    bary_coords: np.ndarray
    sector_ids: np.ndarray


def _build_annulus_surface(mesh):
    """Build the full annulus outer surface as a PyVista PolyData.

    Replicates the sector mesh across all *N* sectors, then extracts
    the triangulated surface.  The result is cached — pass the same
    mesh object to avoid rebuilding.
    """
    import pyvista as pv
    from turbomodal._utils import rotation_matrix_3x3

    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements)
    n_nodes = mesh.num_nodes()
    n_elem = elements.shape[0]
    N = mesh.num_sectors
    sector_angle = 2.0 * np.pi / N
    axis = mesh.rotation_axis

    all_pts = np.empty((N * n_nodes, 3))
    all_cells = np.empty((N * n_elem, 11), dtype=np.int64)
    all_celltypes = np.full(N * n_elem, 24, dtype=np.uint8)  # VTK_QUADRATIC_TETRA

    for s in range(N):
        R = rotation_matrix_3x3(s * sector_angle, axis)
        all_pts[s * n_nodes:(s + 1) * n_nodes] = nodes @ R.T
        offset = s * n_elem
        all_cells[offset:offset + n_elem, 0] = 10
        all_cells[offset:offset + n_elem, 1:] = elements + s * n_nodes

    grid = pv.UnstructuredGrid(all_cells.ravel(), all_celltypes, all_pts)
    return grid.extract_surface(algorithm="dataset_surface")


def _precompute_ray_hits(surface, sensors, is_stat, mesh, sector_angle,
                         n_steps: int = 256) -> dict[int, np.ndarray]:
    """Pre-compute ray-surface intersections for stationary sensors.

    For each stationary sensor, simulate the disk rotating through one
    sector (all sectors are identical by cyclic symmetry).  At each
    angular step, rotate the sensor into the disk's rest frame and cast
    a ray along its measurement direction.  Record hit/miss.

    Returns ``{sensor_idx: bool_array(n_steps)}``.
    """
    from turbomodal._utils import rotation_matrix_3x3

    axis = mesh.rotation_axis
    results: dict[int, np.ndarray] = {}

    for s_idx, sensor in enumerate(sensors):
        if not is_stat[s_idx]:
            continue

        pos = np.asarray(sensor.position, dtype=np.float64)
        direction = np.asarray(sensor.direction, dtype=np.float64)
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-30:
            continue
        direction = direction / d_norm

        hit_mask = np.zeros(n_steps, dtype=bool)

        for i in range(n_steps):
            # Rotate sensor backward by (i / n_steps) * sector_angle
            # (equivalent to disk rotating forward by that amount)
            angle = -i * sector_angle / n_steps
            R = rotation_matrix_3x3(angle, axis)
            pos_rot = R @ pos
            dir_rot = R @ direction

            # Extend ray well beyond the mesh
            end_point = pos_rot + dir_rot * 10.0
            hit_pts, _ = surface.ray_trace(pos_rot, end_point)
            hit_mask[i] = len(hit_pts) > 0

        results[s_idx] = hit_mask

    return results


def _compute_barycentric(p, v0, v1, v2):
    """Compute barycentric coordinates of point *p* in triangle (v0, v1, v2).

    Returns (lam0, lam1, lam2) such that p ≈ lam0*v0 + lam1*v1 + lam2*v2.
    """
    e0 = v1 - v0
    e1 = v2 - v0
    e2 = p - v0
    d00 = np.dot(e0, e0)
    d01 = np.dot(e0, e1)
    d11 = np.dot(e1, e1)
    d20 = np.dot(e2, e0)
    d21 = np.dot(e2, e1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return 1.0 / 3, 1.0 / 3, 1.0 / 3  # degenerate triangle
    lam1 = (d11 * d20 - d01 * d21) / denom
    lam2 = (d00 * d21 - d01 * d20) / denom
    lam0 = 1.0 - lam1 - lam2
    return lam0, lam1, lam2


def _precompute_ray_geometry(surface, sensors, is_stat, mesh, sector_angle,
                             n_steps: int = 256) -> dict[int, RayHitGeometry]:
    """Pre-compute ray-surface intersections with full geometry data.

    Like :func:`_precompute_ray_hits` but also records the hit point,
    surface cell, triangle vertex node IDs (mapped back to sector-0),
    barycentric coordinates, and sector index for each angular bin.

    Returns ``{sensor_idx: RayHitGeometry}``.
    """
    from turbomodal._utils import rotation_matrix_3x3

    axis = mesh.rotation_axis
    n_nodes = mesh.num_nodes()
    N = mesh.num_sectors

    # Surface topology — map surface point IDs back to volume node IDs
    orig_ids = np.asarray(surface.point_data["vtkOriginalPointIds"])
    surf_pts = np.asarray(surface.points)

    # Triangle connectivity: surface.regular_faces is (n_tris, 3) for a
    # fully-triangulated PolyData.  Fall back to manual extraction from
    # the faces array if the property is unavailable.
    if hasattr(surface, "regular_faces") and surface.regular_faces is not None:
        tri_verts = np.asarray(surface.regular_faces)  # (n_tris, 3)
    else:
        raw = np.asarray(surface.faces)
        # VTK format: [n_verts, v0, v1, v2, n_verts, v0, ...]
        tri_verts = raw.reshape(-1, 4)[:, 1:]  # assume all triangles

    results: dict[int, RayHitGeometry] = {}

    for s_idx, sensor in enumerate(sensors):
        if not is_stat[s_idx]:
            continue

        pos = np.asarray(sensor.position, dtype=np.float64)
        direction = np.asarray(sensor.direction, dtype=np.float64)
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-30:
            continue
        direction = direction / d_norm

        hit_mask = np.zeros(n_steps, dtype=bool)
        hit_points = np.full((n_steps, 3), np.nan)
        cell_id_arr = np.full(n_steps, -1, dtype=np.int64)
        local_node_ids = np.full((n_steps, 3), -1, dtype=np.int64)
        bary_coords = np.zeros((n_steps, 3))
        sector_ids = np.full(n_steps, -1, dtype=np.intp)

        for i in range(n_steps):
            angle = -i * sector_angle / n_steps
            R = rotation_matrix_3x3(angle, axis)
            pos_rot = R @ pos
            dir_rot = R @ direction

            end_point = pos_rot + dir_rot * 10.0
            hit_pts, cell_ids = surface.ray_trace(pos_rot, end_point)
            if len(hit_pts) == 0:
                continue

            hit_mask[i] = True
            hit_points[i] = hit_pts[0]  # closest hit
            cid = int(cell_ids[0])
            cell_id_arr[i] = cid

            # Triangle vertex surface-local indices → volume node IDs
            sv = tri_verts[cid]  # (3,) surface-local vertex indices
            vol_ids = orig_ids[sv]  # (3,) volume node IDs in full annulus

            # Map to sector and local node ID within sector 0
            sec = vol_ids // n_nodes
            loc = vol_ids % n_nodes
            # Use first vertex's sector as the representative
            sector_ids[i] = int(sec[0])
            local_node_ids[i] = loc

            # Barycentric coordinates
            v0 = surf_pts[sv[0]]
            v1 = surf_pts[sv[1]]
            v2 = surf_pts[sv[2]]
            lam0, lam1, lam2 = _compute_barycentric(hit_pts[0], v0, v1, v2)
            bary_coords[i] = [lam0, lam1, lam2]

        results[s_idx] = RayHitGeometry(
            hit_mask=hit_mask,
            hit_points=hit_points,
            cell_ids=cell_id_arr,
            local_node_ids=local_node_ids,
            bary_coords=bary_coords,
            sector_ids=sector_ids,
        )

    return results


def _build_time_vector(config: SignalGenerationConfig, rpm: float) -> np.ndarray:
    """Build the time vector from config, respecting priority order.

    Priority: ``config.time`` > ``t_start/t_end`` > ``num_revolutions`` > ``duration``.
    """
    if config.time is not None:
        return np.asarray(config.time, dtype=np.float64)

    if config.t_end > config.t_start:
        t_start = config.t_start
        t_end = config.t_end
    elif config.num_revolutions > 0 and rpm != 0:
        t_start = config.t_start
        t_end = t_start + config.num_revolutions * 60.0 / abs(rpm)
    else:
        t_start = config.t_start
        t_end = t_start + config.duration

    n_samples = max(1, int((t_end - t_start) * config.sample_rate))
    return t_start + np.arange(n_samples) / config.sample_rate


def _sensor_is_stationary(sensor) -> bool:
    """Determine whether a sensor is stationary (casing-mounted) or rotating.

    By default, only strain gauges are rotating (mounted on the blade).
    BTT probes, casing accelerometers, and displacement sensors (eddy
    current, capacitive, laser vibrometer) are stationary.  Override
    with ``SensorLocation(is_stationary=...)``.
    """
    from turbomodal.sensors import SensorType

    if sensor.is_stationary is not None:
        return sensor.is_stationary
    # Only strain gauges rotate with the disk
    return sensor.sensor_type != SensorType.STRAIN_GAUGE


def filter_modal_results(
    modal_results: list,
    mesh=None,
    nd: int | list[int] | None = None,
    nc: int | list[int] | None = None,
    whirl: int | list[int] | None = None,
    modes: int | list[int] | None = None,
) -> list:
    """Filter modal results by nodal diameter, nodal circles, whirl, or mode index.

    Parameters
    ----------
    modal_results : list of ModalResult (one per harmonic index).
    mesh : Mesh object (required when filtering by *nc*).
    nd : Nodal diameter(s) to keep.  ``None`` keeps all.
    nc : Nodal circle(s) to keep.  ``None`` keeps all.
        Requires *mesh* for on-the-fly mode identification.
    whirl : Whirl direction(s) to keep: ``+1`` (FW), ``-1`` (BW),
        ``0`` (degenerate/standing).  ``None`` keeps all.
    modes : Mode index/indices within each ND to keep (0-based,
        ordered by frequency).  For example, ``modes=0`` keeps only
        the lowest-frequency mode per ND; ``modes=[0, 1]`` keeps the
        two lowest.  ``None`` keeps all.  Applied after *whirl* and
        *nc* filtering.

    Returns
    -------
    Filtered list of ModalResult.  When *nc*, *whirl*, or *modes* is
    specified, individual results may contain fewer modes than the
    originals.
    """
    from turbomodal._core import ModalResult

    # Normalise scalar → list
    nd_set: set[int] | None = None
    nc_set: set[int] | None = None
    whirl_set: set[int] | None = None
    modes_set: set[int] | None = None
    if nd is not None:
        nd_set = {nd} if isinstance(nd, int) else set(nd)
    if nc is not None:
        nc_set = {nc} if isinstance(nc, int) else set(nc)
    if whirl is not None:
        whirl_set = {whirl} if isinstance(whirl, int) else set(whirl)
    if modes is not None:
        modes_set = {modes} if isinstance(modes, int) else set(modes)

    # Step 1: filter by ND (harmonic_index)
    if nd_set is not None:
        modal_results = [r for r in modal_results if r.harmonic_index in nd_set]

    # Step 2: filter by whirl direction (per-mode)
    if whirl_set is not None:
        filtered_w: list = []
        for r in modal_results:
            w_arr = np.asarray(r.whirl_direction)
            keep = [i for i in range(len(r.frequencies)) if int(w_arr[i]) in whirl_set]
            if not keep:
                continue
            nr = ModalResult()
            nr.harmonic_index = r.harmonic_index
            nr.rpm = r.rpm
            nr.converged = r.converged
            idx = np.array(keep)
            nr.frequencies = np.asarray(r.frequencies)[idx]
            nr.mode_shapes = np.asarray(r.mode_shapes)[:, idx]
            nr.whirl_direction = np.asarray(r.whirl_direction)[idx]
            filtered_w.append(nr)
        modal_results = filtered_w

    # Step 3: filter by NC (requires mode identification)
    if nc_set is not None:
        if mesh is None:
            raise ValueError("mesh is required when filtering by nc")
        from turbomodal._core import identify_modes

        filtered: list = []
        for r in modal_results:
            ids = identify_modes(r, mesh)
            keep = [i for i, mid in enumerate(ids) if mid.nodal_circle in nc_set]
            if not keep:
                continue
            # Build a new ModalResult with only the kept modes
            nr = ModalResult()
            nr.harmonic_index = r.harmonic_index
            nr.rpm = r.rpm
            nr.converged = r.converged
            idx = np.array(keep)
            nr.frequencies = np.asarray(r.frequencies)[idx]
            nr.mode_shapes = np.asarray(r.mode_shapes)[:, idx]
            nr.whirl_direction = np.asarray(r.whirl_direction)[idx]
            filtered.append(nr)
        modal_results = filtered

    # Step 4: filter by mode index (0-based position within each ND)
    if modes_set is not None:
        filtered_m: list = []
        for r in modal_results:
            n = len(r.frequencies)
            keep = sorted(i for i in modes_set if 0 <= i < n)
            if not keep:
                continue
            nr = ModalResult()
            nr.harmonic_index = r.harmonic_index
            nr.rpm = r.rpm
            nr.converged = r.converged
            idx = np.array(keep)
            nr.frequencies = np.asarray(r.frequencies)[idx]
            nr.mode_shapes = np.asarray(r.mode_shapes)[:, idx]
            nr.whirl_direction = np.asarray(r.whirl_direction)[idx]
            filtered_m.append(nr)
        modal_results = filtered_m

    return modal_results


def generate_signals_for_condition(
    sensor_array,
    modal_results: list,
    rpm: float,
    config: SignalGenerationConfig,
    noise_config=None,
    forced_response_result=None,
    condition=None,
) -> dict:
    """Generate synthetic sensor signals for a single operating condition.

    Uses a **full-annulus virtual probe** model: for each mode the
    displacement field is ``u(r, θ, z, t) = Re[φ_k(r,z) ·
    exp(−j·w·k·θ) · exp(j·ω·t)]``, where *θ* is the sensor's
    circumferential angle and *w* is the whirl direction.  Stationary
    sensors (BTT probes, casing accelerometers) observe the Doppler-
    shifted stationary-frame frequency; rotating sensors (strain
    gauges) observe the rotating-frame frequency directly.

    **Blade passage gating** uses ray tracing against the full-annulus
    surface mesh.  For each stationary sensor a ray is cast along its
    measurement direction; if the ray hits the blade surface the
    displacement is read, otherwise (gap between blades) the signal
    is zero.  The hit pattern is pre-computed once for one sector
    sweep (cyclic symmetry) and tiled across the time series.  This
    naturally handles shrouded blades (surface spans the full sector →
    ray always hits), unshrouded blades (gaps → ray misses), and
    arbitrary sensor orientations.

    Parameters
    ----------
    sensor_array : VirtualSensorArray instance
    modal_results : list of ModalResult (one per harmonic)
    rpm : rotational speed in RPM
    config : signal generation configuration
    noise_config : NoiseConfig (optional, for adding noise)
    forced_response_result : ForcedResponseResult (optional, for amplitudes)
    condition : OperatingCondition (optional, included in output for tracing)

    Returns
    -------
    dict with keys:
        ``'signals'``
            ``(n_sensors, n_samples)`` float64.
        ``'time'``
            ``(n_samples,)`` float64.
        ``'clean_signals'``
            ``(n_sensors, n_samples)`` float64 (before noise).
        ``'active_modes'``
            ``list[ActiveMode]`` — only present when
            ``amplitude_mode='physics'``.  Contains the modes that
            were actually excited (after amplitude thresholding and
            multi-mode sampling).
        ``'condition'``
            OperatingCondition (if provided).
        ``'btt_arrival_times'``
            ``{sensor_idx: ndarray}`` — discrete blade arrival times
            (only for BTT probes).
        ``'btt_deflections'``
            ``{sensor_idx: ndarray}`` — deflection at each arrival.
        ``'btt_blade_indices'``
            ``{sensor_idx: ndarray}`` — which blade produced each
            arrival.
    """
    sensors = sensor_array.config.sensors
    n_sensors = len(sensors)
    t = _build_time_vector(config, rpm)
    n_samples = len(t)

    signals = np.zeros((n_sensors, n_samples))
    rng = np.random.default_rng(config.seed)

    # --- Rotation parameters ---
    omega_hz = abs(rpm) / 60.0          # rev/s
    omega_rad = 2.0 * np.pi * omega_hz  # rad/s

    # --- Sensor classification ---
    is_stat = np.array([_sensor_is_stationary(s) for s in sensors])

    # --- Circumferential geometry (when mesh is available) ---
    mesh = getattr(sensor_array, 'mesh', None)
    has_mesh = mesh is not None and hasattr(mesh, 'num_sectors')

    if has_mesh:
        theta_s = sensor_array.sensor_circumferential_angles()  # (n_sensors,)
        N = mesh.num_sectors
        sector_angle = 2.0 * np.pi / N
    else:
        theta_s = np.zeros(n_sensors)
        N = 0
        sector_angle = 0.0

    # --- Ray geometry for surface displacement lookup ---
    # For stationary sensors looking at discrete blades, pre-compute
    # the full ray-surface intersection geometry (hit points, triangle
    # vertices, barycentric coords) so we can evaluate the actual mode
    # shape displacement at each blade passage instead of using the
    # analytical approximation + binary gating.
    #
    # Only use the ray-based path for sensors that see GAPS (i.e. the
    # hit mask is not all-True).  For continuous surfaces (shrouds,
    # solid disks) the analytical approach is correct and avoids
    # spurious amplitude modulation from mesh discretisation.
    ray_geometry: dict[int, RayHitGeometry] = {}
    has_ray_geom = np.zeros(n_sensors, dtype=bool)
    if has_mesh and np.any(is_stat) and omega_rad > 0:
        try:
            full_geom = sensor_array.ray_hit_geometry()
            for s_idx, rg in full_geom.items():
                if not np.all(rg.hit_mask):  # has gaps → use ray-based path
                    ray_geometry[s_idx] = rg
            has_ray_geom = np.array([s in ray_geometry for s in range(n_sensors)])
        except Exception:
            pass

    # --- Mode shape sampling ---
    # sample_mode_shape uses nearest node in sector 0; the circumferential
    # phase factor exp(-j*w*k*θ) is applied separately below.

    if config.amplitude_mode == "physics":
        # --- Physics-based amplitude model ---
        if config.excitation_model is None:
            raise ValueError(
                "amplitude_mode='physics' requires excitation_model to be set "
                "in SignalGenerationConfig"
            )
        active_modes = compute_physics_amplitudes(
            modal_results, rpm, N, config.excitation_model,
            mesh=mesh, rng=rng,
        )
        if not active_modes:
            import warnings
            em = config.excitation_model
            f_rev = abs(rpm) / 60.0
            max_eo_h = _resolve_max_eo_harmonic(em, modal_results, f_rev)
            eo_list = [h * V for V in em.stator_vane_counts
                       for h in range(1, max_eo_h + 1)]
            eo_freqs = [eo * f_rev for eo in eo_list]
            modal_freq_ranges = []
            for mr in modal_results:
                fs = np.asarray(mr.frequencies)
                if len(fs) > 0:
                    modal_freq_ranges.append(
                        f"k={mr.harmonic_index}: {fs.min():.1f}-{fs.max():.1f} Hz"
                    )
            warnings.warn(
                f"amplitude_mode='physics' produced no active modes at "
                f"{rpm:.0f} RPM. No geometric zig-zag crossings found.\n"
                f"  EOs from stator vanes (max_eo_harmonic={max_eo_h}): "
                f"{eo_list}\n"
                f"  EO excitation frequencies: "
                f"{[f'{f:.1f} Hz' for f in eo_freqs]}\n"
                f"  LEO orders: {em.leo_orders}\n"
                f"  Modal frequency ranges: {modal_freq_ranges}\n"
                f"  num_sectors (N): {N}\n"
                f"Hint: ensure the RPM corresponds to a resonance crossing "
                f"where the EO zig-zag intersects a modal family curve in "
                f"the ZZENF diagram.",
                stacklevel=2,
            )
        # Build a lookup from harmonic_index → ModalResult
        mr_by_k: dict[int, object] = {mr.harmonic_index: mr for mr in modal_results}

        for am in active_modes:
            mr = mr_by_k.get(am.harmonic_index)
            if mr is None:
                continue
            if am.mode_index >= len(mr.frequencies):
                continue
            phi_s = sensor_array.sample_mode_shape(mr.mode_shapes[:, am.mode_index])
            omega_rot_val = 2.0 * np.pi * am.frequency

            if am.blade_amplitudes is not None:
                _add_mistuned_component(
                    signals, t, phi_s, am.amplitude, omega_rot_val, am.phase,
                    am.harmonic_index, am.whirl_direction, omega_rad,
                    theta_s, is_stat, None, config.max_frequency,
                    am.blade_amplitudes, am.blade_phases, N,
                    sensors=sensors,
                    skip_sensors=has_ray_geom,
                )
            else:
                _add_single_component(
                    signals, t, phi_s, am.amplitude, omega_rot_val, am.phase,
                    am.harmonic_index, am.whirl_direction, omega_rad,
                    theta_s, is_stat, None, config.max_frequency,
                    skip_sensors=has_ray_geom,
                )

            # Ray-based displacement for stationary sensors with geometry
            if np.any(has_ray_geom):
                _add_ray_based_component(
                    signals, t, mr.mode_shapes[:, am.mode_index],
                    am.amplitude, omega_rot_val, am.phase,
                    am.harmonic_index, am.whirl_direction, omega_rad,
                    theta_s, is_stat, None, config.max_frequency,
                    ray_geometry, mesh.num_nodes(), N, sensors,
                    blade_amplitudes=am.blade_amplitudes,
                    blade_phases=am.blade_phases,
                )

        # Colored broadband noise (added before sensor noise pipeline)
        if (config.excitation_model.broadband_snr_db > 0
                and active_modes and len(t) > 1):
            peak_amp = max(am.amplitude for am in active_modes)
            _add_colored_broadband(
                signals, peak_amp, config.excitation_model,
                config.sample_rate, rng,
            )
    else:
        # --- Original amplitude modes: unit / random / forced_response ---
        mode_count = 0
        for mr in modal_results:
            n_modes = len(mr.frequencies)
            k = mr.harmonic_index
            whirl_arr = (
                np.asarray(mr.whirl_direction)
                if hasattr(mr, 'whirl_direction')
                else np.zeros(n_modes, dtype=np.int32)
            )

            for m in range(n_modes):
                f_rot = mr.frequencies[m]
                if config.max_modes_per_harmonic > 0 and m >= config.max_modes_per_harmonic:
                    break

                phi_s = sensor_array.sample_mode_shape(mr.mode_shapes[:, m])  # (n_sensors,) complex

                # Amplitude
                if config.amplitude_mode == "forced_response" and forced_response_result is not None:
                    if mode_count < len(forced_response_result.max_response_amplitude):
                        amp = forced_response_result.max_response_amplitude[mode_count]
                    else:
                        amp = config.amplitude_scale
                elif config.amplitude_mode == "random":
                    amp = config.amplitude_scale * rng.exponential(1.0)
                else:
                    amp = config.amplitude_scale

                omega_rot = 2.0 * np.pi * f_rot
                w = int(whirl_arr[m]) if m < len(whirl_arr) else 0
                phase = rng.uniform(0, 2 * np.pi)

                # Damping envelope: exp(-ζ·ω·(t - t0))
                if config.damping_ratio > 0:
                    t0 = t[0]
                    envelope = np.exp(-config.damping_ratio * omega_rot * (t - t0))
                else:
                    envelope = None

                # ---- Synthesise per-mode contribution ----
                # Physics: u(r,θ,z,t) = Re[φ(r,z) · exp(-j·w·k·θ) · exp(j·ω·t)]
                # FW (w=+1): exp(-jkθ) rotates with the disk → lab freq = f_rot + kΩ
                # BW (w=-1): exp(+jkθ) rotates against disk → lab freq = |f_rot − kΩ|

                if w != 0 or k == 0:
                    # --- Single component (Coriolis-split or axisymmetric k=0) ---
                    _add_single_component(
                        signals, t, phi_s, amp, omega_rot, phase,
                        k, w, omega_rad, theta_s, is_stat, envelope,
                        config.max_frequency,
                        skip_sensors=has_ray_geom,
                    )
                    # Ray-based displacement for stationary sensors
                    if np.any(has_ray_geom):
                        _add_ray_based_component(
                            signals, t, mr.mode_shapes[:, m],
                            amp, omega_rot, phase,
                            k, w, omega_rad, theta_s, is_stat, envelope,
                            config.max_frequency,
                            ray_geometry, mesh.num_nodes(), N, sensors,
                        )
                else:
                    # --- Degenerate (w=0, k>0): emit FW + BW at half amplitude ---
                    for w_comp in (+1, -1):
                        _add_single_component(
                            signals, t, phi_s, amp * 0.5, omega_rot, phase,
                            k, w_comp, omega_rad, theta_s, is_stat, envelope,
                            config.max_frequency,
                            skip_sensors=has_ray_geom,
                        )
                        # Ray-based displacement for stationary sensors
                        if np.any(has_ray_geom):
                            _add_ray_based_component(
                                signals, t, mr.mode_shapes[:, m],
                                amp * 0.5, omega_rot, phase,
                                k, w_comp, omega_rad, theta_s, is_stat, envelope,
                                config.max_frequency,
                                ray_geometry, mesh.num_nodes(), N, sensors,
                            )

                mode_count += 1

    # --- Blade passage gating (fallback) and BTT extraction ---
    # Sensors with ray geometry already have physically correct signals
    # (zero where no hit, proper per-blade displacement where hit).
    # Sensors WITHOUT ray geometry still need the old binary gating.
    # BTT arrival extraction works for both paths.
    from turbomodal.sensors import SensorType

    is_btt = np.array([s.sensor_type == SensorType.BTT_PROBE for s in sensors])

    btt_arrival_times: dict[int, np.ndarray] = {}
    btt_deflections: dict[int, np.ndarray] = {}
    btt_blade_indices: dict[int, np.ndarray] = {}

    if has_mesh and np.any(is_stat) and omega_rad > 0:
        T_rev = 2.0 * np.pi / omega_rad

        # Fallback binary gating for stationary sensors without ray geometry
        fallback_sensors = is_stat & ~has_ray_geom
        if np.any(fallback_sensors):
            try:
                ray_hits = sensor_array.ray_hit_pattern()
            except Exception:
                ray_hits = {}

            for s in range(n_sensors):
                if not fallback_sensors[s] or s not in ray_hits:
                    continue

                hit_mask_sector = ray_hits[s]
                n_steps_mask = len(hit_mask_sector)

                theta_local = (theta_s[s] - omega_rad * t) % sector_angle
                bin_idx = (theta_local / sector_angle * n_steps_mask).astype(np.intp)
                np.clip(bin_idx, 0, n_steps_mask - 1, out=bin_idx)
                on_blade = hit_mask_sector[bin_idx]

                signals[s, ~on_blade] = 0.0

        # Discrete blade arrivals (BTT probes only) — works for both paths
        for s in range(n_sensors):
            if not (is_stat[s] and is_btt[s]):
                continue

            # Use ray geometry hit mask if available, else ray_hit_pattern
            if has_ray_geom[s]:
                hit_mask_sector = ray_geometry[s].hit_mask
            else:
                try:
                    rh = sensor_array.ray_hit_pattern()
                    if s not in rh:
                        continue
                    hit_mask_sector = rh[s]
                except Exception:
                    continue

            arrivals: list[float] = []
            deflections_list: list[float] = []
            blade_ids: list[int] = []
            for b in range(N):
                blade_angle = 2.0 * np.pi * b / N
                t0_blade = (theta_s[s] - blade_angle) / omega_rad
                while t0_blade < t[0]:
                    t0_blade += T_rev
                while t0_blade <= t[-1]:
                    arrivals.append(t0_blade)
                    idx_t = min(np.searchsorted(t, t0_blade), n_samples - 1)
                    deflections_list.append(signals[s, idx_t])
                    blade_ids.append(b)
                    t0_blade += T_rev

            btt_arrival_times[s] = np.array(arrivals)
            btt_deflections[s] = np.array(deflections_list)
            btt_blade_indices[s] = np.array(blade_ids, dtype=np.int32)

    clean_signals = signals.copy()

    # Apply noise
    if noise_config is not None:
        signals = apply_noise(signals, noise_config, config.sample_rate, rng)

    result_dict: dict = {
        "signals": signals,
        "time": t,
        "clean_signals": clean_signals,
    }
    if config.amplitude_mode == "physics":
        result_dict["active_modes"] = active_modes
    if condition is not None:
        result_dict["condition"] = condition
    if btt_arrival_times:
        result_dict["btt_arrival_times"] = btt_arrival_times
        result_dict["btt_deflections"] = btt_deflections
        result_dict["btt_blade_indices"] = btt_blade_indices
    return result_dict


def _add_single_component(
    signals: np.ndarray,
    t: np.ndarray,
    phi_s: np.ndarray,
    amp: float,
    omega_rot: float,
    phase: float,
    k: int,
    w: int,
    omega_rad: float,
    theta_s: np.ndarray,
    is_stat: np.ndarray,
    envelope: np.ndarray | None,
    max_frequency: float,
    skip_sensors: np.ndarray | None = None,
) -> None:
    """Add one mode component (FW or BW) to the signal array in-place.

    For stationary sensors the lab-frame frequency is used with the
    circumferential phase factor.  For rotating sensors the rotating-
    frame frequency is used.

    Parameters
    ----------
    skip_sensors : (n_sensors,) bool, optional.  If provided, sensors
        where ``skip_sensors[s]`` is True are skipped (handled by the
        ray-based displacement path instead).
    """
    n_sensors = len(phi_s)

    # Stationary-frame frequency: f_stat = f_rot + w·k·Ω
    f_rot = omega_rot / (2.0 * np.pi)
    f_omega = omega_rad / (2.0 * np.pi)
    f_stat = abs(f_rot + w * k * f_omega)
    omega_stat = 2.0 * np.pi * f_stat

    if max_frequency > 0 and f_stat > max_frequency:
        return

    # Pre-compute trig for stationary and rotating frequencies
    cos_stat = np.cos(omega_stat * t + phase)
    sin_stat = np.sin(omega_stat * t + phase)
    cos_rot = np.cos(omega_rot * t + phase)
    sin_rot = np.sin(omega_rot * t + phase)

    for s in range(n_sensors):
        if skip_sensors is not None and skip_sensors[s]:
            continue

        phi_mag = np.abs(phi_s[s])
        phi_phase = np.angle(phi_s[s])
        if phi_mag < 1e-30:
            continue

        if is_stat[s]:
            # Circumferential phase: -w·k·θ_s
            circ_phase = -w * k * theta_s[s]
            total_phase = phi_phase + circ_phase
            contrib = amp * phi_mag * (
                np.cos(total_phase) * cos_stat - np.sin(total_phase) * sin_stat
            )
        else:
            # Rotating sensor: frequency = f_rot, circumferential phase = -w·k·θ_s
            circ_phase = -w * k * theta_s[s]
            total_phase = phi_phase + circ_phase
            contrib = amp * phi_mag * (
                np.cos(total_phase) * cos_rot - np.sin(total_phase) * sin_rot
            )

        if envelope is not None:
            contrib *= envelope

        signals[s, :] += contrib


def _add_mistuned_component(
    signals: np.ndarray,
    t: np.ndarray,
    phi_s: np.ndarray,
    amp: float,
    omega_rot: float,
    phase: float,
    k: int,
    w: int,
    omega_rad: float,
    theta_s: np.ndarray,
    is_stat: np.ndarray,
    envelope: np.ndarray | None,
    max_frequency: float,
    blade_amplitudes: np.ndarray,
    blade_phases: np.ndarray | None,
    num_sectors: int,
    sensors=None,
    skip_sensors: np.ndarray | None = None,
) -> None:
    """Add a mistuned mode component with per-blade amplitude/phase.

    For stationary sensors, determines which blade is passing at each
    time step and applies that blade's amplitude and phase offset.
    For rotating sensors (strain gauges), uses the blade the sensor is
    mounted on (``SensorLocation.blade_index``).

    Parameters
    ----------
    skip_sensors : (n_sensors,) bool, optional.  If provided, sensors
        where ``skip_sensors[s]`` is True are skipped (handled by the
        ray-based displacement path instead).
    """
    n_sensors = len(phi_s)
    N = num_sectors

    f_rot = omega_rot / (2.0 * np.pi)
    f_omega = omega_rad / (2.0 * np.pi)
    f_stat = abs(f_rot + w * k * f_omega)
    omega_stat = 2.0 * np.pi * f_stat

    if max_frequency > 0 and f_stat > max_frequency:
        return

    for s in range(n_sensors):
        if skip_sensors is not None and skip_sensors[s]:
            continue

        phi_mag = np.abs(phi_s[s])
        phi_phase = np.angle(phi_s[s])
        if phi_mag < 1e-30:
            continue

        circ_phase = -w * k * theta_s[s]
        total_phase_base = phi_phase + circ_phase

        if is_stat[s] and omega_rad > 0 and N > 0:
            # Stationary sensor: determine which blade is passing at each t
            blade_angle_at_sensor = (theta_s[s] - omega_rad * t) % (2.0 * np.pi)
            blade_idx = (blade_angle_at_sensor / (2.0 * np.pi) * N).astype(np.intp)
            blade_idx = blade_idx % N

            b_amp = blade_amplitudes[blade_idx]
            b_phase = blade_phases[blade_idx] if blade_phases is not None else 0.0

            total_phase = total_phase_base + b_phase + phase
            contrib = amp * phi_mag * b_amp * np.cos(omega_stat * t + total_phase)
        else:
            # Rotating sensor: mounted on a specific blade
            b_idx = 0
            if sensors is not None and hasattr(sensors[s], 'blade_index'):
                b_idx = sensors[s].blade_index % N if N > 0 else 0
            b_amp = blade_amplitudes[b_idx]
            b_phase = blade_phases[b_idx] if blade_phases is not None else 0.0
            total_phase = total_phase_base + b_phase + phase
            contrib = amp * phi_mag * b_amp * np.cos(omega_rot * t + total_phase)

        if envelope is not None:
            contrib *= envelope

        signals[s, :] += contrib


def _interpolate_mode_at_ray_hits(
    mode_shape: np.ndarray,
    ray_geom: RayHitGeometry,
    n_nodes_per_sector: int,
    sensor_direction: np.ndarray,
) -> np.ndarray:
    """Interpolate a mode shape at pre-computed ray hit points.

    Evaluates the displacement field at each angular bin's hit location
    using barycentric interpolation on the surface triangle, then projects
    onto the sensor measurement direction.

    The circumferential phase factor is NOT applied here — it is handled
    at runtime based on which sector is under the sensor at each time step.

    Parameters
    ----------
    mode_shape : (3 * n_nodes_per_sector,) complex — one mode vector.
    ray_geom : Pre-computed geometry for one sensor.
    n_nodes_per_sector : Number of nodes in the reference sector mesh.
    sensor_direction : (3,) unit vector — measurement direction.

    Returns
    -------
    (n_steps,) complex — projected displacement at each angular bin.
    Zero where no ray hit.
    """
    n_steps = len(ray_geom.hit_mask)
    result = np.zeros(n_steps, dtype=np.complex128)

    mask = ray_geom.hit_mask
    n_hits = np.count_nonzero(mask)
    if n_hits == 0:
        return result

    # Reshape mode shape from flat DOF vector to (n_nodes, 3) complex
    mode_3d = mode_shape.reshape(n_nodes_per_sector, 3)

    # Gather mode shape values at the three triangle vertices
    loc_ids = ray_geom.local_node_ids[mask]  # (n_hits, 3)
    phi_v = mode_3d[loc_ids]  # (n_hits, 3_vertices, 3_xyz) complex

    # Barycentric interpolation: weighted sum of vertex displacements
    bary = ray_geom.bary_coords[mask]  # (n_hits, 3)
    phi_interp = np.einsum('ij,ijk->ik', bary, phi_v)  # (n_hits, 3_xyz)

    # Project onto sensor direction
    dir_norm = sensor_direction / max(np.linalg.norm(sensor_direction), 1e-30)
    phi_proj = phi_interp @ dir_norm  # (n_hits,) complex

    result[mask] = phi_proj
    return result


def _add_ray_based_component(
    signals: np.ndarray,
    t: np.ndarray,
    mode_shape: np.ndarray,
    amp: float,
    omega_rot: float,
    phase: float,
    k: int,
    w: int,
    omega_rad: float,
    theta_s: np.ndarray,
    is_stat: np.ndarray,
    envelope: np.ndarray | None,
    max_frequency: float,
    ray_geometry: dict,
    n_nodes_per_sector: int,
    num_sectors: int,
    sensors: list,
    blade_amplitudes: np.ndarray | None = None,
    blade_phases: np.ndarray | None = None,
) -> None:
    """Add a mode component using ray-traced surface displacement.

    For each stationary sensor with pre-computed ray geometry, evaluates
    the actual mode shape at the ray-surface intersection point as each
    blade passes.  This captures the correct per-blade amplitude and
    phase variation for nodal diameter patterns.
    """
    N = num_sectors
    if N == 0:
        return
    sector_angle = 2.0 * np.pi / N

    f_rot = omega_rot / (2.0 * np.pi)
    f_omega = omega_rad / (2.0 * np.pi)
    f_stat = abs(f_rot + w * k * f_omega)
    omega_stat = 2.0 * np.pi * f_stat

    if max_frequency > 0 and f_stat > max_frequency:
        return

    for s in range(len(is_stat)):
        if not is_stat[s] or s not in ray_geometry:
            continue

        rg = ray_geometry[s]
        n_steps = len(rg.hit_mask)

        # Interpolate mode shape at each angular bin's hit point
        sensor_dir = np.asarray(sensors[s].direction, dtype=np.float64)
        phi_bins = _interpolate_mode_at_ray_hits(
            mode_shape, rg, n_nodes_per_sector, sensor_dir,
        )

        # Map each time sample to angular bin within one sector
        theta_local = (theta_s[s] - omega_rad * t) % sector_angle
        bin_idx = (theta_local / sector_angle * n_steps).astype(np.intp)
        np.clip(bin_idx, 0, n_steps - 1, out=bin_idx)

        # Determine which sector is under the sensor at each time step
        theta_full = (theta_s[s] - omega_rad * t) % (2.0 * np.pi)
        sector_at_t = (theta_full / sector_angle).astype(np.intp)
        sector_at_t = sector_at_t % N

        # Look up pre-interpolated displacement and apply circumferential phase
        phi_at_t = phi_bins[bin_idx]  # (n_samples,) complex
        circ_phase = np.exp(-1j * w * k * sector_at_t * sector_angle)
        phi_at_t = phi_at_t * circ_phase

        # Time-harmonic signal
        contrib = amp * np.real(
            phi_at_t * np.exp(1j * (omega_stat * t + phase))
        )

        # Per-blade mistuning
        if blade_amplitudes is not None:
            b_amp = blade_amplitudes[sector_at_t]
            contrib *= b_amp
        if blade_phases is not None:
            # Re-compute with the per-blade phase offset
            b_phase = blade_phases[sector_at_t]
            contrib = amp * np.real(
                phi_at_t * np.exp(1j * (omega_stat * t + phase + b_phase))
            )
            if blade_amplitudes is not None:
                contrib *= blade_amplitudes[sector_at_t]

        if envelope is not None:
            contrib *= envelope

        signals[s, :] += contrib


def _add_colored_broadband(
    signals: np.ndarray,
    peak_amplitude: float,
    model: ExcitationModel,
    sample_rate: float,
    rng: np.random.Generator,
) -> None:
    """Add colored broadband noise to all sensor channels in-place.

    Noise level is set by *model.broadband_snr_db* relative to
    *peak_amplitude*.  Spectral shape is ``PSD ~ f^(-gamma)`` with
    gamma = *model.broadband_spectral_exponent*.
    """
    from turbomodal.noise import generate_colored_noise

    n_sensors, n_samples = signals.shape
    noise_rms = peak_amplitude * 10.0 ** (-model.broadband_snr_db / 20.0)

    for s in range(n_sensors):
        noise = generate_colored_noise(
            n_samples, sample_rate, model.broadband_spectral_exponent, rng,
        )
        signals[s, :] += noise * noise_rms


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

    # Use the first condition's RPM for time vector sizing
    ref_rpm = abs(conditions[0].rpm) if len(conditions) > 0 and conditions[0].rpm != 0 else 3000.0
    t_ref = _build_time_vector(config, ref_rpm)
    n_samples = len(t_ref)

    all_signals = np.zeros((n_cond, n_sensors, n_samples))
    all_clean = np.zeros((n_cond, n_sensors, n_samples))
    t_start = time.perf_counter()

    for i in range(n_cond):
        fr = forced_response_results[i] if forced_response_results else None
        cond = conditions[i] if i < len(conditions) else None
        rpm = cond.rpm if cond is not None else 0.0

        result = generate_signals_for_condition(
            sensor_array, modal_results_per_condition[i],
            rpm, config, noise_config, fr, condition=cond,
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

    return {
        "signals": all_signals,
        "clean_signals": all_clean,
        "conditions": conditions,
        "sample_rate": config.sample_rate,
        "time": t_ref,
    }
