"""Virtual sensor array for sampling FEA mode shapes and synthesizing time signals.

Subsystem B: provides configurable sensor placement (BTT probes, strain gauges,
casing accelerometers), interpolation from the FE mesh, and time-domain signal
synthesis with optional noise injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from turbomodal.noise import NoiseConfig, apply_noise


class SensorType(Enum):
    """Types of virtual sensor."""

    STRAIN_GAUGE = "strain_gauge"
    BTT_PROBE = "btt_probe"
    CASING_ACCELEROMETER = "casing_accelerometer"
    DISPLACEMENT = "displacement"


@dataclass
class SensorLocation:
    """Placement and characteristics of a single virtual sensor.

    Parameters
    ----------
    sensor_type : Kind of sensor.
    position : (3,) world-space coordinates of the sensor.
    direction : (3,) unit vector defining the measurement direction
        (displacement is projected onto this axis).
    label : Human-readable identifier (e.g. ``"BTT_probe_03"``).
    bandwidth : Sensor bandwidth in Hz.
    sensitivity : Sensor sensitivity (output per unit displacement).
    noise_floor : Minimum detectable signal level.
    """

    sensor_type: SensorType
    position: NDArray
    direction: NDArray
    label: str = ""
    bandwidth: float = 50_000.0
    sensitivity: float = 1.0
    noise_floor: float = 1e-9
    is_stationary: bool | None = None  # None = infer from sensor_type
    blade_index: int = 0  # Which blade the sensor is mounted on (rotating sensors)


@dataclass
class SensorArrayConfig:
    """Configuration for an array of virtual sensors.

    Parameters
    ----------
    sensors : Ordered list of sensor locations.
    sample_rate : Sampling rate in Hz.
    duration : Total acquisition duration in seconds.
    """

    sensors: list[SensorLocation] = field(default_factory=list)
    sample_rate: float = 100_000.0
    duration: float = 1.0

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @staticmethod
    def default_btt_array(
        num_probes: int,
        casing_radius: float,
        axial_positions: Sequence[float] | NDArray,
        sample_rate: float = 500_000.0,
        duration: float = 1.0,
    ) -> SensorArrayConfig:
        """Create a BTT probe array with probes evenly spaced circumferentially.

        At each axial position in *axial_positions* a ring of *num_probes*
        probes is placed at angular increments of ``2*pi / num_probes``.
        Each probe measures displacement in the radial direction.

        Parameters
        ----------
        num_probes : Number of probes per axial station.
        casing_radius : Radial distance from the rotation axis to the
            probe tip.
        axial_positions : Axial (z) coordinate of each measurement ring.
        sample_rate : Acquisition sampling rate in Hz.
        duration : Acquisition duration in seconds.

        Returns
        -------
        Populated :class:`SensorArrayConfig`.
        """
        axial_positions = np.asarray(axial_positions, dtype=np.float64)
        sensors: list[SensorLocation] = []

        for ax_idx, z in enumerate(axial_positions):
            for p in range(num_probes):
                theta = 2.0 * np.pi * p / num_probes
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                position = np.array([
                    casing_radius * cos_t,
                    casing_radius * sin_t,
                    z,
                ])
                # Radial direction (pointing inward toward the axis)
                direction = np.array([-cos_t, -sin_t, 0.0])

                sensors.append(SensorLocation(
                    sensor_type=SensorType.BTT_PROBE,
                    position=position,
                    direction=direction,
                    label=f"BTT_ax{ax_idx}_p{p:02d}",
                    bandwidth=500_000.0,
                    sensitivity=1.0,
                    noise_floor=1e-6,
                ))

        return SensorArrayConfig(
            sensors=sensors,
            sample_rate=sample_rate,
            duration=duration,
        )

    @staticmethod
    def default_strain_gauge_array(
        num_gauges: int,
        radial_positions: Sequence[float] | NDArray,
        sample_rate: float = 50_000.0,
        duration: float = 1.0,
    ) -> SensorArrayConfig:
        """Create a strain-gauge array at specified radial positions.

        Gauges are placed on the z=0 plane.  At each radial position,
        *num_gauges* gauges are distributed at equal angular increments.
        The measurement direction is tangential (perpendicular to the
        radius in the xy-plane).

        Parameters
        ----------
        num_gauges : Number of gauges per radial ring.
        radial_positions : Radial distance of each gauge ring.
        sample_rate : Acquisition sampling rate in Hz.
        duration : Acquisition duration in seconds.

        Returns
        -------
        Populated :class:`SensorArrayConfig`.
        """
        radial_positions = np.asarray(radial_positions, dtype=np.float64)
        sensors: list[SensorLocation] = []

        for r_idx, r in enumerate(radial_positions):
            for g in range(num_gauges):
                theta = 2.0 * np.pi * g / num_gauges
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                position = np.array([r * cos_t, r * sin_t, 0.0])
                # Tangential direction (perpendicular to radial, in-plane)
                direction = np.array([-sin_t, cos_t, 0.0])

                sensors.append(SensorLocation(
                    sensor_type=SensorType.STRAIN_GAUGE,
                    position=position,
                    direction=direction,
                    label=f"SG_r{r_idx}_g{g:02d}",
                    bandwidth=50_000.0,
                    sensitivity=1.0,
                    noise_floor=1e-9,
                ))

        return SensorArrayConfig(
            sensors=sensors,
            sample_rate=sample_rate,
            duration=duration,
        )

    @staticmethod
    def default_displacement_array(
        positions: Sequence[Sequence[float]] | NDArray,
        direction: str | Sequence[float] | NDArray = "axial",
        sample_rate: float = 50_000.0,
        duration: float = 1.0,
        bandwidth: float = 10_000.0,
        sensitivity: float = 1.0,
        noise_floor: float = 1e-7,
    ) -> SensorArrayConfig:
        """Create a displacement sensor array at specified locations.

        Models non-contact displacement probes (e.g. eddy-current,
        capacitive, or laser vibrometer) that directly measure
        displacement at given points on the structure.

        Parameters
        ----------
        positions : (N, 3) array-like of sensor positions in world
            coordinates.
        direction : Measurement direction.  One of:

            * ``"axial"`` — z-direction ``[0, 0, 1]`` (default, typical
              for disk-face probes).
            * ``"radial"`` — outward radial from the rotation axis,
              computed per sensor from its (x, y) position.
            * ``(3,)`` array-like — a single direction vector shared
              by all sensors (will be normalised internally).

        sample_rate : Acquisition sampling rate in Hz.
        duration : Acquisition duration in seconds.
        bandwidth : Sensor bandwidth in Hz.
        sensitivity : Output per unit displacement.
        noise_floor : Minimum detectable displacement in metres.

        Returns
        -------
        Populated :class:`SensorArrayConfig`.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.shape[1] != 3:
            raise ValueError(
                f"positions must have 3 columns (x, y, z), got {positions.shape[1]}"
            )

        sensors: list[SensorLocation] = []

        for idx, pos in enumerate(positions):
            if isinstance(direction, str):
                if direction == "axial":
                    d = np.array([0.0, 0.0, 1.0])
                elif direction == "radial":
                    r_xy = np.linalg.norm(pos[:2])
                    if r_xy < 1e-12:
                        # On-axis: fall back to x-direction
                        d = np.array([1.0, 0.0, 0.0])
                    else:
                        d = np.array([pos[0] / r_xy, pos[1] / r_xy, 0.0])
                else:
                    raise ValueError(
                        f"direction must be 'axial', 'radial', or a "
                        f"(3,) vector, got '{direction}'"
                    )
            else:
                d = np.asarray(direction, dtype=np.float64).ravel()
                if len(d) != 3:
                    raise ValueError(
                        f"direction vector must have 3 components, got {len(d)}"
                    )

            sensors.append(SensorLocation(
                sensor_type=SensorType.DISPLACEMENT,
                position=pos,
                direction=d,
                label=f"DISP_{idx:03d}",
                bandwidth=bandwidth,
                sensitivity=sensitivity,
                noise_floor=noise_floor,
            ))

        return SensorArrayConfig(
            sensors=sensors,
            sample_rate=sample_rate,
            duration=duration,
        )


class VirtualSensorArray:
    """Sample FE mode shapes at discrete sensor locations and synthesize
    time-domain signals.

    Parameters
    ----------
    mesh : turbomodal ``Mesh`` object (must expose ``.nodes`` and
        ``.num_nodes()``).
    config : :class:`SensorArrayConfig` describing the sensor layout.
    """

    def __init__(self, mesh, config: SensorArrayConfig) -> None:
        self._mesh = mesh
        self._config = config

        # Lazily built sparse interpolation matrix
        self._interp_matrix: sparse.csr_matrix | None = None

        # Lazily built ray-tracing cache (geometry-only, RPM-independent)
        self._annulus_surface = None       # PyVista PolyData
        self._ray_hit_cache: dict | None = None  # {sensor_idx: bool_array}
        self._ray_geometry_cache: dict | None = None  # {sensor_idx: RayHitGeometry}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SensorArrayConfig:
        """Return the sensor array configuration."""
        return self._config

    @property
    def mesh(self):
        """Return the underlying mesh object."""
        return self._mesh

    @property
    def n_sensors(self) -> int:
        """Number of sensors in the array."""
        return len(self._config.sensors)

    @property
    def interpolation_matrix(self) -> sparse.csr_matrix:
        """Sparse (n_sensors, n_dof) interpolation matrix, built on first access."""
        if self._interp_matrix is None:
            self._interp_matrix = self.build_interpolation_matrix()
        return self._interp_matrix

    def sensor_circumferential_angles(self) -> NDArray:
        """Compute circumferential angle (radians) of each sensor.

        The angle is measured in the plane perpendicular to the mesh
        rotation axis.  For ``rotation_axis=2`` (Z) this is
        ``atan2(y, x)``.

        Returns
        -------
        ``(n_sensors,)`` float64 array of angles in ``[0, 2*pi)``.
        """
        axis = self._mesh.rotation_axis
        c1 = (axis + 1) % 3
        c2 = (axis + 2) % 3
        angles = np.empty(self.n_sensors, dtype=np.float64)
        for i, sensor in enumerate(self._config.sensors):
            pos = np.asarray(sensor.position, dtype=np.float64)
            angles[i] = np.arctan2(pos[c2], pos[c1]) % (2.0 * np.pi)
        return angles

    def blade_tip_profile(self, radius_tol: float = 0.02) -> tuple[float, float]:
        """Determine the angular extent of the blade tip from mesh geometry.

        Finds nodes at the outer radius (within *radius_tol* fraction of
        the maximum radius) and returns the angular span of those nodes
        within one sector ``[0, 2*pi/N)``.

        Parameters
        ----------
        radius_tol : Fractional tolerance for identifying tip nodes.
            A node is considered a tip node if its radius is within
            ``radius_tol * max_radius`` of the maximum.

        Returns
        -------
        ``(blade_start_angle, blade_end_angle)`` in radians within
        the sector.  The gap between sectors spans
        ``[blade_end_angle, 2*pi/N)``.
        """
        axis = self._mesh.rotation_axis
        c1 = (axis + 1) % 3
        c2 = (axis + 2) % 3
        nodes = np.asarray(self._mesh.nodes)
        N = self._mesh.num_sectors
        sector_angle = 2.0 * np.pi / N

        # Compute radial distances in the rotation plane
        radii = np.sqrt(nodes[:, c1] ** 2 + nodes[:, c2] ** 2)
        max_r = radii.max()
        tip_mask = radii >= max_r * (1.0 - radius_tol)

        if not np.any(tip_mask):
            # Fallback: full sector
            return (0.0, sector_angle)

        # Angular positions of tip nodes within the sector
        tip_angles = np.arctan2(nodes[tip_mask, c2], nodes[tip_mask, c1])
        tip_angles = tip_angles % sector_angle  # fold into [0, sector_angle)

        return (float(tip_angles.min()), float(tip_angles.max()))

    # ------------------------------------------------------------------
    # Ray-tracing cache
    # ------------------------------------------------------------------

    @property
    def annulus_surface(self):
        """Full-annulus surface mesh (PyVista PolyData), built on first access.

        The surface depends only on mesh geometry and is reused across
        all calls to :func:`~turbomodal.signal_gen.generate_signals_for_condition`.
        """
        if self._annulus_surface is None:
            from turbomodal.signal_gen import _build_annulus_surface
            self._annulus_surface = _build_annulus_surface(self._mesh)
        return self._annulus_surface

    def ray_hit_geometry(self, n_steps: int = 256) -> dict:
        """Pre-computed ray-surface intersection geometry, built on first access.

        Returns ``{sensor_idx: RayHitGeometry}`` with full intersection
        data (hit points, triangle vertices, barycentric coords, sector
        IDs) for each stationary sensor.  Cached because it depends only
        on mesh geometry and sensor placement.
        """
        if self._ray_geometry_cache is None:
            from turbomodal.signal_gen import (
                _precompute_ray_geometry, _sensor_is_stationary,
            )
            sensors = self._config.sensors
            is_stat = np.array([_sensor_is_stationary(s) for s in sensors])
            sector_angle = 2.0 * np.pi / self._mesh.num_sectors
            self._ray_geometry_cache = _precompute_ray_geometry(
                self.annulus_surface, sensors, is_stat, self._mesh,
                sector_angle, n_steps=n_steps,
            )
        return self._ray_geometry_cache

    def ray_hit_pattern(self, n_steps: int = 256) -> dict[int, np.ndarray]:
        """Pre-computed ray hit masks for stationary sensors, built on first access.

        Returns ``{sensor_idx: bool_array(n_steps)}`` indicating where
        a ray from each stationary sensor hits the blade surface during
        one sector of rotation.  Delegates to :meth:`ray_hit_geometry`
        and extracts the boolean mask.
        """
        if self._ray_hit_cache is None:
            geom = self.ray_hit_geometry(n_steps=n_steps)
            self._ray_hit_cache = {
                s_idx: rg.hit_mask for s_idx, rg in geom.items()
            }
        return self._ray_hit_cache

    def invalidate_ray_cache(self) -> None:
        """Clear the cached annulus surface and ray hit patterns.

        Call this if the mesh or sensor positions have changed.
        """
        self._annulus_surface = None
        self._ray_hit_cache = None
        self._ray_geometry_cache = None

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def build_interpolation_matrix(self) -> sparse.csr_matrix:
        """Build a sparse interpolation matrix of shape ``(n_sensors, n_dof)``.

        For each sensor the nearest mesh node is located.  The
        corresponding row in the matrix projects the three displacement
        DOFs at that node onto the sensor measurement direction.

        Returns
        -------
        Sparse CSR matrix with shape ``(n_sensors, 3 * n_nodes)``.
        """
        from scipy.spatial import cKDTree

        nodes = np.asarray(self._mesh.nodes)  # (n_nodes, 3)
        n_nodes = self._mesh.num_nodes()
        n_dof = 3 * n_nodes
        n_sensors = self.n_sensors

        tree = cKDTree(nodes)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for s_idx, sensor in enumerate(self._config.sensors):
            pos = np.asarray(sensor.position, dtype=np.float64)
            direction = np.asarray(sensor.direction, dtype=np.float64)
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-30:  # effectively zero; guard against degenerate direction vectors
                raise ValueError(
                    f"Sensor {s_idx}: zero-norm direction vector {sensor.direction}"
                )
            direction = direction / dir_norm

            # Find nearest node via KD-tree (O(log N) per query)
            _, nearest = tree.query(pos)
            nearest = int(nearest)

            # Project displacement at *nearest* onto sensor direction.
            # DOFs for node *nearest* are at indices [3*nearest, 3*nearest+1, 3*nearest+2].
            base_dof = 3 * nearest
            for ax in range(3):
                if abs(direction[ax]) > 1e-15:  # skip negligible components for sparse interpolation
                    rows.append(s_idx)
                    cols.append(base_dof + ax)
                    vals.append(float(direction[ax] * sensor.sensitivity))

        interp = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_sensors, n_dof),
        )
        self._interp_matrix = interp
        return interp

    # ------------------------------------------------------------------
    # Mode shape sampling
    # ------------------------------------------------------------------

    def sample_mode_shape(self, mode_shape: NDArray) -> NDArray:
        """Sample a mode shape at the sensor locations.

        Parameters
        ----------
        mode_shape : Complex mode-shape vector of length ``n_dof``
            (i.e. ``3 * n_nodes``), as stored column-wise in
            ``ModalResult.mode_shapes``.

        Returns
        -------
        Complex array of shape ``(n_sensors,)`` with the projected
        reading at each sensor.
        """
        H = self.interpolation_matrix  # (n_sensors, n_dof), real
        phi = np.asarray(mode_shape).ravel()

        if np.iscomplexobj(phi):
            # Sparse @ complex: split into real/imag to guarantee
            # compatibility across scipy versions.
            return H @ phi.real + 1j * (H @ phi.imag)
        return H @ phi

    # ------------------------------------------------------------------
    # Time signal synthesis
    # ------------------------------------------------------------------

    def generate_time_signal(
        self,
        modal_results: Sequence,
        rpm: float,
        amplitudes: NDArray | None = None,
        noise_config: NoiseConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray:
        """Synthesize a multi-channel time-domain signal from modal results.

        The signal at the sensors is:

        .. math::

            x_s(t) = \\sum_r A_r \\, \\phi^s_r \\, \\cos(\\omega_r t + \\varphi_r)

        where the sum runs over all modes across all harmonic indices in
        *modal_results*, :math:`\\phi^s_r` is the (complex) sensor
        reading for mode *r*, and :math:`\\omega_r = 2\\pi f_r`.

        Parameters
        ----------
        modal_results : Sequence of ``ModalResult`` objects (one per
            harmonic index), each providing ``.frequencies``,
            ``.mode_shapes``, and ``.harmonic_index``.
        rpm : Rotational speed in RPM (used only for metadata /
            downstream compatibility; the modal frequencies already
            account for spin-softening).
        amplitudes : Optional 1-D array whose length equals the total
            number of modes across all harmonic indices.  If ``None``,
            unit amplitude is assumed for every mode.
        noise_config : Optional :class:`NoiseConfig`.  If provided, noise
            is applied to the synthesized signal via
            :func:`turbomodal.noise.apply_noise`.
        rng : Optional NumPy random generator for reproducibility.

        Returns
        -------
        Real array of shape ``(n_sensors, n_samples)``.
        """
        sr = self._config.sample_rate
        dur = self._config.duration
        n_samples = int(sr * dur)
        n_sensors = self.n_sensors
        t = np.arange(n_samples) / sr  # (n_samples,)

        if rng is None:
            rng = np.random.default_rng()

        # Collect all (frequency, sensor_reading, amplitude) triples
        mode_list: list[tuple[float, NDArray, float]] = []
        for result in modal_results:
            freqs = np.asarray(result.frequencies)
            shapes = np.asarray(result.mode_shapes)  # (n_dof, n_modes)
            n_modes = len(freqs)
            for m in range(n_modes):
                phi_s = self.sample_mode_shape(shapes[:, m])  # (n_sensors,) complex
                mode_list.append((float(freqs[m]), phi_s, 1.0))

        # Assign amplitudes
        if amplitudes is not None:
            amplitudes = np.asarray(amplitudes, dtype=np.float64).ravel()
            if len(amplitudes) != len(mode_list):
                raise ValueError(
                    f"Length of amplitudes ({len(amplitudes)}) does not match "
                    f"total number of modes ({len(mode_list)})."
                )
            mode_list = [
                (freq, phi_s, float(amp))
                for (freq, phi_s, _), amp in zip(mode_list, amplitudes)
            ]

        # Synthesise: x_s(t) = sum_r A_r * Re(phi_s_r) * cos(omega_r*t)
        #                     - A_r * Im(phi_s_r) * sin(omega_r*t)
        # which is equivalent to A_r * Re(phi_s_r * exp(j*omega_r*t))
        signal = np.zeros((n_sensors, n_samples), dtype=np.float64)

        for freq, phi_s, amp in mode_list:
            if amp == 0.0:
                continue
            omega = 2.0 * np.pi * freq
            cos_wt = np.cos(omega * t)  # (n_samples,)
            sin_wt = np.sin(omega * t)  # (n_samples,)
            phi_real = np.real(phi_s)   # (n_sensors,)
            phi_imag = np.imag(phi_s)   # (n_sensors,)

            # outer products: (n_sensors,1) * (1,n_samples)
            signal += amp * (
                phi_real[:, np.newaxis] * cos_wt[np.newaxis, :]
                - phi_imag[:, np.newaxis] * sin_wt[np.newaxis, :]
            )

        # Apply noise if configured
        if noise_config is not None:
            signal = apply_noise(signal, noise_config, sr, rng=rng)

        return signal
