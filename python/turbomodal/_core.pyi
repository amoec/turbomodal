"""Type stubs for the turbomodal._core C++ extension module."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

# --- Material ---

class Material:
    E: float
    nu: float
    rho: float
    T_ref: float
    E_slope: float

    def __init__(self, E: float, nu: float, rho: float) -> None: ...
    def __init__(  # type: ignore[misc]
        self, E: float, nu: float, rho: float, T_ref: float, E_slope: float
    ) -> None: ...
    def at_temperature(self, T: float) -> Material: ...
    def validate(self) -> None: ...

# --- NodeSet ---

class NodeSet:
    name: str
    node_ids: list[int]

    def __init__(self) -> None: ...

# --- BCType ---

class BCType:
    FIXED: BCType
    DISPLACEMENT: BCType
    FRICTIONLESS: BCType
    ELASTIC_SUPPORT: BCType
    CYLINDRICAL: BCType

# --- ConstraintGroup ---

class ConstraintGroup:
    name: str
    node_ids: list[int]
    type: BCType
    constrained_components: list[bool]
    surface_normal: npt.NDArray[np.float64]
    spring_stiffness: npt.NDArray[np.float64]
    cylinder_axis: npt.NDArray[np.float64]
    cylinder_origin: npt.NDArray[np.float64]

    def __init__(self) -> None: ...

# --- Mesh ---

class Mesh:
    nodes: npt.NDArray[np.float64]
    elements: npt.NDArray[np.int32]
    node_sets: list[NodeSet]
    left_boundary: NodeSet
    right_boundary: NodeSet
    matched_pairs: list[tuple[int, int]]
    free_boundary: NodeSet
    num_sectors: int
    rotation_axis: int

    def __init__(self) -> None: ...
    def load_from_gmsh(self, filename: str) -> None: ...
    def load_from_arrays(
        self,
        node_coords: npt.NDArray[np.float64],
        element_connectivity: npt.NDArray[np.int32],
        node_sets: list[NodeSet],
        num_sectors: int,
        rotation_axis: int = 2,
    ) -> None: ...
    def identify_cyclic_boundaries(self, tolerance: float = 1e-6) -> None: ...
    def match_boundary_nodes(self) -> None: ...
    def find_node_set(self, name: str) -> NodeSet: ...
    def select_nodes_by_plane(
        self,
        plane_point: npt.NDArray[np.float64],
        plane_normal: npt.NDArray[np.float64],
        tolerance: float = 1e-6,
    ) -> list[int]: ...
    def compute_surface_normal(
        self, node_ids: list[int]
    ) -> npt.NDArray[np.float64]: ...
    def num_nodes(self) -> int: ...
    def num_elements(self) -> int: ...
    def num_dof(self) -> int: ...

# --- GlobalAssembler ---

class GlobalAssembler:
    @property
    def K(self) -> object: ...  # scipy.sparse.csc_matrix
    @property
    def M(self) -> object: ...  # scipy.sparse.csc_matrix
    @property
    def K_sigma(self) -> object: ...  # scipy.sparse.csc_matrix
    @property
    def G(self) -> object: ...  # scipy.sparse.csc_matrix
    @property
    def K_omega(self) -> object: ...  # scipy.sparse.csc_matrix

    def __init__(self) -> None: ...
    def assemble(self, mesh: Mesh, material: Material) -> None: ...
    def assemble_stress_stiffening(
        self,
        mesh: Mesh,
        material: Material,
        displacement: npt.NDArray[np.float64],
        omega: float,
    ) -> None: ...
    def assemble_rotating_effects(
        self, mesh: Mesh, material: Material, omega: float
    ) -> None: ...
    def assemble_centrifugal_load(
        self, mesh: Mesh, material: Material, omega: float, axis: int
    ) -> npt.NDArray[np.float64]: ...

# --- SolverConfig ---

class SolverConfig:
    nev: int
    ncv: int
    shift: float
    tolerance: float
    max_iterations: int

    def __init__(self) -> None: ...

# --- SolverStatus ---

class SolverStatus:
    converged: bool
    num_converged: int
    iterations: int
    max_residual: float
    message: str

    def __init__(self) -> None: ...

# --- SolverProgress (rich callback data from solve_at_rpm) ---

class SolverProgress:
    completed: int
    total: int
    harmonic_k: int
    converged: bool
    iterations: int
    num_converged: int
    num_modes: int
    max_residual: float
    min_freq_hz: float
    max_freq_hz: float
    elapsed_s: float

    def __init__(self) -> None: ...

# --- ModalResult ---

class ModalResult:
    harmonic_index: int
    rpm: float
    frequencies: npt.NDArray[np.float64]
    mode_shapes: npt.NDArray[np.complex128]
    whirl_direction: npt.NDArray[np.int32]
    converged: bool

    def __init__(self) -> None: ...
    def wave_propagation_velocity(
        self, radius: float
    ) -> npt.NDArray[np.float64]: ...

# --- StationaryFrameResult ---

class StationaryFrameResult:
    frequencies: npt.NDArray[np.float64]
    whirl_direction: npt.NDArray[np.int32]

    def __init__(self) -> None: ...

# --- FluidType enum ---

class FluidType:
    NONE: FluidType
    GAS_AIC: FluidType
    LIQUID_ANALYTICAL: FluidType
    KWAK_ANALYTICAL: FluidType
    POTENTIAL_FLOW_BEM: FluidType
    LIQUID_ACOUSTIC_FEM: FluidType

# --- FluidConfig ---

class FluidConfig:
    type: FluidType
    fluid_density: float
    disk_radius: float
    disk_thickness: float
    speed_of_sound: float

    def __init__(self) -> None: ...

# --- ParametricCondition ---

class ParametricCondition:
    rpm: float
    temperature: float
    mistuning: list[float]

    def __init__(
        self,
        rpm: float,
        temperature: float = 293.15,
        mistuning: list[float] | None = None,
    ) -> None: ...

# --- CyclicSymmetrySolver ---

class CyclicSymmetrySolver:
    def __init__(
        self,
        mesh: Mesh,
        material: Material,
        fluid: FluidConfig = ...,
        apply_hub_constraint: bool = True,
    ) -> None: ...
    def __init__(  # type: ignore[misc]
        self,
        mesh: Mesh,
        material: Material,
        constraints: list[ConstraintGroup],
        fluid: FluidConfig = ...,
    ) -> None: ...
    def solve_at_rpm(
        self,
        rpm: float,
        num_modes_per_harmonic: int,
        harmonic_indices: list[int] = ...,
        max_threads: int = 0,
        include_coriolis: bool = False,
        min_frequency: float = 0.0,
        progress_cb: object = None,  # Callable[[SolverProgress], None]
        allow_condensation: bool = False,
        memory_reserve_fraction: float = 0.2,
    ) -> list[ModalResult]: ...
    def solve_rpm_sweep(
        self,
        rpm_values: npt.NDArray[np.float64],
        num_modes_per_harmonic: int,
        harmonic_indices: list[int] = ...,
        max_threads: int = 0,
        include_coriolis: bool = False,
        min_frequency: float = 0.0,
    ) -> list[list[ModalResult]]: ...
    def solve_parametric(
        self,
        conditions: list[ParametricCondition],
        num_modes_per_harmonic: int,
        harmonic_indices: list[int] = ...,
        max_threads: int = 0,
        include_coriolis: bool = False,
        min_frequency: float = 0.0,
        allow_condensation: bool = False,
        memory_reserve_fraction: float = 0.2,
        progress_cb: object = None,  # Callable[[SolverProgress], None]
    ) -> list[list[list[ModalResult]]]: ...
    @staticmethod
    def compute_stationary_frame(
        rotating_result: ModalResult, num_sectors: int
    ) -> StationaryFrameResult: ...
    def export_campbell_csv(
        self, filename: str, results: list[list[ModalResult]]
    ) -> None: ...
    def export_zzenf_csv(
        self, filename: str, results: list[ModalResult]
    ) -> None: ...
    def export_mode_shape_vtk(
        self, filename: str, result: ModalResult, mode_index: int
    ) -> None: ...

# --- AddedMassModel ---

class AddedMassModel:
    @staticmethod
    def kwak_avmi(
        nodal_diameter: int,
        rho_fluid: float,
        rho_structure: float,
        thickness: float,
        radius: float,
    ) -> float: ...
    @staticmethod
    def frequency_ratio(
        nodal_diameter: int,
        rho_fluid: float,
        rho_structure: float,
        thickness: float,
        radius: float,
    ) -> float: ...

# --- ModalSolver ---

class ModalSolver:
    def __init__(self) -> None: ...

# --- DampingType enum ---

class DampingType:
    NONE: DampingType
    MODAL: DampingType
    RAYLEIGH: DampingType

# --- DampingConfig ---

class DampingConfig:
    type: DampingType
    modal_damping_ratios: npt.NDArray[np.float64]
    rayleigh_alpha: float
    rayleigh_beta: float
    aero_damping_ratios: npt.NDArray[np.float64]

    def __init__(self) -> None: ...
    def effective_damping(self, mode_index: int, omega_r: float) -> float: ...

# --- ExcitationType enum ---

class ExcitationType:
    UNIFORM_PRESSURE: ExcitationType
    POINT_FORCE: ExcitationType
    SPATIAL_DISTRIBUTION: ExcitationType

# --- ForcedResponseConfig ---

class ForcedResponseConfig:
    engine_order: int
    force_amplitude: float
    excitation_type: ExcitationType
    force_node_id: int
    force_direction: npt.NDArray[np.float64]
    force_vector: npt.NDArray[np.float64]
    freq_min: float
    freq_max: float
    num_freq_points: int

    def __init__(self) -> None: ...

# --- ForcedResponseResult ---

class ForcedResponseResult:
    engine_order: int
    rpm: float
    natural_frequencies: npt.NDArray[np.float64]
    modal_forces: npt.NDArray[np.complex128]
    modal_damping_ratios: npt.NDArray[np.float64]
    participation_factors: npt.NDArray[np.float64]
    effective_modal_mass: npt.NDArray[np.float64]
    sweep_frequencies: npt.NDArray[np.float64]
    modal_amplitudes: npt.NDArray[np.complex128]
    max_response_amplitude: float
    resonance_frequencies: npt.NDArray[np.float64]

    def __init__(self) -> None: ...

# --- ForcedResponseSolver ---

class ForcedResponseSolver:
    def __init__(
        self, mesh: Mesh, damping: DampingConfig = ...
    ) -> None: ...
    def solve(
        self,
        modal_results: list[ModalResult],
        rpm: float,
        config: ForcedResponseConfig,
    ) -> ForcedResponseResult: ...
    @staticmethod
    def compute_modal_forces(
        mode_shapes: npt.NDArray[np.complex128],
        force_vector: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]: ...
    @staticmethod
    def modal_frf(
        omega: float, omega_r: float, Q_r: float, zeta_r: float
    ) -> complex: ...
    @staticmethod
    def compute_participation_factors(
        mode_shapes: npt.NDArray[np.complex128],
        M: object,
        direction: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def compute_effective_modal_mass(
        mode_shapes: npt.NDArray[np.complex128],
        M: object,
        direction: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def build_eo_excitation(
        self,
        engine_order: int,
        amplitude: float,
        type: ExcitationType = ...,
        force_node_id: int = -1,
        force_dir: npt.NDArray[np.float64] = ...,
    ) -> npt.NDArray[np.complex128]: ...

# --- MistuningConfig ---

class MistuningConfig:
    blade_frequency_deviations: npt.NDArray[np.float64]
    tuned_frequencies: npt.NDArray[np.float64]
    mode_family_index: int

    def __init__(self) -> None: ...

# --- MistuningResult ---

class MistuningResult:
    frequencies: npt.NDArray[np.float64]
    blade_amplitudes: npt.NDArray[np.complex128]
    amplitude_magnification: npt.NDArray[np.float64]
    peak_magnification: float
    localization_ipr: float

    def __init__(self) -> None: ...

# --- FMMSolver ---

class FMMSolver:
    @staticmethod
    def solve(
        num_sectors: int,
        tuned_frequencies: npt.NDArray[np.float64],
        blade_frequency_deviations: npt.NDArray[np.float64],
    ) -> MistuningResult: ...
    @staticmethod
    def random_mistuning(
        num_sectors: int, sigma: float, seed: int = 42
    ) -> npt.NDArray[np.float64]: ...

# --- ModeIdentification ---

class ModeIdentification:
    nodal_diameter: int
    nodal_circle: int
    whirl_direction: int
    frequency: float
    wave_velocity: float
    participation_factor: float
    family_label: str

    def __init__(self) -> None: ...

# --- GroundTruthLabel ---

class GroundTruthLabel:
    rpm: float
    temperature: float
    nodal_diameter: int
    nodal_circle: int
    whirl_direction: int
    frequency: float
    wave_velocity: float
    family_label: str
    participation_factor: float
    effective_modal_mass: float
    amplitude: float
    damping_ratio: float
    amplitude_magnification: float
    is_localized: bool
    condition_id: int
    mode_index: int

    def __init__(self) -> None: ...

# --- Free functions ---

def identify_nodal_circles(
    mode_shape: npt.NDArray[np.complex128], mesh: Mesh
) -> int: ...

def classify_mode_family(
    mode_shape: npt.NDArray[np.complex128], mesh: Mesh
) -> str: ...

def identify_modes(
    result: ModalResult,
    mesh: Mesh,
    characteristic_radius: float = 0.0,
) -> list[ModeIdentification]: ...
