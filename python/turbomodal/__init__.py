"""Turbomodal: Cyclic symmetry FEA solver for turbomachinery modal analysis."""

from turbomodal._core import (
    Material,
    NodeSet,
    BCType,
    ConstraintGroup,
    Mesh,
    GlobalAssembler,
    SolverConfig,
    SolverStatus,
    ModalResult,
    FluidConfig,
    FluidType,
    CyclicSymmetrySolver,
    AddedMassModel,
    ModalSolver,
    # Damping, forced response, mistuning, and mode identification
    DampingConfig,
    DampingType,
    ForcedResponseConfig,
    ForcedResponseResult,
    ForcedResponseSolver,
    ExcitationType,
    MistuningConfig,
    MistuningResult,
    FMMSolver,
    ModeIdentification,
    GroundTruthLabel,
    identify_nodal_circles,
    classify_mode_family,
    identify_modes,
)

from turbomodal.io import load_mesh, load_cad, inspect_cad, CadInfo
from turbomodal.solver import solve, rpm_sweep, campbell_data, BoundaryCondition
from turbomodal.viz import (
    DiagramStyle,
    plot_mesh,
    plot_mode,
    plot_full_mesh,
    _removed_func,
    plot_cad,
    plot_campbell,
    plot_zzenf,
    diagnose_frequencies,
    format_condition_label,
    interactive_plane_selector,
    bc_editor,
    plot_boundary_conditions,
    _removed_func,
    _removed_func,
)

# Subsystem B: Signal generation pipeline
    _RemovedEnum,
    _RemovedClass,
    _RemovedClass,
    _RemovedClass,
)
    _RemovedClass,
    _removed,
    _removed,
    filter_modal_results,
)

# Subsystem A extensions: dataset & parametric sweep
    _RemovedClass,
    _RemovedClass,
    export_modal_results,
    load_modal_results,
)
    _RemovedClass,
    _RemovedClass,
    generate_conditions,
    _removed,
)

try:
    from turbomodal._version import version as __version__
except ImportError:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("turbomodal")

__all__ = [
    # Core C++ classes
    "Material",
    "NodeSet",
    "BCType",
    "ConstraintGroup",
    "BoundaryCondition",
    "Mesh",
    "GlobalAssembler",
    "SolverConfig",
    "SolverStatus",
    "ModalResult",
    "FluidConfig",
    "FluidType",
    "CyclicSymmetrySolver",
    "AddedMassModel",
    "ModalSolver",
    # Damping, forced response, mistuning, and mode identification
    "DampingConfig",
    "DampingType",
    "ForcedResponseConfig",
    "ForcedResponseResult",
    "ForcedResponseSolver",
    "ExcitationType",
    "MistuningConfig",
    "MistuningResult",
    "FMMSolver",
    "ModeIdentification",
    "GroundTruthLabel",
    "identify_nodal_circles",
    "classify_mode_family",
    "identify_modes",
    # I/O
    "load_mesh",
    "load_cad",
    "inspect_cad",
    "CadInfo",
    # Solver
    "solve",
    "rpm_sweep",
    "campbell_data",
    # Visualization
    "DiagramStyle",
    "plot_mesh",
    "plot_mode",
    "plot_full_mesh",
    "_removed_func",
    "plot_cad",
    "plot_campbell",
    "plot_zzenf",
    "diagnose_frequencies",
    "format_condition_label",
    "interactive_plane_selector",
    "bc_editor",
    "plot_boundary_conditions",
    "_removed_func",
    "_removed_func",
    # Sensors & noise
    "_RemovedEnum",
    "_RemovedClass",
    "_RemovedClass",
    "_RemovedClass",
    "_RemovedClass",
    "_removed",
    # Signal generation
    "_RemovedClass",
    "_removed",
    "_removed",
    "filter_modal_results",
    # Dataset & parametric
    "_RemovedClass",
    "_RemovedClass",
    "export_modal_results",
    "load_modal_results",
    "_RemovedClass",
    "_RemovedClass",
    "generate_conditions",
    "_removed",
]
