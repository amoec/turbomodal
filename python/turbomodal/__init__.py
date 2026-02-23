"""Turbomodal: Cyclic symmetry FEA solver for turbomachinery modal analysis."""

from turbomodal._core import (
    Material,
    NodeSet,
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
    # New C++ classes
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
from turbomodal.solver import solve, rpm_sweep, campbell_data
from turbomodal.viz import (
    plot_mesh,
    plot_mode,
    plot_full_annulus,
    plot_full_mesh,
    plot_cad,
    animate_mode,
    plot_campbell,
    plot_zzenf,
    format_condition_label,
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
    # New C++ classes
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
    "plot_mesh",
    "plot_mode",
    "plot_full_annulus",
    "plot_full_mesh",
    "plot_cad",
    "animate_mode",
    "plot_campbell",
    "plot_zzenf",
    "format_condition_label",
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
