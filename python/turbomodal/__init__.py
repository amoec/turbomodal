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

from turbomodal.io import load_mesh, load_cad
from turbomodal.solver import solve, rpm_sweep, campbell_data
from turbomodal.viz import (
    plot_mesh,
    plot_mode,
    plot_full_annulus,
    animate_mode,
    plot_campbell,
    plot_zzenf,
)

# Subsystem B: Signal generation pipeline
from turbomodal.sensors import (
    SensorType,
    SensorLocation,
    SensorArrayConfig,
    VirtualSensorArray,
)
from turbomodal.noise import NoiseConfig, apply_noise
from turbomodal.signal_gen import (
    SignalGenerationConfig,
    generate_signals_for_condition,
    generate_dataset_signals,
)

# Subsystem A extensions: dataset & parametric sweep
from turbomodal.dataset import (
    OperatingCondition,
    DatasetConfig,
    export_modal_results,
    load_modal_results,
)
from turbomodal.parametric import (
    ParametricRange,
    ParametricSweepConfig,
    generate_conditions,
    run_parametric_sweep,
)

__version__ = "0.1.0"

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
    # Solver
    "solve",
    "rpm_sweep",
    "campbell_data",
    # Visualization
    "plot_mesh",
    "plot_mode",
    "plot_full_annulus",
    "animate_mode",
    "plot_campbell",
    "plot_zzenf",
    # Sensors & noise
    "SensorType",
    "SensorLocation",
    "SensorArrayConfig",
    "VirtualSensorArray",
    "NoiseConfig",
    "apply_noise",
    # Signal generation
    "SignalGenerationConfig",
    "generate_signals_for_condition",
    "generate_dataset_signals",
    # Dataset & parametric
    "OperatingCondition",
    "DatasetConfig",
    "export_modal_results",
    "load_modal_results",
    "ParametricRange",
    "ParametricSweepConfig",
    "generate_conditions",
    "run_parametric_sweep",
]
