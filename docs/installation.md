# Installation

## Prerequisites

- **Python 3.9** or later
- **C++17** compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- **CMake 3.20** or later

The C++ core uses Eigen 3.4.0 (fetched automatically via CMake FetchContent) and the
Spectra eigenvalue library (included as a git submodule). You do not need to install
these manually.

---

## Quick Install

> **Note:** turbomodal is not yet published to PyPI. Install from source using
> the instructions in [Building from Source](#building-from-source) below.

The core package requires the following dependencies (installed automatically):

| Package      | Minimum Version | Purpose                              |
|------------- |---------------- |------------------------------------- |
| numpy        | 1.22            | Array operations, linear algebra     |
| scipy        | 1.8             | Signal processing, sparse matrices   |
| pyvista      | 0.43            | 3-D mesh visualization               |
| matplotlib   | 3.5             | 2-D plotting (Campbell, ZZENF)       |
| gmsh         | 4.12            | CAD import and mesh generation       |
| meshio       | 5.3             | Mesh file format I/O                 |
| h5py         | 3.7             | HDF5 dataset storage                 |

---

## Install with ML Support

To use the machine learning pipeline (feature extraction, complexity ladder
training, SHAP explainability, sensor optimization), install with the `ml` extra:

```bash
pip install -e ".[ml]"
```

This adds:

| Package       | Minimum Version | Purpose                                    |
|-------------- |---------------- |------------------------------------------- |
| scikit-learn  | 1.2             | Tiers 1-3 models, metrics, data splitting  |
| xgboost       | 1.7             | Tier 2 gradient-boosted tree models         |
| torch         | 2.0             | Tiers 4-6 neural network models            |
| shap          | 0.42            | SHAP value computation for explainability   |
| optuna        | 3.0             | Bayesian hyperparameter optimization        |
| mlflow        | 2.10            | Experiment tracking and model registry      |

---

## Install Everything

For development (including test dependencies), install all extras:

```bash
pip install -e ".[ml,dev]"
```

The `dev` extra adds:

| Package    | Minimum Version | Purpose             |
|----------- |---------------- |-------------------- |
| pytest     | 7               | Test runner         |
| pytest-cov | (any)           | Coverage reporting  |

---

## Building from Source

### 1. Clone the repository

```bash
git clone https://github.com/amoec/turbomodal.git
cd turbomodal
```

### 2. Initialize the Spectra submodule

The Spectra eigenvalue library is vendored as a git submodule under
`external/spectra/`. You must initialize it before building:

```bash
git submodule update --init
```

If this step is skipped, CMake will fail with:

```
FATAL_ERROR: Spectra not found. Run: git submodule update --init
```

### 3. Install build dependencies

The build system uses **scikit-build-core** with **pybind11** for the
C++/Python bridge:

```bash
pip install scikit-build-core pybind11
```

### 4. Build and install

```bash
pip install .
```

This triggers the following build flow:

1. scikit-build-core invokes CMake with the options specified in
   `pyproject.toml` (Release mode, Python bindings enabled).
2. CMake downloads **Eigen 3.4.0** via `FetchContent` (a local copy,
   independent of any system-installed Eigen).
3. The C++ static library `turbomodal_core` is compiled from the sources
   in `src/` (material, element, mesh, assembler, solvers, etc.).
4. pybind11 builds the `_core` Python extension module linking against
   `turbomodal_core`.
5. The extension is installed alongside the Python package under
   `turbomodal/`.

For an editable (development) install:

```bash
pip install -e ".[ml,dev]"
```

### 5. Build C++ tests separately (optional)

If you want to run the C++ unit tests (GTest), configure CMake directly:

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON=OFF
cmake --build .
ctest --output-on-failure
```

### Building on Windows

On Windows, use Visual Studio 2019+ or the Build Tools for Visual Studio.
Open a **Developer Command Prompt** (or **x64 Native Tools Command Prompt**)
and follow the same steps above. CMake will auto-detect the MSVC compiler.

If you prefer a specific generator:

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTS=ON -DBUILD_PYTHON=OFF
cmake --build . --config Release
ctest --output-on-failure -C Release
```

For the Python package:

```cmd
pip install -e ".[ml,dev]"
```

scikit-build-core will invoke CMake with MSVC automatically. No compiler
paths need to be specified.

---

## CMake Options

The following options can be passed to CMake via `-D`:

| Option                   | Default | Description                                      |
|------------------------- |-------- |------------------------------------------------- |
| `BUILD_PYTHON`           | `ON`    | Build the pybind11 Python bindings               |
| `BUILD_TESTS`            | `ON`    | Build C++ GTest unit tests                       |
| `BUILD_VALIDATION_TESTS` | `OFF`   | Build slow validation and integration tests      |
| `USE_OPENMP`             | `ON`    | Enable OpenMP parallelism in the solver          |

When building through `pip install`, the defaults in `pyproject.toml` set
`BUILD_TESTS=OFF`, `BUILD_PYTHON=ON`, and `USE_OPENMP=OFF`. To override,
pass CMake arguments via the `SKBUILD_CMAKE_ARGS` environment variable:

```bash
SKBUILD_CMAKE_ARGS="-DUSE_OPENMP=ON" pip install .
```

---

## External Dependencies (C++)

| Library     | Version | Source                                       |
|------------ |-------- |--------------------------------------------- |
| Eigen       | 3.4.0   | CMake FetchContent (downloaded at configure) |
| Spectra     | latest  | Git submodule at `external/spectra/`         |
| GTest       | 1.14.0  | CMake FetchContent (downloaded at configure) |
| OpenMP      | system  | Optional, enabled by `USE_OPENMP=ON`         |

Eigen 3.4.0 is pinned explicitly because the Spectra library is
incompatible with Eigen 5.x. The FetchContent approach ensures the correct
version is used regardless of what is installed on the system.

---

## Verifying Installation

After installing, verify that both the Python package and the C++ backend
load correctly:

```python
import turbomodal
print(turbomodal.__version__)  # Should print "0.1.0"

# Verify the C++ extension loads
from turbomodal import Material, Mesh, CyclicSymmetrySolver

# Create a test material
mat = Material(E=200e9, nu=0.3, rho=7800)
print(f"Material: E={mat.E/1e9:.0f} GPa, nu={mat.nu}, rho={mat.rho}")
```

To verify that the ML extras are available:

```python
from turbomodal.ml import FeatureConfig, TrainingConfig, train_mode_id_model
from turbomodal.optimization import (
    SensorOptimizationConfig,
    optimize_sensor_placement,
    compute_shap_values,
    calibrate_confidence,
)
print("ML and optimization modules loaded successfully.")
```

Run the test suite:

```bash
pytest python/tests/ -v
```

---

## Troubleshooting

### Eigen not found / wrong version

If CMake reports that Eigen is not found or you see errors related to Eigen
5.x incompatibility with Spectra, ensure that the FetchContent download
succeeded. Check your internet connection and verify that the URL
`https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz`
is accessible. The build does not use any system-installed Eigen.

### Spectra not found

Ensure you ran `git submodule update --init` before building. The Spectra
headers must be present at `external/spectra/include/Spectra/`.

### pybind11 version mismatch

The build requires pybind11 >= 2.12. If you see binding compilation errors,
upgrade pybind11:

```bash
pip install "pybind11>=2.12"
```

### BLAS/LAPACK issues on Linux

Eigen can optionally use system BLAS/LAPACK for performance. If you see
linker errors related to BLAS, install the development packages:

```bash
# Debian/Ubuntu
sudo apt install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel
```

### GPU support for PyTorch (Tiers 4-6)

The `torch` dependency installed via `pip install "turbomodal[ml]"` will
default to CPU-only on most systems. For CUDA GPU acceleration:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "turbomodal[ml]"
```

The `TrainingConfig.device` parameter controls device selection. Set it to
`"auto"` (the default) to use CUDA or MPS when available, or explicitly
to `"cuda"`, `"mps"`, or `"cpu"`.

### macOS Apple Silicon (MPS)

PyTorch MPS backend is supported on macOS with Apple Silicon. The
`TrainingConfig(device="auto")` setting will automatically select MPS
when available. If you encounter MPS-related errors, fall back to CPU:

```python
config = TrainingConfig(device="cpu")
```

### OpenMP not found on macOS

macOS ships with a Clang that does not include OpenMP by default. Either
install `libomp` via Homebrew:

```bash
brew install libomp
```

Or build without OpenMP (the default for pip installs):

```bash
SKBUILD_CMAKE_ARGS="-DUSE_OPENMP=OFF" pip install .
```

### Windows: Visual Studio not detected

If CMake cannot find a C++ compiler on Windows, install the
**Build Tools for Visual Studio** from
https://visualstudio.microsoft.com/visual-cpp-build-tools/ and select
the "Desktop development with C++" workload. Then run the build from
a **Developer Command Prompt** or set the environment with:

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
pip install .
```

### Windows: long path errors

If you encounter path-too-long errors during the Eigen FetchContent
download, enable long paths in Windows:

```cmd
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

Then restart your terminal and rebuild.

### Windows: setting CMake arguments

On Windows, use `set` instead of environment variable prefixes:

```cmd
set SKBUILD_CMAKE_ARGS=-DUSE_OPENMP=ON
pip install .
```

Or in PowerShell:

```powershell
$env:SKBUILD_CMAKE_ARGS="-DUSE_OPENMP=ON"
pip install .
```
