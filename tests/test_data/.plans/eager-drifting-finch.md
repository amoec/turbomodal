# Plan: Update Documentation to Reflect All Changes

## Context

The turbomodal project has undergone significant development — new C++ modules (damping, forced response, mistuning, mode identification), a comprehensive V&V test suite (14 C++ test suites, 14 Python test files), and build system updates. The documentation is out of date in several areas:

1. **`docs/installation.md`** falsely claims the package is on PyPI (`pip install turbomodal`)
2. **`docs/validation.md`** lists only 5 Python test files — there are now 14
3. **`README.md`** understates the test suite and implies PyPI availability
4. **`docs/installation.md`** is missing the `BUILD_VALIDATION_TESTS` CMake option and incorrectly says GTest is "system"
5. **`pyproject.toml`** is missing project metadata (URLs, license, authors)

## Changes

### 1. `docs/installation.md`

**a. Replace "Quick Install" section (lines 15-34)** — Remove `pip install turbomodal` claim. Replace with a note that the package is not yet on PyPI, and that installation is from source only. Keep the dependency table.

**b. Replace "Install with ML Support" section (lines 37-56)** — Change `pip install "turbomodal[ml]"` to `pip install -e ".[ml]"` (from-source). Keep the dependency table.

**c. Replace "Install Everything" section (lines 59-73)** — Change `pip install "turbomodal[ml,dev]"` to `pip install -e ".[ml,dev]"` (from-source).

**d. Update CMake Options table (lines 171-187)** — Add `BUILD_VALIDATION_TESTS` (default OFF, "Build slow validation/integration tests").

**e. Fix External Dependencies table (line 197)** — Change GTest source from "system" to "CMake FetchContent (downloaded at configure)".

### 2. `docs/validation.md`

**a. Update "Test Suite Overview" section (lines 229-248)** — Replace the 5-file table with the full 14-file inventory covering all subsystems (A through D), including the new test_solver.py, test_dataset.py, test_parametric.py, test_noise.py, test_sensors.py, test_signal_gen.py, test_validation_python.py, and test_integration.py.

**b. Add new "C++ Test Suites" section** — Document all 14 C++ test executables (MaterialTests through IntegrationTests), noting the new suites: DampingTests (14 tests), ForcedResponseTests (18 tests), MistuningTests (18 tests), ModeIdentificationTests (13 tests), ValidationTests (10 tests), IntegrationTests (3 tests).

**c. Add new subsections documenting the new Python test files** — Brief tables for each new test file, matching the existing format used for ML and optimization tests.

**d. Update "Running Tests" section (lines 244-268)** — Add commands for:
- C++ validation tests: `cmake -DBUILD_VALIDATION_TESTS=ON`
- Python validation tests: `pytest -m validation`
- Coverage: `pytest --cov=turbomodal --cov-report=term-missing`

**e. Update "CI Configuration Notes" (lines 370-394)** — Add notes about `@pytest.mark.validation` / `@pytest.mark.slow` markers, session-scoped fixtures, and C++ validation test runtime (~10 min).

### 3. `README.md`

**a. Add PyPI notice** — After line 5 (platform badges), add:
> **Note:** This package is not yet published to PyPI. Install from source (see below).

**b. Update "Running Tests" section (lines 261-275)** — Expand to show the full test suite (14 C++ suites, 14 Python test files, commands for validation tests and coverage).

**c. Fix line 190** — Change `pip install turbomodal[ml]` to `pip install -e ".[ml]"`.

### 4. `pyproject.toml`

**a. Add project metadata** after `requires-python`:
- `license = {text = "MIT"}`
- `authors = [{name = "Adam Moëc"}]`
- `[project.urls]` section with Repository and Documentation links

**b. Add pytest markers configuration**:
```toml
[tool.pytest.ini_options]
markers = [
    "validation: slow validation tests against analytical benchmarks",
    "slow: tests taking more than a few seconds",
]
```

### 5. `docs/quickstart.md`

**a. Update "Next Steps" section (lines 396-407)** — Add mention of validation tests (`pytest -m validation` and `BUILD_VALIDATION_TESTS=ON`).

### Files NOT modified

- `docs/architecture.md` — accurate as-is
- `docs/ml-guide.md` — accurate as-is
- `docs/api/*.md` — APIs haven't changed

## Files to Modify

1. `turbomodal/docs/installation.md`
2. `turbomodal/docs/validation.md`
3. `turbomodal/README.md`
4. `turbomodal/pyproject.toml`
5. `turbomodal/docs/quickstart.md`

## Verification

1. Confirm no broken markdown links: `grep -r '](docs/' README.md`
2. Verify pyproject.toml parses: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
3. Confirm `BUILD_VALIDATION_TESTS` mentioned in both CMakeLists.txt and docs
4. Confirm test counts match: `ctest --test-dir build -N` and `pytest --collect-only python/tests/`
