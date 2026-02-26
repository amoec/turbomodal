# Contributing to turbomodal

Thank you for your interest in contributing to turbomodal.

## Development Setup

### Prerequisites

- Python 3.9+
- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.20+

### Clone and build

```bash
git clone https://github.com/amoec/turbomodal.git
cd turbomodal
git submodule update --init --recursive

# Install in editable mode with all extras
pip install -e ".[dev,ml]"
```

### Build C++ tests

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON=OFF
cmake --build . --parallel
ctest --output-on-failure
```

## Running Tests

```bash
# Python tests
pytest python/tests/ -v

# Python validation tests only (slower)
pytest python/tests/ -v -m validation

# Python tests with coverage
pytest python/tests/ -v --cov=turbomodal --cov-report=term-missing

# C++ unit tests
cd build && ctest --output-on-failure

# C++ validation tests (requires rebuild)
cmake .. -DBUILD_TESTS=ON -DBUILD_VALIDATION_TESTS=ON
cmake --build . && ctest --output-on-failure
```

All tests must pass before submitting a pull request.

## Code Style

### Python

- Follow PEP 8.
- Use type annotations for all public function signatures.
- Keep lines under 100 characters.

### C++

- C++17 standard.
- Use `clang-format` with the project's configuration.
- Prefer `const` references for input parameters.

## Pull Request Process

1. Fork the repository and create a feature branch from `dev`.
2. Make your changes with clear, descriptive commit messages.
3. Add or update tests for any new or changed functionality.
4. Ensure all tests pass locally on your platform.
5. Update documentation if your change affects the public API.
6. Submit a pull request against the `dev` branch.

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/amoec/turbomodal/issues)
to report bugs or request features. Include:

- A clear description of the problem or feature.
- Steps to reproduce (for bugs).
- Platform, Python version, and turbomodal version.
- Relevant error messages or stack traces.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.
