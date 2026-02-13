@echo off
REM Build and run C++ tests on Windows
REM Usage: buildtests.bat [build_type]
REM   build_type: Release (default) or Debug

setlocal
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

echo Configuring CMake (%BUILD_TYPE%)...
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

echo Building...
cmake --build . --config %BUILD_TYPE%

echo Running tests...
ctest --output-on-failure -C %BUILD_TYPE%
endlocal
