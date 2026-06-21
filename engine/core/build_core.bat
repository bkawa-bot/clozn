@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH%;C:\Program Files\CMake\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
cd /d "%~dp0"
echo === tools ===
where cl || (echo CL_MISSING & exit /b 1)
where cmake || (echo CMAKE_MISSING & exit /b 1)
where ninja || (echo NINJA_MISSING & exit /b 1)
echo === configure ===
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 1)
echo === build ===
cmake --build build-cpu
if errorlevel 1 (echo BUILD_FAILED & exit /b 1)
echo === ctest ===
ctest --test-dir build-cpu --output-on-failure
