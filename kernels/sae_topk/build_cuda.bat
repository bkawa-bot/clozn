@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH%;C:\Program Files\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
cd /d "%~dp0"
echo === configure (sae_topk kernel) ===
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (echo CONFIGURE_FAILED & exit /b 2)
echo === build (sae_topk lib + sae_validate + sae_test) ===
cmake --build build-cuda
if errorlevel 1 (echo BUILD_FAILED & exit /b 3)
echo === ctest (sae smoke) ===
ctest --test-dir build-cuda --output-on-failure
echo === DONE exit=%errorlevel% ===
