@echo off
rem MOA Renderer - configure and build (Release). Double-click or run from a terminal.
setlocal
cd /d "%~dp0"

where cmake >nul 2>nul
if errorlevel 1 (
    echo ERROR: CMake was not found. Install CMake 4.x from cmake.org and tick "Add CMake to the PATH".
    goto :fail
)
where nvcc >nul 2>nul
if errorlevel 1 (
    echo ERROR: The CUDA compiler nvcc was not found. Install the CUDA Toolkit, then open a new terminal.
    goto :fail
)

echo === Configuring ===
cmake -S . -B build -A x64
if errorlevel 1 goto :fail

echo.
echo === Building ===
cmake --build build --config Release --parallel
if errorlevel 1 goto :fail

echo.
echo === Build succeeded: build\Release\moa.exe ===
echo Run run.bat to render the test scene.
if not defined MOA_NO_PAUSE pause
exit /b 0

:fail
echo.
echo === Build FAILED. Copy the messages above and send them to Claude. ===
if not defined MOA_NO_PAUSE pause
exit /b 1
