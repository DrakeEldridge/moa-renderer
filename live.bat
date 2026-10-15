@echo off
rem MOA Renderer - open the live render window.
setlocal
cd /d "%~dp0"

if not exist "build\Release\moa.exe" (
    echo moa.exe not found. Run build.bat first.
    pause
    exit /b 1
)

"build\Release\moa.exe" %*
if errorlevel 1 (
    echo.
    echo === MOA Renderer stopped with an error. Copy the messages above and send them to Claude. ===
    pause
)
