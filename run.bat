@echo off
rem MOA Renderer - render the Cornell box test scene and open the output folder.
rem Any extra options are passed through, e.g.  run.bat --tall mirror --short glass --background 0.3
setlocal
cd /d "%~dp0"

if not exist "build\Release\moa.exe" (
    echo moa.exe not found. Run build.bat first.
    pause
    exit /b 1
)

if not exist renders mkdir renders
"build\Release\moa.exe" --out renders\cornell %*
if errorlevel 1 (
    echo.
    echo === Render FAILED. Copy the messages above and send them to Claude. ===
    pause
    exit /b 1
)

start "" "%~dp0renders"
pause
