@echo off
setlocal

set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_prod.ps1"

if errorlevel 1 (
  echo.
  echo Failed to start production server.
  pause
)

endlocal
