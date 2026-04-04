@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
set PYTHONUTF8=1

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
set EC=%ERRORLEVEL%

echo.
echo ========================================
echo [RAG-Lite] Exit code: %EC%
if not "%EC%"=="0" (
  echo Log: %~dp0last-run.log
  if exist "%~dp0last-run-error.txt" echo Error: %~dp0last-run-error.txt
)
echo ========================================
echo Press any key to close this window...
pause >nul

exit /b %EC%
