@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo [RAG-Lite] %CD%

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
set EC=%ERRORLEVEL%

echo.
echo [RAG-Lite] Exit code: %EC%
if not "%EC%"=="0" (
  echo Log: %~dp0last-run.log
  if exist "%~dp0last-run-error.txt" echo Error: %~dp0last-run-error.txt
)
echo Press any key to close...
pause >nul
exit /b %EC%
