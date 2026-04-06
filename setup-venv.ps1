# Create .venv + pip install only (same automation as run.ps1)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot

$logPath = Join-Path $PSScriptRoot "last-run.log"
$errPath = Join-Path $PSScriptRoot "last-run-error.txt"
Remove-Item $logPath, $errPath -ErrorAction SilentlyContinue
try { Start-Transcript -Path $logPath -Force | Out-Null } catch { }

$exitCode = 0
try {
    . (Join-Path $PSScriptRoot "venv-common.ps1")

    $bootstrap = Get-OrInstall-BootstrapPython
    Invoke-RagLitePythonGate $bootstrap

    $venvDir = Join-Path $PSScriptRoot ".venv"
    $venvPy = Join-Path $venvDir "Scripts\python.exe"

    if (-not (Test-Path $venvPy)) {
        Write-Host "[RAG-Lite] Creating $venvDir ..." -ForegroundColor Cyan
        & $bootstrap.Exe @($bootstrap.Prefix + @("-m", "venv", $venvDir))
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed, exit $($LASTEXITCODE)" }
        if (-not (Test-Path $venvPy)) {
            throw "venv missing: $venvPy"
        }
        Write-Host "[RAG-Lite] .venv created." -ForegroundColor Green
    }

    . (Join-Path $PSScriptRoot "bootstrap-venv-pip.ps1")
    Ensure-VenvPip -VenvPythonExe $venvPy -ProjectRoot $PSScriptRoot -BootstrapPip $bootstrap

    Write-Host "[RAG-Lite] pip install -r requirements.txt ..." -ForegroundColor Cyan
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r (Join-Path $PSScriptRoot "requirements.txt") --default-timeout=120
    if ($LASTEXITCODE -ne 0) { throw "pip install failed, exit $($LASTEXITCODE)" }

    $ensureImg = Join-Path $PSScriptRoot "rag_lite\ensure_image_deps.py"
    if (Test-Path $ensureImg) {
        & $venvPy $ensureImg
    }

    Write-Host "[RAG-Lite] Done. Interpreter: $venvPy"
}
catch {
    $exitCode = 1
    $msg = if ($_.Exception.Message) { $_.Exception.Message } else { "$_" }
    Write-Host "ERROR: $msg" -ForegroundColor Red
    $_ | Out-String | Set-Content -Path $errPath -Encoding UTF8
    Write-Host "See $errPath and $logPath" -ForegroundColor Yellow
}
finally {
    try { Stop-Transcript -ErrorAction SilentlyContinue | Out-Null } catch { }
}

exit $exitCode
