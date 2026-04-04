# RAG-Lite: pick Python 3.12/3.11/3.10, auto-remove bad .venv, pip bootstrap, run main
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"

$logPath = Join-Path $PSScriptRoot "last-run.log"
$errPath = Join-Path $PSScriptRoot "last-run-error.txt"
Remove-Item $logPath, $errPath -ErrorAction SilentlyContinue
try {
    Start-Transcript -Path $logPath -Force | Out-Null
} catch { }

$exitCode = 0
try {
    . (Join-Path $PSScriptRoot "venv-common.ps1")

    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

    $bootstrap = Get-OrInstall-BootstrapPython
    Invoke-RagLitePythonGate $bootstrap

    if (-not (Test-Path $venvPython)) {
        $tag = ($bootstrap.Prefix -join " ")
        Write-Host "[RAG-Lite] Creating .venv (bootstrap: $($bootstrap.Exe) $tag)..."
        $venvDir = Join-Path $PSScriptRoot ".venv"
        & $bootstrap.Exe @($bootstrap.Prefix + @("-m", "venv", $venvDir))
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed, exit code $($LASTEXITCODE)" }
        if (-not (Test-Path $venvPython)) {
            throw "venv not found after create: $venvPython"
        }
        Write-Host "[RAG-Lite] .venv ready."
    }

    function Invoke-ProjectPython {
        param([string[]]$PyArgs)
        & $venvPython @PyArgs
    }

    . (Join-Path $PSScriptRoot "bootstrap-venv-pip.ps1")
    Ensure-VenvPip -VenvPythonExe $venvPython -ProjectRoot $PSScriptRoot -BootstrapPip $bootstrap

    Invoke-ProjectPython @("-c", "import gradio")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[RAG-Lite] pip install -r requirements.txt (first run may take a while)..."
        $req = Join-Path $PSScriptRoot "requirements.txt"
        Invoke-ProjectPython @("-m", "pip", "install", "-r", $req, "--default-timeout=120")
        if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed, exit $($LASTEXITCODE)" }
    }

    Write-Host "[RAG-Lite] Using: $venvPython"
    $mainPy = Join-Path $PSScriptRoot "main.py"
    Invoke-ProjectPython @("-u", $mainPy)
    if ($LASTEXITCODE -ne 0) {
        throw "main.py failed, exit code $($LASTEXITCODE)"
    }
}
catch {
    $exitCode = 1
    $msg = if ($_.Exception.Message) { $_.Exception.Message } else { "$_" }
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    $detail = @"
$msg

ScriptStackTrace:
$($_.ScriptStackTrace)

Exception:
$($_.Exception | Format-List * | Out-String)
"@
    Set-Content -Path $errPath -Value $detail -Encoding UTF8
    Write-Host "Details saved to: $errPath" -ForegroundColor Yellow
    Write-Host "Full session log: $logPath" -ForegroundColor Yellow
}
finally {
    try { Stop-Transcript -ErrorAction SilentlyContinue | Out-Null } catch { }
}

exit $exitCode
