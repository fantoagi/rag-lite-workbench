# Dot-source:  Ensure-VenvPip -VenvPythonExe $x -ProjectRoot $y  [-BootstrapPip $hashtable]
function Ensure-VenvPip {
    param(
        [Parameter(Mandatory = $true)][string]$VenvPythonExe,
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [hashtable]$BootstrapPip = $null
    )

    & $VenvPythonExe -m pip --version *>$null
    if ($LASTEXITCODE -eq 0) { return }

    $tgt = Join-Path $ProjectRoot ".venv\Lib\site-packages"

    function Invoke-SystemPipTarget {
        param([string[]]$PipArgs)
        if ($BootstrapPip) {
            & $BootstrapPip.Exe @($BootstrapPip.Prefix + @("-m", "pip") + $PipArgs)
            return
        }
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -m pip @PipArgs
            return
        }
        if (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m pip @PipArgs
        }
    }

    Write-Host "[RAG-Lite] (1/3) System pip -> venv site-packages (should show download progress)..." -ForegroundColor Cyan
    Invoke-SystemPipTarget @("install", "pip", "setuptools", "wheel", "--target", $tgt, "--default-timeout", "180")
    & $VenvPythonExe -m pip --version *>$null
    if ($LASTEXITCODE -eq 0) { return }

    Write-Host "[RAG-Lite] (2/3) ensurepip (may be slow)..." -ForegroundColor Yellow
    & $VenvPythonExe -m ensurepip --upgrade --default-pip
    & $VenvPythonExe -m pip --version *>$null
    if ($LASTEXITCODE -eq 0) { return }

    Write-Host "[RAG-Lite] (3/3) get-pip.py (needs network)..." -ForegroundColor Cyan
    $gp = Join-Path $env:TEMP "rag_lite_get_pip.py"
    try {
        Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $gp -UseBasicParsing
    } catch {
        Write-Host "[RAG-Lite] Download failed: $_" -ForegroundColor Red
        throw "get-pip download failed: $_"
    }
    & $VenvPythonExe $gp --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[RAG-Lite] get-pip failed." -ForegroundColor Red
        throw "get-pip.py failed, exit code $($LASTEXITCODE)"
    }
}
