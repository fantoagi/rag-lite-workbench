# Dot-sourced by run.ps1 / setup-venv.ps1. Caller: Set-Location $PSScriptRoot first.

function Invoke-BootstrapPythonProbe {
    param(
        [string]$Exe,
        [string[]]$Prefix,
        [string]$Code
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    $prevNative = $null
    if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
        $prevNative = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        if ($Prefix -and @($Prefix).Count -gt 0) {
            $all = @() + $Prefix + @("-c", $Code)
            & $Exe @all *>&1 | Out-Null
        }
        else {
            & $Exe -c $Code *>&1 | Out-Null
        }
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $prevEap
        if ($null -ne $prevNative) {
            $PSNativeCommandUseErrorActionPreference = $prevNative
        }
    }
}

function Get-BootstrapPythonForRagLite {
    $test = "import sys; raise SystemExit(0 if (3,10) <= sys.version_info < (3,14) else 1)"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($ver in @("3.12", "3.11", "3.10")) {
            if (Invoke-BootstrapPythonProbe -Exe "py" -Prefix @("-$ver") -Code $test) {
                return @{ Exe = "py"; Prefix = @("-$ver") }
            }
        }
        if (Invoke-BootstrapPythonProbe -Exe "py" -Prefix @("-3") -Code $test) {
            return @{ Exe = "py"; Prefix = @("-3") }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        if (Invoke-BootstrapPythonProbe -Exe "python" -Prefix @() -Code $test) {
            return @{ Exe = "python"; Prefix = @() }
        }
    }
    return $null
}

function Remove-RagLiteVenvIfPythonTooNew {
    $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) { return }
    if (Invoke-BootstrapPythonProbe -Exe $venvPy -Prefix @() -Code "import sys; raise SystemExit(1 if sys.version_info >= (3, 14) else 0)") {
        return
    }
    Write-Host "[RAG-Lite] Removing .venv (Python 3.14+); will recreate..." -ForegroundColor Yellow
    Remove-Item (Join-Path $PSScriptRoot ".venv") -Recurse -Force -ErrorAction Stop
}

function Invoke-RagLitePythonGate {
    param($Bootstrap)
    if (-not $Bootstrap) { return }
    $chk = Join-Path $PSScriptRoot "check_python_supported.py"
    & $Bootstrap.Exe @($Bootstrap.Prefix + @($chk))
    if ($LASTEXITCODE -ne 0) {
        throw "check_python_supported.py failed, exit code $($LASTEXITCODE)"
    }
}

function Try-WingetPython312 {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) { return $false }
    Write-Host "[RAG-Lite] Trying winget install Python.Python.3.12 ..." -ForegroundColor Cyan
    $proc = Start-Process -FilePath "winget" -ArgumentList @(
        "install", "-e", "--id", "Python.Python.3.12",
        "--accept-package-agreements", "--accept-source-agreements"
    ) -Wait -PassThru -NoNewWindow
    if ($proc.ExitCode -ne 0) {
        Write-Host "[RAG-Lite] winget exit $($proc.ExitCode). Install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor Red
        return $false
    }
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
    Start-Sleep -Seconds 3
    return $true
}

function Get-OrInstall-BootstrapPython {
    Remove-RagLiteVenvIfPythonTooNew
    $bs = Get-BootstrapPythonForRagLite
    if ($bs) { return $bs }
    if (Try-WingetPython312) {
        $bs = Get-BootstrapPythonForRagLite
        if ($bs) { return $bs }
    }
    Write-Host "[RAG-Lite] Need Python 3.10-3.13. Install 3.12 and enable 'py' launcher, then re-run." -ForegroundColor Red
    throw "RAG-Lite: no Python 3.10-3.13 found (and winget did not help)."
}
