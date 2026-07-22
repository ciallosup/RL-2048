param(
    [ValidateSet("venv", "conda")]
    [string]$Mode = "venv"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Find-PythonWithTkinter {
    $candidates = @()
    if ($env:CONDA_PREFIX) {
        $candidates += Join-Path $env:CONDA_PREFIX "python.exe"
    }
    $miniconda = "D:\Miniconda\python.exe"
    if (Test-Path $miniconda) { $candidates += $miniconda }
    $candidates += @("python", "py -3.11", "py -3.12", "py -3.13")

    foreach ($cmd in $candidates) {
        try {
            & cmd /c "$cmd -c `"import tkinter`"" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                if ($cmd -eq "python" -or $cmd -like "py *") { return $cmd }
                return $cmd
            }
        } catch {
            continue
        }
    }
    return $null
}

if ($Mode -eq "venv") {
    $pythonCmd = Find-PythonWithTkinter
    if (-not $pythonCmd) {
        Write-Error "No Python with tkinter found. Install Miniconda or Python with tcl/tk, or run: .\scripts\setup_env.ps1 -Mode conda"
    }
    Write-Host "Using Python: $pythonCmd"
    if (Test-Path ".venv") {
        Write-Host "Existing .venv found; recreate if tkinter import fails after setup."
    } else {
        if ($pythonCmd -eq "python") {
            python -m venv .venv
        } elseif ($pythonCmd -like "py *") {
            Invoke-Expression "$pythonCmd -m venv .venv"
        } else {
            & $pythonCmd -m venv .venv
        }
    }
    & .\.venv\Scripts\Activate.ps1
    python -m pip install -U pip
    pip install -e ".[dev,train]"
    python -c "import tkinter; print('tkinter OK')"
    Write-Host ""
    Write-Host "Virtual environment ready: .venv"
    Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
} else {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        Write-Error "conda not found. Install Miniconda/Anaconda or use -Mode venv."
    }
    if (-not (Test-Path ".conda\env")) {
        conda create -p .\.conda\env python=3.11 -y
    }
    conda activate .\.conda\env
    python -m pip install -U pip
    pip install -e ".[dev,train]"
    python -c "import tkinter; print('tkinter OK')"
    Write-Host ""
    Write-Host "Conda environment ready: .conda/env"
    Write-Host "Activate: conda activate ./.conda/env"
}

Write-Host "Run visualizer: rl2048-play"
