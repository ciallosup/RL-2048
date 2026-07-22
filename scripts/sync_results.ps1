# Pull training results from AutoDL to local machine.
# Usage: .\scripts\sync_results.ps1 -Host autodl-rl2048
param(
    [string]$Host = "autodl-rl2048",
    [string]$RemoteDir = "/root/autodl-tmp/RL-2048/results",
    [string]$LocalDir = "F:\RL-2048\results"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

# Requires rsync (Git for Windows, WSL, or cwRsync).
$rsync = Get-Command rsync -ErrorAction SilentlyContinue
if (-not $rsync) {
    Write-Error "rsync not found. Install Git for Windows or use WSL, then retry."
}

& rsync -avz "${Host}:${RemoteDir}/" "${LocalDir}/"
Write-Host "Results synced to $LocalDir"
