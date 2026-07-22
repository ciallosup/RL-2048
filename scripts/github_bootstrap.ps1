# First-time GitHub setup (run locally after creating a private repo on GitHub).
# Usage:
#   .\scripts\github_bootstrap.ps1 -RemoteUrl https://github.com/<user>/RL-2048.git
param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl
)

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path .git)) {
    git init
}

git add .
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "Initial commit: RL-2048 DQN baseline, eval pipeline, and AutoDL cloud support"
}

$remotes = git remote 2>$null
if ($remotes -notcontains "origin") {
    git remote add origin $RemoteUrl
}

git branch -M main
git push -u origin main
Write-Host "Pushed to $RemoteUrl"
