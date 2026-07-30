param(
    [switch]$SkipBuild,
    [switch]$SkipInstall,
    [switch]$NoServe,
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

function Ensure-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "'$Name' is not available in PATH. $InstallHint"
    }
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $repoRoot "frontend"
$backendDir = Join-Path $repoRoot "backend"

if (-not (Test-Path $frontendDir)) {
    throw "Frontend directory not found: $frontendDir"
}
if (-not (Test-Path $backendDir)) {
    throw "Backend directory not found: $backendDir"
}

# Add standard Node location if present.
$nodeDir = Join-Path $env:ProgramFiles "nodejs"
if (Test-Path (Join-Path $nodeDir "node.exe")) {
    $env:Path = "$nodeDir;$env:Path"
}

Ensure-Command -Name "python" -InstallHint "Install Python 3.10+ and reopen terminal."
if (-not $SkipBuild) {
    Ensure-Command -Name "npm" -InstallHint "Install Node.js LTS and reopen terminal."
}

if (-not $SkipBuild) {
    Write-Host "==> Building frontend..."
    Push-Location $frontendDir
    try {
        if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
            Write-Host "==> Installing frontend dependencies..."
            npm install
        }
        npm run build
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host "==> Skipping frontend build."
}

Write-Host "==> Installing backend dependencies..."
Push-Location $backendDir
try {
    if (-not $SkipInstall) {
        python -m pip install -r requirements.txt waitress
    }
    else {
        Write-Host "==> Skipping backend dependency install."
    }

    if ($NoServe) {
        Write-Host "==> NoServe enabled. Exiting after setup."
        exit 0
    }

    Write-Host "==> Starting production server on http://localhost:$Port"
    python -m waitress --host=0.0.0.0 --port=$Port api_server:app
}
finally {
    Pop-Location
}
