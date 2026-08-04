$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found."
    }
}

$PythonCommand = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }

Require-Command $PythonCommand
Require-Command node
Require-Command npm
Require-Command git
Require-Command docker
Require-Command aws

if ($PythonCommand -eq "py") {
    py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
    $PythonArgs = @("-3.12")
} else {
    python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
    $PythonArgs = @()
}

node -e 'const major = Number(process.versions.node.split(".")[0]); if (major < 22) { throw new Error(`Node.js 22+ required; found ${process.versions.node}`); }'
if ($LASTEXITCODE -ne 0) { throw "Node.js 22 or newer is required." }

$AwsVersion = aws --version 2>&1
if ($LASTEXITCODE -ne 0 -or $AwsVersion -notmatch "^aws-cli/2") {
    throw "AWS CLI v2 is required. Found: $AwsVersion"
}

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is installed, but its daemon is not running."
}

if (-not (Test-Path ".venv")) {
    & $PythonCommand @PythonArgs -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev,infra]"
npm install
if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
if (-not (Test-Path ".git")) {
    git init
}
& ".\.venv\Scripts\pre-commit.exe" install

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

New-Item -ItemType Directory -Force -Path "data\raw", "data\processed", "models" | Out-Null
New-Item -ItemType File -Force -Path "data\raw\.gitkeep", "data\processed\.gitkeep", "models\.gitkeep" | Out-Null

& ".\.venv\Scripts\python.exe" scripts\check_environment.py

Write-Host ""
Write-Host "Ready. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Edit .env before deploying."
