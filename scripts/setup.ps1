# ML Project setup script (Windows PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "=== ML Project Setup ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Installing dependencies..."
& .\.venv\Scripts\pip install -q -r requirements-dev.txt

Write-Host "Training models (this may take a few minutes)..."
& .\.venv\Scripts\python ML/train.py

$models = @("stacking_model.pkl", "lgbm_model.pkl", "xgb_model.pkl", "locations.json", "metrics.json")
$modelDir = Join-Path $Root "backend\models"
$missing = @()
foreach ($f in $models) {
    if (-not (Test-Path (Join-Path $modelDir $f))) { $missing += $f }
}

if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing artifacts: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "Setup complete. Start the API with:" -ForegroundColor Green
Write-Host "  cd backend; ..\.venv\Scripts\uvicorn main:app --reload"
