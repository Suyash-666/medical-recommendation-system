# Heroku Deployment Script
# Run this after rotating your Firebase key

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Medical Recommendation System - Heroku Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Heroku CLI is installed
try {
    $herokuVersion = heroku --version 2>&1
    Write-Host "✓ Heroku CLI installed: $herokuVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Heroku CLI not found. Please install it first." -ForegroundColor Red
    Write-Host "  Install: winget install Heroku.HerokuCLI" -ForegroundColor Yellow
    Write-Host "  Then restart PowerShell and run this script again." -ForegroundColor Yellow
    exit 1
}

# Check if firebase-credentials.json exists
if (-not (Test-Path "firebase-credentials.json")) {
    Write-Host "`n✗ firebase-credentials.json not found!" -ForegroundColor Red
    Write-Host "  Please ensure you have rotated your Firebase key and saved it as:" -ForegroundColor Yellow
    Write-Host "  C:\WebPython\firebase-credentials.json" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n✓ firebase-credentials.json found" -ForegroundColor Green

# Login to Heroku
Write-Host "`n[Step 1/5] Logging into Heroku..." -ForegroundColor Cyan
Write-Host "A browser will open for authentication.`n" -ForegroundColor Yellow
heroku login
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Heroku login failed" -ForegroundColor Red
    exit 1
}

# Create or link to app
Write-Host "`n[Step 2/5] Heroku App Setup" -ForegroundColor Cyan
$appName = Read-Host "Enter Heroku app name (or press Enter to create new)"

if ($appName) {
    Write-Host "Linking to existing app: $appName" -ForegroundColor Yellow
    heroku git:remote -a $appName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to link to app. Creating new one..." -ForegroundColor Yellow
        heroku create $appName
    }
} else {
    Write-Host "Creating new Heroku app..." -ForegroundColor Yellow
    heroku create
}

# Generate and set Flask secret key
Write-Host "`n[Step 3/5] Setting Flask secret key..." -ForegroundColor Cyan
$flaskSecret = python -c "import secrets; print(secrets.token_urlsafe(32))"
heroku config:set FLASK_SECRET_KEY="$flaskSecret"
Write-Host "✓ FLASK_SECRET_KEY set" -ForegroundColor Green

# Set Firebase credentials
Write-Host "`n[Step 4/5] Setting Firebase credentials..." -ForegroundColor Cyan
$cred = Get-Content .\firebase-credentials.json -Raw | ConvertFrom-Json | ConvertTo-Json -Compress
heroku config:set FIREBASE_CREDENTIALS="$cred"
Write-Host "✓ FIREBASE_CREDENTIALS set" -ForegroundColor Green

# Deploy
Write-Host "`n[Step 5/5] Deploying to Heroku..." -ForegroundColor Cyan
git push heroku main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "✓ Deployment Successful!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    
    Write-Host "Opening your app in browser..." -ForegroundColor Cyan
    heroku open
    
    Write-Host "`nUseful commands:" -ForegroundColor Yellow
    Write-Host "  View logs: heroku logs --tail" -ForegroundColor White
    Write-Host "  Restart app: heroku restart" -ForegroundColor White
    Write-Host "  View app info: heroku apps:info" -ForegroundColor White
} else {
    Write-Host "`n✗ Deployment failed. Check logs:" -ForegroundColor Red
    Write-Host "  heroku logs --tail" -ForegroundColor Yellow
}
