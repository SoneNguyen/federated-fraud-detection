# Run all 3 federated clients simultaneously
# Usage: .\run_all_clients.ps1

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
$DataBasePath = Join-Path $ProjectRoot "data\processed"

# Verify venv exists
if (-not (Test-Path $VenvPath)) {
    Write-Error "Virtual environment not found at: $VenvPath"
    Write-Host "Please run 'uv install' first"
    exit 1
}

# Verify data exists
if (-not (Test-Path $DataBasePath)) {
    Write-Error "Data directory not found at: $DataBasePath"
    Write-Host "Please run 'uv run python data/load_ieee_cis.py' first"
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Federated Learning - Launching 3 Clients" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$jobs = @()

foreach ($client_id in 0, 1, 2) {
    $data_path = Join-Path $DataBasePath "client_$client_id\transactions_normalized.parquet"
    
    # Verify client data exists
    if (-not (Test-Path $data_path)) {
        Write-Error "Client $client_id data not found at: $data_path"
        exit 1
    }
    
    $job_name = "client_$client_id"
    
    Write-Host "[Client $client_id] Starting..." -ForegroundColor Yellow
    Write-Host "  Data: $data_path"
    Write-Host ""
    
    # Start client in background
    $job = Start-Job -Name $job_name -ScriptBlock {
        param($ProjectRoot, $VenvPath, $ClientID, $DataPath)
        
        # Activate venv and run client
        Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -ErrorAction SilentlyContinue
        & $VenvPath
        Set-Location $ProjectRoot
        
        # Set environment variables
        $env:CLIENT_ID = $ClientID
        $env:DATA_PATH = $DataPath
        $env:SERVER_ADDRESS = "localhost:8080"
        $env:LOCAL_EPOCHS = 5
        
        # Run client
        & uv run python client/run_client.py
    } -ArgumentList $ProjectRoot, $VenvPath, $client_id, $data_path
    
    $jobs += $job
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All 3 clients launched" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all clients" -ForegroundColor Magenta
Write-Host ""

# Monitor all jobs
while ($jobs | Where-Object { $_.State -eq "Running" }) {
    Start-Sleep -Seconds 1
}

# Show final status
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Client Status:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

foreach ($job in $jobs) {
    $status = if ($job.State -eq "Completed") { "✓ Completed" } else { "✗ $($job.State)" }
    Write-Host "$($job.Name): $status" -ForegroundColor Green
    
    # Show any errors
    $errors = Receive-Job -Job $job -ErrorVariable $true 2>&1 | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }
    if ($errors) {
        Write-Host "  Errors:" -ForegroundColor Red
        foreach ($err in $errors) {
            Write-Host "    - $err" -ForegroundColor Red
        }
    }
}

# Clean up
Remove-Job -Job $jobs -ErrorAction SilentlyContinue
