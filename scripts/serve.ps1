# Serve the API for the hosted frontend: uvicorn on this machine, published by
# Tailscale Funnel at https://<machine>.<tailnet>.ts.net (see DEPLOYMENT.md).
#
#   .\scripts\serve.ps1            # from the Backend folder
#
# No --reload and a single worker on purpose: the background workers
# (processing, transcription, generation) are threads inside this process, and
# several uvicorn workers would each run them and process the same jobs twice.

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$port = 8000

if (Get-Command tailscale -ErrorAction SilentlyContinue) {
    # Idempotent: publishes https://<this machine>.ts.net -> http://127.0.0.1:8000 and
    # keeps it on in the background (also after reboots) until `tailscale funnel reset`.
    tailscale funnel --bg $port | Out-Null
    Write-Host "Public API:" -ForegroundColor Green
    tailscale funnel status
} else {
    Write-Warning "Tailscale is not installed: the API will only be reachable on this machine."
}

Write-Host "Starting the API on http://127.0.0.1:$port (Ctrl+C to stop)..." -ForegroundColor Green
python -m uvicorn app.main:app --host 127.0.0.1 --port $port --proxy-headers --forwarded-allow-ips 127.0.0.1
