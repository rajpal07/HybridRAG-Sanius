# ============================================================
#  Healthcare Intelligence — Launch Script
#  Starts: FastAPI backend (8000), Streamlit frontend (8501),
#          and an ngrok tunnel pointing at Streamlit (8501).
#
#  Usage:  .\start.ps1
#  Stop:   Ctrl+C  (or close the three terminal windows)
# ============================================================

$root   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venv   = Join-Path $root ".venv\Scripts\python.exe"
$uvicorn = Join-Path $root ".venv\Scripts\uvicorn.exe"
$streamlit = Join-Path $root ".venv\Scripts\streamlit.exe"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Healthcare Intelligence — Starting up..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. FastAPI backend ────────────────────────────────────────
Write-Host "[1/3] Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    param($root, $uvicorn)
    Set-Location $root
    & $uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload 2>&1
} -ArgumentList $root, $uvicorn

Start-Sleep -Seconds 3   # give uvicorn time to bind

# ── 2. Streamlit frontend ─────────────────────────────────────
Write-Host "[2/3] Starting Streamlit frontend on http://localhost:8501 ..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    param($root, $streamlit)
    Set-Location $root
    & $streamlit run frontend/app.py --server.port 8501 --server.headless true 2>&1
} -ArgumentList $root, $streamlit

Start-Sleep -Seconds 4   # give Streamlit time to start

# ── 3. ngrok tunnel → Streamlit ──────────────────────────────
Write-Host "[3/3] Opening ngrok tunnel to port 8501 ..." -ForegroundColor Yellow
$ngrokJob = Start-Job -ScriptBlock {
    ngrok http 8501 2>&1
}

Start-Sleep -Seconds 4   # give ngrok time to establish the tunnel

# ── 4. Fetch public URL from ngrok local API ──────────────────
$publicUrl = $null
$attempts  = 0
while (-not $publicUrl -and $attempts -lt 10) {
    Start-Sleep -Seconds 2
    $attempts++
    try {
        $tunnels = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -ErrorAction SilentlyContinue
        $publicUrl = ($tunnels.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1).public_url
        if (-not $publicUrl) {
            $publicUrl = ($tunnels.tunnels | Select-Object -First 1).public_url
        }
    } catch {
        # ngrok API not ready yet — retry
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  All services running!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Backend  : http://localhost:8000" -ForegroundColor White
Write-Host "  Frontend : http://localhost:8501" -ForegroundColor White
Write-Host "  Ngrok UI : http://localhost:4040" -ForegroundColor White
Write-Host ""

if ($publicUrl) {
    Write-Host "  PUBLIC URL (share this):" -ForegroundColor Cyan
    Write-Host "  $publicUrl" -ForegroundColor Green -BackgroundColor DarkGray
} else {
    Write-Host "  Could not auto-detect ngrok URL." -ForegroundColor Red
    Write-Host "  Open http://localhost:4040 in your browser to find it." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Press Ctrl+C to stop all services." -ForegroundColor Gray
Write-Host ""

# ── 5. Stream logs and keep alive until Ctrl+C ───────────────
try {
    while ($true) {
        # Print any new output from jobs
        Receive-Job $backendJob  | ForEach-Object { Write-Host "[backend]  $_" -ForegroundColor DarkGray }
        Receive-Job $frontendJob | ForEach-Object { Write-Host "[frontend] $_" -ForegroundColor DarkGray }
        Receive-Job $ngrokJob    | ForEach-Object { Write-Host "[ngrok]    $_" -ForegroundColor DarkGray }
        Start-Sleep -Seconds 3
    }
} finally {
    Write-Host "`nStopping all services..." -ForegroundColor Yellow
    Stop-Job $backendJob, $frontendJob, $ngrokJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob, $frontendJob, $ngrokJob -Force -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
}
