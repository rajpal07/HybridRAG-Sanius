param(
  [Parameter(Mandatory = $false)]
  [string]$OllamaDir = "D:\\Apps\\Ollama",

  [Parameter(Mandatory = $false)]
  [string]$OllamaModelsDir = "D:\\Apps\\Ollama\\.models",

  [Parameter(Mandatory = $false)]
  [string]$LogPath = "D:\\HybridRAG_Sanius\\.omx\\logs\\windows_admin_setup.log"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
Start-Transcript -Path $LogPath -Append | Out-Null

function Assert-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $p = New-Object Security.Principal.WindowsPrincipal($id)
  if (-not $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an elevated PowerShell (Run as Administrator)."
  }
}

Write-Host "Sanius Windows admin setup" -ForegroundColor Cyan
Assert-Admin
Write-Host "Logging to: $LogPath"

# ---- Ollama install (admin) ----
if (Get-Command ollama -ErrorAction SilentlyContinue) {
  Write-Host "Ollama already installed: $( (Get-Command ollama).Source )"
} else {
  $ollamaInstaller = Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue `
    -Path (Join-Path $env:TEMP "chocolatey\\Ollama") -Filter "OllamaSetup.exe" `
    | Sort-Object LastWriteTime -Descending | Select-Object -First 1

  if ($ollamaInstaller) {
    Write-Host "Found Ollama installer: $($ollamaInstaller.FullName)"
    Write-Host "Running Ollama installer (silent)..."
    # NSIS installers typically support /D=<path> to select the install directory.
    # It must be the last argument. This requires admin rights for a machine install.
    $args = @("/S", "/D=$OllamaDir")
    Write-Host "Install dir: $OllamaDir"
    $p = Start-Process -FilePath $ollamaInstaller.FullName -ArgumentList $args -PassThru -Wait
    Write-Host "Ollama installer exit code: $($p.ExitCode)"
  } else {
    Write-Warning "OllamaSetup.exe not found in $env:TEMP\\chocolatey\\Ollama"
    Write-Warning "Option A: download from https://ollama.com/download and install"
    Write-Warning "Option B: choco install ollama -y (in an elevated shell)"
  }
}

# Refresh PATH from registry in this session
$env:PATH = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

Write-Host "`nConfiguring Ollama model storage..."
try {
  New-Item -ItemType Directory -Force -Path $OllamaModelsDir | Out-Null
  [Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $OllamaModelsDir, "Machine")
  $env:OLLAMA_MODELS = $OllamaModelsDir
  Write-Host "OLLAMA_MODELS (Machine) = $OllamaModelsDir"
} catch {
  Write-Warning "Could not set OLLAMA_MODELS: $($_.Exception.Message)"
}

Write-Host "`nChecking Ollama port 11434..."
try {
  $tnc = Test-NetConnection -ComputerName localhost -Port 11434
  Write-Host ("11434 listening: " + $tnc.TcpTestSucceeded)
} catch {
  Write-Warning $_.Exception.Message
}

# ---- Postgres: create sanius role/db ----
$svc = Get-CimInstance Win32_Service | Where-Object { $_.Name -like "postgresql-x64-*" } | Select-Object -First 1
if (-not $svc) { throw "No PostgreSQL Windows service found (expected name like postgresql-x64-17)." }

$dataMatch = [regex]::Match($svc.PathName, '-D\s+"([^"]+)"')
if (!$dataMatch.Success) { throw "Could not parse Postgres data dir from service PathName." }
$dataDir = $dataMatch.Groups[1].Value
$hba = Join-Path $dataDir "pg_hba.conf"
if (!(Test-Path $hba)) { throw "pg_hba.conf not found at $hba" }

$binMatch = [regex]::Match($svc.PathName, '^"([^"]+pg_ctl\.exe)"')
if (!$binMatch.Success) { throw "Could not parse pg_ctl.exe path from service PathName." }
$pgCtlPath = $binMatch.Groups[1].Value
$pgBin = Split-Path -Parent $pgCtlPath
$psql = Join-Path $pgBin "psql.exe"
if (!(Test-Path $psql)) { throw "psql.exe not found at $psql" }

Write-Host "`nPostgres service: $($svc.Name)"
Write-Host "Postgres data dir: $dataDir"

Write-Host "Backing up pg_hba.conf..."
$backup = Join-Path $dataDir ("pg_hba.conf.bak_codex_" + (Get-Date -Format yyyyMMdd_HHmmss))
Copy-Item -LiteralPath $hba -Destination $backup -Force
Write-Host "Backup: $backup"

try {
  Write-Host "Temporarily enabling trust auth for local postgres user..."
  $lines = Get-Content -LiteralPath $hba
  $insertAt = -1
  for ($i = 0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -match '^local\s+all\s+all\s+') { $insertAt = $i; break }
  }
  if ($insertAt -lt 0) { throw "Could not find 'local all all ...' line in pg_hba.conf" }

  $stamp = (Get-Date -Format yyyyMMdd_HHmmss)
  $markerStart = "# CODEX TEMP TRUST START $stamp"
  $markerEnd = "# CODEX TEMP TRUST END $stamp"
  $trustBlock = @(
    $markerStart,
    "local   all             postgres                                 trust",
    "host    all             postgres         127.0.0.1/32            trust",
    "host    all             postgres         ::1/128                 trust",
    $markerEnd
  )

  $new = @()
  if ($insertAt -gt 0) { $new += $lines[0..($insertAt-1)] }
  $new += $trustBlock
  $new += $lines[$insertAt..($lines.Length-1)]

  # Use ASCII to avoid BOM issues (Postgres on Windows can be picky here)
  Set-Content -LiteralPath $hba -Value $new -Encoding Ascii

  Write-Host "Restarting Postgres service to apply pg_hba.conf..."
  Restart-Service -Name $svc.Name -Force

  Write-Host "Creating/ensuring role+db 'sanius'..."
  $sql = @'
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sanius') THEN
    CREATE ROLE sanius LOGIN PASSWORD 'sanius';
  ELSE
    ALTER ROLE sanius WITH LOGIN PASSWORD 'sanius';
  END IF;
END$$;
'@

  $sql | & $psql -w -h localhost -p 5432 -U postgres -d postgres -v ON_ERROR_STOP=1 -f -

  $dbExists = & $psql -w -h localhost -p 5432 -U postgres -d postgres -tAc "select 1 from pg_database where datname='sanius';"
  if ($dbExists -match "1") {
    Write-Host "Database 'sanius' already exists."
  } else {
    Write-Host "Creating database 'sanius' owned by 'sanius'..."
    & $psql -w -h localhost -p 5432 -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE sanius OWNER sanius;"
  }

  Write-Host "Ensuring pgvector extension exists in 'sanius'..."
  "CREATE EXTENSION IF NOT EXISTS vector;" | & $psql -w -h localhost -p 5432 -U postgres -d sanius -v ON_ERROR_STOP=1 -f -
} finally {
  Write-Host "Restoring original pg_hba.conf..."
  Copy-Item -LiteralPath $backup -Destination $hba -Force
  Restart-Service -Name $svc.Name -Force
}

Write-Host "`nVerifying login as sanius..."
$env:PGPASSWORD = "sanius"
"select current_user, current_database();" | & $psql -w -h localhost -p 5432 -U sanius -d sanius -v ON_ERROR_STOP=1 -f -
Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue

Write-Host "`nDone." -ForegroundColor Green

Stop-Transcript | Out-Null
