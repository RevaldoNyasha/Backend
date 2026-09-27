# Back up the database and the uploaded files into a dated folder.
#
#   .\scripts\backup.ps1                          # into ..\backups next to Backend
#   .\scripts\backup.ps1 -Destination D:\Backups  # another drive is safer
#   .\scripts\backup.ps1 -Keep 30                 # how many backups to keep
#
# Safe to run while serve.ps1 is running: the database is copied with SQLite's
# backup API (a consistent snapshot, WAL included), not by copying the file.

param(
    [string]$Destination = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'backups'),
    [int]$Keep = 14
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$target = Join-Path $Destination (Get-Date -Format 'yyyy-MM-dd_HHmmss')
New-Item -ItemType Directory -Force $target | Out-Null

python -c @'
import sqlite3, sys
source = sqlite3.connect("data/app.db")
backup = sqlite3.connect(sys.argv[1])
with backup:
    source.backup(backup)
backup.close()
source.close()
'@ (Join-Path $target 'app.db')
if ($LASTEXITCODE -ne 0) { throw "Database backup failed." }

if (Test-Path uploads) {
    Compress-Archive -Path 'uploads\*' -DestinationPath (Join-Path $target 'uploads.zip')
}

# Drop the oldest backups beyond -Keep.
Get-ChildItem $Destination -Directory |
    Sort-Object Name -Descending |
    Select-Object -Skip $Keep |
    Remove-Item -Recurse -Force

Write-Host "Backup saved to $target" -ForegroundColor Green
