<#
Commit and push Paper Library changes created by scheduled tasks.

This script is intentionally small and conservative:
- exits cleanly when there are no changes
- stages the whole repository, including newly generated files
- rebases before pushing so remote-only commits are not overwritten
- uses a local lock to avoid concurrent automated pushes
#>

[CmdletBinding()]
param(
  [string]$Message = 'Automated paper library update',
  [int]$LockTimeoutSeconds = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$GitDir = Join-Path $Root '.git'
$LockPath = Join-Path $GitDir 'paper-library-upload.lock'

function Invoke-Git {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments
  )

  $previousErrorActionPreference = $ErrorActionPreference
  $hasNativePreference = Test-Path Variable:PSNativeCommandUseErrorActionPreference
  if ($hasNativePreference) {
    $previousNativePreference = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
  }

  try {
    $ErrorActionPreference = 'Continue'
    $output = & git -C $Root @Arguments 2>&1 | ForEach-Object { $_.ToString() }
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
    if ($hasNativePreference) {
      $PSNativeCommandUseErrorActionPreference = $previousNativePreference
    }
  }

  if ($exitCode -ne 0) {
    $command = 'git ' + ($Arguments -join ' ')
    throw "$command failed with exit code ${exitCode}:`n$($output -join "`n")"
  }
  return $output
}

if (-not (Test-Path $GitDir)) {
  throw "This directory is not a Git repository: $Root"
}

$deadline = (Get-Date).AddSeconds($LockTimeoutSeconds)
$lockStream = $null
while ($null -eq $lockStream) {
  try {
    $lockStream = [System.IO.File]::Open(
      $LockPath,
      [System.IO.FileMode]::OpenOrCreate,
      [System.IO.FileAccess]::ReadWrite,
      [System.IO.FileShare]::None
    )
  } catch [System.IO.IOException] {
    if ((Get-Date) -ge $deadline) {
      throw "Timed out waiting for automated Git upload lock: $LockPath"
    }
    Start-Sleep -Seconds 5
  }
}

try {
  $branch = (Invoke-Git @('branch', '--show-current') | Out-String).Trim()
  if ([string]::IsNullOrWhiteSpace($branch)) {
    throw 'Cannot publish from a detached HEAD.'
  }

  $status = (Invoke-Git @('status', '--porcelain') | Out-String).Trim()
  if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Output 'No Git changes to upload.'
    exit 0
  }

  Invoke-Git @('add', '-A') | Out-Null

  & git -C $Root diff --cached --quiet
  $diffExitCode = $LASTEXITCODE
  if ($diffExitCode -eq 0) {
    Write-Output 'No staged Git changes to upload.'
    exit 0
  }
  if ($diffExitCode -ne 1) {
    throw "git diff --cached --quiet failed with exit code $diffExitCode"
  }

  $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
  Invoke-Git @('commit', '-m', "$Message ($timestamp)") | Out-Null
  Invoke-Git @('pull', '--rebase', '--autostash', 'origin', $branch) | Out-Null
  Invoke-Git @('push', 'origin', $branch) | Out-Null

  $head = (Invoke-Git @('rev-parse', '--short', 'HEAD') | Out-String).Trim()
  Write-Output "Uploaded Git changes to origin/$branch at $head."
} finally {
  if ($null -ne $lockStream) {
    $lockStream.Dispose()
  }
  Remove-Item -LiteralPath $LockPath -ErrorAction SilentlyContinue
}
