param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$OutFile,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Test-PdfFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    $file = Get-Item -LiteralPath $Path
    if ($file.Length -lt 1024) {
        return $false
    }

    $stream = [System.IO.File]::OpenRead($file.FullName)
    try {
        $buffer = New-Object byte[] 4
        $read = $stream.Read($buffer, 0, 4)
        if ($read -ne 4) {
            return $false
        }

        $header = [System.Text.Encoding]::ASCII.GetString($buffer)
        return $header -eq "%PDF"
    }
    finally {
        $stream.Dispose()
    }
}

$resolvedOutFile = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutFile)
$parent = Split-Path -Parent $resolvedOutFile
if (-not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
}

if ((Test-Path -LiteralPath $resolvedOutFile) -and -not $Force) {
    if (Test-PdfFile -Path $resolvedOutFile) {
        Write-Host "Exists and looks valid: $resolvedOutFile"
        exit 0
    }

    throw "Output already exists but is not a valid PDF. Use -Force to replace it: $resolvedOutFile"
}

$tmpPath = "$resolvedOutFile.download"
if (Test-Path -LiteralPath $tmpPath) {
    Remove-Item -LiteralPath $tmpPath -Force
}

try {
    $curlArgs = @(
        "-L",
        "--fail",
        "--retry", "3",
        "--retry-delay", "1",
        "--connect-timeout", "30",
        "-o", $tmpPath,
        $Url
    )

    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl.exe exited with code $LASTEXITCODE"
    }

    if (-not (Test-PdfFile -Path $tmpPath)) {
        throw "Downloaded file is not a valid PDF: $Url"
    }

    Move-Item -LiteralPath $tmpPath -Destination $resolvedOutFile -Force
    Write-Host "Downloaded PDF: $resolvedOutFile"
}
catch {
    if (Test-Path -LiteralPath $tmpPath) {
        Remove-Item -LiteralPath $tmpPath -Force
    }

    Write-Error (
        "PDF download failed. If this runs inside Codex sandbox and the error mentions " +
        "arxiv.org:443, 127.0.0.1, proxy, DNS, or connection failure, rerun this same " +
        "script with escalated network access. Url: $Url. Output: $resolvedOutFile. " +
        "Original error: $($_.Exception.Message)"
    )
}
