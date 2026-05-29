<#
Crop a raster image (PNG/JPG) to a rectangle or to an auto-detected content box.

Use cases:
  - Crop a PDF-rendered full page into a "screenshot" for a specific figure/table.
  - Auto-trim whitespace around content (heuristic).

Examples:
  # Manual crop (pixels)
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\crop_image.ps1 `
    -Input figures\\<paper_id>\\render_page_004.png `
    -Output figures\\<paper_id>\\fig2.png `
    -X 520 -Y 210 -Width 1000 -Height 520

  # Auto-trim within the top part of the page
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\crop_image.ps1 `
    -Input figures\\<paper_id>\\render_page_004.png `
    -Output figures\\<paper_id>\\fig2.png `
    -AutoTrim -Region Top -RegionPercent 62 -Margin 12
#>

[CmdletBinding(PositionalBinding = $false)]
param(
  [Parameter(Mandatory = $true)]
  [Alias('Input')]
  [string]$InputPath,

  [Parameter(Mandatory = $true)]
  [Alias('Output')]
  [string]$OutputPath,

  [int]$X,
  [int]$Y,
  [int]$Width,
  [int]$Height,

  [switch]$AutoTrim,

  [ValidateSet('All','Top','Bottom')]
  [string]$Region = 'Top',

  [ValidateRange(5,100)]
  [int]$RegionPercent = 65,

  [ValidateRange(0,200)]
  [int]$Margin = 10,

  [ValidateRange(0,255)]
  [int]$WhiteThreshold = 245,

  [ValidateRange(0,2000)]
  [int]$SkipTop = 0,

  [ValidateRange(0,2000)]
  [int]$SkipBottom = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ensure-Dir([string]$path) {
  $dir = Split-Path -Parent $path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
}

function Clamp([int]$v, [int]$min, [int]$max) {
  if ($v -lt $min) { return $min }
  if ($v -gt $max) { return $max }
  return $v
}

Add-Type -AssemblyName System.Drawing | Out-Null

$inPath = (Resolve-Path -LiteralPath $InputPath).Path
Ensure-Dir $OutputPath

$bitmap = [System.Drawing.Bitmap]::FromFile($inPath)
try {
  $imgW = $bitmap.Width
  $imgH = $bitmap.Height

  if (-not $AutoTrim) {
    if ($Width -le 0 -or $Height -le 0) { throw "Manual crop requires -Width and -Height." }
    $x0 = Clamp $X 0 ($imgW - 1)
    $y0 = Clamp $Y 0 ($imgH - 1)
    $w0 = Clamp $Width 1 ($imgW - $x0)
    $h0 = Clamp $Height 1 ($imgH - $y0)
  } else {
    # Select initial scan region.
    $scanX = 0
    $scanY = 0
    $scanW = $imgW
    $scanH = $imgH
    $slice = [math]::Floor($imgH * ($RegionPercent / 100.0))
    if ($Region -eq 'Top') {
      $scanH = [math]::Max(1, $slice)
    } elseif ($Region -eq 'Bottom') {
      $scanY = [math]::Max(0, $imgH - $slice)
      $scanH = $imgH - $scanY
    }

    if ($SkipTop -gt 0) {
      $scanY = Clamp ($scanY + $SkipTop) 0 ($imgH - 1)
    }
    if ($SkipBottom -gt 0) {
      $scanH = [math]::Max(1, ($scanH - $SkipBottom))
    }

    $minX = $scanX + $scanW
    $minY = $scanY + $scanH
    $maxX = $scanX
    $maxY = $scanY
    $found = $false

    for ($yy = $scanY; $yy -lt ($scanY + $scanH); $yy++) {
      for ($xx = $scanX; $xx -lt ($scanX + $scanW); $xx++) {
        $c = $bitmap.GetPixel($xx, $yy)
        # Consider non-white pixels as content.
        if ($c.R -lt $WhiteThreshold -or $c.G -lt $WhiteThreshold -or $c.B -lt $WhiteThreshold) {
          $found = $true
          if ($xx -lt $minX) { $minX = $xx }
          if ($yy -lt $minY) { $minY = $yy }
          if ($xx -gt $maxX) { $maxX = $xx }
          if ($yy -gt $maxY) { $maxY = $yy }
        }
      }
    }

    if (-not $found) { throw "AutoTrim found no non-white content in region $Region ($RegionPercent%)." }

    $x0 = Clamp ($minX - $Margin) 0 ($imgW - 1)
    $y0 = Clamp ($minY - $Margin) 0 ($imgH - 1)
    $x1 = Clamp ($maxX + $Margin) 0 ($imgW - 1)
    $y1 = Clamp ($maxY + $Margin) 0 ($imgH - 1)
    $w0 = [math]::Max(1, ($x1 - $x0 + 1))
    $h0 = [math]::Max(1, ($y1 - $y0 + 1))
  }

  $rect = New-Object System.Drawing.Rectangle($x0, $y0, $w0, $h0)
  $cropped = $bitmap.Clone($rect, $bitmap.PixelFormat)
  try {
    $parent = Split-Path -Parent $OutputPath
    if ($parent) {
      Ensure-Dir $OutputPath
    }
    $finalOut = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
      $OutputPath
    } else {
      Join-Path (Get-Location).Path $OutputPath
    }
    $cropped.Save($finalOut)
    Write-Output ("Cropped: {0} -> {1} (x={2},y={3},w={4},h={5})" -f $InputPath,$finalOut,$x0,$y0,$w0,$h0)
  } finally {
    $cropped.Dispose()
  }
} finally {
  $bitmap.Dispose()
}
