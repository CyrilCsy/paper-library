<# 
Fallback Paper Library manager (PowerShell).

This vault originally uses `scripts/paper_manager.py`, but some environments
may not have a runnable Python interpreter. This script implements a compatible
subset of the CLI used by the daily automation:

  scan | sync-obsidian | pick --reserve | extract | extract-images | mark-note | export-xlsx
#>

[CmdletBinding(PositionalBinding = $false)]
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [ValidateSet('scan', 'sync-obsidian', 'pick', 'extract', 'extract-images', 'mark-note', 'mark-read', 'export-xlsx', 'stats')]
  [string]$Command,

  [switch]$Reserve,
  [switch]$RefreshClassification,
  [string]$PaperId,
  [int]$MaxChars = 45000,
  [string]$Output,
  [string]$OutputDir,
  [int]$MaxImages = 24,
  [int]$MinWidth = 180,
  [int]$MinHeight = 120,
  [string]$NotePath,
  [string]$Status = 'read'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:ROOT = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:PAPER_DIR = Join-Path $script:ROOT 'papers'
$script:INDEX_PATH = Join-Path $script:ROOT 'papers.csv'
$script:XLSX_PATH = Join-Path $script:ROOT 'paper_library.xlsx'
$script:NOTES_DIR = Join-Path $script:ROOT 'notes'
$script:EXTRACT_DIR = Join-Path $script:ROOT 'extracted_text'
$script:FIGURE_DIR = Join-Path $script:ROOT 'figures'
$script:OBSIDIAN_DIR = Join-Path $script:ROOT 'literature'
$script:OBSIDIAN_PAPERS_DIR = Join-Path $script:OBSIDIAN_DIR 'papers'
$script:OBSIDIAN_MAPS_DIR = Join-Path $script:OBSIDIAN_DIR 'maps'
$script:OBSIDIAN_YEARS_DIR = Join-Path $script:OBSIDIAN_DIR 'years'
$script:OBSIDIAN_VENUES_DIR = Join-Path $script:OBSIDIAN_DIR 'venues'
$script:OBSIDIAN_TOPICS_DIR = Join-Path $script:OBSIDIAN_DIR 'topics'

$script:CSV_FIELDS = @(
  'paper_id',
  'filename',
  'title',
  'year',
  'venue',
  'subfield',
  'keywords',
  'importance',
  'importance_reason',
  'read_status',
  'note_status',
  'note_path',
  'selected_on',
  'last_reviewed_on',
  'added_on',
  'updated_on',
  'file_size_mb'
)

$script:VENUES = @(
  'AAAI','ACL','CVPR','ECCV','ICCV','ICLR','ICML','IJCV','MMASIA','NEURIPS','NIPS','PMLR','WACV'
)

$script:READ_DONE = @('read','done','finished','yes','已读','完成')
$script:NOTE_DONE = @('generated','done','yes','已生成')

$script:LANDMARK_PATTERNS = [ordered]@{
  'denoising diffusion probabilistic models'               = 'foundational diffusion model'
  'high-resolution image synthesis with latent diffusion models' = 'latent diffusion foundation'
  'taming transformers for high-resolution image synthesis' = 'VQGAN and transformer foundation'
  'maskgit'                                               = 'masked visual token generation foundation'
  'make-a-video'                                          = 'early large-scale text-to-video model'
  'cogvideo'                                              = 'large text-to-video transformer line'
  'cogview'                                               = 'large text-to-image transformer line'
  'diffusion models beat gans'                            = 'important diffusion scaling result'
  'scalable diffusion models with transformers'           = 'DiT foundation'
  'visual autoregressive modeling'                        = 'major autoregressive image generation result'
  'wan open and advanced'                                 = 'major open video generation system'
  'hunyuanvideo'                                          = 'major video generation system'
  'open-sora'                                             = 'open video generation system'
  'sana efficient high-resolution'                        = 'efficient high-resolution T2I system'
  'soundstream'                                           = 'neural audio codec foundation'
}

function Get-Today {
  (Get-Date).ToString('yyyy-MM-dd')
}

function To-StringOrEmpty([object]$value) {
  if ($null -eq $value) { return '' }
  return [string]$value
}

function Normalize-Space([string]$text) {
  if ($null -eq $text) { return '' }
  return ([regex]::Replace($text, '\s+', ' ')).Trim()
}

function Slugify([string]$text, [int]$limit = 80) {
  $text = (To-StringOrEmpty $text).ToLowerInvariant()
  $text = [regex]::Replace($text, '[^a-z0-9]+', '-').Trim('-')
  if ($text.Length -gt $limit) { $text = $text.Substring(0, $limit) }
  $text = $text.Trim('-')
  if ([string]::IsNullOrWhiteSpace($text)) { return 'paper' }
  return $text
}

function Tag-Slug([string]$text) {
  return (Slugify $text 80).Replace('-', '_')
}

function Safe-Int([object]$value, [int]$default = 0) {
  try {
    $s = (To-StringOrEmpty $value).Trim()
    if ($s -eq '') { return $default }
    return [int]$s
  } catch {
    return $default
  }
}

function Yaml-String([string]$value) {
  $value = (To-StringOrEmpty $value).Replace('\', '\\').Replace('"', '\"')
  return '"' + $value + '"'
}

function Yaml-Sequence([string]$key, [object[]]$values) {
  $cleaned = @()
  foreach ($value in $values) {
    $text = (To-StringOrEmpty $value).Trim()
    if ($text) { $cleaned += $text }
  }
  if ($cleaned.Count -eq 0) { return @("${key}: []") }
  $lines = @("${key}:")
  foreach ($value in $cleaned) { $lines += "  - $(Yaml-String $value)" }
  return $lines
}

function Markdown-Escape([string]$value) {
  return (To-StringOrEmpty $value).Replace('|', '\|').Replace('$', '\$').Replace("`n", ' ')
}

function Wiki-Target-Escape([string]$value) {
  return (To-StringOrEmpty $value).Replace('|', '\|').Replace('$', '\$').Replace("`n", ' ')
}

function Resolve-Library-Path([string]$pathText) {
  $pathText = To-StringOrEmpty $pathText
  if ([System.IO.Path]::IsPathRooted($pathText)) {
    return [System.IO.Path]::GetFullPath($pathText)
  }
  return [System.IO.Path]::GetFullPath((Join-Path $script:ROOT $pathText))
}

function Library-Relative-Path([string]$pathText) {
  $pathText = To-StringOrEmpty $pathText
  $fullPath = Resolve-Library-Path $pathText
  $rootPath = ([System.IO.Path]::GetFullPath($script:ROOT)).TrimEnd('\', '/')
  if ($fullPath.StartsWith($rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $fullPath.Substring($rootPath.Length).TrimStart('\', '/') -replace '\\','/'
  }
  return $pathText -replace '\\','/'
}

function Wiki-Link([string]$target, [string]$alias, [switch]$Table) {
  $separator = if ($Table) { '\|' } else { '|' }
  return "[[$(Wiki-Target-Escape $target)$separator$(Markdown-Escape $alias)]]"
}

function Note-Link-Target([string]$notePath) {
  $target = Library-Relative-Path $notePath
  if ($target.ToLowerInvariant().EndsWith('.md')) {
    $target = $target.Substring(0, $target.Length - 3)
  }
  return $target
}

function Parse-Filename([string]$filename) {
  $stem = [System.IO.Path]::GetFileNameWithoutExtension($filename)
  $m = [regex]::Match($stem, '^(?<year>\d{4})_(?<rest>.+)$')
  $year = if ($m.Success) { $m.Groups['year'].Value } else { '' }
  $rest = if ($m.Success) { $m.Groups['rest'].Value } else { $stem }

  $rest = Normalize-Space ($rest -replace '_', ' ')

  $regexOpts = [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
  $supplementary = [regex]::IsMatch($rest, '\b(supplementary material|supplemental|supplementary)\b', $regexOpts)
  $rest = Normalize-Space ([regex]::Replace($rest, '\b(supplementary material|supplemental|supplementary)\b', '', $regexOpts))

  $venue = ''
  $parts = $rest -split ' '
  if ($parts.Count -ge 2) {
    $maybeVenue = $parts[-1].ToUpperInvariant().Replace('-', '')
    if ($script:VENUES -contains $maybeVenue) {
      $venue = $maybeVenue
      $rest = Normalize-Space (($parts[0..($parts.Count - 2)] -join ' '))
    }
  }
  if ($stem -like '*ICLR Reject*') { $venue = 'ICLR Reject' }

  $title = Normalize-Space $rest
  if ($supplementary) { $title = "$title (supplement)" }

  return [ordered]@{
    year = $year
    title = $title
    venue = $venue
    is_supplement = if ($supplementary) { 'yes' } else { '' }
  }
}

function Tag-Title([string]$title) {
  $low = (To-StringOrEmpty $title).ToLowerInvariant()
  $tags = New-Object System.Collections.Generic.List[string]

  function Add-Tag([string]$name, [string[]]$needles) {
    foreach ($needle in $needles) {
      if ($low.Contains($needle)) { $tags.Add($name); break }
    }
  }

  Add-Tag 'survey' @('survey','prospects')
  Add-Tag 'benchmark' @('benchmark','vbench')
  Add-Tag 'dataset' @('dataset','datacuration','data curation','captions')
  Add-Tag 'text-to-video' @('text-to-video','text to video','t2v','video generation','video generative','cogvideo')
  Add-Tag 'image-to-video' @('image-to-video','image to video','i2v','real image animation')
  Add-Tag 'long-video' @('long video','minute-long','infinite videos','longer video','streamingt2v')
  Add-Tag 'text-to-image' @('text-to-image','text to image','t2i','image synthesis','image generation','image generative','cogview')
  Add-Tag 'diffusion-flow' @('diffusion','rectified flow','flow matching','denoising')
  Add-Tag 'autoregressive' @('autoregressive','next-scale','next-frame')
  Add-Tag 'visual-tokenizer-vq' @('vector quant','vq-','vqgan','vq-vae','codebook','tokenizer','quantization')
  Add-Tag 'inpainting-completion' @('inpainting','completion')
  Add-Tag 'prompt-alignment' @('prompt','text understanding','semantic fidelity','attention regulation')
  Add-Tag 'vision-representation' @('beit','dinov2','efficientvit','masked autoencoders','vision transformer','navit')
  Add-Tag 'audio-speech' @('audio','soundstream','speech')
  Add-Tag 'time-series' @('time series','forecaster')
  Add-Tag 'protein' @('protein')

  $tagSet = $tags | Sort-Object -Unique
  $subfield =
    if ($tagSet -contains 'survey') { 'Survey / Roadmap' }
    elseif ($tagSet -contains 'benchmark') { 'Benchmark / Evaluation' }
    elseif ($tagSet -contains 'dataset') { 'Dataset / Data Curation' }
    elseif ($tagSet -contains 'long-video') { 'Long Video Generation' }
    elseif (($tagSet -contains 'text-to-video') -or ($tagSet -contains 'image-to-video')) { 'Video Generation' }
    elseif ($tagSet -contains 'inpainting-completion') { 'Image Inpainting / Completion' }
    elseif ($tagSet -contains 'visual-tokenizer-vq') { 'Visual Tokenization / Vector Quantization' }
    elseif ($tagSet -contains 'autoregressive') { 'Autoregressive Visual Generation' }
    elseif ($tagSet -contains 'diffusion-flow') { 'Diffusion / Flow Models' }
    elseif ($tagSet -contains 'text-to-image') { 'Text-to-Image Generation' }
    elseif ($tagSet -contains 'vision-representation') { 'Vision Representation / Architecture' }
    elseif ($tagSet -contains 'audio-speech') { 'Audio / Speech Generation' }
    elseif ($tagSet -contains 'time-series') { 'Time Series Modeling' }
    elseif ($tagSet -contains 'protein') { 'Protein / Biosequence Modeling' }
    else { 'Other Generative Modeling' }

  return @{
    subfield = $subfield
    tags = $tagSet
  }
}

function Infer-Importance([string]$title, [string]$year, [string]$venue, [string]$subfield, [string[]]$tags) {
  $low = (To-StringOrEmpty $title).ToLowerInvariant()
  $tagSet = if ($null -ne $tags) { @($tags) } else { @() }
  $score = 3
  $reasons = New-Object System.Collections.Generic.List[string]

  foreach ($key in $script:LANDMARK_PATTERNS.Keys) {
    if ($low.Contains($key)) { $score += 2; $reasons.Add($script:LANDMARK_PATTERNS[$key]); break }
  }

  $venueUp = (To-StringOrEmpty $venue).ToUpperInvariant()
  if (@('CVPR','ICCV','ECCV','ICLR','ICML','NEURIPS','NIPS','PMLR') -contains $venueUp) {
    $score += 1
    $reasons.Add("top venue: $venue")
  }

  if ($tagSet | Where-Object { $_ -in @('survey','benchmark','dataset') } | Select-Object -First 1) {
    $score += 1
    $reasons.Add('useful as overview/evaluation/data reference')
  }

  try {
    if (([int]$year) -ge 2024 -and @('Video Generation','Long Video Generation') -contains (To-StringOrEmpty $subfield)) {
      $score += 1
      $reasons.Add('recent video-generation work')
    }
  } catch { }

  if ($low.Contains('supplement')) { $score -= 2; $reasons.Add('supplementary material') }
  if ((To-StringOrEmpty $venue).ToLowerInvariant().Contains('reject')) { $score -= 1; $reasons.Add('marked as rejected version') }
  if (@('Time Series Modeling','Protein / Biosequence Modeling') -contains (To-StringOrEmpty $subfield)) {
    $score -= 1
    $reasons.Add('peripheral to the main visual generation library')
  }

  if ($score -lt 1) { $score = 1 }
  if ($score -gt 5) { $score = 5 }
  if ($reasons.Count -eq 0) { $reasons.Add('baseline relevance from filename heuristics') }
  return @{ importance = $score; reason = ($reasons -join '; ') }
}

function Read-Index {
  if (-not (Test-Path $script:INDEX_PATH)) { return @() }
  return Import-Csv -Path $script:INDEX_PATH
}

function Normalize-Row([hashtable]$row) {
  $out = [ordered]@{}
  foreach ($field in $script:CSV_FIELDS) {
    if ($row.ContainsKey($field) -and $null -ne $row[$field]) { $out[$field] = '' + $row[$field] }
    else { $out[$field] = '' }
  }
  return $out
}

function Csv-Escape([string]$value) {
  $value = To-StringOrEmpty $value
  if ($value -match '[,"\r\n]') {
    return '"' + ($value -replace '"','""') + '"'
  }
  return $value
}

function Write-Index([object[]]$rows) {
  $normalized = foreach ($row in $rows) {
    $h = @{}
    foreach ($field in $script:CSV_FIELDS) {
      $val = $null
      if ($row -is [System.Collections.IDictionary] -and $row.Contains($field)) { $val = $row[$field] }
      else {
        $member = $row | Get-Member -Name $field -MemberType NoteProperty,Property -ErrorAction SilentlyContinue
        if ($null -ne $member) { $val = $row.$field }
      }
      $h[$field] = To-StringOrEmpty $val
    }
    $h
  }

  $sorted = $normalized | Sort-Object -Property @{
      Expression = { ($script:READ_DONE -contains ((To-StringOrEmpty $_.read_status).Trim().ToLowerInvariant())) }
      Ascending = $true
    }, @{
      Expression = { - (Safe-Int $_.importance 0) }
      Ascending = $true
    }, @{
      Expression = { $_.year }
      Ascending = $true
    }, @{
      Expression = { $_.title }
      Ascending = $true
    }

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add(($script:CSV_FIELDS -join ','))
  foreach ($row in $sorted) {
    $lines.Add((($script:CSV_FIELDS | ForEach-Object { Csv-Escape ($row[$_]) }) -join ','))
  }
  $enc = New-Object System.Text.UTF8Encoding($true)
  [System.IO.File]::WriteAllLines($script:INDEX_PATH, $lines, $enc)
}

function Scan([switch]$refreshClassification) {
  $existing = @{}
  foreach ($row in (Read-Index)) { if ($row.paper_id) { $existing[$row.paper_id] = $row } }
  $now = Get-Today
  $rows = New-Object System.Collections.Generic.List[hashtable]
  $seen = New-Object System.Collections.Generic.HashSet[string]

  $searchDir = if (Test-Path $script:PAPER_DIR) { $script:PAPER_DIR } else { $script:ROOT }
  $pdfs = Get-ChildItem -Path $searchDir -Filter *.pdf -File | Sort-Object { $_.Name.ToLowerInvariant() }

  foreach ($pdf in $pdfs) {
    $parsed = Parse-Filename $pdf.Name
    $paperIdBase = if ($parsed.year) { "$($parsed.year)-$(Slugify $parsed.title)" } else { (Slugify $parsed.title) }
    $paperId = $paperIdBase
    $counter = 2
    while ($seen.Contains($paperId)) { $paperId = "$paperIdBase-$counter"; $counter++ }
    $seen.Add($paperId) | Out-Null

    $row = @{}
    if ($existing.ContainsKey($paperId)) {
      foreach ($field in $script:CSV_FIELDS) { $row[$field] = To-StringOrEmpty $existing[$paperId].$field }
      $isNew = $false
    } else {
      foreach ($field in $script:CSV_FIELDS) { $row[$field] = '' }
      $isNew = $true
    }

    $tagInfo = Tag-Title $parsed.title
    $subfieldGuess = $tagInfo.subfield
    $tagsGuess = $tagInfo.tags
    $importanceInfo = Infer-Importance $parsed.title $parsed.year $parsed.venue $subfieldGuess $tagsGuess

    $row.paper_id = $paperId
    $row.filename = $pdf.Name
    $row.title = $parsed.title
    $row.year = $parsed.year
    $row.venue = $parsed.venue
    $row.file_size_mb = ('{0:N2}' -f ($pdf.Length / 1MB))
    $row.updated_on = $now

    if ($isNew) {
      $row.added_on = $now
      if (-not $row.read_status) { $row.read_status = 'unread' }
      if (-not $row.note_status) { $row.note_status = '' }
    }

    if ($refreshClassification -or -not $row.subfield) { $row.subfield = $subfieldGuess }
    if ($refreshClassification -or -not $row.keywords) { $row.keywords = ($tagsGuess -join ', ') }
    if ($refreshClassification -or -not $row.importance) { $row.importance = '' + $importanceInfo.importance }
    if ($refreshClassification -or -not $row.importance_reason) { $row.importance_reason = $importanceInfo.reason }

    $rows.Add((Normalize-Row $row)) | Out-Null
  }

  Write-Index $rows
  return $rows
}

function Unread-Candidates([object[]]$rows) {
  $candidates = @()
  foreach ($row in $rows) {
    $readStatus = (To-StringOrEmpty $row.read_status).Trim().ToLowerInvariant()
    $noteStatus = (To-StringOrEmpty $row.note_status).Trim().ToLowerInvariant()
    if ($script:READ_DONE -contains $readStatus) { continue }
    if ($script:NOTE_DONE -contains $noteStatus) { continue }
    $candidates += $row
  }
  return $candidates
}

function Pick([switch]$reserve) {
  $rows = Scan -refreshClassification:$false
  $candidates = Unread-Candidates $rows
  if ($candidates.Count -eq 0) { return $null }

  $sorted = $candidates | Sort-Object -Property @{
      Expression = { Safe-Int $_.importance 0 }
      Ascending = $false
    }, @{
      Expression = { Safe-Int $_.year 0 }
      Ascending = $false
    }, @{
      Expression = { -not [bool]($_.selected_on) }
      Ascending = $false
    }, @{
      Expression = { $_.title }
      Ascending = $true
    }

  $selected = $sorted[0]
  if ($reserve) {
    foreach ($row in $rows) {
      if ($row.paper_id -eq $selected.paper_id) {
        $row.selected_on = Get-Today
        $row.note_status = 'reserved'
        $row.updated_on = Get-Today
        $selected = $row
        break
      }
    }
    Write-Index $rows
  }
  return $selected
}

function Find-Row([string]$paperId) {
  $rows = Scan -refreshClassification:$false
  foreach ($row in $rows) { if ($row.paper_id -eq $paperId) { return $row } }
  throw "Paper id not found: $paperId"
}

function Replace-Frontmatter([string]$content, [string]$frontmatter) {
  $body = $content.TrimStart([char]0xFEFF).Replace("`r`n", "`n")
  while ($body.StartsWith("---`n")) {
    $end = $body.IndexOf("`n---`n", 4)
    if ($end -lt 0) { break }
    $body = ($body.Substring($end + ("`n---`n").Length)).TrimStart("`n")
  }
  return $frontmatter + $body
}

function Parse-Frontmatter([string]$content) {
  $normalized = $content.TrimStart([char]0xFEFF)
  if (-not $normalized.StartsWith("---`n")) { return @{} }
  $end = $normalized.IndexOf("`n---`n", 4)
  if ($end -lt 0) { return @{} }
  $meta = @{}
  foreach ($line in $normalized.Substring(4, $end - 4).Split("`n")) {
    $l = $line.TrimEnd("`r")
    if ($l -eq '') { continue }
    if ($l.StartsWith(' ')) { continue }
    if ($l -notmatch ':') { continue }
    $parts = $l.Split(':', 2)
    $key = $parts[0].Trim()
    $value = $parts[1].Trim()
    if ($value -eq 'null') { $value = '' }
    if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) {
      $inner = $value.Substring(1, $value.Length - 2)
      $inner = $inner.Replace('\"','"').Replace('\\\\','\')
      $value = $inner
    }
    $meta[$key] = $value
  }
  return $meta
}

function Row-Tags([object]$row) {
  $tags = New-Object System.Collections.Generic.List[string]
  $tags.Add('paper')
  if ($row.subfield) { $tags.Add("paper/$(Tag-Slug $row.subfield)") }
  if ($row.venue) { $tags.Add("venue/$(Tag-Slug $row.venue)") }
  if ($row.year) { $tags.Add("year/$($row.year)") }
  $keywords = @()
  foreach ($part in (To-StringOrEmpty $row.keywords).Split(',')) {
    $p = $part.Trim()
    if ($p) { $keywords += $p }
  }
  foreach ($keyword in ($keywords | Select-Object -First 6)) { $tags.Add("topic/$(Tag-Slug $keyword)") }
  return ($tags | Sort-Object -Unique)
}

function Frontmatter-For-Row([object]$row) {
  $tagLines = (Row-Tags $row | ForEach-Object { "  - $_" }) -join "`n"
  if (-not $tagLines) { $tagLines = "  - paper" }
  $pdfLink = "[[$(Wiki-Target-Escape "papers/$($row.filename)")]]"
  $paperLink = "[[$(Wiki-Target-Escape "literature/papers/$($row.paper_id)")]]"
  $lines = @('---', 'type: paper-note')
  $lines += (Yaml-Sequence 'aliases' @($row.title))
  $lines += @(
    "paper_id: $(Yaml-String $row.paper_id)",
    "title: $(Yaml-String $row.title)",
    ('year: ' + ($(if ($row.year) { $row.year } else { 'null' }))),
    "venue: $(Yaml-String $row.venue)",
    "subfield: $(Yaml-String $row.subfield)"
  )
  $lines += (Yaml-Sequence 'topics' @(Split-Keywords $row))
  $lines += @(
    ('importance: ' + (Safe-Int $row.importance 0)),
    "read_status: $(Yaml-String $row.read_status)",
    "note_status: $(Yaml-String $row.note_status)",
    "selected_on: $(Yaml-String $row.selected_on)",
    "last_reviewed_on: $(Yaml-String $row.last_reviewed_on)",
    "paper: $(Yaml-String $paperLink)",
    "pdf: $(Yaml-String $pdfLink)",
    'tags:',
    $tagLines,
    '---',
    ''
  )
  return ($lines -join "`n")
}

function Apply-Note-Frontmatter-To-Row([hashtable]$row) {
  if (-not $row.note_path) { return $false }
  $path = Resolve-Library-Path $row.note_path
  if (-not (Test-Path $path) -or ([System.IO.Path]::GetExtension($path).ToLowerInvariant() -ne '.md')) { return $false }
  $content = Get-Content -Path $path -Raw -Encoding utf8
  $meta = Parse-Frontmatter $content
  if ($meta.Count -eq 0) { return $false }

  $changed = $false
  foreach ($field in @('read_status','importance','subfield','venue','selected_on','last_reviewed_on')) {
    if (-not $meta.ContainsKey($field)) { continue }
    $value = $meta[$field]
    if ($field -eq 'importance') { $value = '' + (Safe-Int $value (Safe-Int $row.importance 0)) }
    $current = ''
    if ($row.ContainsKey($field) -and $null -ne $row[$field]) { $current = '' + $row[$field] }
    if ($value -ne $current) { $row[$field] = $value; $changed = $true }
  }

  if ($changed) {
    $rs = (To-StringOrEmpty $row.read_status).ToLowerInvariant()
    if (($script:READ_DONE -contains $rs) -and -not $row.last_reviewed_on) {
      $row.last_reviewed_on = Get-Today
    }
    $row.updated_on = Get-Today
  }
  return $changed
}

function Sync-Note-Frontmatter([object]$row) {
  if (-not $row.note_path) { return $false }
  $path = Resolve-Library-Path $row.note_path
  if (-not (Test-Path $path) -or ([System.IO.Path]::GetExtension($path).ToLowerInvariant() -ne '.md')) { return $false }
  $content = Get-Content -Path $path -Raw -Encoding utf8
  $updated = Replace-Frontmatter $content (Frontmatter-For-Row $row)
  if ($updated -ne $content) {
    Set-Content -Path $path -Value $updated -Encoding utf8
  }
  return $true
}

function Note-Wiki-Link([object]$row, [switch]$Table) {
  if (-not $row.note_path) { return '' }
  $title = if ($row.title) { $row.title } elseif ($row.paper_id) { $row.paper_id } else { 'note' }
  return (Wiki-Link (Note-Link-Target $row.note_path) $title -Table:$Table)
}

function Paper-Wiki-Link([object]$row, [switch]$Table) {
  $paperId = To-StringOrEmpty $row.paper_id
  $title = if ($row.title) { $row.title } elseif ($paperId) { $paperId } else { 'paper' }
  if (-not $paperId) { return (Markdown-Escape $title) }
  return (Wiki-Link "literature/papers/$paperId" $title -Table:$Table)
}

function Pdf-Wiki-Link([object]$row, [switch]$Table) {
  if (-not $row.filename) { return '' }
  return (Wiki-Link "papers/$($row.filename)" 'PDF' -Table:$Table)
}

function Field-Wiki-Link([string]$subfield, [switch]$Table) {
  if (-not $subfield) { return '' }
  return (Wiki-Link "literature/fields/$(Slugify $subfield)" $subfield -Table:$Table)
}

function Year-Wiki-Link([string]$year, [switch]$Table) {
  if (-not $year) { return '' }
  return (Wiki-Link "literature/years/$(Slugify $year)" $year -Table:$Table)
}

function Venue-Wiki-Link([string]$venue, [switch]$Table) {
  if (-not $venue) { return '' }
  return (Wiki-Link "literature/venues/$(Slugify $venue)" $venue -Table:$Table)
}

function Topic-Wiki-Link([string]$topic, [switch]$Table) {
  if (-not $topic) { return '' }
  return (Wiki-Link "literature/topics/$(Slugify $topic)" $topic -Table:$Table)
}

function Split-Keywords([object]$row) {
  $out = @()
  foreach ($part in (To-StringOrEmpty $row.keywords).Split(',')) {
    $p = $part.Trim()
    if ($p) { $out += $p }
  }
  return $out
}

function Sort-Rows-For-Reading([object[]]$rows) {
  return $rows | Sort-Object -Property @{
      Expression = { $script:READ_DONE -contains ((To-StringOrEmpty $_.read_status).ToLowerInvariant()) }
      Ascending = $true
    }, @{
      Expression = { - (Safe-Int $_.importance 0) }
      Ascending = $true
    }, @{
      Expression = { - (Safe-Int $_.year 0) }
      Ascending = $true
    }, @{
      Expression = { $_.title }
      Ascending = $true
    }
}

function Markdown-Table([object[]]$rows, [switch]$IncludeNote) {
  if ($IncludeNote) {
    $header = '| 论文 | 年份 | 领域 | 重要度 | 状态 | 笔记 | PDF |'
    $divider = '|---|---:|---|---:|---|---|---|'
  } else {
    $header = '| 论文 | 年份 | 领域 | 重要度 | 状态 | PDF |'
    $divider = '|---|---:|---|---:|---|---|'
  }
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add($header)
  $lines.Add($divider)
  foreach ($row in $rows) {
    $title = Paper-Wiki-Link $row -Table
    $year = Year-Wiki-Link (To-StringOrEmpty $row.year) -Table
    $subfield = Field-Wiki-Link (To-StringOrEmpty $row.subfield) -Table
    $importance = Markdown-Escape (To-StringOrEmpty $row.importance)
    $status = Markdown-Escape (To-StringOrEmpty $row.read_status)
    $pdf = Pdf-Wiki-Link $row -Table
    if ($IncludeNote) {
      $note = Note-Wiki-Link $row -Table
      $lines.Add("| $title | $year | $subfield | $importance | $status | $note | $pdf |")
    } else {
      $lines.Add("| $title | $year | $subfield | $importance | $status | $pdf |")
    }
  }
  return ($lines -join "`n")
}

function Write-Markdown([string]$path, [string]$content) {
  $dir = Split-Path -Parent $path
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

  $payload = (($content.TrimEnd()) + "`n")
  $attempts = 12
  $delayMs = 250
  for ($i = 1; $i -le $attempts; $i++) {
    $tmp = $null
    try {
      $tmp = Join-Path $dir ('.tmp.' + ([guid]::NewGuid().ToString('N')) + '.' + (Split-Path -Leaf $path))
      Set-Content -Path $tmp -Value $payload -Encoding utf8
      Move-Item -LiteralPath $tmp -Destination $path -Force
      return
    } catch {
      if ($tmp -and (Test-Path $tmp)) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
      if ($i -eq $attempts) { throw }
      Start-Sleep -Milliseconds $delayMs
      if ($delayMs -lt 2000) { $delayMs = [Math]::Min(2000, $delayMs * 2) }
    }
  }
}

function Literature-Frontmatter([string]$pageType) {
  return @(
    '---',
    "type: $pageType",
    "generated_on: $(Get-Today)",
    'tags:',
    '  - paper-library',
    '---',
    ''
  ) -join "`n"
}

function Paper-Frontmatter([object]$row) {
  $tagLines = (Row-Tags $row | ForEach-Object { "  - $_" }) -join "`n"
  if (-not $tagLines) { $tagLines = "  - paper" }
  $pdfLink = "[[$(Wiki-Target-Escape "papers/$($row.filename)")]]"
  $lines = @('---', 'type: paper')
  $lines += (Yaml-Sequence 'aliases' @($row.title))
  $lines += @(
    "paper_id: $(Yaml-String $row.paper_id)",
    "title: $(Yaml-String $row.title)",
    ('year: ' + ($(if ($row.year) { $row.year } else { 'null' }))),
    "venue: $(Yaml-String $row.venue)",
    "subfield: $(Yaml-String $row.subfield)"
  )
  $lines += (Yaml-Sequence 'topics' @(Split-Keywords $row))
  $lines += @(
    ('importance: ' + (Safe-Int $row.importance 0)),
    "read_status: $(Yaml-String $row.read_status)",
    "note_status: $(Yaml-String $row.note_status)",
    "added_on: $(Yaml-String $row.added_on)",
    "updated_on: $(Yaml-String $row.updated_on)",
    "pdf: $(Yaml-String $pdfLink)",
    'tags:',
    $tagLines,
    '---',
    ''
  )
  return ($lines -join "`n")
}

function Topic-Frequency([object[]]$rows) {
  $counts = @{}
  foreach ($r in $rows) {
    foreach ($kw in (Split-Keywords $r)) {
      if ($counts.ContainsKey($kw)) { $counts[$kw] = [int]$counts[$kw] + 1 }
      else { $counts[$kw] = 1 }
    }
  }
  return $counts
}

function Graph-Topic-Weight([string]$keyword, [hashtable]$counts) {
  $count = 0
  if ($counts.ContainsKey($keyword)) { $count = [int]$counts[$keyword] }
  if ($count -le 3) { return 5 }
  if ($count -le 8) { return 4 }
  if ($count -le 20) { return 2 }
  return 1
}

function Related-Papers([object]$row, [object[]]$rows, [int]$limit = 4) {
  $rowKeywords = [System.Collections.Generic.HashSet[string]]::new()
  foreach ($kw in @(Split-Keywords $row)) { $rowKeywords.Add([string]$kw) | Out-Null }
  $keywordCounts = Topic-Frequency $rows
  $ignored = [System.Collections.Generic.HashSet[string]]::new([string[]]@('with','from','using','based','model','models','generation','generative','image','video','text','towards','learning','large','scale','open','paper'))
  $rowTitleTerms = [System.Collections.Generic.HashSet[string]]::new()
  foreach ($m in [regex]::Matches((To-StringOrEmpty $row.title).ToLowerInvariant(), '[a-z0-9]+')) {
    $term = $m.Value
    if ($term.Length -ge 4 -and -not $ignored.Contains($term)) { $rowTitleTerms.Add($term) | Out-Null }
  }

  $scored = New-Object System.Collections.Generic.List[object]
  foreach ($cand in $rows) {
    if ($cand.paper_id -eq $row.paper_id) { continue }
    $candKeywords = [System.Collections.Generic.HashSet[string]]::new()
    foreach ($kw in @(Split-Keywords $cand)) { $candKeywords.Add([string]$kw) | Out-Null }
    $candTitleTerms = [System.Collections.Generic.HashSet[string]]::new()
    foreach ($m in [regex]::Matches((To-StringOrEmpty $cand.title).ToLowerInvariant(), '[a-z0-9]+')) {
      $term = $m.Value
      if ($term.Length -ge 4 -and -not $ignored.Contains($term)) { $candTitleTerms.Add($term) | Out-Null }
    }
    $score = 0
    if ($row.subfield -and ($row.subfield -eq $cand.subfield)) { $score += 2 }
    foreach ($kw in $rowKeywords) {
      if ($candKeywords.Contains($kw)) { $score += (Graph-Topic-Weight $kw $keywordCounts) }
    }
    $overlapTitle = 0
    foreach ($t in $rowTitleTerms) { if ($candTitleTerms.Contains($t)) { $overlapTitle++ } }
    if ($overlapTitle -gt 3) { $overlapTitle = 3 }
    $score += $overlapTitle
    if ($score -lt 3) { continue }
    $scored.Add([pscustomobject]@{
      score = $score
      importance = Safe-Int $cand.importance 0
      year = Safe-Int $cand.year 0
      title = $cand.title
      row = $cand
    }) | Out-Null
  }
  $scoredSorted = $scored | Sort-Object -Property @{Expression='score';Descending=$true}, @{Expression='importance';Descending=$true}, @{Expression='year';Descending=$true}, @{Expression='title';Descending=$false}
  return @($scoredSorted | Select-Object -First $limit | ForEach-Object { $_.row })
}

function Paper-Page-Content([object]$row, [object[]]$rows) {
  $keywords = @(Split-Keywords $row)
  $topicLinks = if ($keywords.Count -gt 0) { ($keywords | ForEach-Object { Topic-Wiki-Link $_ } | Where-Object { $_ }) -join ', ' } else { '-' }
  $note = (Note-Wiki-Link $row); if (-not $note) { $note = '-' }
  $venue = (Venue-Wiki-Link (To-StringOrEmpty $row.venue)); if (-not $venue) { $venue = '-' }
  $related = Related-Papers $row $rows 4
  $relatedLines = @()
  foreach ($cand in $related) {
    $candVenue = if ($cand.venue) { $cand.venue } else { 'no venue' }
    $relatedLines += "- $(Paper-Wiki-Link $cand) ($($cand.year), $candVenue)"
  }
  if ($relatedLines.Count -eq 0) { $relatedLines = @('- No close local match yet.') }

  $pageTitleRaw = if ($row.title) { $row.title } elseif ($row.paper_id) { $row.paper_id } else { 'Paper' }
  $pageTitle = Markdown-Escape $pageTitleRaw
  $fieldLink = Field-Wiki-Link (To-StringOrEmpty $row.subfield); if (-not $fieldLink) { $fieldLink = '-' }
  $yearLink = Year-Wiki-Link (To-StringOrEmpty $row.year); if (-not $yearLink) { $yearLink = '-' }
  $pdfLink = Pdf-Wiki-Link $row; if (-not $pdfLink) { $pdfLink = '-' }
  $importanceValue = if ($row.importance) { $row.importance } else { '-' }
  $statusValue = if ($row.read_status) { $row.read_status } else { '-' }
  $addedValue = if ($row.added_on) { $row.added_on } else { '-' }
  $updatedValue = if ($row.updated_on) { $row.updated_on } else { '-' }
  $reasonValue = if ($row.importance_reason) { $row.importance_reason } else { 'No local rationale recorded yet.' }

  return (Paper-Frontmatter $row) +
    "# $pageTitle`n`n" +
    "## Metadata`n`n" +
    "- Field: $fieldLink`n" +
    "- Year: $yearLink`n" +
    "- Venue: $venue`n" +
    "- Topics: $topicLinks`n" +
    "- Importance: $importanceValue / 5`n" +
    "- Status: $statusValue`n" +
    "- Note: $note`n" +
    "- PDF: $pdfLink`n" +
    "- Added: $addedValue`n" +
    "- Updated: $updatedValue`n`n" +
    "## Why It Is Here`n`n" +
    "$reasonValue`n`n" +
    "## Related Papers`n`n" +
    ($relatedLines -join "`n") + "`n"
}

function Write-Group-Index([string]$directory, [string]$key, [object[]]$rows, [string]$pageType, [string]$title, [string[]]$extraLinks) {
  $extra = ''
  if ($extraLinks -and $extraLinks.Count -gt 0) { $extra = (($extraLinks | Where-Object { $_ } | ForEach-Object { "- $_" }) -join "`n") + "`n`n" }
  $content = (Literature-Frontmatter $pageType) + "# $title`n`n" + $extra + "- Papers: $($rows.Count)`n`n" + (Markdown-Table (Sort-Rows-For-Reading $rows) -IncludeNote)
  Write-Markdown (Join-Path $directory "$(Slugify $key).md") $content
}

function Graph-Paper-Line([object]$row) {
  $keywords = @(Split-Keywords $row)
  $topics = @($keywords | Select-Object -First 3 | ForEach-Object { Topic-Wiki-Link $_ } | Where-Object { $_ })
  $topicText = if ($topics.Count -gt 0) { $topics -join ', ' } else { '-' }
  $venue = if ($row.venue) { $row.venue } else { 'no venue' }
  $year = if ($row.year) { $row.year } else { 'no year' }
  $field = Field-Wiki-Link (To-StringOrEmpty $row.subfield)
  if (-not $field) { $field = '-' }
  return "- $(Paper-Wiki-Link $row) ($year, $venue; $field; $topicText)"
}

function Graph-Overview-Content([object[]]$rows, [hashtable]$bySubfield, [hashtable]$byTopic) {
  $fieldLines = @()
  foreach ($entry in ($bySubfield.GetEnumerator() | Sort-Object @{Expression={$_.Value.Count};Descending=$true}, @{Expression={$_.Key};Descending=$false})) {
    $fieldLines += "- $(Field-Wiki-Link $entry.Key) ($($entry.Value.Count) 篇)"
  }

  $topicLines = @()
  foreach ($entry in ($byTopic.GetEnumerator() | Sort-Object @{Expression={$_.Value.Count};Descending=$true}, @{Expression={$_.Key};Descending=$false} | Select-Object -First 18)) {
    $topicLines += "- $(Topic-Wiki-Link $entry.Key) ($($entry.Value.Count) 篇)"
  }

  $anchors = @(Sort-Rows-For-Reading @($rows | Where-Object { (Safe-Int $_.importance 0) -ge 5 }) | Select-Object -First 18)
  $anchorLines = @($anchors | ForEach-Object { Graph-Paper-Line $_ })

  return (Literature-Frontmatter 'paper-graph-map') +
    "# 论文知识图谱总览`n`n" +
    "这页是全局关系图谱的主入口：领域负责分层，主题负责横向连接，重点论文负责把图谱锚定到具体工作。`n`n" +
    "## 主干领域`n`n" +
    ($fieldLines -join "`n") + "`n`n" +
    "## 主题轴`n`n" +
    ($topicLines -join "`n") + "`n`n" +
    "## 核心论文锚点`n`n" +
    ($anchorLines -join "`n") + "`n`n" +
    "## 其他图谱页`n`n" +
    "- [[literature/maps/topic-bridges|跨主题桥接]]`n" +
    "- [[literature/maps/research-frontier|重点阅读路线]]`n"
}

function Graph-Topic-Bridge-Content([object[]]$rows, [hashtable]$byTopic) {
  $topicLines = @()
  foreach ($entry in ($byTopic.GetEnumerator() | Sort-Object @{Expression={$_.Value.Count};Descending=$true}, @{Expression={$_.Key};Descending=$false} | Select-Object -First 24)) {
    $fields = @($entry.Value | ForEach-Object { $_.subfield } | Where-Object { $_ } | Sort-Object -Unique | Select-Object -First 5 | ForEach-Object { Field-Wiki-Link $_ } | Where-Object { $_ })
    $fieldText = if ($fields.Count -gt 0) { $fields -join ', ' } else { '-' }
    $topicLines += "- $(Topic-Wiki-Link $entry.Key) -> $fieldText"
  }

  $bridgeRows = @($rows | Where-Object { (Safe-Int $_.importance 0) -ge 4 -and @(Split-Keywords $_).Count -ge 2 })
  $bridgeLines = @(Sort-Rows-For-Reading $bridgeRows | Select-Object -First 28 | ForEach-Object { Graph-Paper-Line $_ })
  if ($bridgeLines.Count -eq 0) { $bridgeLines = @('- No bridge papers yet.') }

  return (Literature-Frontmatter 'paper-graph-map') +
    "# 跨主题桥接`n`n" +
    "这页只保留能跨领域或跨方法连接的主题和论文，避免全局图被年份、会场和普通列表边淹没。`n`n" +
    "## 主题到领域`n`n" +
    ($topicLines -join "`n") + "`n`n" +
    "## 桥接论文`n`n" +
    ($bridgeLines -join "`n") + "`n"
}

function Graph-Frontier-Content([hashtable]$bySubfield) {
  $sections = @()
  foreach ($sf in ($bySubfield.Keys | Sort-Object)) {
    $priority = @(Sort-Rows-For-Reading $bySubfield[$sf] | Where-Object { (Safe-Int $_.importance 0) -ge 4 } | Select-Object -First 6)
    if ($priority.Count -eq 0) { continue }
    $sections += "## $sf`n"
    $sections += @($priority | ForEach-Object { Graph-Paper-Line $_ })
    $sections += ''
  }

  return (Literature-Frontmatter 'paper-graph-map') +
    "# 重点阅读路线`n`n" +
    "这页按领域挑选高重要度论文，适合在图谱中从领域节点跳到代表性论文，再沿相关论文继续展开。`n`n" +
    (($sections -join "`n").TrimEnd()) + "`n"
}

function Generate-Obsidian-Pages {
  $rows = Scan -refreshClassification:$false

  foreach ($dir in @($script:OBSIDIAN_DIR, (Join-Path $script:OBSIDIAN_DIR 'fields'), $script:OBSIDIAN_PAPERS_DIR, $script:OBSIDIAN_MAPS_DIR, $script:OBSIDIAN_YEARS_DIR, $script:OBSIDIAN_VENUES_DIR, $script:OBSIDIAN_TOPICS_DIR)) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  }

  $importedNotes = 0
  $rowHash = @()
  foreach ($r in $rows) {
    $h = @{}
    foreach ($f in $script:CSV_FIELDS) { $h[$f] = To-StringOrEmpty $r.$f }
    if (Apply-Note-Frontmatter-To-Row $h) { $importedNotes++ }
    $rowHash += $h
  }
  if ($importedNotes -gt 0) {
    Write-Index $rowHash
    $rows = Scan -refreshClassification:$false
  }

  $syncedNotes = 0
  foreach ($r in $rows) { if (Sync-Note-Frontmatter $r) { $syncedNotes++ } }

  $unread = @($rows | Where-Object { -not ($script:READ_DONE -contains ((To-StringOrEmpty $_.read_status).ToLowerInvariant())) })
  $read = @($rows | Where-Object { $script:READ_DONE -contains ((To-StringOrEmpty $_.read_status).ToLowerInvariant()) })
  $noted = @($rows | Where-Object { $script:NOTE_DONE -contains ((To-StringOrEmpty $_.note_status).ToLowerInvariant()) })
  $highPriority = @($rows | Where-Object { (Safe-Int $_.importance 0) -ge 5 })

  $bySubfield = @{}
  $byYear = @{}
  $byVenue = @{}
  $byTopic = @{}
  foreach ($r in $rows) {
    $sf = if ($r.subfield) { $r.subfield } else { 'Uncategorized' }
    if (-not $bySubfield.ContainsKey($sf)) { $bySubfield[$sf] = @() }
    $bySubfield[$sf] += $r

    if ($r.year) { if (-not $byYear.ContainsKey($r.year)) { $byYear[$r.year] = @() }; $byYear[$r.year] += $r }
    if ($r.venue) { if (-not $byVenue.ContainsKey($r.venue)) { $byVenue[$r.venue] = @() }; $byVenue[$r.venue] += $r }
    foreach ($kw in (Split-Keywords $r)) { if (-not $byTopic.ContainsKey($kw)) { $byTopic[$kw] = @() }; $byTopic[$kw] += $r }
  }

  Write-Group-Index $script:OBSIDIAN_PAPERS_DIR 'index' $rows 'paper-index' 'All Papers' @()
  foreach ($r in $rows) {
    Write-Markdown (Join-Path $script:OBSIDIAN_PAPERS_DIR "$($r.paper_id).md") (Paper-Page-Content $r $rows)
  }

  foreach ($year in ($byYear.Keys | Sort-Object -Descending)) {
    Write-Group-Index $script:OBSIDIAN_YEARS_DIR $year $byYear[$year] 'paper-year-index' "$year Papers" @('[[literature/papers/index|All papers]]')
  }
  foreach ($venue in ($byVenue.Keys | Sort-Object)) {
    Write-Group-Index $script:OBSIDIAN_VENUES_DIR $venue $byVenue[$venue] 'paper-venue-index' "$venue Papers" @('[[literature/papers/index|All papers]]')
  }
  foreach ($topic in ($byTopic.Keys | Sort-Object)) {
    $topicFields = @($byTopic[$topic] | ForEach-Object { $_.subfield } | Where-Object { $_ } | Sort-Object -Unique | Select-Object -First 8 | ForEach-Object { Field-Wiki-Link $_ } | Where-Object { $_ })
    Write-Group-Index $script:OBSIDIAN_TOPICS_DIR $topic $byTopic[$topic] 'paper-topic-index' "$topic Papers" $topicFields
  }

  Write-Markdown (Join-Path $script:OBSIDIAN_MAPS_DIR 'overview.md') (Graph-Overview-Content $rows $bySubfield $byTopic)
  Write-Markdown (Join-Path $script:OBSIDIAN_MAPS_DIR 'topic-bridges.md') (Graph-Topic-Bridge-Content $rows $byTopic)
  Write-Markdown (Join-Path $script:OBSIDIAN_MAPS_DIR 'research-frontier.md') (Graph-Frontier-Content $bySubfield)

  $fieldsDir = Join-Path $script:OBSIDIAN_DIR 'fields'
  $subfieldLines = New-Object System.Collections.Generic.List[string]
  foreach ($sf in ($bySubfield.Keys | Sort-Object)) {
    $group = $bySubfield[$sf]
    $fieldUnread = @($group | Where-Object { -not ($script:READ_DONE -contains ((To-StringOrEmpty $_.read_status).ToLowerInvariant())) }).Count
    $fieldRead = @($group | Where-Object { $script:READ_DONE -contains ((To-StringOrEmpty $_.read_status).ToLowerInvariant()) }).Count
    $topicLine = ''
    $topics = @{}
    foreach ($r in $group) {
      foreach ($kw in (Split-Keywords $r)) {
        if ($topics.ContainsKey($kw)) { $topics[$kw] = [int]$topics[$kw] + 1 }
        else { $topics[$kw] = 1 }
      }
    }
    $topTopics = @($topics.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 8 | ForEach-Object { Topic-Wiki-Link $_.Key } | Where-Object { $_ })
    if ($topTopics.Count -gt 0) { $topicLine = ($topTopics -join ', ') } else { $topicLine = '-' }

    $slug = Slugify $sf
    $fieldLink = Field-Wiki-Link $sf -Table
    $subfieldLines.Add("| $fieldLink | $($group.Count) | $fieldUnread | $fieldRead |") | Out-Null

    $fieldPage = (Literature-Frontmatter 'paper-field-index') +
      "# $sf`n`n" +
      "- 总数: $($group.Count)`n" +
      "- 未读: $fieldUnread`n" +
      "- 已读: $fieldRead`n`n" +
      "- Topics: $topicLine`n`n" +
      (Markdown-Table (Sort-Rows-For-Reading $group) -IncludeNote)
    Write-Markdown (Join-Path $fieldsDir "$slug.md") $fieldPage
  }

  $indexContent = (Literature-Frontmatter 'paper-library-index') +
    "# 论文库`n`n" +
    "- 总论文数: $($rows.Count)`n" +
    "- 未读: $($unread.Count)`n" +
    "- 已读: $($read.Count)`n" +
    "- 已生成精讲笔记: $($noted.Count)`n`n" +
    "## 快速入口`n`n" +
    "- [[literature/unread|未读论文]]`n" +
    "- [[literature/read|已读论文]]`n" +
    "- [[literature/high-priority|高优先级论文]]`n" +
    "- [[literature/notes|已生成精讲笔记]]`n`n" +
    "## 知识图谱入口`n`n" +
    "- [[literature/maps/overview|论文知识图谱总览]]`n" +
    "- [[literature/maps/topic-bridges|跨主题桥接]]`n" +
    "- [[literature/maps/research-frontier|重点阅读路线]]`n`n" +
    "## 细分领域`n`n" +
    "| 领域 | 篇数 | 未读 | 已读 |`n|---|---:|---:|---:|`n" +
    ($subfieldLines -join "`n") + "`n`n" +
    "## 最近生成的精讲笔记`n`n" +
    (Markdown-Table (Sort-Rows-For-Reading $noted | Select-Object -First 12) -IncludeNote) + "`n`n" +
    "## 检索索引`n`n" +
    "- [[literature/papers/index|全部论文页]]`n" +
    "- [[literature/years/2025|2025 论文]]`n" +
    "- [[literature/venues/cvpr|CVPR 论文]]`n" +
    "- [[literature/topics/text-to-video|text-to-video 论文]]`n"
  Write-Markdown (Join-Path $script:OBSIDIAN_DIR 'index.md') $indexContent

  $unreadContent = (Literature-Frontmatter 'paper-reading-list') +
    "# 未读论文`n`n" +
    "按重要程度和年份排序。`n`n" +
    (Markdown-Table (Sort-Rows-For-Reading $unread) -IncludeNote)
  Write-Markdown (Join-Path $script:OBSIDIAN_DIR 'unread.md') $unreadContent

  $readContent = (Literature-Frontmatter 'paper-reading-list') +
    "# 已读论文`n`n" +
    (Markdown-Table (Sort-Rows-For-Reading $read) -IncludeNote)
  Write-Markdown (Join-Path $script:OBSIDIAN_DIR 'read.md') $readContent

  $hpContent = (Literature-Frontmatter 'paper-reading-list') +
    "# 高优先级论文`n`n" +
    "包含 `importance >= 5` 的论文。`n`n" +
    (Markdown-Table (Sort-Rows-For-Reading $highPriority) -IncludeNote)
  Write-Markdown (Join-Path $script:OBSIDIAN_DIR 'high-priority.md') $hpContent

  $notesContent = (Literature-Frontmatter 'paper-note-index') +
    "# 已生成精讲笔记`n`n" +
    (Markdown-Table (Sort-Rows-For-Reading $noted) -IncludeNote)
  Write-Markdown (Join-Path $script:OBSIDIAN_DIR 'notes.md') $notesContent

  $dataviewBodyLines = @(
    '# Dataview 查询示例',
    '',
    '如果 Obsidian 安装了 Dataview 插件，可以使用下面的查询。',
    '',
    '## 未读论文笔记',
    '',
    '```dataview',
    'TABLE year, venue, subfield, importance, pdf',
    'FROM "literature/papers"',
    'WHERE type = "paper" AND read_status != "read"',
    'SORT importance DESC, year DESC',
    '```',
    '',
    '## 视频生成方向',
    '',
    '```dataview',
    'TABLE year, venue, importance, read_status',
    'FROM "literature/papers"',
    'WHERE type = "paper" AND subfield = "Video Generation"',
    'SORT year DESC',
    '```',
    ''
  )
  $dataviewBody = ($dataviewBodyLines -join "`n")
  $dataviewContent = (Literature-Frontmatter 'paper-library-help') + $dataviewBody
  Write-Markdown (Join-Path $script:OBSIDIAN_DIR 'dataview.md') $dataviewContent

  return @{
    papers = $rows.Count
    unread = $unread.Count
    read = $read.Count
    notes = $noted.Count
    subfields = $bySubfield.Keys.Count
    maps = 3
    paper_pages = $rows.Count
    years = $byYear.Keys.Count
    venues = $byVenue.Keys.Count
    topics = $byTopic.Keys.Count
    imported_notes = $importedNotes
    synced_notes = $syncedNotes
  }
}

function Mark-Note([string]$paperId, [string]$notePath) {
  $rows = Scan -refreshClassification:$false
  $found = $false
  foreach ($row in $rows) {
    if ($row.paper_id -eq $paperId) {
      $row.note_status = 'generated'
      $row.note_path = Library-Relative-Path $notePath
      $row.updated_on = Get-Today
      $found = $true
      break
    }
  }
  if (-not $found) { throw "Paper id not found: $paperId" }
  Write-Index $rows
  $updatedRow = Find-Row $paperId
  Sync-Note-Frontmatter $updatedRow | Out-Null
}

function Mark-Read([string]$paperId, [string]$status) {
  $rows = Scan -refreshClassification:$false
  $found = $false
  foreach ($row in $rows) {
    if ($row.paper_id -eq $paperId) {
      $row.read_status = $status
      $row.last_reviewed_on = Get-Today
      $row.updated_on = Get-Today
      $found = $true
      break
    }
  }
  if (-not $found) { throw "Paper id not found: $paperId" }
  Write-Index $rows
  $updatedRow = Find-Row $paperId
  Sync-Note-Frontmatter $updatedRow | Out-Null
}

function Clean-Extracted-Text([string]$text) {
  $text = (To-StringOrEmpty $text).Replace([string][char]0, '')
  $lines = $text -split "`n"
  $out = New-Object System.Collections.Generic.List[string]
  $blank = $false
  foreach ($line in $lines) {
    $t = Normalize-Space ($line.TrimEnd("`r"))
    if (-not $t) {
      if (-not $blank) { $out.Add('') | Out-Null }
      $blank = $true
      continue
    }
    $out.Add($t) | Out-Null
    $blank = $false
  }
  return (($out -join "`n").Trim())
}

function Extract-Text([string]$paperId, [int]$maxChars, [string]$outputPath) {
  $row = Find-Row $paperId
  $pdfPath = Join-Path $script:PAPER_DIR $row.filename
  if (-not (Test-Path $pdfPath)) { $pdfPath = Join-Path $script:ROOT $row.filename }
  if (-not (Test-Path $pdfPath)) { throw "PDF not found: $pdfPath" }

  $info = & pdfinfo.exe $pdfPath 2>$null
  $pageCount = 0
  foreach ($line in $info) {
    if ($line -match '^Pages:\s+(?<n>\d+)\s*$') { $pageCount = [int]$Matches['n']; break }
  }
  if ($pageCount -le 0) { $pageCount = 30 }

  $chunks = New-Object System.Collections.Generic.List[string]
  for ($p = 1; $p -le $pageCount; $p++) {
    $pageText = & pdftotext.exe -f $p -l $p -q $pdfPath - 2>$null
    $pageText = ($pageText -join "`n").Trim()
    if ($pageText) { $chunks.Add("`n`n--- Page $p ---`n$pageText") | Out-Null }
    $len = 0
    foreach ($c in $chunks) { $len += $c.Length }
    if ($len -ge $maxChars) { break }
  }

  $text = Clean-Extracted-Text ($chunks -join "`n")
  if ($text.Length -gt $maxChars) { $text = $text.Substring(0, $maxChars) }

  if ($outputPath) {
    $outPath = Resolve-Path -LiteralPath (Split-Path -Parent $outputPath) -ErrorAction SilentlyContinue
    $dir = if ($outPath) { $outPath.Path } else { (Split-Path -Parent $outputPath) }
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Set-Content -Path $outputPath -Value $text -Encoding utf8
  }
  return $text
}

function Get-Image-Dimensions([string]$path) {
  Add-Type -AssemblyName System.Drawing | Out-Null
  $img = [System.Drawing.Image]::FromFile($path)
  try {
    return @{ width = $img.Width; height = $img.Height }
  } finally {
    $img.Dispose()
  }
}

function Extract-Images([string]$paperId, [string]$outputDir, [int]$maxImages, [int]$minWidth, [int]$minHeight) {
  $row = Find-Row $paperId
  $pdfPath = Join-Path $script:PAPER_DIR $row.filename
  if (-not (Test-Path $pdfPath)) { $pdfPath = Join-Path $script:ROOT $row.filename }
  if (-not (Test-Path $pdfPath)) { throw "PDF not found: $pdfPath" }

  $targetDir = if ($outputDir) { $outputDir } else { (Join-Path $script:FIGURE_DIR $paperId) }
  if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir -Force | Out-Null }

  $tmpDir = Join-Path $env:TEMP ("paper_images_" + [guid]::NewGuid().ToString('n'))
  New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

  try {
    $root = Join-Path $tmpDir 'img'
    $files = & pdfimages.exe -all -p -print-filenames -q $pdfPath $root 2>$null
    $allFiles = @()
    foreach ($line in $files) {
      $p = ('' + $line).Trim()
      if ($p) { $allFiles += $p }
    }

    $candidates = New-Object System.Collections.Generic.List[object]
    foreach ($src in $allFiles) {
      if (-not (Test-Path $src)) { continue }
      $ext = [System.IO.Path]::GetExtension($src).ToLowerInvariant()
      if ($ext -notin @('.png','.jpg','.jpeg','.tif','.tiff')) { continue }

      $dims = $null
      try { $dims = Get-Image-Dimensions $src } catch { continue }
      if ($dims.width -lt $minWidth -or $dims.height -lt $minHeight) { continue }

      $base = [System.IO.Path]::GetFileNameWithoutExtension($src)
      $page = 0
      $m = [regex]::Match($base, 'img-(?<page>\d{1,4})-(?<idx>\d{1,4})', 'IgnoreCase')
      if ($m.Success) { $page = [int]$m.Groups['page'].Value } else { $page = 0 }

      $candidates.Add([pscustomobject]@{
        src = $src
        page = $page
        width = [int]$dims.width
        height = [int]$dims.height
        area = [int]$dims.width * [int]$dims.height
        base = $base
        ext = $ext
        source_name = [System.IO.Path]::GetFileName($src)
      }) | Out-Null
    }

    $selected = New-Object System.Collections.Generic.List[object]
    $perPage = 2
    foreach ($group in ($candidates | Group-Object -Property page | Sort-Object Name)) {
      $top = @($group.Group | Sort-Object -Property area -Descending | Select-Object -First $perPage)
      foreach ($item in $top) { $selected.Add($item) | Out-Null }
      if ($selected.Count -ge $maxImages) { break }
    }
    if ($selected.Count -lt $maxImages) {
      $selectedSet = New-Object System.Collections.Generic.HashSet[string]
      foreach ($item in $selected) { $selectedSet.Add($item.src) | Out-Null }
      $rest = @($candidates | Where-Object { -not $selectedSet.Contains($_.src) } | Sort-Object -Property area -Descending)
      foreach ($item in $rest) {
        if ($selected.Count -ge $maxImages) { break }
        $selected.Add($item) | Out-Null
      }
    }

    $extracted = New-Object System.Collections.Generic.List[hashtable]
    $pageCounters = @{}
    foreach ($item in $selected) {
      $page = [int]$item.page
      if (-not $pageCounters.ContainsKey($page)) { $pageCounters[$page] = 0 }
      $pageCounters[$page] = [int]$pageCounters[$page] + 1
      $idx = [int]$pageCounters[$page]

      $safeStem = [regex]::Replace(($item.base), '[^A-Za-z0-9_.-]+', '_')
      if (-not $safeStem) { $safeStem = 'image' }
      $outName = "page_{0:000}_{1:00}_{2}{3}" -f $page, $idx, $safeStem, $item.ext
      $dst = Join-Path $targetDir $outName
      Copy-Item -LiteralPath $item.src -Destination $dst -Force

      $rel = (Resolve-Path $dst).Path.Substring(($script:ROOT.Length + 1)) -replace '\\','/'
      $extracted.Add(@{
        path = $rel
        page = '' + $page
        width = '' + $item.width
        height = '' + $item.height
        source_name = $item.source_name
      }) | Out-Null
    }

    $metaPath = Join-Path $targetDir 'images.json'
    ($extracted | ConvertTo-Json -Depth 6) | Set-Content -Path $metaPath -Encoding utf8
    return $extracted
  } finally {
    Remove-Item -LiteralPath $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
  }
}

function Export-Xlsx {
  $rows = Scan -refreshClassification:$false

  Add-Type -AssemblyName System.IO.Compression | Out-Null

  function Xml-Escape([string]$value) {
    $value = To-StringOrEmpty $value
    return ($value.Replace('&', '&amp;').Replace('<', '&lt;').Replace('>', '&gt;').Replace('\"', '&quot;'))
  }

  function Col-Letter([int]$index0) {
    $n = $index0 + 1
    $s = ''
    while ($n -gt 0) {
      $n--
      $s = ([char](65 + ($n % 26))) + $s
      $n = [math]::Floor($n / 26)
    }
    return $s
  }

  $rowCount = $rows.Count + 1
  $colCount = $script:CSV_FIELDS.Count
  $lastCell = "$(Col-Letter ($colCount - 1))$rowCount"

  $sheetRows = New-Object System.Text.StringBuilder
  for ($r = 0; $r -lt $rowCount; $r++) {
    $excelRow = $r + 1
    [void]$sheetRows.Append("<row r=`"$excelRow`">")
    for ($c = 0; $c -lt $colCount; $c++) {
      $addr = "$(Col-Letter $c)$excelRow"
      $value = if ($r -eq 0) { $script:CSV_FIELDS[$c] } else { To-StringOrEmpty $rows[$r - 1].($script:CSV_FIELDS[$c]) }
      $escaped = Xml-Escape $value
      [void]$sheetRows.Append("<c r=`"$addr`" t=`"inlineStr`"><is><t xml:space=`"preserve`">$escaped</t></is></c>")
    }
    [void]$sheetRows.Append("</row>")
  }

  $sheetXml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" +
    "<worksheet xmlns=`"http://schemas.openxmlformats.org/spreadsheetml/2006/main`">" +
    "<dimension ref=`"A1:$lastCell`"/>" +
    "<sheetData>$($sheetRows.ToString())</sheetData>" +
    "</worksheet>"

  $workbookXml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" +
    "<workbook xmlns=`"http://schemas.openxmlformats.org/spreadsheetml/2006/main`" xmlns:r=`"http://schemas.openxmlformats.org/officeDocument/2006/relationships`">" +
    "<sheets><sheet name=`"papers`" sheetId=`"1`" r:id=`"rId1`"/></sheets>" +
    "</workbook>"

  $workbookRelsXml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" +
    "<Relationships xmlns=`"http://schemas.openxmlformats.org/package/2006/relationships`">" +
    "<Relationship Id=`"rId1`" Type=`"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet`" Target=`"worksheets/sheet1.xml`"/>" +
    "<Relationship Id=`"rId2`" Type=`"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles`" Target=`"styles.xml`"/>" +
    "</Relationships>"

  $stylesXml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" +
    "<styleSheet xmlns=`"http://schemas.openxmlformats.org/spreadsheetml/2006/main`" xmlns:mc=`"http://schemas.openxmlformats.org/markup-compatibility/2006`" mc:Ignorable=`"x14ac`" xmlns:x14ac=`"http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac`">" +
    "<fonts count=`"1`" x14ac:knownFonts=`"1`"><font><sz val=`"11`"/><name val=`"Calibri`"/><family val=`"2`"/><scheme val=`"minor`"/></font></fonts>" +
    "<fills count=`"2`"><fill><patternFill patternType=`"none`"/></fill><fill><patternFill patternType=`"gray125`"/></fill></fills>" +
    "<borders count=`"1`"><border><left/><right/><top/><bottom/><diagonal/></border></borders>" +
    "<cellStyleXfs count=`"1`"><xf numFmtId=`"0`" fontId=`"0`" fillId=`"0`" borderId=`"0`"/></cellStyleXfs>" +
    "<cellXfs count=`"1`"><xf numFmtId=`"0`" fontId=`"0`" fillId=`"0`" borderId=`"0`" xfId=`"0`"/></cellXfs>" +
    "<cellStyles count=`"1`"><cellStyle name=`"Normal`" xfId=`"0`" builtinId=`"0`"/></cellStyles>" +
    "<dxfs count=`"0`"/><tableStyles count=`"0`" defaultTableStyle=`"TableStyleMedium2`" defaultPivotStyle=`"PivotStyleLight16`"/>" +
    "</styleSheet>"

  $contentTypesXml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" +
    "<Types xmlns=`"http://schemas.openxmlformats.org/package/2006/content-types`">" +
    "<Default Extension=`"rels`" ContentType=`"application/vnd.openxmlformats-package.relationships+xml`"/>" +
    "<Default Extension=`"xml`" ContentType=`"application/xml`"/>" +
    "<Override PartName=`"/xl/workbook.xml`" ContentType=`"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml`"/>" +
    "<Override PartName=`"/xl/worksheets/sheet1.xml`" ContentType=`"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml`"/>" +
    "<Override PartName=`"/xl/styles.xml`" ContentType=`"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml`"/>" +
    "<Override PartName=`"/docProps/core.xml`" ContentType=`"application/vnd.openxmlformats-package.core-properties+xml`"/>" +
    "<Override PartName=`"/docProps/app.xml`" ContentType=`"application/vnd.openxmlformats-officedocument.extended-properties+xml`"/>" +
    "</Types>"

  $packageRelsXml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" +
    "<Relationships xmlns=`"http://schemas.openxmlformats.org/package/2006/relationships`">" +
    "<Relationship Id=`"rId1`" Type=`"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument`" Target=`"xl/workbook.xml`"/>" +
    "<Relationship Id=`"rId2`" Type=`"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties`" Target=`"docProps/core.xml`"/>" +
    "<Relationship Id=`"rId3`" Type=`"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties`" Target=`"docProps/app.xml`"/>" +
    "</Relationships>"

  $now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  $coreXml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" +
    "<cp:coreProperties xmlns:cp=`"http://schemas.openxmlformats.org/package/2006/metadata/core-properties`" xmlns:dc=`"http://purl.org/dc/elements/1.1/`" xmlns:dcterms=`"http://purl.org/dc/terms/`" xmlns:dcmitype=`"http://purl.org/dc/dcmitype/`" xmlns:xsi=`"http://www.w3.org/2001/XMLSchema-instance`">" +
    "<dc:creator>paper_manager</dc:creator><cp:lastModifiedBy>paper_manager</cp:lastModifiedBy>" +
    "<dcterms:created xsi:type=`"dcterms:W3CDTF`">$now</dcterms:created>" +
    "<dcterms:modified xsi:type=`"dcterms:W3CDTF`">$now</dcterms:modified>" +
    "</cp:coreProperties>"

  $appXml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?>" +
    "<Properties xmlns=`"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties`" xmlns:vt=`"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes`">" +
    "<Application>Paper Library Manager</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>" +
    "<HeadingPairs><vt:vector size=`"2`" baseType=`"variant`"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs>" +
    "<TitlesOfParts><vt:vector size=`"1`" baseType=`"lpstr`"><vt:lpstr>papers</vt:lpstr></vt:vector></TitlesOfParts>" +
    "</Properties>"

  $path = $script:XLSX_PATH
  $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Create, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
  try {
    $archive = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create, $true)
    try {
      function Add-ZipEntry([System.IO.Compression.ZipArchive]$zip, [string]$name, [string]$content) {
        $entry = $zip.CreateEntry($name)
        $stream = $entry.Open()
        try {
          $writer = New-Object System.IO.StreamWriter($stream, (New-Object System.Text.UTF8Encoding($false)))
          try { $writer.Write($content) } finally { $writer.Dispose() }
        } finally {
          $stream.Dispose()
        }
      }

      Add-ZipEntry $archive '[Content_Types].xml' $contentTypesXml
      Add-ZipEntry $archive '_rels/.rels' $packageRelsXml
      Add-ZipEntry $archive 'docProps/core.xml' $coreXml
      Add-ZipEntry $archive 'docProps/app.xml' $appXml
      Add-ZipEntry $archive 'xl/workbook.xml' $workbookXml
      Add-ZipEntry $archive 'xl/_rels/workbook.xml.rels' $workbookRelsXml
      Add-ZipEntry $archive 'xl/styles.xml' $stylesXml
      Add-ZipEntry $archive 'xl/worksheets/sheet1.xml' $sheetXml
    } finally {
      $archive.Dispose()
    }
  } finally {
    $fs.Dispose()
  }
}

function Stats {
  $rows = Scan -refreshClassification:$false
  $result = @{
    read_status = @{}
    note_status = @{}
    subfield = @{}
    importance = @{}
  }
  foreach ($row in $rows) {
    foreach ($key in $result.Keys) {
      $value = $row.$key
      if (-not $value) { $value = '(blank)' }
      if ($result[$key].ContainsKey($value)) { $result[$key][$value] = [int]$result[$key][$value] + 1 }
      else { $result[$key][$value] = 1 }
    }
  }
  return $result
}

switch ($Command) {
  'scan' {
    $rows = Scan -refreshClassification:$RefreshClassification
    "Indexed $($rows.Count) PDFs into $(Split-Path -Leaf $script:INDEX_PATH)" | Write-Output
  }
  'pick' {
    $selected = Pick -reserve:$Reserve
    if ($null -eq $selected) { '{}' | Write-Output }
    else { $selected | ConvertTo-Json -Depth 6 | Write-Output }
  }
  'extract' {
    if (-not $PaperId) { throw '--PaperId required' }
    $text = Extract-Text $PaperId $MaxChars $Output
    if ($Output) { "Wrote $($text.Length) chars to $Output" | Write-Output }
    else { $text | Write-Output }
  }
  'extract-images' {
    if (-not $PaperId) { throw '--PaperId required' }
    $images = Extract-Images $PaperId $OutputDir $MaxImages $MinWidth $MinHeight
    $images | ConvertTo-Json -Depth 6 | Write-Output
  }
  'mark-note' {
    if (-not $PaperId -or -not $NotePath) { throw '--PaperId and --NotePath required' }
    Mark-Note $PaperId $NotePath
    "Marked note generated for $PaperId" | Write-Output
  }
  'mark-read' {
    if (-not $PaperId) { throw '--PaperId required' }
    Mark-Read $PaperId $Status
    "Marked $PaperId as $Status" | Write-Output
  }
  'export-xlsx' {
    Export-Xlsx
    "Exported $(Split-Path -Leaf $script:XLSX_PATH)" | Write-Output
  }
  'sync-obsidian' {
    $stats = Generate-Obsidian-Pages
    $stats | ConvertTo-Json -Depth 6 | Write-Output
  }
  'stats' {
    (Stats) | ConvertTo-Json -Depth 6 | Write-Output
  }
}

