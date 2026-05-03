# Overnight-resilient downloader for badrex/malagasy-speech-full.
# - Pins all HF caches to E:\hf-cache (HF_HOME, HF_HUB_CACHE, HF_XET_CACHE).
# - Retries on failure (hf download is idempotent: resumes via local-dir).
# - Watchdog: aborts if C:\Users\miana\.cache\huggingface ever grows past 1 GB
#   (means caches leaked to C: and would fill the system drive).
# - Stops only on full success or fatal C: leak.

$ErrorActionPreference = 'Continue'
$env:HF_HOME         = 'E:\hf-cache'
$env:HF_HUB_CACHE    = 'E:\hf-cache\hub'
$env:HF_XET_CACHE    = 'E:\hf-cache\xet'
$env:HF_ASSETS_CACHE = 'E:\hf-cache\assets'
New-Item -ItemType Directory -Force -Path E:\hf-cache\hub, E:\hf-cache\xet, E:\hf-data | Out-Null

$repo      = 'badrex/malagasy-speech-full'
$localDir  = 'E:\hf-data\malagasy-speech-full'
$logDir    = 'E:\projects\weak\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$mainLog   = Join-Path $logDir 'badrex_download.log'
$cLeakLimitGB = 1.0
$maxAttempts  = 50

function Get-DirGB($p) {
  if (-not (Test-Path $p)) { return 0 }
  $b = (Get-ChildItem $p -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  if (-not $b) { 0 } else { [math]::Round($b/1GB, 2) }
}

function Log($msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
  $line | Tee-Object -FilePath $mainLog -Append | Out-Host
}

Log "=== overnight download start ==="
Log "target: $localDir"
Log "HF_HUB_CACHE=$env:HF_HUB_CACHE  HF_XET_CACHE=$env:HF_XET_CACHE"

for ($i = 1; $i -le $maxAttempts; $i++) {
  Log "attempt #$i - invoking hf download"
  $attemptLog = Join-Path $logDir ("badrex_attempt_{0:D2}.log" -f $i)

  $proc = Start-Process -FilePath 'uvx' `
    -ArgumentList @('--from','huggingface_hub','hf','download',$repo,'--repo-type','dataset','--local-dir',$localDir) `
    -RedirectStandardOutput $attemptLog -RedirectStandardError "$attemptLog.err" `
    -PassThru -NoNewWindow

  while (-not $proc.HasExited) {
    Start-Sleep 30
    $cLeak = Get-DirGB 'C:\Users\miana\.cache\huggingface'
    $eData = Get-DirGB $localDir
    $eCache = Get-DirGB 'E:\hf-cache'
    $cFree = [math]::Round((Get-PSDrive C).Free/1GB, 1)
    Log ("  cFree={0}GB  Cleak={1}GB  Edata={2}GB  Ecache={3}GB" -f $cFree, $cLeak, $eData, $eCache)
    if ($cLeak -gt $cLeakLimitGB) {
      Log "!! C: leak detected ($cLeak GB) - killing process to protect system drive"
      try { Stop-Process -Id $proc.Id -Force } catch {}
      Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
      Log "!! aborting overnight script. Investigate before retrying."
      exit 2
    }
  }

  $code = $proc.ExitCode
  Log "attempt #$i exited with code $code"

  if ($code -eq 0) {
    $finalGB = Get-DirGB $localDir
    Log "=== SUCCESS - $localDir is $finalGB GB ==="
    exit 0
  }

  Log "retrying in 30s..."
  Start-Sleep 30
}

Log "!! exhausted $maxAttempts attempts without success"
exit 1
