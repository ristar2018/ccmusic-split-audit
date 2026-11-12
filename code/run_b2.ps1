# B2: negative controls with a uniform 3 seeds (the noise objection raised in external review M6)
#   3 libraries (chest_falsetto / erhu_playing_tech / GZ_IsoTech) x 2 backbones x {A0,A2} x seed0/1/2 = 36 runs
#   Results land in a separate runs_b2/, leaving the sealed single-seed evidence of the E1f round in runs/ untouched.
#   Leakage caliber: each recording of a negative-control library has 1 slice (K̄≈1); A0/A2 both split at recording level and the training side holds no Q_test recording;
#   this script creates no new split, it only changes the seed.
# Usage: pwsh -File code/run_b2.ps1 [tag]
param([string]$Tag = "b2")
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$py = "D:/XiaBao/AIArtDetect/envs/fapn/python.exe"
$env:TORCH_HOME = (Get-Location).Path + "/data/torch_home"
$libs = @("chest_falsetto", "erhu_playing_tech", "GZ_IsoTech")
$arms = @("A0", "A2")
$bbs = @("alexnet", "swin_t")
$out = "runs_b2"
$drv = "logs/${Tag}_driver.log"
Set-Content -Path $drv -Value "" -Encoding utf8
$total = $libs.Count * $bbs.Count * $arms.Count
$n = 0
$t0 = Get-Date
foreach ($lib in $libs) {
  foreach ($bb in $bbs) {
    foreach ($arm in $arms) {
      $n++
      $log = "logs/${Tag}_${lib}_${arm}_${bb}.log"
      $done = $true
      foreach ($s in 0..2) {
        if (-not (Test-Path "$out/$lib/$arm/$bb/seed$s/result.json")) { $done = $false }
      }
      if ($done) {
        "[$n/$total] $lib $arm $bb SKIP (3 seeds done)" | Tee-Object -FilePath $drv -Append
        continue
      }
      $ts = Get-Date
      & $py code/train_e1.py --lib $lib --arm $arm --bb $bb --seeds 0,1,2 --out $out --resume --tag $Tag *> $log
      $rc = $LASTEXITCODE
      $mins = [int]((Get-Date) - $ts).TotalMinutes
      $now = (Get-Date).ToString("HH:mm:ss")
      "[$n/$total] $lib $arm $bb rc=$rc ${mins}min $now" | Tee-Object -FilePath $drv -Append
      if ($rc -ne 0) {
        "!! failed, log tail:" | Tee-Object -FilePath $drv -Append
        Get-Content $log -Tail 5 | Tee-Object -FilePath $drv -Append
      }
    }
  }
}
$mins = [int]((Get-Date) - $t0).TotalMinutes
$now = (Get-Date).ToString("HH:mm:ss")
"ALL DONE $now total ${mins}min" | Tee-Object -FilePath $drv -Append
