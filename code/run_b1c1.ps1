# B1 + C1 (the two supplementary experiments requested in external review M7 / M1)
#   C1: CTIS unrelated-recording injection control -> 2 backbones x 3 seeds = 6 runs (written to runs/, same directory as the other CTIS arms)
#   B1: P3 recording-grouped fold baseline A1r -> 2 backbones x 2 seeds x 3 folds = 12 runs (written to runs_p3/)
#   The two run serially to avoid contention on a single GPU.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File code/run_b1c1.ps1 [tag]
param([string]$Tag = "b1c1")
Set-Location (Join-Path $PSScriptRoot "..")
$py = "D:/XiaBao/AIArtDetect/envs/fapn/python.exe"
$env:TORCH_HOME = (Get-Location).Path + "/data/torch_home"
$drv = "logs/" + $Tag + "_driver.log"
Set-Content -Path $drv -Value "" -Encoding utf8
$t0 = Get-Date

foreach ($bb in @("alexnet", "swin_t")) {
    $log = "logs/" + $Tag + "_C1_" + $bb + ".log"
    $ts = Get-Date
    & $py code/train_e1.py --lib CTIS --arm C1 --bb $bb --seeds 0,1,2 --resume --tag c1 *> $log
    $rc = $LASTEXITCODE
    $mins = [int]((Get-Date) - $ts).TotalMinutes
    $line = "[C1] CTIS C1 " + $bb + " rc=" + $rc + " " + $mins + "min " + (Get-Date).ToString("HH:mm:ss")
    Add-Content -Path $drv -Value $line
    Write-Output $line
    if ($rc -ne 0) {
        Get-Content $log -Tail 5 | Add-Content -Path $drv
    }
}

foreach ($bb in @("alexnet", "swin_t")) {
    foreach ($sd in @(0, 1)) {
        $log = "logs/" + $Tag + "_A1r_" + $bb + "_seed" + $sd + ".log"
        $ts = Get-Date
        & $py code/train_p3.py --arms A1r --bbs $bb --seeds $sd --folds 1,2,3 --epochs 10 --fp32 --tag b1 *> $log
        $rc = $LASTEXITCODE
        $mins = [int]((Get-Date) - $ts).TotalMinutes
        $line = "[B1] A1r " + $bb + " seed" + $sd + " rc=" + $rc + " " + $mins + "min " + (Get-Date).ToString("HH:mm:ss")
        Add-Content -Path $drv -Value $line
        Write-Output $line
        if ($rc -ne 0) {
            Get-Content $log -Tail 5 | Add-Content -Path $drv
        }
    }
}

$line = "ALL DONE " + (Get-Date).ToString("HH:mm:ss") + " total " + [int]((Get-Date) - $t0).TotalMinutes + "min"
Add-Content -Path $drv -Value $line
Write-Output $line
