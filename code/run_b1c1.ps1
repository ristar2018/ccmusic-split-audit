# B1 + C1（外审 M7 / M1 的两个补充实验）
#   C1：CTIS 无关录音注入对照 -> 2 骨干 x 3 seed = 6 runs（写 runs/，与其它 CTIS 臂同目录）
#   B1：P3 录音分组折基线 A1r -> 2 骨干 x 2 seed x 3 折 = 12 runs（写 runs_p3/）
#   两者串行跑，避免单卡争抢。
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File code/run_b1c1.ps1 [tag]
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
