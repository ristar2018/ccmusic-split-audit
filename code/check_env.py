"""环境体检：确认本机能否跑实验（重点：Blackwell sm_120 与 GPU 版 torch）。

用法（在目标 conda 环境里）：
    python code/check_env.py

判据：
    [1] torch 版本 >= 2.7 且 cuda 构建 >= 12.8
    [2] cuda 可用、(12, 0) 能力、arch_list 含 sm_120
    [3] fp16 矩阵乘能跑出数值（真正验证内核存在）
    [4] 依赖齐全（librosa/soundfile/sklearn/pandas/timm/transformers）
全部 OK 才算环境就绪；任一项 FAIL 按提示处理。
"""
from __future__ import annotations

import platform
import shutil
import sys


def head(t: str) -> None:
    print("\n" + "=" * 58 + f"\n{t}\n" + "=" * 58)


def main() -> int:
    fails: list[str] = []
    head("0. 基本信息")
    print("python      :", sys.version.split()[0], f"({platform.machine()})")
    print("platform    :", platform.platform())
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print("torch       : 导入失败 ->", str(exc)[:120])
        print("\n常见原因：base 环境的 OMP 冲突（设 KMP_DUPLICATE_LIB_OK=TRUE 只治标），"
              "或未安装。请按 §6.3 新建环境。")
        return 1

    head("1. torch 与 CUDA 构建")
    print("torch       :", torch.__version__)
    print("cuda build  :", torch.version.cuda)
    print("cudnn       :", torch.backends.cudnn.version())
    if torch.version.cuda is None:
        fails.append("装的是 CPU 版 torch（无 CUDA 构建）")
    elif tuple(int(x) for x in torch.version.cuda.split(".")[:2]) < (12, 8):
        fails.append(f"CUDA 构建 {torch.version.cuda} < 12.8，Blackwell 可能缺内核")

    head("2. GPU 与架构")
    ok = torch.cuda.is_available()
    print("cuda available:", ok)
    if not ok:
        fails.append("torch.cuda.is_available() 为 False")
    else:
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"device      : {name}  {total:.1f} GB  capability={cap}")
        arch = torch.cuda.get_arch_list()
        print("arch_list   :", arch)
        if cap >= (12, 0) and "sm_120" not in arch:
            fails.append("设备是 sm_120 但 arch_list 无 sm_120（需重装 cu128+ 轮子）")

    head("3. fp16 内核实测（决定性检查）")
    try:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        y = x @ x
        v = float(y.float().mean())
        print("matmul ok   :", v)
    except Exception as exc:  # noqa: BLE001
        print("matmul FAIL :", str(exc)[:200])
        fails.append("fp16 矩阵乘失败（典型报错 no kernel image → 换 cu129 索引重装）")

    head("4. 依赖")
    for mod in ("librosa", "soundfile", "sklearn", "pandas", "numpy",
                "torchvision", "timm", "transformers"):
        try:
            m = __import__(mod)
            print(f"  {mod:14s} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:14s} MISSING ({str(exc)[:60]})")
            fails.append(f"缺少依赖 {mod}")

    head("5. 其他")
    print("ffmpeg      :", shutil.which("ffmpeg") or "未在 PATH（音频解码可能失败）")
    print("num cpus    :", __import__("os").cpu_count())

    head("结论")
    if fails:
        print("未就绪：")
        for f in fails:
            print("  -", f)
        return 1
    print("环境就绪，可以开始 E0 冒烟实验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
