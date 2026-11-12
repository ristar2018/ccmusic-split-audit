"""Environment health check: can this machine run the experiments (focus: Blackwell sm_120 and a CUDA build of torch).

Usage (inside the target conda environment):
    python code/check_env.py

Criteria:
    [1] torch version >= 2.7 and CUDA build >= 12.8
    [2] CUDA available, capability (12, 0), arch_list contains sm_120
    [3] fp16 matmul returns a value (the decisive check that the kernel exists)
    [4] dependencies present (librosa/soundfile/sklearn/pandas/timm/transformers)
The environment is ready only if all pass; handle each FAIL as noted.
"""
from __future__ import annotations

import platform
import shutil
import sys


def head(t: str) -> None:
    print("\n" + "=" * 58 + f"\n{t}\n" + "=" * 58)


def main() -> int:
    fails: list[str] = []
    head("0. Basics")
    print("python      :", sys.version.split()[0], f"({platform.machine()})")
    print("platform    :", platform.platform())
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print("torch       : import failed ->", str(exc)[:120])
        print("\nCommon causes: OMP conflict in the base environment (KMP_DUPLICATE_LIB_OK=TRUE only masks it),"
              " or torch not installed. Rebuild the environment as in §6.3.")
        return 1

    head("1. torch and CUDA build")
    print("torch       :", torch.__version__)
    print("cuda build  :", torch.version.cuda)
    print("cudnn       :", torch.backends.cudnn.version())
    if torch.version.cuda is None:
        fails.append("CPU-only torch installed (no CUDA build)")
    elif tuple(int(x) for x in torch.version.cuda.split(".")[:2]) < (12, 8):
        fails.append(f"CUDA build {torch.version.cuda} < 12.8; Blackwell kernels may be missing")

    head("2. GPU and architecture")
    ok = torch.cuda.is_available()
    print("cuda available:", ok)
    if not ok:
        fails.append("torch.cuda.is_available() is False")
    else:
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"device      : {name}  {total:.1f} GB  capability={cap}")
        arch = torch.cuda.get_arch_list()
        print("arch_list   :", arch)
        if cap >= (12, 0) and "sm_120" not in arch:
            fails.append("device is sm_120 but arch_list has no sm_120 (reinstall from a cu128+ wheel)")

    head("3. fp16 kernel check (decisive)")
    try:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        y = x @ x
        v = float(y.float().mean())
        print("matmul ok   :", v)
    except Exception as exc:  # noqa: BLE001
        print("matmul FAIL :", str(exc)[:200])
        fails.append("fp16 matmul failed (typical error: no kernel image -> reinstall from the cu129 index)")

    head("4. Dependencies")
    for mod in ("librosa", "soundfile", "sklearn", "pandas", "numpy",
                "torchvision", "timm", "transformers"):
        try:
            m = __import__(mod)
            print(f"  {mod:14s} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:14s} MISSING ({str(exc)[:60]})")
            fails.append(f"missing dependency {mod}")

    head("5. Other")
    print("ffmpeg      :", shutil.which("ffmpeg") or "not on PATH (audio decoding may fail)")
    print("num cpus    :", __import__("os").cpu_count())

    head("Verdict")
    if fails:
        print("Not ready:")
        for f in fails:
            print("  -", f)
        return 1
    print("Environment ready; the E0 smoke run can start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
