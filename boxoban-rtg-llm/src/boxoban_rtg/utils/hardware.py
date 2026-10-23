"""Hardware/software inventory. Every probe fails soft and reports what it could not check."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata

PKGS = ["torch", "transformers", "peft", "accelerate", "numpy", "pandas", "pyarrow", "trl", "vllm", "bitsandbytes"]


def _run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def _nccl_worker(rank, world, port, out):
    import os

    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    x = torch.full((1024,), float(rank + 1), device="cuda")
    dist.all_reduce(x)
    ok = bool(torch.allclose(x, torch.full_like(x, world * (world + 1) / 2)))
    if rank == 0:
        out.put(ok)
    dist.destroy_process_group()


def check_nccl(n_gpus: int) -> dict:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_nccl_worker, args=(r, n_gpus, 29511, q)) for r in range(n_gpus)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(120)
    ok = (not q.empty()) and q.get()
    return {"all_reduce_ok": bool(ok), "exit_codes": [p.exitcode for p in ps]}


def hardware_report(expect_gpus: int = 4, run_nccl: bool = True) -> dict:
    rep = {"python": sys.version.split()[0], "platform": platform.platform(), "problems": []}
    try:
        rep["wsl"] = "microsoft" in open("/proc/version").read().lower()
    except OSError:
        rep["wsl"] = False
    rep["packages"] = {}
    for n in PKGS:
        try:
            rep["packages"][n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            rep["packages"][n] = None
    du = shutil.disk_usage(".")
    rep["disk_gb"] = {"total": du.total // 2**30, "free": du.free // 2**30}
    try:
        import os

        rep["cpu_threads"] = os.cpu_count()
        rep["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        pass
    code, out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
                      "--format=csv,noheader,nounits"])
    rep["nvidia_smi"] = out if code == 0 else None
    if code != 0:
        rep["problems"].append("nvidia-smi unavailable (in WSL2 it comes from the Windows driver; do not install a Linux display driver)")
    try:
        import torch

        rep["torch_cuda"] = torch.cuda.is_available()
        n = torch.cuda.device_count() if rep["torch_cuda"] else 0
        rep["gpus"] = []
        for i in range(n):
            free, total = torch.cuda.mem_get_info(i)
            rep["gpus"].append({"index": i, "name": torch.cuda.get_device_name(i), "total_gb": round(total / 2**30, 1),
                                "free_gb": round(free / 2**30, 1), "bf16": torch.cuda.is_bf16_supported()})
        rep["cuda_version"] = torch.version.cuda
        if n < expect_gpus:
            rep["problems"].append(f"expected {expect_gpus} GPUs, PyTorch sees {n}")
        elif run_nccl and n > 1:
            rep["nccl"] = check_nccl(n)
            if not rep["nccl"]["all_reduce_ok"]:
                rep["problems"].append("NCCL all_reduce failed")
    except ImportError:
        rep["torch_cuda"] = None
        rep["problems"].append("PyTorch not installed; CUDA/VRAM/NCCL not checked")
    return rep


def format_report(rep: dict) -> str:
    return json.dumps(rep, indent=2)
