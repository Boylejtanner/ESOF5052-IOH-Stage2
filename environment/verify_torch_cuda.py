"""Verify a CUDA PyTorch install: device, VRAM, matmul throughput, cuDNN/channels_last sanity, torch.compile availability.

usage: python verify_torch_cuda.py [--json out.json] [--quick]
Exit code 0 indicates usable CUDA; 1 indicates an unavailable device. Requires PyTorch; NumPy is optional.
"""
# Check CUDA correctness and speed before using it for model training.
import argparse
import importlib.util
import json
import subprocess
import sys
import time


def matmul_tflops(torch, dtype, n=4096, iters=30, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n ** 3 * iters / (time.perf_counter() - t0) / 1e12


def conv_sanity(torch):
    """channels_last bf16 conv on a BSHI-shaped batch against the fp32 result; also times a forward+backward step."""
    torch.manual_seed(5052)
    x = torch.randn(64, 7, 60, 200, device="cuda")
    conv = torch.nn.Conv2d(7, 32, 3, padding=1).cuda()
    ref = conv(x)
    xc, cc = x.contiguous(memory_format=torch.channels_last), conv.to(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = cc(xc)
    err = float((out.detach().float() - ref.detach()).abs().max())
    torch.backends.cudnn.benchmark = True
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cc(xc).float().mean().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cc(xc).float().mean().backward()
    torch.cuda.synchronize()
    return {"max_abs_diff_bf16_vs_fp32": err, "channels_last": bool(out.is_contiguous(memory_format=torch.channels_last)), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "fwd_bwd_ms_per_batch64": round((time.perf_counter() - t0) / 10 * 1000, 2)}


def compile_check():
    """torch.compile with the default (inductor) backend needs Triton, which has no official Windows build: confirm in a subprocess with a time limit."""
    code = "import torch\nf=torch.compile(lambda x: (x*2).sum())\nprint(float(f(torch.ones(8,device='cuda'))))"
    res = {"triton_installed": importlib.util.find_spec("triton") is not None}
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        res.update(inductor_works=p.returncode == 0, detail=(p.stderr.strip().splitlines() or [""])[-1][:160])
    except subprocess.TimeoutExpired:
        res.update(inductor_works=False, detail="timeout after 120 s")
    return res


def main():
    # Read command-line options instead of hard-coding this run's settings.
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="skip the torch.compile subprocess")
    args = ap.parse_args()
    import torch
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available()}
    if not out["cuda_available"]:
        print(json.dumps(out, indent=1))
        return 1
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    out.update(device=props.name, capability=f"{props.major}.{props.minor}", vram_total_gb=round(total / 1e9, 2), vram_free_gb_torch=round(free / 1e9, 2),
               bf16_supported=bool(torch.cuda.is_bf16_supported()))
    try:
        # Run the external command and inspect its completion status.
        used, tot = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.split(",")
        out["vram_used_by_all_programs_gb_nvidia_smi"] = round(float(used) / 1024, 2)
        out["vram_free_gb_plan_with_this"] = round((float(tot) - float(used)) / 1024, 2)     # torch.cuda.mem_get_info can overstate free memory under WDDM
    except Exception as exc:
        out["nvidia_smi"] = f"unavailable: {exc!r}"[:100]
    out["tflops"] = {"fp32_no_tf32": round(matmul_tflops(torch, torch.float32), 1), "fp32_tf32": round(matmul_tflops(torch, torch.float32, tf32=True), 1),
                     "bf16": round(matmul_tflops(torch, torch.bfloat16), 1)}
    torch.backends.cuda.matmul.allow_tf32 = False
    out["conv"] = conv_sanity(torch)
    if not args.quick:
        out["torch_compile"] = compile_check()
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
