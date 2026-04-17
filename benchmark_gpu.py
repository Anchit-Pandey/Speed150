"""
benchmark_gpu.py  —  A100 throughput benchmark for CS3T-UNet
=============================================================
Measures:
  - Forward pass throughput at various batch sizes
  - Memory usage per batch size
  - Comparison: FP32 vs BF16 vs BF16+compile
  - Optimal batch size recommendation for training

Run before starting training to calibrate batch_size and accum_steps.
"""

import time
import math
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from model_gpu import CS3TUNet, enable_a100_flags, count_parameters


def warmup(model, device, batch_size, T=10, Nf=64, Nt=64, n=3):
    """Warmup iterations for torch.compile and cuDNN autotuning."""
    x = torch.randn(batch_size, 2*T, Nf, Nt, device=device, dtype=torch.bfloat16)
    for _ in range(n):
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(x)
    torch.cuda.synchronize()


def measure_throughput(model, device, batch_size, T=10, Nf=64, Nt=64,
                        n_iters=50, dtype=torch.bfloat16, backward=False):
    """Returns (samples/sec, peak_mem_GB)."""
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(batch_size, 2*T, Nf, Nt, device=device, dtype=dtype)
    y = torch.randn(batch_size, 2*5,  Nf, Nt, device=device, dtype=dtype)
    crit = nn.MSELoss()

    # Warmup
    for _ in range(5):
        with autocast(device_type='cuda', dtype=dtype):
            out = model(x)
            if backward:
                loss = crit(out, y); loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with autocast(device_type='cuda', dtype=dtype):
            out = model(x)
            if backward:
                loss = crit(out, y); loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps  = batch_size * n_iters / elapsed
    mem  = torch.cuda.max_memory_allocated(device) / 1e9
    return sps, mem


def run_benchmark(args):
    enable_a100_flags()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("No CUDA device found — benchmark requires GPU"); return

    print(f"\n{'═'*65}")
    print(f"  CS3T-UNet A100 GPU Benchmark")
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'═'*65}\n")

    T = args.T; Nf = args.Nf; Nt = args.Nt

    # ── Benchmark 1: Forward pass throughput vs batch size ────────────────────
    print("─── Forward pass throughput (BF16, no compile) ───────────────────")
    print(f"{'Batch':>8} {'Samp/s':>10} {'VRAM (GB)':>12} {'Status':>10}")
    model_nc = CS3TUNet(in_channels=2*T, out_channels=2*5,
                        embed_dim=64, num_blocks=(2,2,6,2)).to(device)
    model_nc = model_nc.to(memory_format=torch.channels_last)

    recommended_batch = 32
    for bs in [32, 64, 128, 192, 256, 320, 384, 512]:
        try:
            warmup(model_nc, device, bs, T, Nf, Nt)
            sps, mem = measure_throughput(model_nc, device, bs, T, Nf, Nt,
                                           n_iters=30, dtype=torch.bfloat16)
            status = "OK"
            recommended_batch = bs
            print(f"{bs:>8} {sps:>10.0f} {mem:>12.2f} {status:>10}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{bs:>8} {'OOM':>10} {'---':>12} {'SKIP':>10}")
            break

    print(f"\n  → Max feasible batch size: {recommended_batch}")
    eff_batch = recommended_batch * 4
    print(f"  → Recommended: batch={recommended_batch}, accum_steps=4 "
          f"→ effective batch={eff_batch}\n")

    # ── Benchmark 2: FP32 vs BF16 vs BF16+compile (forward only) ─────────────
    bs = min(128, recommended_batch)
    print(f"─── Precision comparison (batch={bs}, forward only) ──────────────")
    print(f"{'Mode':>25} {'Samp/s':>10} {'VRAM (GB)':>12} {'Speedup':>10}")

    results = {}
    for mode, dtype, compile_m in [
        ('FP32 (baseline)',   torch.float32, False),
        ('BF16',              torch.bfloat16, False),
        ('BF16 + channels_last', torch.bfloat16, False),
        ('BF16 + compile',    torch.bfloat16, True),
    ]:
        try:
            m = CS3TUNet(in_channels=2*T, out_channels=2*5,
                         embed_dim=64, num_blocks=(2,2,6,2)).to(device)
            m = m.to(memory_format=torch.channels_last)
            if compile_m and hasattr(torch, 'compile'):
                m = torch.compile(m, mode='reduce-overhead')  # faster compile for bench
            warmup(m, device, bs, T, Nf, Nt)
            sps, mem = measure_throughput(m, device, bs, T, Nf, Nt,
                                           n_iters=50, dtype=dtype)
            results[mode] = sps
            base = results.get('FP32 (baseline)', sps)
            print(f"{mode:>25} {sps:>10.0f} {mem:>12.2f} {sps/base:>9.2f}×")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{mode:>25} {'ERROR':>10}  {str(e)[:30]}")

    # ── Benchmark 3: Forward + Backward throughput ────────────────────────────
    bs_bwd = min(64, recommended_batch)
    print(f"\n─── Forward + Backward throughput (BF16, batch={bs_bwd}) ─────────")
    for mode, use_ckpt in [('No gradient checkpointing', False),
                             ('With grad. checkpointing',  True)]:
        try:
            m = CS3TUNet(in_channels=2*T, out_channels=2*5, embed_dim=64,
                         num_blocks=(2,2,6,2), use_checkpoint=use_ckpt).to(device)
            m = m.to(memory_format=torch.channels_last)
            sps, mem = measure_throughput(m, device, bs_bwd, T, Nf, Nt,
                                           n_iters=20, dtype=torch.bfloat16,
                                           backward=True)
            print(f"  {mode:35s} {sps:>8.0f} samp/s  {mem:.2f} GB")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {mode}: ERROR — {e}")

    # ── Benchmark 4: CUDA Graph overhead reduction ────────────────────────────
    print(f"\n─── CUDA Graph capture (batch={bs_bwd}, BF16) ────────────────────")
    try:
        m = CS3TUNet(in_channels=2*T, out_channels=2*5, embed_dim=64,
                     num_blocks=(2,2,6,2)).to(device)
        m = m.to(memory_format=torch.channels_last)

        # Standard (no graph)
        sps_std, _ = measure_throughput(m, device, bs_bwd, T, Nf, Nt,
                                         n_iters=50, dtype=torch.bfloat16)

        # With CUDA Graph
        x_s = torch.randn(bs_bwd, 2*T, Nf, Nt, device=device, dtype=torch.bfloat16)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                _ = m(x_s)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            g.replay()
        torch.cuda.synchronize()
        sps_graph = bs_bwd * 50 / (time.perf_counter() - t0)

        print(f"  Standard         : {sps_std:>8.0f} samp/s")
        print(f"  CUDA Graph replay: {sps_graph:>8.0f} samp/s  "
              f"({sps_graph/sps_std:.2f}×)")
        del m, g; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CUDA Graph benchmark error: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  RECOMMENDED TRAINING CONFIGURATION")
    print(f"{'═'*65}")
    print(f"  python train_gpu.py \\")
    print(f"    --batch_size {recommended_batch} \\")
    print(f"    --accum_steps 4 \\")
    print(f"    --lr 2e-3 \\")
    print(f"    --epochs 400 \\")
    print(f"    --num_workers 8 \\")
    print(f"    --compile_model \\")
    print(f"    [--use_cuda_graph]  # add for extra ~20% speed")
    print(f"\n  Effective batch size: {recommended_batch*4}")
    print(f"{'═'*65}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--T',  type=int, default=10)
    p.add_argument('--Nf', type=int, default=64)
    p.add_argument('--Nt', type=int, default=64)
    run_benchmark(p.parse_args())
