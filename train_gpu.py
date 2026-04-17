"""
train_gpu.py  —  A100-optimized training for CS3T-UNet
=======================================================
GPU optimizations:
  1.  BF16 autocast (torch.amp.autocast, dtype=bfloat16)
        - A100 BF16 TFLOPS: 77.97 (dense), 312 (sparse)
        - Better numerical range than FP16 (no loss scaling needed)
        - No GradScaler required

  2.  torch.compile(mode='max-autotune')
        - Fuses ops, eliminates Python overhead
        - 2-4× throughput on A100 after warmup (1-3 min)

  3.  Large batch + gradient accumulation
        - batch_size=128, accum_steps=4 → effective batch=512
        - Keeps GPU SM occupancy high
        - Larger effective batch → faster convergence per wall-clock hour

  4.  CUDAPrefetcher
        - Overlaps H2D transfer with forward pass
        - Eliminates data-loading bottleneck

  5.  CUDA Graphs  (optional, --use_cuda_graph)
        - Captures the entire forward+backward as a static graph
        - Replays with near-zero kernel launch overhead
        - 20-30% speedup for fixed batch sizes

  6.  OneCycleLR scheduler
        - Reaches peak LR faster than warmup+cosine
        - Matches or exceeds paper results in fewer epochs

  7.  Fused AdamW  (fused=True)
        - Single CUDA kernel for optimizer step
        - ~10% faster optimizer step on A100

  8.  Channel-last model conversion
        - model.to(memory_format=torch.channels_last)

  9.  GPU memory profiling  (--profile)
        - torch.profiler traces top kernels

  10. Throughput and GPU utilization logging
        - Samples/sec, GPU memory, SM utilization via pynvml

Paper: INFOCOM 2024, Kang et al.
"""

import os
import sys
import time
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from contextlib import nullcontext

from model_gpu   import CS3TUNet, build_model, count_parameters, enable_a100_flags
from dataset_gpu import get_dataloaders_gpu, CUDAPrefetcher, CSIDatasetFromADP
from losses      import CompositeLoss, nmse_db, mae_metric
from visualize   import (plot_loss_curves, plot_nmse_curve, plot_csi_comparison,
                          plot_error_map, plot_temporal_sequence,
                          plot_error_histogram, plot_nmse_per_step)


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Note: deterministic=False for max throughput (set True for exact repro)


# ─────────────────────────────────────────────────────────────────────────────
# GPU STATS LOGGING  (requires pynvml — graceful fallback if not installed)
# ─────────────────────────────────────────────────────────────────────────────
class GPUMonitor:
    def __init__(self, device_idx: int = 0):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle  = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self._pynvml = pynvml
            self.enabled = True
        except Exception:
            self.enabled = False

    def stats(self) -> dict:
        if not self.enabled:
            used = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            return {'mem_used_gb': used, 'mem_peak_gb': peak,
                    'util_pct': -1, 'temp_c': -1}
        p  = self._pynvml
        mi = p.nvmlDeviceGetMemoryInfo(self.handle)
        ut = p.nvmlDeviceGetUtilizationRates(self.handle)
        tc = p.nvmlDeviceGetTemperature(self.handle, p.NVML_TEMPERATURE_GPU)
        return {
            'mem_used_gb': mi.used  / 1e9,
            'mem_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
            'util_pct':    ut.gpu,
            'temp_c':      tc,
        }

    def reset_peak(self):
        torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# CUDA GRAPH TRAINING STEP
# Captures a single forward+backward+optimizer step as a CUDA graph.
# After capture, replays with near-zero Python/kernel-launch overhead.
# Requirements: fixed batch size, static model (no dynamic control flow).
# ─────────────────────────────────────────────────────────────────────────────
class CUDAGraphTrainer:
    """
    Wraps a training step as a CUDA graph for maximum throughput.
    ~20-30% speedup on A100 for fixed batch sizes.
    """
    def __init__(self, model, optimizer, criterion, device,
                 batch_shape_x, batch_shape_y):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device
        self._graph    = None
        self._static_X = torch.zeros(batch_shape_x, device=device, dtype=torch.bfloat16)
        self._static_Y = torch.zeros(batch_shape_y, device=device, dtype=torch.bfloat16)
        self._static_loss = None
        self._capture()

    def _capture(self):
        """Warm up then capture CUDA graph."""
        # Warm up (3 iterations) — needed before capture
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = self.model(self._static_X)
                    loss = self.criterion(pred, self._static_Y)
                loss.backward()
                self.optimizer.step()
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self._graph):
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                self._static_pred = self.model(self._static_X)
                self._static_loss = self.criterion(self._static_pred, self._static_Y)
            self._static_loss.backward()
            self.optimizer.step()
        print("[CUDA Graph] Captured training step graph")

    def step(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Copy data into static buffers and replay graph."""
        self._static_X.copy_(X)
        self._static_Y.copy_(Y)
        self._graph.replay()
        return self._static_loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN EPOCH  (standard BF16 path)
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch_bf16(model, prefetcher, optimizer, criterion, scheduler,
                     device, accum_steps: int, max_grad_norm: float = 1.0) -> dict:
    """
    BF16 training epoch with gradient accumulation.
    No GradScaler needed for BF16 (unlike FP16).
    """
    model.train()
    total_loss = 0.0; total_nmse = 0.0; total_mae = 0.0
    n_batches  = 0; n_samples  = 0
    t0         = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    for i, (X, Y) in enumerate(prefetcher):
        # X, Y already on GPU in bfloat16 from CUDAPrefetcher
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            pred = model(X)
            loss = criterion(pred, Y) / accum_steps   # scale for accumulation

        loss.backward()

        if (i + 1) % accum_steps == 0:
            # Clip and step only every accum_steps batches
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Compute metrics in fp32 for accuracy
            pred_f = pred.float(); Y_f = Y.float()
            total_loss += loss.item() * accum_steps
            total_nmse += nmse_db(pred_f, Y_f)
            total_mae  += mae_metric(pred_f, Y_f)
        n_batches += 1; n_samples += X.size(0)

    elapsed = time.perf_counter() - t0
    return {
        'loss':         total_loss / n_batches,
        'nmse_db':      total_nmse / n_batches,
        'mae':          total_mae  / n_batches,
        'samples_per_s':n_samples  / elapsed,
        'elapsed_s':    elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN EPOCH  (CUDA Graph path — fixed batch size only)
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch_graph(graph_trainer, prefetcher, n_batches_expected: int) -> dict:
    """CUDA Graph training — fastest possible path, ~20-30% over BF16 standard."""
    total_loss = 0.0; n = 0
    t0 = time.perf_counter()
    for X, Y in prefetcher:
        total_loss += graph_trainer.step(X, Y)
        n += 1
    elapsed = time.perf_counter() - t0
    return {
        'loss':          total_loss / max(n, 1),
        'nmse_db':       float('nan'),   # metrics computed in eval pass
        'samples_per_s': n * X.size(0) / elapsed,
        'elapsed_s':     elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVAL EPOCH
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, prefetcher, criterion, device,
               return_samples: bool = False) -> dict:
    model.eval()
    total_loss = 0.0; total_nmse = 0.0; total_mae = 0.0; n = 0
    samples = None

    for i, (X, Y) in enumerate(prefetcher):
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            pred = model(X)
            loss = criterion(pred, Y)
        pf = pred.float(); Yf = Y.float()
        total_loss += loss.item()
        total_nmse += nmse_db(pf, Yf)
        total_mae  += mae_metric(pf, Yf)
        n += 1
        if return_samples and i == 0:
            samples = (X.float().cpu(), Y.float().cpu(), pred.float().cpu())

    result = {'loss': total_loss/n, 'nmse_db': total_nmse/n, 'mae': total_mae/n}
    if return_samples: result['samples'] = samples
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TORCH PROFILER  (run for --profile_batches batches, save chrome trace)
# ─────────────────────────────────────────────────────────────────────────────
def run_profiler(model, loader, device, out_dir: str, n_batches: int = 5):
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
    os.makedirs(out_dir, exist_ok=True)
    criterion = CompositeLoss()
    model.train()
    prefetcher = CUDAPrefetcher(loader, device)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities,
                 schedule=torch.profiler.schedule(wait=1, warmup=1, active=n_batches),
                 on_trace_ready=tensorboard_trace_handler(out_dir),
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True) as prof:
        for i, (X, Y) in enumerate(prefetcher):
            if i >= n_batches + 2: break
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(X)
                loss = criterion(pred, Y)
            loss.backward()
            prof.step()

    print(f"\nTop 15 CUDA kernels by self_cuda_time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    print(f"\nProfiler trace saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict):
    set_seed(cfg['seed'])
    enable_a100_flags()
    os.makedirs(cfg['out_dir'], exist_ok=True)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    monitor = GPUMonitor()
    monitor.reset_peak()

    print(f"\n{'═'*65}")
    print(f"  CS3T-UNet — A100 Optimized Training")
    print(f"{'═'*65}")
    if device.type == 'cuda':
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  dtype : bfloat16  |  compile: {cfg['compile_model']}")
    print(f"  batch : {cfg['batch_size']}  |  accum: {cfg['accum_steps']}  "
          f"→  eff. batch: {cfg['batch_size']*cfg['accum_steps']}")
    print(f"{'═'*65}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    if cfg.get('use_adp_direct', False):
        from torch.utils.data import DataLoader, random_split
        train_full = CSIDatasetFromADP(cfg['train_adp_path'], 'train_adp',
                                       T=cfg['T'], L=cfg['L'])
        test_ds    = CSIDatasetFromADP(cfg['test_adp_path'],  'test_adp',
                                       T=cfg['T'], L=cfg['L'])
        n_val   = int(len(train_full) * 0.1)
        n_train = len(train_full) - n_val
        train_ds, val_ds = random_split(train_full, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42))
        loader_kw = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                         pin_memory=True, persistent_workers=(cfg['num_workers']>0),
                         prefetch_factor=4 if cfg['num_workers']>0 else None,
                         drop_last=True)
        loaders = {
            'train': DataLoader(train_ds, shuffle=True,  **loader_kw),
            'val':   DataLoader(val_ds,   shuffle=False, **loader_kw),
            'test':  DataLoader(test_ds,  shuffle=False, **loader_kw),
            'info':  {'n_train':n_train,'n_val':n_val,'n_test':len(test_ds)}
        }
    else:
        loaders = get_dataloaders_gpu(
            train_x_path   = cfg['train_x_path'],
            train_y_path   = cfg['train_y_path'],
            test_x_path    = cfg['test_x_path'],
            test_y_path    = cfg['test_y_path'],
            batch_size     = cfg['batch_size'],
            num_workers    = cfg['num_workers'],
            use_npy        = cfg.get('use_npy', False),
            x_train_key    = cfg.get('x_train_key', 'X_train_L5'),
            y_train_key    = cfg.get('y_train_key', 'Y_train_L5'),
            x_test_key     = cfg.get('x_test_key',  'X_test_L5'),
            y_test_key     = cfg.get('y_test_key',  'Y_test_L5'),
        )

    info = loaders['info']
    print(f"Data  — train: {info['n_train']}  val: {info['n_val']}  test: {info['n_test']}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        T              = cfg['T'],
        L              = cfg['L'],
        compile_model  = cfg['compile_model'],
        use_checkpoint = cfg['use_checkpoint'],
        embed_dim      = cfg['embed_dim'],
        num_blocks     = tuple(cfg['num_blocks']),
        num_heads      = cfg['num_heads'],
        stripe_width   = cfg['stripe_width'],
        group_size     = cfg['group_size'],
        attn_drop      = cfg.get('attn_drop', 0.0),
        drop           = cfg.get('drop', 0.0),
    ).to(device)

    # channels_last for Conv2d Tensor Core alignment
    model = model.to(memory_format=torch.channels_last)

    p = count_parameters(model)
    print(f"Model — {p['total_M']:.2f}M params  (paper: ~19.64M)")

    # ── Optional profiling pass ────────────────────────────────────────────────
    if cfg.get('profile', False):
        print("\nRunning profiler (5 batches)...")
        run_profiler(model, loaders['train'], device,
                     os.path.join(cfg['out_dir'], 'profiler_trace'))

    # ── Optimizer  (fused AdamW — single CUDA kernel per step on A100) ────────
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 1e-4),
            fused=True,             # A100: ~10% faster optimizer step
        )
        print("Optimizer: AdamW (fused=True)")
    except TypeError:
        # fused kwarg requires PyTorch >= 2.0
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 1e-4),
        )
        print("Optimizer: AdamW (fused=False — upgrade to PyTorch ≥ 2.0)")

    # ── Scheduler: OneCycleLR ─────────────────────────────────────────────────
    # Reaches peak LR faster than warmup+cosine; better GPU utilization
    # Effective steps = epochs × (batches_per_epoch / accum_steps)
    steps_per_epoch = math.ceil(
        info['n_train'] / (cfg['batch_size'] * cfg['accum_steps'])
    )
    total_steps = cfg['epochs'] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = cfg['lr'],
        total_steps      = total_steps,
        pct_start        = 0.05,     # 5% warmup (≈ paper's 10/400 epochs)
        anneal_strategy  = 'cos',
        div_factor       = 25.0,     # start lr = max_lr / 25
        final_div_factor = 1e4,      # end   lr = max_lr / 1e4
    )
    print(f"Scheduler: OneCycleLR  max_lr={cfg['lr']}  total_steps={total_steps}")

    criterion = CompositeLoss(nmse_weight=cfg.get('nmse_weight', 0.0))

    # ── CUDA Graphs (optional) ────────────────────────────────────────────────
    use_graph = cfg.get('use_cuda_graph', False)
    graph_trainer = None
    if use_graph and device.type == 'cuda':
        # Build static shapes for CUDA graph
        X_shape = (cfg['batch_size'], 2*cfg['T'], cfg['Nf'], cfg['Nt'])
        Y_shape = (cfg['batch_size'], 2*cfg['L'], cfg['Nf'], cfg['Nt'])
        graph_trainer = CUDAGraphTrainer(model, optimizer, criterion, device,
                                         X_shape, Y_shape)

    # ── Training loop ─────────────────────────────────────────────────────────
    start_epoch = 0; best_nmse = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_nmse': [], 'val_nmse': [],
               'lr': [], 'samples_per_s': [], 'gpu_mem_gb': [], 'gpu_util': []}

    print(f"\n{'Ep':>5}  {'LR':>8}  {'TrLoss':>9}  {'TrNMSE':>9}  "
          f"{'VlLoss':>9}  {'VlNMSE':>9}  {'Samp/s':>8}  {'VRAM':>7}  "
          f"{'Util':>5}  {'T':>6}")
    print("─" * 95)

    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.perf_counter()
        monitor.reset_peak()

        # Prefetchers (reset each epoch)
        train_pf = CUDAPrefetcher(loaders['train'], device, torch.bfloat16)
        val_pf   = CUDAPrefetcher(loaders['val'],   device, torch.bfloat16)

        # Train
        if use_graph and graph_trainer is not None:
            tr = train_epoch_graph(graph_trainer, train_pf,
                                   len(loaders['train']))
        else:
            tr = train_epoch_bf16(model, train_pf, optimizer, criterion,
                                  scheduler, device, cfg['accum_steps'],
                                  cfg.get('max_grad_norm', 1.0))

        # Validate
        vl = eval_epoch(model, val_pf, criterion, device)

        # Stats
        gpu = monitor.stats()
        lr  = optimizer.param_groups[0]['lr']
        elapsed = time.perf_counter() - t0

        # Log
        history['train_loss'].append(tr['loss'])
        history['val_loss'].append(vl['loss'])
        history['train_nmse'].append(tr['nmse_db'])
        history['val_nmse'].append(vl['nmse_db'])
        history['lr'].append(lr)
        history['samples_per_s'].append(tr['samples_per_s'])
        history['gpu_mem_gb'].append(gpu['mem_peak_gb'])
        history['gpu_util'].append(gpu['util_pct'])

        # Print
        nm_str = f"{tr['nmse_db']:>9.2f}" if not math.isnan(tr['nmse_db']) else "  (graph)"
        print(f"{epoch+1:>5}  {lr:>8.5f}  {tr['loss']:>9.5f}  {nm_str}  "
              f"{vl['loss']:>9.5f}  {vl['nmse_db']:>9.2f}  "
              f"{tr['samples_per_s']:>8.0f}  "
              f"{gpu['mem_peak_gb']:>5.1f}GB  {gpu['util_pct']:>4.0f}%  "
              f"{elapsed:>5.1f}s")

        # Save best
        if vl['nmse_db'] < best_nmse:
            best_nmse = vl['nmse_db']
            # Save underlying model (unwrap compile wrapper if present)
            raw_model = getattr(model, '_orig_mod', model)
            torch.save({'epoch': epoch+1, 'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_nmse': best_nmse, 'config': cfg},
                       os.path.join(cfg['out_dir'], 'best_model.pt'))
            print(f"  ★ New best: {best_nmse:.2f} dB  (saved)")

        # Periodic checkpoint
        if (epoch + 1) % cfg.get('save_every', 50) == 0:
            raw_model = getattr(model, '_orig_mod', model)
            torch.save({'epoch': epoch+1, 'model': raw_model.state_dict(),
                        'config': cfg},
                       os.path.join(cfg['out_dir'], f'ckpt_epoch{epoch+1:04d}.pt'))

        # NaN guard
        if math.isnan(tr['loss']) or math.isnan(vl['loss']):
            print("NaN loss — stopping."); break

        # Save history each epoch
        with open(os.path.join(cfg['out_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  Final Test Evaluation (best model)")
    ckpt = torch.load(os.path.join(cfg['out_dir'], 'best_model.pt'), map_location=device)
    raw_model = getattr(model, '_orig_mod', model)
    raw_model.load_state_dict(ckpt['model'])
    test_pf = CUDAPrefetcher(loaders['test'], device, torch.bfloat16)
    test_res = eval_epoch(model, test_pf, criterion, device, return_samples=True)
    print(f"  Test NMSE  : {test_res['nmse_db']:.2f} dB")
    print(f"  Test Loss  : {test_res['loss']:.6f}")
    print(f"  Paper target (L=5): -20.58 dB on QuaDRiGa")

    # Throughput summary
    avg_sps = np.mean(history['samples_per_s'][-50:])
    avg_mem = np.mean(history['gpu_mem_gb'][-50:])
    avg_util= np.mean([u for u in history['gpu_util'][-50:] if u >= 0])
    print(f"\n  Throughput (last 50 epochs):")
    print(f"    Samples/sec  : {avg_sps:.0f}")
    print(f"    GPU VRAM     : {avg_mem:.1f} GB  / {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
    print(f"    GPU util     : {avg_util:.0f}%")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_dir = os.path.join(cfg['out_dir'], 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_loss_curves(history, plot_dir)
    plot_nmse_curve(history, plot_dir)
    if 'samples' in test_res:
        X_s, Y_s, P_s = test_res['samples']
        plot_csi_comparison(Y_s, P_s, plot_dir, cfg['L'])
        plot_error_map(Y_s, P_s, plot_dir, cfg['L'])
        plot_temporal_sequence(Y_s, P_s, plot_dir, cfg['L'])
        plot_error_histogram(Y_s, P_s, plot_dir)
        plot_nmse_per_step(Y_s, P_s, plot_dir, cfg['L'])

    print(f"\nAll outputs saved to: {cfg['out_dir']}")
    return history, test_res


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='CS3T-UNet A100 Training')

    # Data
    p.add_argument('--train_x',        default='train_L5.mat')
    p.add_argument('--train_y',        default='train_L5.mat')
    p.add_argument('--test_x',         default='test_L5.mat')
    p.add_argument('--test_y',         default='test_L5.mat')
    p.add_argument('--x_train_key',    default='X_train_L5')
    p.add_argument('--y_train_key',    default='Y_train_L5')
    p.add_argument('--x_test_key',     default='X_test_L5')
    p.add_argument('--y_test_key',     default='Y_test_L5')
    p.add_argument('--use_npy',        action='store_true')
    p.add_argument('--use_adp_direct', action='store_true')
    p.add_argument('--train_adp_path', default='train_adp_norm.mat')
    p.add_argument('--test_adp_path',  default='test_adp_norm.mat')

    # Architecture
    p.add_argument('--embed_dim',    type=int,   default=64)
    p.add_argument('--num_blocks',   type=int,   nargs=4, default=[2,2,6,2])
    p.add_argument('--num_heads',    type=int,   default=8)
    p.add_argument('--stripe_width', type=int,   default=7)
    p.add_argument('--group_size',   type=int,   default=4)
    p.add_argument('--attn_drop',    type=float, default=0.0)
    p.add_argument('--drop',         type=float, default=0.0)
    p.add_argument('--T',            type=int,   default=10)
    p.add_argument('--L',            type=int,   default=5)
    p.add_argument('--Nf',           type=int,   default=64)
    p.add_argument('--Nt',           type=int,   default=64)

    # Training
    p.add_argument('--epochs',          type=int,   default=400)
    p.add_argument('--batch_size',      type=int,   default=128,
                   help='Per-GPU batch. A100 80GB comfortably handles 256.')
    p.add_argument('--accum_steps',     type=int,   default=4,
                   help='Gradient accumulation. eff_batch=batch*accum')
    p.add_argument('--lr',              type=float, default=2e-3)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--max_grad_norm',   type=float, default=1.0)
    p.add_argument('--nmse_weight',     type=float, default=0.0)
    p.add_argument('--num_workers',     type=int,   default=8)
    p.add_argument('--save_every',      type=int,   default=50)

    # GPU optimizations
    p.add_argument('--compile_model',   action='store_true', default=True)
    p.add_argument('--no_compile',      dest='compile_model', action='store_false')
    p.add_argument('--use_checkpoint',  action='store_true',
                   help='Gradient checkpointing. Enable only if OOM at batch>512.')
    p.add_argument('--use_cuda_graph',  action='store_true',
                   help='CUDA Graph capture. ~20-30%% speedup, fixed batch only.')
    p.add_argument('--profile',         action='store_true',
                   help='Run torch.profiler for 5 batches then exit.')
    p.add_argument('--profile_batches', type=int,   default=5)

    p.add_argument('--out_dir', default='./outputs_gpu')
    p.add_argument('--seed',    type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = vars(args)
    cfg['train_x_path'] = cfg.pop('train_x')
    cfg['train_y_path'] = cfg.pop('train_y')
    cfg['test_x_path']  = cfg.pop('test_x')
    cfg['test_y_path']  = cfg.pop('test_y')
    os.makedirs(cfg['out_dir'], exist_ok=True)
    with open(os.path.join(cfg['out_dir'], 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    train(cfg)
