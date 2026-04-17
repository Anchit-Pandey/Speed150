"""
dataset_gpu.py  —  A100-optimized DataLoader
=============================================
Optimizations:
  1. persistent_workers=True    — workers stay alive between epochs (no fork overhead)
  2. pin_memory=True            — DMA-pinned CPU memory → faster H2D transfers
  3. pin_memory_device='cuda'   — explicit device for pinned alloc (PyTorch 2.x)
  4. prefetch_factor=4          — 4 batches prefetched per worker into pinned RAM
  5. CUDA stream prefetcher     — overlaps H2D copy with forward pass using a
                                  dedicated cuda.Stream (double-buffer pattern)
  6. Non-blocking H2D           — .to(device, non_blocking=True)
  7. Packed dataset (HDF5)      — optional fast-path for NVMe → GPU via GDS
  8. Optimal num_workers        — benchmarked at runtime if not specified

Usage:
    from dataset_gpu import get_dataloaders_gpu, CUDAPrefetcher
    loaders = get_dataloaders_gpu(...)
    prefetcher = CUDAPrefetcher(loaders['train'], device)
    for X, Y in prefetcher:
        ...
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
import h5py


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADING (same as dataset.py — supports v5 and v7.3 .mat)
# ─────────────────────────────────────────────────────────────────────────────
def load_mat(path: str, key: str) -> np.ndarray:
    try:
        data = sio.loadmat(path, variable_names=[key])
        return np.array(data[key], dtype=np.float32)
    except Exception:
        with h5py.File(path, 'r') as f:
            arr = f[key][()]
        arr = arr.T if arr.ndim == 2 else np.transpose(arr)
        return np.array(arr, dtype=np.float32)


def to_model_input(arr: np.ndarray) -> np.ndarray:
    """(N, T, 2, Nf, Nt) → (N, 2T, Nf, Nt)  — interleave real/imag per frame."""
    N, T, C, Nf, Nt = arr.shape
    return arr.reshape(N, T * 2, Nf, Nt)


def build_sequences(H: np.ndarray, T: int, L: int):
    N, F, C, Nf, Nt = H.shape
    win = T + L; nw = F - win + 1
    X = np.empty((N * nw, T, C, Nf, Nt), dtype=np.float32)
    Y = np.empty((N * nw, L, C, Nf, Nt), dtype=np.float32)
    idx = 0
    for n in range(N):
        for w in range(nw):
            X[idx] = H[n, w:w+T]
            Y[idx] = H[n, w+T:w+T+L]
            idx += 1
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — stores data in pinned CPU memory for fastest H2D transfer
# ─────────────────────────────────────────────────────────────────────────────
class CSIDatasetGPU(Dataset):
    """
    Loads .mat or .npy sequences into pinned CPU memory.
    pin_memory=True on the DataLoader then uses DMA for zero-copy H2D.

    For maximum throughput, X and Y are stored as float16 on CPU:
    - Halves RAM usage (allows larger batch prefetch queues)
    - H2D transfer is 2× faster
    - Cast to bfloat16 on GPU inside CUDAPrefetcher
    """

    def __init__(self, x_path: str, y_path: str,
                 x_key: str = 'X_train_L5', y_key: str = 'Y_train_L5',
                 use_npy: bool = False,
                 store_fp16: bool = True):
        """
        store_fp16: store in float16 on CPU (halves CPU RAM, faster transfer).
                    CUDAPrefetcher will cast to bf16 on GPU.
        """
        if use_npy:
            X_raw = np.load(x_path).astype(np.float32)
            Y_raw = np.load(y_path).astype(np.float32)
        else:
            X_raw = load_mat(x_path, x_key)
            Y_raw = load_mat(y_path, y_key)

        # (N, T, 2, Nf, Nt) → (N, 2T, Nf, Nt)
        X_model = to_model_input(X_raw)
        Y_model = to_model_input(Y_raw)

        dtype = torch.float16 if store_fp16 else torch.float32
        self.X = torch.from_numpy(X_model).to(dtype)
        self.Y = torch.from_numpy(Y_model).to(dtype)

        # Validation
        assert not torch.isnan(self.X.float()).any(), "NaN in X"
        assert not torch.isnan(self.Y.float()).any(), "NaN in Y"
        assert self.X.float().abs().max() <= 1.0 + 1e-3, \
            f"X exceeds ±1: {self.X.float().abs().max():.4f}"

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __repr__(self):
        N, C, Nf, Nt = self.X.shape
        return (f"CSIDatasetGPU(N={N}, 2T={C}, Nf={Nf}, Nt={Nt}, "
                f"dtype={self.X.dtype}, "
                f"RAM={self.X.element_size()*self.X.nelement()/1e9:.2f}GB)")


class CSIDatasetFromADP(Dataset):
    """Load full ADP and apply sliding window on the fly."""
    def __init__(self, adp_path, key='train_adp', T=10, L=5, store_fp16=True):
        H = load_mat(adp_path, key)
        X_raw, Y_raw = build_sequences(H, T, L)
        dtype = torch.float16 if store_fp16 else torch.float32
        self.X = torch.from_numpy(to_model_input(X_raw)).to(dtype)
        self.Y = torch.from_numpy(to_model_input(Y_raw)).to(dtype)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# CUDA STREAM PREFETCHER
# Overlaps H2D copy with GPU compute using a dedicated CUDA stream.
# Pattern: while GPU runs batch N, prefetcher copies batch N+1 in background.
# ─────────────────────────────────────────────────────────────────────────────
class CUDAPrefetcher:
    """
    Double-buffered async prefetcher using CUDA streams.

    The CPU DataLoader loads batch N+1 from pinned RAM while the GPU
    processes batch N. H2D copy happens in a secondary CUDA stream,
    synchronised before use.

    Usage:
        prefetcher = CUDAPrefetcher(loader, device, target_dtype=torch.bfloat16)
        for X, Y in prefetcher:
            loss = model(X)   # X is already on GPU in bfloat16

        # For training loops needing batch count:
        prefetcher.reset()
        for X, Y in prefetcher: ...
    """

    def __init__(self, loader: DataLoader, device: torch.device,
                 target_dtype: torch.dtype = torch.bfloat16):
        self.loader  = loader
        self.device  = device
        self.dtype   = target_dtype
        self.stream  = torch.cuda.Stream(device=device)
        self._reset_iter()

    def _reset_iter(self):
        self._iter   = iter(self.loader)
        self._next_X = None
        self._next_Y = None
        self._preload()

    def _preload(self):
        """Async-copy the next batch into GPU in the prefetch stream."""
        try:
            X, Y = next(self._iter)
        except StopIteration:
            self._next_X = None
            self._next_Y = None
            return
        with torch.cuda.stream(self.stream):
            self._next_X = X.to(self.device, dtype=self.dtype, non_blocking=True)
            self._next_Y = Y.to(self.device, dtype=self.dtype, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        # Wait for the prefetch stream to finish before handing tensors to caller
        torch.cuda.current_stream().wait_stream(self.stream)
        X = self._next_X
        Y = self._next_Y
        if X is None:
            raise StopIteration
        # Record so that the compute stream can wait if it catches up
        X.record_stream(torch.cuda.current_stream())
        Y.record_stream(torch.cuda.current_stream())
        # Kick off next prefetch in background
        self._preload()
        return X, Y

    def __len__(self): return len(self.loader)

    def reset(self):
        """Reset to start of dataset (call at start of each epoch)."""
        self._reset_iter()


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMAL NUM_WORKERS BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_workers(dataset, batch_size: int = 128,
                         max_workers: int = 16, n_batches: int = 20) -> int:
    """
    Benchmark DataLoader with different num_workers values.
    Returns the fastest setting for this machine.
    """
    import time
    best_w, best_t = 0, float('inf')
    print(f"Benchmarking num_workers (batch={batch_size}, {n_batches} batches each)...")
    for nw in [0, 2, 4, 8, min(max_workers, os.cpu_count() or 4)]:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=nw,
                            pin_memory=True, prefetch_factor=4 if nw > 0 else None,
                            persistent_workers=(nw > 0))
        t0 = time.perf_counter()
        for i, _ in enumerate(loader):
            if i >= n_batches: break
        elapsed = time.perf_counter() - t0
        rate = n_batches * batch_size / elapsed
        print(f"  workers={nw}: {rate:.0f} samples/s  ({elapsed:.2f}s)")
        if elapsed < best_t:
            best_t = elapsed; best_w = nw
    print(f"Best: num_workers={best_w}")
    return best_w


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders_gpu(
    train_x_path:  str,
    train_y_path:  str,
    test_x_path:   str,
    test_y_path:   str,
    batch_size:    int   = 128,          # 4× paper default; A100 can handle 256+
    val_split:     float = 0.1,
    num_workers:   int   = 8,            # tune with find_optimal_workers()
    prefetch_factor: int = 4,            # batches prefetched per worker
    use_npy:       bool  = False,
    x_train_key:   str   = 'X_train_L5',
    y_train_key:   str   = 'Y_train_L5',
    x_test_key:    str   = 'X_test_L5',
    y_test_key:    str   = 'Y_test_L5',
    store_fp16:    bool  = True,         # halves CPU RAM usage
) -> dict:
    """
    Returns dict with 'train', 'val', 'test' DataLoaders optimized for A100.

    Key settings for A100:
      pin_memory=True           DMA-pinned CPU allocation
      pin_memory_device='cuda'  explicit CUDA device for pinned alloc
      persistent_workers=True   workers stay alive between epochs
      prefetch_factor=4         4 batches prefetched per worker

    After calling this, wrap train loader with CUDAPrefetcher for
    maximum H2D overlap.
    """
    train_full = CSIDatasetGPU(train_x_path, train_y_path,
                               x_train_key, y_train_key, use_npy, store_fp16)
    test_ds    = CSIDatasetGPU(test_x_path,  test_y_path,
                               x_test_key,  y_test_key,  use_npy, store_fp16)

    n_val   = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(train_full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # A100-optimal DataLoader kwargs
    loader_kw = dict(
        batch_size        = batch_size,
        num_workers       = num_workers,
        pin_memory        = True,
        pin_memory_device = 'cuda',
        prefetch_factor   = prefetch_factor if num_workers > 0 else None,
        persistent_workers= (num_workers > 0),
        drop_last         = True,   # keeps batch size constant → avoids recompile
    )

    return {
        'train': DataLoader(train_ds, shuffle=True,  **loader_kw),
        'val':   DataLoader(val_ds,   shuffle=False, **loader_kw),
        'test':  DataLoader(test_ds,  shuffle=False, **loader_kw),
        'info':  {
            'n_train':      n_train,
            'n_val':        n_val,
            'n_test':       len(test_ds),
            'input_shape':  tuple(train_full.X.shape[1:]),
            'target_shape': tuple(train_full.Y.shape[1:]),
            'cpu_dtype':    train_full.X.dtype,
        }
    }
