"""
visualize.py — All result visualisation for CS3T-UNet
Paper: INFOCOM 2024 — Kang et al.

Produces:
  (a) Training vs Validation loss curves
  (b) NMSE vs Epoch curve
  (c) Ground Truth vs Predicted CSI heatmaps (side-by-side)
  (d) Error magnitude maps |pred - gt|
  (e) Temporal prediction sequence
  (f) Error histogram
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


# ─────────────────────────────────────────────
# SHARED STYLE
# ─────────────────────────────────────────────
CMAP_AMP   = 'viridis'
CMAP_ERR   = 'hot'
CMAP_DIFF  = 'RdBu_r'
DPI        = 150
FIGSIZE_SM = (10, 4)
FIGSIZE_LG = (14, 5)

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.titlesize':  11,
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi':      DPI,
})


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)


def _complex_amp(arr_2L_Nf_Nt, L: int, frame: int = 0):
    """
    arr: (2L, Nf, Nt)  — real/imag interleaved
    Returns amplitude of frame `frame`: (Nf, Nt)
    """
    r = arr_2L_Nf_Nt[2*frame]
    i = arr_2L_Nf_Nt[2*frame + 1]
    return np.sqrt(r**2 + i**2)


# ─────────────────────────────────────────────
# (A) TRAINING / VALIDATION LOSS CURVES
# ─────────────────────────────────────────────
def plot_loss_curves(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    ep = range(1, len(history['train_loss']) + 1)
    ax.plot(ep, history['train_loss'], label='Train loss', linewidth=1.5)
    ax.plot(ep, history['val_loss'],   label='Val loss',   linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Learning rate
    ax = axes[1]
    ax.plot(ep, history['lr'], color='orange', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (warmup + cosine)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, 'loss_curves.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# (B) NMSE vs EPOCH
# ─────────────────────────────────────────────
def plot_nmse_curve(history: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ep = range(1, len(history['train_nmse']) + 1)
    ax.plot(ep, history['train_nmse'], label='Train NMSE', linewidth=1.5)
    ax.plot(ep, history['val_nmse'],   label='Val NMSE',   linewidth=1.5)

    # Reference line: paper result for L=5 on QuaDRiGa
    ax.axhline(y=-20.58, color='red', linestyle='--', alpha=0.7,
               label='Paper result L=5 (−20.58 dB)')

    best_val = min(history['val_nmse'])
    best_ep  = history['val_nmse'].index(best_val) + 1
    ax.scatter([best_ep], [best_val], color='green', zorder=5,
               label=f'Best val: {best_val:.2f} dB @ ep{best_ep}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('NMSE vs Epoch  [Paper Eq.(10)]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, 'nmse_curve.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# (C) GROUND TRUTH vs PREDICTED CSI HEATMAPS
# ─────────────────────────────────────────────
def plot_csi_comparison(Y: torch.Tensor, pred: torch.Tensor,
                        save_dir: str, L: int, n_samples: int = 3):
    """
    Y, pred: (N, 2L, Nf, Nt)
    Shows amplitude heatmaps side-by-side for first n_samples × L frames.
    """
    Y_np    = _to_numpy(Y)
    pred_np = _to_numpy(pred)
    n_show  = min(n_samples, Y_np.shape[0])

    for frame in range(min(L, 3)):   # show up to 3 prediction steps
        fig, axes = plt.subplots(n_show, 3,
                                 figsize=(12, 3.5 * n_show))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        for s in range(n_show):
            gt_amp   = _complex_amp(Y_np[s],    L, frame)
            pr_amp   = _complex_amp(pred_np[s], L, frame)
            err_amp  = np.abs(gt_amp - pr_amp)

            vmin = min(gt_amp.min(), pr_amp.min())
            vmax = max(gt_amp.max(), pr_amp.max())

            im0 = axes[s,0].imshow(gt_amp,  aspect='auto',
                                   cmap=CMAP_AMP, vmin=vmin, vmax=vmax)
            axes[s,0].set_title(f'Ground Truth — sample {s}, step {frame+1}')
            axes[s,0].set_xlabel('Antenna index')
            axes[s,0].set_ylabel('Subcarrier index')
            fig.colorbar(im0, ax=axes[s,0], fraction=0.046)

            im1 = axes[s,1].imshow(pr_amp,  aspect='auto',
                                   cmap=CMAP_AMP, vmin=vmin, vmax=vmax)
            axes[s,1].set_title(f'Predicted — sample {s}, step {frame+1}')
            axes[s,1].set_xlabel('Antenna index')
            axes[s,1].set_ylabel('Subcarrier index')
            fig.colorbar(im1, ax=axes[s,1], fraction=0.046)

            im2 = axes[s,2].imshow(err_amp, aspect='auto', cmap=CMAP_ERR)
            axes[s,2].set_title(f'|Error| = |GT − Pred|')
            axes[s,2].set_xlabel('Antenna index')
            axes[s,2].set_ylabel('Subcarrier index')
            fig.colorbar(im2, ax=axes[s,2], fraction=0.046)

        fig.suptitle(f'ADP Amplitude: Ground Truth vs Predicted (step={frame+1})',
                     fontsize=12, y=1.01)
        fig.tight_layout()
        path = os.path.join(save_dir, f'csi_comparison_step{frame+1}.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# (D) ERROR MAP
# ─────────────────────────────────────────────
def plot_error_map(Y: torch.Tensor, pred: torch.Tensor,
                   save_dir: str, L: int):
    """Signed error (real channel) across all L steps for one sample."""
    Y_np    = _to_numpy(Y)
    pred_np = _to_numpy(pred)

    fig, axes = plt.subplots(2, L, figsize=(3.5*L, 7))

    for step in range(L):
        gt_r  = Y_np[0,    2*step]        # real component, step
        pr_r  = pred_np[0, 2*step]
        err   = pr_r - gt_r              # signed error
        vabs  = max(abs(err).max(), 1e-8)

        im = axes[0, step].imshow(gt_r, aspect='auto', cmap=CMAP_AMP)
        axes[0, step].set_title(f'GT real  step {step+1}')
        axes[0, step].set_xlabel('Antenna')
        axes[0, step].set_ylabel('Subcarrier')
        fig.colorbar(im, ax=axes[0, step], fraction=0.046)

        im2 = axes[1, step].imshow(err, aspect='auto',
                                    cmap=CMAP_DIFF, vmin=-vabs, vmax=vabs)
        axes[1, step].set_title(f'Error = Pred − GT  step {step+1}')
        axes[1, step].set_xlabel('Antenna')
        axes[1, step].set_ylabel('Subcarrier')
        fig.colorbar(im2, ax=axes[1, step], fraction=0.046)

    fig.suptitle('Signed Prediction Error per Step (sample 0)', fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, 'error_map.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# (E) TEMPORAL SEQUENCE VISUALISATION
# ─────────────────────────────────────────────
def plot_temporal_sequence(Y: torch.Tensor, pred: torch.Tensor,
                           save_dir: str, L: int, ant_idx: int = 0):
    """
    Plots the amplitude of the ADP at a fixed angle bin (ant_idx)
    across all L prediction steps, GT vs Predicted.
    """
    Y_np    = _to_numpy(Y)
    pred_np = _to_numpy(pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Waterfall: rows = prediction step, cols = subcarrier (delay)
    gt_steps   = np.stack([_complex_amp(Y_np[0],    L, s)[:, ant_idx]
                            for s in range(L)], axis=0)    # (L, Nf)
    pred_steps = np.stack([_complex_amp(pred_np[0], L, s)[:, ant_idx]
                            for s in range(L)], axis=0)

    vmin = min(gt_steps.min(), pred_steps.min())
    vmax = max(gt_steps.max(), pred_steps.max())

    im0 = axes[0].imshow(gt_steps, aspect='auto', cmap=CMAP_AMP,
                          vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Ground Truth — antenna {ant_idx}, steps 1–{L}')
    axes[0].set_xlabel('Subcarrier (delay) index')
    axes[0].set_ylabel('Prediction step')
    axes[0].set_yticks(range(L))
    axes[0].set_yticklabels([f'Step {i+1}' for i in range(L)])
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pred_steps, aspect='auto', cmap=CMAP_AMP,
                          vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Predicted — antenna {ant_idx}, steps 1–{L}')
    axes[1].set_xlabel('Subcarrier (delay) index')
    axes[1].set_ylabel('Prediction step')
    axes[1].set_yticks(range(L))
    axes[1].set_yticklabels([f'Step {i+1}' for i in range(L)])
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle('Temporal Prediction Sequence (ADP amplitude, sample 0)',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, 'temporal_sequence.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# (F) ERROR HISTOGRAM
# ─────────────────────────────────────────────
def plot_error_histogram(Y: torch.Tensor, pred: torch.Tensor,
                         save_dir: str, n_samples: int = 100):
    Y_np    = _to_numpy(Y[:n_samples])
    pred_np = _to_numpy(pred[:n_samples])

    errors = (pred_np - Y_np).ravel()
    abs_err = np.abs(errors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(errors, bins=100, color='steelblue', edgecolor='none',
                 alpha=0.8, density=True)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Prediction error')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Signed Error Distribution (n={n_samples} samples)')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(abs_err, bins=100, color='darkorange', edgecolor='none',
                 alpha=0.8, density=True)
    axes[1].set_xlabel('|Prediction error|')
    axes[1].set_ylabel('Density')
    p95 = np.percentile(abs_err, 95)
    axes[1].axvline(p95, color='red', linestyle='--', linewidth=1,
                    label=f'95th pct: {p95:.4f}')
    axes[1].set_title('Absolute Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Prediction Error Statistics', fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, 'error_histogram.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# (G) NMSE PER STEP  (multi-step breakdown)
# ─────────────────────────────────────────────
def plot_nmse_per_step(Y: torch.Tensor, pred: torch.Tensor,
                       save_dir: str, L: int, eps: float = 1e-10):
    """
    Computes NMSE separately for each of the L prediction steps.
    Mirrors Fig. 10 in the paper.
    """
    Y_np    = _to_numpy(Y)
    pred_np = _to_numpy(pred)
    step_nmse = []

    for step in range(L):
        gt_r = Y_np[:, 2*step]
        gt_i = Y_np[:, 2*step+1]
        pr_r = pred_np[:, 2*step]
        pr_i = pred_np[:, 2*step+1]

        gt_c = gt_r + 1j*gt_i
        pr_c = pr_r + 1j*pr_i
        diff = pr_c - gt_c

        # Per-sample NMSE then average
        per_sample = (np.abs(diff).reshape(len(Y_np), -1)**2).sum(1) / \
                     (np.abs(gt_c).reshape(len(Y_np), -1)**2).sum(1)
        nmse_db_val = 10 * np.log10(per_sample.mean() + eps)
        step_nmse.append(nmse_db_val)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, L+1), step_nmse, 'o-', linewidth=2,
            markersize=6, label='CS3T-UNet (reproduced)')

    # Paper Fig.10 reference values for QuaDRiGa (approximate read from figure)
    paper_ref = {1: -27.47, 2: -25.0, 3: -23.0, 4: -21.5, 5: -20.58}
    ref_steps = [s for s in paper_ref if s <= L]
    ax.plot(ref_steps, [paper_ref[s] for s in ref_steps], 's--',
            linewidth=1.5, markersize=6, color='gray', alpha=0.7,
            label='Paper Table I / Fig.10 (QuaDRiGa)')

    ax.set_xlabel('Prediction step')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('NMSE per Prediction Step  [cf. Paper Fig. 10]')
    ax.set_xticks(range(1, L+1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, 'nmse_per_step.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return step_nmse
