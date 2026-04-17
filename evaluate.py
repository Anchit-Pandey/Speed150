"""
evaluate.py — Standalone evaluation of a trained CS3T-UNet checkpoint
Paper: INFOCOM 2024 — Kang et al.

Usage:
  python evaluate.py --ckpt outputs/best_model.pt \
                     --test_x test_L5.mat --test_y test_L5.mat \
                     --L 5 --out_dir eval_results
"""

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from model     import CS3TUNet, count_parameters
from dataset   import CSIDataset
from losses    import CompositeLoss, nmse_db, mae_metric
from visualize import (plot_csi_comparison, plot_error_map,
                        plot_temporal_sequence, plot_error_histogram,
                        plot_nmse_per_step)


@torch.no_grad()
def evaluate(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['out_dir'], exist_ok=True)

    # Load checkpoint
    ckpt   = torch.load(cfg['ckpt'], map_location='cpu')
    train_cfg = ckpt.get('config', {})

    # Build model
    model = CS3TUNet(
        in_channels  = 2 * cfg['T'],
        out_channels = 2 * cfg['L'],
        embed_dim    = train_cfg.get('embed_dim',    64),
        num_blocks   = tuple(train_cfg.get('num_blocks', [2,2,6,2])),
        num_heads    = train_cfg.get('num_heads',    8),
        stripe_width = train_cfg.get('stripe_width', 7),
        group_size   = train_cfg.get('group_size',   4),
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()

    p = count_parameters(model)
    print(f"Model loaded. Parameters: {p['total_M']:.2f}M")
    print(f"Checkpoint epoch: {ckpt.get('epoch','?')}  "
          f"Best val NMSE: {ckpt.get('best_nmse','?'):.2f} dB\n")

    # Dataset
    ds = CSIDataset(cfg['test_x'], cfg['test_y'],
                    cfg.get('x_key','X_test_L5'),
                    cfg.get('y_key','Y_test_L5'),
                    use_npy=cfg.get('use_npy', False))
    loader = DataLoader(ds, batch_size=cfg['batch_size'],
                        shuffle=False, num_workers=4)

    criterion = CompositeLoss()
    total_loss = 0.0
    all_preds  = []
    all_gts    = []

    for X, Y in loader:
        X = X.to(device); Y = Y.to(device)
        pred = model(X)
        total_loss += criterion(pred, Y).item()
        all_preds.append(pred.cpu())
        all_gts.append(Y.cpu())

    P = torch.cat(all_preds)
    G = torch.cat(all_gts)
    n = len(loader)

    nmse_all = nmse_db(P, G)
    mae_all  = mae_metric(P, G)

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Test Loss  : {total_loss/n:.6f}")
    print(f"  Test NMSE  : {nmse_all:.2f} dB")
    print(f"  Test MAE   : {mae_all:.6f}")
    print()
    print("Paper Table I (QuaDRiGa dataset):")
    print("  L=1: -27.47 dB  |  L=5: -20.58 dB  |  Avg: -24.03 dB")
    print("=" * 50)

    # Per-step NMSE
    step_nmse = plot_nmse_per_step(G, P, cfg['out_dir'], cfg['L'])
    print("\nPer-step NMSE:")
    for i, v in enumerate(step_nmse):
        print(f"  Step {i+1}: {v:.2f} dB")

    # Visualisations
    plot_dir = cfg['out_dir']
    plot_csi_comparison(G, P, plot_dir, cfg['L'])
    plot_error_map(G, P, plot_dir, cfg['L'])
    plot_temporal_sequence(G, P, plot_dir, cfg['L'])
    plot_error_histogram(G, P, plot_dir)

    # Save summary
    summary = {
        'test_loss':   total_loss / n,
        'test_nmse_db': nmse_all,
        'test_mae':    mae_all,
        'per_step_nmse': step_nmse,
        'paper_target_L5': -20.58,
    }
    with open(os.path.join(cfg['out_dir'], 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {cfg['out_dir']}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--test_x',     default='test_L5.mat')
    p.add_argument('--test_y',     default='test_L5.mat')
    p.add_argument('--x_key',      default='X_test_L5')
    p.add_argument('--y_key',      default='Y_test_L5')
    p.add_argument('--T',          type=int,   default=10)
    p.add_argument('--L',          type=int,   default=5)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--use_npy',    action='store_true')
    p.add_argument('--out_dir',    default='eval_results')
    args = p.parse_args()
    evaluate(vars(args))
