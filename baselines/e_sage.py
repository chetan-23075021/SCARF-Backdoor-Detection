"""
E-SAGE: Explainability-based Defense Against Backdoor Attacks on GNNs
(Yuan et al., WAICS 2024) -- adapted for skeleton ST-GCN.

Includes AUTO-RESUME CHECKPOINTING to survive Google Colab timeouts.
"""

import os
import pickle
import numpy as np
import torch


def _cell_importance(model, x_single, pred_class, steps=8):
    """Integrated-gradients-style importance from a zeroed baseline."""
    device = x_single.device
    baseline = torch.zeros_like(x_single)
    total_grad = torch.zeros_like(x_single)

    for s in range(1, steps + 1):
        alpha = s / steps
        interpolated = (baseline + alpha * (x_single - baseline)).unsqueeze(0).requires_grad_(True)
        logits = model(interpolated)
        target_logit = logits[0, pred_class]
        target_logit.backward()
        total_grad += interpolated.grad[0].detach()

    avg_grad = total_grad / steps
    importance = (avg_grad * (x_single - baseline)).abs().sum(dim=0)[:, :, 0]  # (T, V), actor 0
    return importance


def detect(model, x_batch, pred_labels, device, prune_steps=3, prune_frac_per_step=0.02,
           checkpoint_interval=100, checkpoint_file='./results/baselines/e_sage_checkpoint.pkl'):
    """
    x_batch: (N, C, T, V, M)

    Returns: scores (N,)
    Features automatic checkpointing every `checkpoint_interval` samples.
    """
    model.eval()
    n = len(x_batch)
    scores = np.zeros(n, dtype=float)
    completed_mask = np.zeros(n, dtype=bool)

    # Ensure output directory exists for checkpoints
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    # =========================================================
    # RESUME LOGIC: Check for previous unfinished execution
    # =========================================================
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                ckpt = pickle.load(f)
            if len(ckpt.get('scores', [])) == n:
                scores = ckpt['scores']
                completed_mask = ckpt['completed_mask']
                done_count = int(np.sum(completed_mask))
                print(f"\n🔄 Found existing E-SAGE checkpoint!")
                print(f"▶️ Resuming from sample {done_count}/{n} ({done_count/n*100:.1f}% already done).\n")
        except Exception as e:
            print(f"[Warning] Could not load checkpoint ({e}), starting fresh.")

    for i in range(n):
        # Skip already completed samples
        if completed_mask[i]:
            continue

        x_i = x_batch[i].to(device)
        c = int(pred_labels[i])

        importance = _cell_importance(model, x_i, c)  # (T, V)
        T, V = importance.shape
        order = torch.argsort(importance.flatten(), descending=True)
        cells_per_step = max(1, int(prune_frac_per_step * T * V))

        x_current = x_i.clone()
        with torch.no_grad():
            logits = model(x_current.unsqueeze(0))
            prev_logit = logits[0, c].item()

        first_step_drop = None
        total_drop = 0.0

        for step in range(prune_steps):
            idx = order[step*cells_per_step:(step+1)*cells_per_step]
            t_idx = torch.div(idx, V, rounding_mode='floor')
            v_idx = idx % V
            x_current = x_current.clone()
            x_current[:, t_idx, v_idx, 0] = 0.0

            with torch.no_grad():
                logits = model(x_current.unsqueeze(0))
                new_logit = logits[0, c].item()

            drop = prev_logit - new_logit
            if step == 0:
                first_step_drop = drop
            total_drop += drop
            prev_logit = new_logit

        # Protect against negative drops and division by zero
        first_step_drop = max(0.0, first_step_drop if first_step_drop is not None else 0.0)
        avg_drop = max(0.0, total_drop / prune_steps)
        scores[i] = (first_step_drop / (avg_drop + 1e-8)) if avg_drop > 0 else 0.0
        completed_mask[i] = True

        # =========================================================
        # SAVE PROGRESS EVERY K SAMPLES (ATOMIC WRITE)
        # =========================================================
        if (i + 1) % checkpoint_interval == 0 or (i + 1) == n:
            done_total = int(np.sum(completed_mask))
            print(f"    E-SAGE: {done_total}/{n} samples processed ({(done_total/n)*100:.1f}%) [Checkpoint Saved]")
            
            # Write to a temporary file first so an interruption cannot corrupt the checkpoint
            tmp_ckpt = checkpoint_file + '.tmp'
            with open(tmp_ckpt, 'wb') as f:
                pickle.dump({'scores': scores, 'completed_mask': completed_mask}, f)
            os.replace(tmp_ckpt, checkpoint_file)

    print(f"✅ E-SAGE finished all {n} samples successfully!")
    return scores