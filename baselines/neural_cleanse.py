"""
Neural Cleanse (Wang et al., 2019) -- adapted for skeleton ST-GCN.

Original idea: for each class, optimize a small, universal mask+pattern
that -- when applied to ANY input -- flips the model's prediction to
that class. If one class needs a MUCH smaller mask (by L1 norm) than all
the others, that class is very likely the backdoor target and the
recovered mask approximates the real trigger. A per-class MAD (median
absolute deviation) outlier index flags the anomalous class.

Adaptation notes:
- The "patch" here is a learned (T, V)-shaped soft mask over the
  skeleton (shared across channels, applied only to actor 0) plus a
  learned per-cell replacement pattern -- the skeleton analog of Neural
  Cleanse's pixel-mask + pixel-pattern formulation for images.
- Runs on the FROZEN best_victim.pth -- no retraining, matching the
  original method's per-class optimization search.
"""

import torch
import torch.nn.functional as F
import numpy as np


def _reverse_engineer_trigger(model, ref_batch, target_class, device,
                              steps=300, lr=0.05, lambda_reg=1e-2):
    """
    ref_batch: (B, C, T, V, M) carrier samples -- the mask+pattern must
               flip THEIR prediction to target_class.
    Returns: mask (T, V) in [0,1], pattern (C, T, V), final L1 norm of mask.
    """
    B, C, T, V, M = ref_batch.shape
    mask_logits = torch.zeros(T, V, device=device, requires_grad=True)
    pattern = torch.zeros(C, T, V, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([mask_logits, pattern], lr=lr)
    target = torch.full((B,), target_class, dtype=torch.long, device=device)

    for _ in range(steps):
        mask = torch.sigmoid(mask_logits)  # (T, V) in (0, 1)
        
        # Clamp pattern to physical skeleton coordinate bounds to prevent optimizer cheating
        clamped_pattern = torch.clamp(pattern, -2.0, 2.0)
        blended = mask.unsqueeze(0) * clamped_pattern + (1 - mask.unsqueeze(0)) * ref_batch[:, :, :, :, 0]

        x_adv = ref_batch.clone()
        x_adv[:, :, :, :, 0] = blended

        logits = model(x_adv)
        ce_loss = F.cross_entropy(logits, target)
        reg_loss = mask.sum()
        loss = ce_loss + lambda_reg * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_mask = torch.sigmoid(mask_logits).detach()
    final_pattern = torch.clamp(pattern, -2.0, 2.0).detach()
    return final_mask.cpu().numpy(), final_pattern.cpu().numpy(), final_mask.sum().item()


def detect(model, reference_samples, num_classes, device, steps=300):
    """
    reference_samples: (B, C, T, V, M) small carrier batch (e.g. 32
                       samples drawn from the test/train set).
    Returns dict with suspected_class, per-class L1 norms, and the
    recovered trigger mask/pattern for the suspected class.
    """
    model.eval()
    ref_batch = reference_samples.to(device)
    per_class_l1 = {}
    per_class_result = {}

    print(f"  Reverse-engineering a trigger candidate for all {num_classes} "
          f"classes ({steps} steps each -- this is the expensive part)...")
    for c in range(num_classes):
        mask, pattern, l1 = _reverse_engineer_trigger(model, ref_batch, c, device, steps=steps)
        per_class_l1[c] = l1
        per_class_result[c] = (mask, pattern)
        if c % 10 == 0:
            print(f"    class {c}/{num_classes}: mask L1 = {l1:.2f}")

    l1_values = np.array(list(per_class_l1.values()))
    median = np.median(l1_values)
    mad = np.median(np.abs(l1_values - median)) + 1e-8
    anomaly_index = np.abs(l1_values - median) / (1.4826 * mad)  # consistent MAD estimator

    # Flag the class with an anomalously SMALL mask (easy-to-hide trigger),
    # not just any outlier -- a class needing an anomalously LARGE mask is
    # not evidence of a backdoor.
    candidates = np.where(l1_values < median)[0]
    if len(candidates) == 0:
        suspected_class = int(np.argmin(l1_values))
    else:
        suspected_class = int(candidates[np.argmax(anomaly_index[candidates])])

    trigger_mask, trigger_pattern = per_class_result[suspected_class]

    return {
        'suspected_class': suspected_class,
        'anomaly_index': float(anomaly_index[suspected_class]),
        'per_class_l1': per_class_l1,
        'trigger_mask': trigger_mask,
        'trigger_pattern': trigger_pattern,
    }


def score_samples(model, x_batch, suspected_class, trigger_mask, device, batch_size=32):
    """
    Per-sample suspicion score. Evaluated in mini-batches to prevent OOM.
    """
    model.eval()
    mask_t = torch.from_numpy(trigger_mask).float().to(device)  # (T, V)
    
    # Adaptive threshold: zero out cells with top 10% highest mask intensity
    threshold = np.percentile(trigger_mask, 90)
    keep = (mask_t < max(0.1, threshold)).float()

    all_scores = []
    n = len(x_batch)

    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            xb = x_batch[start_idx:end_idx].to(device)

            orig_logits = model(xb)
            pred = orig_logits.argmax(dim=1)
            orig_target_logit = orig_logits[:, suspected_class]

            x_masked = xb.clone()
            x_masked[:, :, :, :, 0] = xb[:, :, :, :, 0] * keep.unsqueeze(0)
            masked_logits = model(x_masked)
            masked_target_logit = masked_logits[:, suspected_class]

            # Prevent negative drop values from artificially skewing scores
            drop = torch.clamp(orig_target_logit - masked_target_logit, min=0.0)
            is_target_pred = (pred == suspected_class).float()
            
            score = (drop * is_target_pred).cpu().numpy()
            all_scores.append(score)

    return np.concatenate(all_scores)