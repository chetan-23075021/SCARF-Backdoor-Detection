import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def detect(model, x_batch, pred_labels, device, keep_frac=0.15, batch_size=32):
    """Batched XGBD implementation to prevent Colab timeout."""
    model.eval()
    n = len(x_batch)
    scores = np.zeros(n, dtype=float)

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        xb = x_batch[start_idx:end_idx].to(device)
        cb = pred_labels[start_idx:end_idx]

        xb.requires_grad_(True)
        logits = model(xb)
        
        # Batch loss accumulation for saliency
        loss = sum(F.cross_entropy(logits[i:i+1], torch.tensor([cb[i]], device=device)) for i in range(end_idx - start_idx))
        loss.backward()
        
        saliencies = xb.grad.abs()[:, :, :, :, 0].mean(dim=1) 
        x_explained = xb.clone().detach()

        for i in range(end_idx - start_idx):
            sal = saliencies[i]
            T, V = sal.shape
            k = max(1, int(keep_frac * T * V))
            flat = sal.flatten()
            topk_idx = torch.topk(flat, k).indices
            keep_mask = torch.zeros_like(flat)
            keep_mask[topk_idx] = 1.0
            keep_mask = keep_mask.view(T, V)

            x_explained[i, :, :, :, 0] = xb[i, :, :, :, 0].detach() * keep_mask.unsqueeze(0)

        with torch.no_grad():
            logits_exp = model(x_explained)
            for i in range(end_idx - start_idx):
                loss_exp = F.cross_entropy(logits_exp[i:i+1], torch.tensor([cb[i]], device=device)).item()
                scores[start_idx + i] = -loss_exp

        if start_idx % 500 < batch_size:
            print(f"    XGBD: {start_idx}/{n} samples explained")

    return scores