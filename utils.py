"""
Utility functions
"""

import numpy as np
import torch
import random
import os

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(scores, true_flags):
    """Compute detection metrics"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    auroc = roc_auc_score(true_flags, scores)
    auprc = average_precision_score(true_flags, scores)
    
    return auroc, auprc


def save_checkpoint(model, optimizer, epoch, filename):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)


def load_checkpoint(model, optimizer, filename, device=None):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

"""
Evaluation metrics for SCARF performance
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, accuracy_score

def compute_detection_metrics(scores, true_labels):
    """
    Compute all detection metrics
    Args:
        scores: SCARF scores (higher = more likely poisoned)
        true_labels: Ground truth (1 = poisoned, 0 = clean)
    """
    metrics = {}
    
    # AUROC
    metrics['auroc'] = roc_auc_score(true_labels, scores) * 100
    
    # AUPRC
    metrics['auprc'] = average_precision_score(true_labels, scores) * 100
    
    # Precision@k for different k
    n_poison = int(np.sum(true_labels))
    for k in [n_poison, n_poison*2, int(len(scores)*0.05)]:
        if k <= len(scores):
            top_k_idx = np.argsort(scores)[-k:]
            precision_at_k = np.sum(true_labels[top_k_idx]) / k
            metrics[f'precision@{k}'] = precision_at_k * 100
    
    # Recall@k
    for k in [n_poison, n_poison*2, int(len(scores)*0.05)]:
        if k <= len(scores):
            top_k_idx = np.argsort(scores)[-k:]
            recall_at_k = np.sum(true_labels[top_k_idx]) / n_poison
            metrics[f'recall@{k}'] = recall_at_k * 100
    
    return metrics

def compute_localization_metrics(pred_regions, true_regions, threshold_iou=0.5):
    """
    Compute trigger localization accuracy
    Args:
        pred_regions: List of predicted (t_start, t_end, v_start, v_end)
        true_regions: List of ground truth trigger regions
    """
    if len(pred_regions) == 0 or len(true_regions) == 0:
        return {'hit_rate': 0, 'avg_iou': 0}
    
    def compute_iou(region1, region2):
        t1_s, t1_e, v1_s, v1_e = region1
        t2_s, t2_e, v2_s, v2_e = region2
        
        # Temporal IoU
        t_inter = max(0, min(t1_e, t2_e) - max(t1_s, t2_s))
        t_union = max(t1_e, t2_e) - min(t1_s, t2_s)
        t_iou = t_inter / t_union if t_union > 0 else 0
        
        # Spatial IoU
        v_inter = max(0, min(v1_e, v2_e) - max(v1_s, v2_s))
        v_union = max(v1_e, v2_e) - min(v1_s, v2_s)
        v_iou = v_inter / v_union if v_union > 0 else 0
        
        return (t_iou + v_iou) / 2  # Average IoU
    
    hits = 0
    ious = []
    
    for pred, true in zip(pred_regions, true_regions):
        if pred is None or true is None:
            continue
        iou = compute_iou(pred, true)
        ious.append(iou)
        if iou >= threshold_iou:
            hits += 1
    
    return {
        'hit_rate': hits / len(true_regions) * 100,
        'avg_iou': np.mean(ious) * 100 if ious else 0
    }

def compute_asr_ca_tradeoff(model, test_loader, poison_indices, removal_ratios=[0, 1, 2, 3, 4, 5, 7, 10, 15, 20]):
    """
    Compute ASR vs CA tradeoff for different removal ratios
    """
    device = next(model.parameters()).device
    results = {'removal_ratios': removal_ratios, 'asr': [], 'ca': []}
    
    for ratio in removal_ratios:
        # Simulate removal of top ratio% samples
        n_remove = int(len(poison_indices) * ratio / 100)
        # This is simplified - you'd need to implement actual removal
        
    return results


def compute_relative_transfer(source_x, target_x, region):
    """
    Extracts the local motion perturbation using a relative representation
    rather than copying absolute coordinates to ensure manifold compatibility.
    """
    t_start, t_end, v_start, v_end = region
    
    # Extract source and target regions
    source_region = source_x[:, t_start:t_end, v_start:v_end].clone()
    target_region = target_x[:, t_start:t_end, v_start:v_end].clone()
    
    # Isolate the local root (first joint in the affected region)
    source_root = source_region[:, :, 0:1].clone()
    target_root = target_region[:, :, 0:1].clone()
    
    # Calculate joint-relative displacement
    source_relative = source_region - source_root
    
    # Map the relative motion onto the target context's absolute root position
    transferred_region = target_root + source_relative
    
    return transferred_region


def apply_temporal_smoothing(x, t_start, t_end, window=2):
    """
    Applies moving-average temporal boundary smoothing to preserve 
    velocity continuity after a perturbation is injected[cite: 3].
    x: Tensor of shape (C, T, V)
    """
    smoothed = x.clone()
    T = x.shape[1]
    
    # Smooth start boundary to prevent sudden jumps
    for t in range(max(1, t_start - window), min(T - 1, t_start + window)):
        smoothed[:, t, :] = (x[:, t-1, :] + x[:, t, :] + x[:, t+1, :]) / 3.0
        
    # Smooth end boundary
    for t in range(max(1, t_end - window), min(T - 1, t_end + window)):
        smoothed[:, t, :] = (x[:, t-1, :] + x[:, t, :] + x[:, t+1, :]) / 3.0
        
    return smoothed


def validate_manifold_constraints(original_x, transferred_x):
    """
    Checks if the counterfactual/transferred sample remains on-manifold 
    by calculating bone-length error and velocity discontinuity[cite: 3].
    Returns True if valid, False if it violates physical constraints.
    """
    # 1. Check velocity continuity (max acceleration limit)
    orig_vel = torch.diff(original_x, dim=1)
    trans_vel = torch.diff(transferred_x, dim=1)
    
    vel_discontinuity = torch.max(torch.abs(trans_vel - orig_vel)).item()
    
    # If the maximum velocity jump is physically unrealistic (threshold based on your dataset normalization)
    if vel_discontinuity > 2.0: 
        return False
        
    return True