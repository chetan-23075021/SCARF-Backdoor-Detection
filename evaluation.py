"""
Evaluation metrics for SCARF performance
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def compute_detection_metrics(scores, true_labels):
    """
    Compute all detection metrics with a safety net for 0 poison samples.
    Includes FPR@90, FPR@95, and Precision@budget.
    """
    metrics = {}
    n_poison = int(np.sum(true_labels))
    
    # --- THE SAFETY NET ---
    if n_poison == 0:
        print("\n[Evaluation] ⚠️ No poisoned samples found in ground truth. Skipping division math.")
        return {
            'auroc': 0.0, 'auprc': 0.0, 
            'fpr@95': 0.0, 'fpr@90': 0.0,
            'precision@budget': 0.0
        }
    
    # AUROC & AUPRC
    metrics['auroc'] = roc_auc_score(true_labels, scores) * 100
    metrics['auprc'] = average_precision_score(true_labels, scores) * 100
    
    # False Positive Rates at specific TPR thresholds
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    
    # FPR@95% TPR
    idx_95 = np.where(tpr >= 0.95)[0]
    metrics['fpr@95'] = fpr[idx_95[0]] * 100 if len(idx_95) > 0 else 0.0
    
    # FPR@90% TPR
    idx_90 = np.where(tpr >= 0.90)[0]
    metrics['fpr@90'] = fpr[idx_90[0]] * 100 if len(idx_90) > 0 else 0.0
    
    # Precision@poison budget
    if n_poison <= len(scores):
        top_k_idx = np.argsort(scores)[-n_poison:]
        precision_at_budget = np.sum(true_labels[top_k_idx]) / n_poison
        metrics['precision@budget'] = precision_at_budget * 100
    else:
        metrics['precision@budget'] = 0.0
        
    return metrics


def compute_localization_metrics(pred_regions, true_regions):
    """
    Compute spatial IoU, temporal IoU, and unified spatio-temporal IoU.
    """
    if not pred_regions or not true_regions:
        return {'avg_spatial_iou': 0, 'avg_temporal_iou': 0, 'avg_st_iou': 0}
        
    spatial_ious, temporal_ious, st_ious = [], [], []
    
    for i, pred in enumerate(pred_regions):
        if i >= len(true_regions):
            break
        true = true_regions[i]
        
        if pred is None or true is None:
            continue
            
        t1_s, t1_e, v1_s, v1_e = pred
        t2_s, t2_e, v2_s, v2_e = true
        
        # Temporal IoU
        t_inter = max(0, min(t1_e, t2_e) - max(t1_s, t2_s))
        t_union = max(t1_e, t2_e) - min(t1_s, t2_s)
        t_iou = t_inter / t_union if t_union > 0 else 0
        
        # Spatial IoU
        v_inter = max(0, min(v1_e, v2_e) - max(v1_s, v2_s))
        v_union = max(v1_e, v2_e) - min(v1_s, v2_s)
        v_iou = v_inter / v_union if v_union > 0 else 0
        
        # Spatio-Temporal IoU (Area Intersection / Area Union)
        st_inter_area = t_inter * v_inter
        st_union_area = (t1_e - t1_s) * (v1_e - v1_s) + (t2_e - t2_s) * (v2_e - v2_s) - st_inter_area
        st_iou = st_inter_area / st_union_area if st_union_area > 0 else 0
        
        temporal_ious.append(t_iou)
        spatial_ious.append(v_iou)
        st_ious.append(st_iou)
        
    n_true_regions = sum(1 for r in true_regions if r is not None)

    return {
        'avg_temporal_iou': np.mean(temporal_ious) * 100 if temporal_ious else 0,
        'avg_spatial_iou': np.mean(spatial_ious) * 100 if spatial_ious else 0,
        'avg_st_iou': np.mean(st_ious) * 100 if st_ious else 0,
        'recall@1': (sum(1 for iou in st_ious if iou > 0.5) / max(n_true_regions, 1)) * 100
    }