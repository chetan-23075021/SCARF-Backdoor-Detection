"""
Spectral Signatures (Tran, Li & Madry, 2018) -- adapted for skeleton ST-GCN.

Original idea: backdoor samples leave a detectable trace in the top
right-singular-vector of the (centered) penultimate-layer activation
matrix, because the network learns a strong, low-rank direction
correlated with the trigger. Per predicted class, project each sample's
centered activation onto that top singular vector; the squared
projection is the "outlier score" -- exactly the original method,
unchanged, just applied to ST-GCN's pooled features instead of a CNN's
FC-layer activations.
"""

import numpy as np


def detect(features, pred_labels, num_classes, top_k_singular=1):
    """
    features: (N, D) pooled activations
    pred_labels: (N,) predicted class per sample

    Returns: scores (N,) normalized to [0, 1], higher = more suspicious.
    """
    n = len(features)
    scores = np.zeros(n, dtype=float)

    for c in range(num_classes):
        idx = np.where(pred_labels == c)[0]
        if len(idx) < 10:
            continue

        feats_c = features[idx]
        mean_c = feats_c.mean(axis=0, keepdims=True)
        centered = feats_c - mean_c

        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        top_v = vt[:top_k_singular]  # (top_k_singular, D)

        proj = centered @ top_v.T  # (len(idx), top_k_singular)
        outlier_score = np.sum(proj ** 2, axis=1)

        scores[idx] = outlier_score

    if scores.max() > 0:
        scores = scores / scores.max()
    return scores