"""
Activation Clustering (Chen et al., 2018) -- adapted for skeleton ST-GCN.

Original idea: on an image/text classifier, backdoored samples of the
target class activate the penultimate layer differently from genuinely
target-class clean samples, even though both get the same predicted
label. Cluster each predicted class's activations into 2 groups (k-means
on a low-dim PCA projection); the smaller cluster is flagged as poisoned.

Adaptation notes:
- "Activations" here = the 256-dim pooled ST-GCN feature (see
  baselines/common.py:pooled_features), the direct analog of the
  penultimate FC-layer activations used in the original paper.
- Everything else (PCA -> 2-means -> minority-cluster flag) is unchanged
  from the original method.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def detect(features, pred_labels, num_classes, min_cluster_size_frac=0.35, pca_dim=10):
    """
    features: (N, D) pooled activations
    pred_labels: (N,) predicted class per sample (NOT ground truth --
                 Activation Clustering doesn't use ground truth, that's
                 the whole point)

    Returns: scores (N,) in [0, 1]. Samples assigned to a class's
             minority cluster get a continuous suspicion score based on
             how far they sit from the majority cluster's centroid;
             everyone else gets 0.
    """
    n = len(features)
    scores = np.zeros(n, dtype=float)

    for c in range(num_classes):
        idx = np.where(pred_labels == c)[0]
        if len(idx) < 10:
            continue  # too few samples in this predicted class to cluster meaningfully

        feats_c = features[idx]
        d = min(pca_dim, feats_c.shape[1], len(idx) - 1)
        if d < 1:
            continue
        reduced = PCA(n_components=d, random_state=42).fit_transform(feats_c)

        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(reduced)

        size0 = int(np.sum(cluster_labels == 0))
        size1 = int(np.sum(cluster_labels == 1))
        minority_label = 0 if size0 < size1 else 1
        minority_frac = min(size0, size1) / len(idx)

        if minority_frac > min_cluster_size_frac:
            # Balanced clusters -- Activation Clustering would NOT flag
            # this class at all.
            continue

        centers = km.cluster_centers_
        majority_label = 1 - minority_label
        dist_to_majority = np.linalg.norm(reduced - centers[majority_label], axis=1)
        dist_to_own = np.linalg.norm(reduced - centers[cluster_labels], axis=1)
        # Continuous suspicion in [0, 1): how much closer to the OTHER
        # (majority) cluster's centroid a point sits relative to its own.
        local_score = dist_to_majority / (dist_to_majority + dist_to_own + 1e-8)

        is_minority = (cluster_labels == minority_label).astype(float)
        scores[idx] = is_minority * local_score

    return scores