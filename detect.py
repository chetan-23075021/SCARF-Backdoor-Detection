"""
SCARF detection pipeline
"""

import torch
import numpy as np
from models import STGCN, CompletionModel, SCARFDetector
from config import config
import os
from sklearn.metrics import roc_auc_score, average_precision_score

def load_models():
    """Load trained models"""
    device = torch.device(config.DEVICE)
    
    # Load victim model
    victim = STGCN(num_classes=config.NUM_CLASSES).to(device)
    victim_path = os.path.join(config.CHECKPOINT_DIR, 'best_victim.pth')
    if os.path.exists(victim_path):
        victim.load_state_dict(torch.load(victim_path, map_location=device))
        print("✓ Victim model loaded")
    victim.eval()
    
    # Load completion model
    completion = CompletionModel(
        in_channels=config.INPUT_CHANNELS
    ).to(device)
    comp_path = os.path.join(config.CHECKPOINT_DIR, 'completion_final.pth')
    if os.path.exists(comp_path):
        completion.load_state_dict(torch.load(comp_path, map_location=device))
        print("✓ Completion model loaded")
    completion.eval()
    
    return victim, completion


def _build_reference_pool(train_loader, device, num_samples=15):
    """
    Extract a small, semantically diverse pool of reference contexts 
    from the available training set for cross-context transferability tests.
    """
    if train_loader is None:
        print("⚠️ No train_loader provided. Cross-context transferability will be skipped.")
        return None
        
    print(f"🔄 Building reference pool of {num_samples} diverse samples...")
    reference_pool = []
    collected_classes = set()
    
    # Iterate through training data to sample diverse classes
    for x, labels, _ in train_loader:
        x = x.to(device)
        
        for i in range(len(x)):
            label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
            
            # Prioritize semantic diversity: only add if we haven't seen this class much
            if label not in collected_classes or len(reference_pool) < num_samples:
                # Keep the FULL (C, T, V, M) shape -- this must match exactly
                # what the victim model was trained on and what SCARFDetector's
                # other components now expect. Do NOT collapse the person axis.
                sample = x[i]
                reference_pool.append(sample)
                collected_classes.add(label)
                
            if len(reference_pool) >= num_samples:
                print(f"✓ Reference pool built covering {len(collected_classes)} unique classes.")
                return reference_pool
                
    return reference_pool


def run_detection(test_loader, train_loader=None):
    """Run SCARF detection on test data"""
    victim, completion = load_models()
    detector = SCARFDetector(victim, completion, config)
    
    # Build the reference context pool for the cross-context transfer experiments
    device = torch.device(config.DEVICE)
    reference_pool = _build_reference_pool(train_loader, device, num_samples=15)
    
    print("\n🔍 Running SCARF detection...")
    
    # Pass the reference pool into the detect method
    scores, poison_flags, regions = detector.detect(test_loader, reference_pool=reference_pool)
    
    # Compute detection metrics
    n_samples = len(scores)
    n_poison_true = int(n_samples * config.POISON_RATE)
    n_poison_pred = np.sum(poison_flags)
    
    print(f"\n📊 Detection Results:")
    print(f"  Samples processed: {n_samples}")
    print(f"  Detected as poisoned: {n_poison_pred} ({n_poison_pred/n_samples*100:.1f}%)")
    print(f"  Expected poisoned: {n_poison_true} ({config.POISON_RATE*100}%)")
    
    # # If we have ground truth poison flags, compute metrics
    # if hasattr(test_loader.dataset, 'poison_flags'):
    #     true_flags = test_loader.dataset.poison_flags[:len(scores)]
    #     if np.sum(true_flags) > 0:
    #         auroc = roc_auc_score(true_flags, scores)
    #         auprc = average_precision_score(true_flags, scores)
    #         print(f"\n📈 Detection Performance:")
    #         print(f"  AUROC: {auroc*100:.2f}%")
    #         print(f"  AUPRC: {auprc*100:.2f}%")
    if hasattr(test_loader.dataset, 'poison_flags'):
        true_flags = test_loader.dataset.poison_flags[:len(scores)]
        if np.sum(true_flags) > 0:
            auroc = roc_auc_score(true_flags, scores)
            auprc = average_precision_score(true_flags, scores)
            print(f"\n📈 Detection Performance:")
            print(f"  AUROC: {auroc*100:.2f}%")
            print(f"  AUPRC: {auprc*100:.2f}%")

            # --- DIAGNOSTIC: check label convention and score direction ---
            true_flags_arr = np.array(true_flags)
            scores_arr = np.array(scores)
            print(f"\n🔬 Diagnostic:")
            print(f"  true_flags unique values: {np.unique(true_flags_arr)}, dtype: {true_flags_arr.dtype}")
            print(f"  Count flagged poisoned (true): {int(np.sum(true_flags_arr))}")
            print(f"  Mean score | poisoned=True : {scores_arr[true_flags_arr == 1].mean():.4f}")
            print(f"  Mean score | poisoned=False: {scores_arr[true_flags_arr == 0].mean():.4f}")
            print(f"  Median score | poisoned=True : {np.median(scores_arr[true_flags_arr == 1]):.4f}")
            print(f"  Median score | poisoned=False: {np.median(scores_arr[true_flags_arr == 0]):.4f}")
    
    # Cluster signatures to find recurring patterns
    print("\n🔄 Clustering region signatures...")
    cluster_labels = detector.cluster_signatures()
    if cluster_labels is not None:
        n_clusters = len(set(cluster_labels))
        print(f"  Found {n_clusters} distinct region clusters")
    
    return scores, poison_flags, regions, detector