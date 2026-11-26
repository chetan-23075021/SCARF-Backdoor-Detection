"""
Summarize SCARF detection results
"""

import pickle
import numpy as np
import os
from config import config

def summarize_results():
    results_file = os.path.join(config.CHECKPOINT_DIR, 'detection_results.pkl')
    
    if not os.path.exists(results_file):
        print("No results found. Run detection first!")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print("\n" + "="*60)
    print("SCARF FINAL RESULTS SUMMARY")
    print("="*60)
    
    metrics = results['metrics']
    print(f"\n📊 Detection Metrics:")
    print(f"  AUROC: {metrics.get('auroc', 0):.2f}%")
    print(f"  AUPRC: {metrics.get('auprc', 0):.2f}%")
    
    for key in metrics:
        if 'precision' in key or 'recall' in key:
            print(f"  {key}: {metrics[key]:.2f}%")
    
    # Compare with random baseline
    n_samples = len(results['scores'])
    n_poison = int(np.sum(results['true_flags']))
    random_auprc = n_poison / n_samples * 100
    print(f"\n📈 Improvement over Random:")
    print(f"  Random AUPRC: {random_auprc:.2f}%")
    print(f"  SCARF AUPRC: {metrics.get('auprc', 0):.2f}%")
    print(f"  Gain: {metrics.get('auprc', 0) - random_auprc:.2f} points")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    summarize_results()