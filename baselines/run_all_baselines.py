"""
Modular SOTA Baseline Runner for Experiment 1 (List-of-experiments.pdf)

Usage:
    # Run fast baselines first
    python baselines/run_all_baselines.py --method ac
    python baselines/run_all_baselines.py --method ss

    # Run heavy baselines in their own separate Colab cells
    python baselines/run_all_baselines.py --method nc
    python baselines/run_all_baselines.py --method xgbd
    python baselines/run_all_baselines.py --method e_sage
    python baselines/run_all_baselines.py --method abl

    # Or compile and print the final table once all have run:
    python baselines/run_all_baselines.py --aggregate
"""

import os
import sys
import time
import pickle
import csv
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from utils import set_seed
from baselines.common import load_victim, get_poisoned_loaders, collect_test_arrays, summarize
from baselines import activation_clustering, spectral_signatures, neural_cleanse, xgbd, e_sage, abl

RESULTS_DIR = os.path.join('.', 'results', 'baselines')
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_single_result(name, summary_dict, raw_scores, poison_flags, extra_data=None):
    """Persist individual baseline results to prevent losing progress on timeout."""
    payload = {
        'summary': summary_dict,
        'scores': raw_scores,
        'poison_flags': poison_flags,
        'extra_data': extra_data or {}
    }
    filename = os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(payload, f)
    print(f" Saved individual result to: {filename}")


def load_scarf_results():
    path = os.path.join(config.CHECKPOINT_DIR, 'detection_results.pkl')
    if not os.path.exists(path):
        print(f"[Warning] {path} not found. Run `python main.py --poison --detect` first.")
        return None
    with open(path, 'rb') as f:
        results = pickle.load(f)
    
    loc_iou = results.get('loc_metrics', {}).get('avg_st_iou', "N/A")
    if loc_iou != "N/A":
        loc_iou = f"{loc_iou:.2f}%"
        
    summary = summarize('SCARF (ours)', results['scores'], results['true_flags'], loc_iou=loc_iou)
    save_single_result('scarf', summary, results['scores'], results['true_flags'], extra_data={'regions': results.get('regions')})
    return summary


# def aggregate_all():
#     """Reads all existing baseline .pkl files and prints the master comparison table."""
#     load_scarf_results()
#     all_summaries = []
    
#     expected = ['activation_clustering', 'spectral_signatures', 'neural_cleanse', 
#                 'xgbd', 'e_sage', 'abl_(train-set)', 'scarf']
    
#     for method_file in os.listdir(RESULTS_DIR):
#         if method_file.endswith('.pkl') and not method_file.startswith('experiment1'):
#             with open(os.path.join(RESULTS_DIR, method_file), 'rb') as f:
#                 data = pickle.load(f)
#                 all_summaries.append(data['summary'])

#     if not all_summaries:
#         print("No baseline results found in results/baselines/.")
#         return

#     # Print Table
#     print("\n" + "=" * 88)
#     print(f"{'Method':<25}{'AUROC':>9}{'AUPRC':>9}{'FPR@95TPR':>12}{'Prec@Budget':>14}{'Loc IoU':>12}")
#     print("=" * 88)
#     for r in all_summaries:
#         loc_str = r['loc_iou'] if isinstance(r['loc_iou'], str) else f"{r['loc_iou']:.2f}%"
#         print(f"{r['method']:<25}{r['auroc']:>8.2f}%{r['auprc']:>8.2f}%"
#               f"{r['fpr@95tpr']:>11.2f}%{r['precision@budget']:>13.2f}%{loc_str:>12}")
#     print("=" * 88)

#     # Save to CSV
#     csv_path = os.path.join('.', 'results', 'experiment1_sota_comparison.csv')
#     with open(csv_path, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=['method', 'auroc', 'auprc', 'fpr@95tpr', 'precision@budget', 'loc_iou'])
#         writer.writeheader()
#         writer.writerows(all_summaries)
#     print(f"Aggregated table exported to {csv_path}")

def aggregate_all():
    """Reads all existing baseline .pkl files and prints the master comparison table."""
    load_scarf_results()
    all_summaries = []
    
    for method_file in os.listdir(RESULTS_DIR):
        # FIX: Ignore checkpoint files and non-summary files
        if method_file.endswith('.pkl') and not method_file.startswith('experiment1') and 'checkpoint' not in method_file:
            with open(os.path.join(RESULTS_DIR, method_file), 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'summary' in data:
                    all_summaries.append(data['summary'])

    if not all_summaries:
        print("No baseline results found in results/baselines/.")
        return

    # Print Table
    print("\n" + "=" * 88)
    print(f"{'Method':<25}{'AUROC':>9}{'AUPRC':>9}{'FPR@95TPR':>12}{'Prec@Budget':>14}{'Loc IoU':>12}")
    print("=" * 88)
    for r in all_summaries:
        loc_str = r['loc_iou'] if isinstance(r['loc_iou'], str) else f"{r['loc_iou']:.2f}%"
        print(f"{r['method']:<25}{r['auroc']:>8.2f}%{r['auprc']:>8.2f}%"
              f"{r['fpr@95tpr']:>11.2f}%{r['precision@budget']:>13.2f}%{loc_str:>12}")
    print("=" * 88)

    csv_path = os.path.join('.', 'results', 'experiment1_sota_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'auroc', 'auprc', 'fpr@95tpr', 'precision@budget', 'loc_iou'])
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"Aggregated table exported to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SOTA Baselines modularly.")
    parser.add_argument('--method', type=str, choices=['ac', 'ss', 'nc', 'xgbd', 'e_sage', 'abl', 'all'],
                        default='all', help="Which baseline to execute.")
    parser.add_argument('--aggregate', action='store_true', help="Aggregate all existing results without running.")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_all()
        return

    set_seed(42)
    device = torch.device(config.DEVICE)
    victim = load_victim(device)
    train_loader, train_dataset, test_loader, test_dataset = get_poisoned_loaders()

    # Load arrays only if test-time baseline is selected
    if args.method != 'abl':
        print("\nExtracting shared test-set features and activations...")
        arrays = collect_test_arrays(victim, test_loader, device)

    # 1. Activation Clustering
    if args.method in ['ac', 'all']:
        print("\n--- [1/6] Running Activation Clustering ---")
        t0 = time.time()
        scores = activation_clustering.detect(arrays['features'], arrays['pred'], config.NUM_CLASSES)
        summ = summarize('Activation Clustering', scores, arrays['poison_flags'])
        save_single_result('activation_clustering', summ, scores, arrays['poison_flags'])
        print(f"Done in {time.time()-t0:.1f}s | AUROC: {summ['auroc']:.2f}%")

    # 2. Spectral Signatures
    if args.method in ['ss', 'all']:
        print("\n--- [2/6] Running Spectral Signatures ---")
        t0 = time.time()
        scores = spectral_signatures.detect(arrays['features'], arrays['pred'], config.NUM_CLASSES)
        summ = summarize('Spectral Signatures', scores, arrays['poison_flags'])
        save_single_result('spectral_signatures', summ, scores, arrays['poison_flags'])
        print(f"Done in {time.time()-t0:.1f}s | AUROC: {summ['auroc']:.2f}%")

    # 3. Neural Cleanse
    if args.method in ['nc', 'all']:
        print("\n--- [3/6] Running Neural Cleanse ---")
        t0 = time.time()
        ref_batch = arrays['x'][:32]
        nc_result = neural_cleanse.detect(victim, ref_batch, config.NUM_CLASSES, device, steps=50)
        scores = neural_cleanse.score_samples(victim, arrays['x'], nc_result['suspected_class'], nc_result['trigger_mask'], device)
        summ = summarize('Neural Cleanse', scores, arrays['poison_flags'])
        save_single_result('neural_cleanse', summ, scores, arrays['poison_flags'], extra_data=nc_result)
        print(f"Done in {time.time()-t0:.1f}s | AUROC: {summ['auroc']:.2f}%")

    # 4. XGBD
    if args.method in ['xgbd', 'all']:
        print("\n--- [4/6] Running XGBD ---")
        t0 = time.time()
        scores = xgbd.detect(victim, arrays['x'], arrays['pred'], device)
        summ = summarize('XGBD', scores, arrays['poison_flags'])
        save_single_result('xgbd', summ, scores, arrays['poison_flags'])
        print(f"Done in {time.time()-t0:.1f}s | AUROC: {summ['auroc']:.2f}%")

    # 5. E-SAGE
    if args.method in ['e_sage', 'all']:
        print("\n--- [5/6] Running E-SAGE ---")
        t0 = time.time()
        scores = e_sage.detect(victim, arrays['x'], arrays['pred'], device, prune_steps=3)
        summ = summarize('E-SAGE', scores, arrays['poison_flags'])
        save_single_result('e_sage', summ, scores, arrays['poison_flags'])
        print(f"Done in {time.time()-t0:.1f}s | AUROC: {summ['auroc']:.2f}%")

    # 6. ABL
    if args.method in ['abl', 'all']:
        print("\n--- [6/6] Running Anti-Backdoor Learning (ABL) ---")
        t0 = time.time()
        abl_scores, abl_poison_flags, _ = abl.run_abl(train_loader, device=device, isolation_epochs=5, total_epochs=10)
        summ = summarize('ABL (train-set)', abl_scores, abl_poison_flags)
        save_single_result('abl_(train-set)', summ, abl_scores, abl_poison_flags)
        print(f"Done in {time.time()-t0:.1f}s | AUROC: {summ['auroc']:.2f}%")

    aggregate_all()


if __name__ == "__main__":
    main()