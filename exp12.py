import os
import copy
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. Project directory setup
project_dir = "/content/drive/MyDrive/BTP/Skeleton project/SCARF_Complete"
if os.path.exists(project_dir):
    os.chdir(project_dir)

from config import config
config.NUM_WORKERS = 0  
config.DEVICE = 'cuda'
config.POISON_RATE = 0.10

from data_loader import get_dataloader
from inject_poison import poison_dataset
from detect import run_detection, load_models, _build_reference_pool
from evaluation import compute_detection_metrics
from models import SCARFDetector
from torch.utils.data import DataLoader

print("="*60)
print("🚀 EXP 12: ORACLE REGION ANALYSIS (FINAL EXPERIMENT)")
print("="*60)

# Load clean subset locked at 1500 samples
train_loader, train_dataset = get_dataloader(config, split='train', shuffle=True)
_, val_dataset = get_dataloader(config, split='test', shuffle=False)

test_dataset = copy.deepcopy(val_dataset)
test_dataset.data = test_dataset.data[:1500]
test_dataset.labels = test_dataset.labels[:1500]

# Poison dataset
test_dataset, _ = poison_dataset(config, test_dataset, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

# =================================================================
# 1. RUN STANDARD SCARF
# =================================================================
print("\n" + "="*50 + "\n🚀 Running Standard SCARF (with RPN)\n" + "="*50)
scores_std, poison_flags_std, regions_std, detector_std = run_detection(test_loader, train_loader)
metrics_std = compute_detection_metrics(scores_std, test_dataset.poison_flags[:len(scores_std)])
auc_std = metrics_std.get('auroc', 0.0)
print(f"\n✅ STANDARD AUROC: {auc_std:.2f}%")


# =================================================================
# 2. RUN ORACLE SCARF (Monkey-patching the RPN)
# =================================================================
print("\n" + "="*50 + "\n🚀 Running Oracle SCARF (Ground-Truth Regions)\n" + "="*50)

victim, completion = load_models()
device = torch.device(config.DEVICE)
reference_pool = _build_reference_pool(train_loader, device, num_samples=15)

detector_oracle = SCARFDetector(victim, completion, config)
original_propose = detector_oracle.region_proposal.propose
true_regions = test_dataset.true_regions

# Stateful mock to inject the exact ground truth region for poisoned samples
class OracleRPN:
    def __init__(self, orig_fn, truth):
        self.orig_fn = orig_fn
        self.truth = truth
        self.idx = 0
        
    def propose(self, model, x, target_class=None):
        # Prevent index out of bounds if the dataloader drops the last batch
        if self.idx >= len(self.truth):
            return self.orig_fn(model, x, target_class)
            
        reg = self.truth[self.idx]
        self.idx += 1
        
        # If it's a poisoned sample, return the exact Oracle region.
        # If it's clean, let the standard RPN run to generate a baseline.
        if reg is not None:
            return [reg]
        return self.orig_fn(model, x, target_class)

# Inject the Oracle
detector_oracle.region_proposal.propose = OracleRPN(original_propose, true_regions).propose

print("\n🔍 Running Oracle SCARF detection...")
scores_or, poison_flags_or, regions_or = detector_oracle.detect(test_loader, reference_pool=reference_pool)
metrics_or = compute_detection_metrics(scores_or, test_dataset.poison_flags[:len(scores_or)])
auc_or = metrics_or.get('auroc', 0.0)
print(f"\n✅ ORACLE AUROC: {auc_or:.2f}%")


# =================================================================
# 3. GENERATE GRAPH & CSV
# =================================================================
os.makedirs('./results/exp12_data', exist_ok=True)
csv_path = './results/experiment12_oracle_analysis.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Method', 'AUROC(%)', 'AUPRC(%)'])
    writer.writerow(['Standard (RPN)', f"{auc_std:.2f}", f"{metrics_std.get('auprc', 0.0):.2f}"])
    writer.writerow(['Oracle (Ground Truth)', f"{auc_or:.2f}", f"{metrics_or.get('auprc', 0.0):.2f}"])

categories = ['Standard RPN', 'Oracle RPN']
auroc_scores = [auc_std, auc_or]

plt.figure(figsize=(6, 5))
bars = plt.bar(categories, auroc_scores, color=['#1f77b4', '#d62728'], width=0.5, edgecolor='black', linewidth=1.2)

plt.title('Exp 12: SCARF Oracle Region Analysis', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('AUROC (%)', fontsize=11)
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),  
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

os.makedirs('./figures/NTU60/exp12', exist_ok=True)
save_path = './figures/NTU60/exp12/exp12_oracle_analysis.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n🎉 FINAL EXPERIMENT COMPLETE! Graph saved to {save_path}")