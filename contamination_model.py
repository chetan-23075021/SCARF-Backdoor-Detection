import os
import copy
import shutil
import pickle
import torch
import csv
import matplotlib.pyplot as plt

# 1. Project directory
project_dir = "/content/drive/MyDrive/BTP/Skeleton project/SCARF_Complete"
if os.path.exists(project_dir):
    os.chdir(project_dir)

# 2. Config Natively
from config import config
config.NUM_WORKERS = 0  
config.DEVICE = 'cuda'

from data_loader import get_dataloader
from inject_poison import poison_dataset
from detect import run_detection
from evaluation import compute_detection_metrics
from torch.utils.data import DataLoader

print("="*60)
print("🚀 LOADING DATA & CREATING 1500-SAMPLE SUBSET...")
print("="*60)

# 3. Load Data & Create Subset
train_loader, train_dataset = get_dataloader(config, split='train', shuffle=True)
_, val_dataset = get_dataloader(config, split='test', shuffle=False)

# SUBSET: Take exactly 1500 samples (approx 47 batches) to finish in ~25 mins
test_dataset = copy.deepcopy(val_dataset)
test_dataset.data = test_dataset.data[:1500]
test_dataset.labels = test_dataset.labels[:1500]

print("🧪 Poisoning Subset...")
test_dataset, _ = poison_dataset(config, test_dataset, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
print(f"✅ Subset Ready! Test Samples: {len(test_dataset.data)} ({len(test_loader)} batches)\n")

rates = [0, 1, 5, 10, 20]
ckpt_dir = config.CHECKPOINT_DIR
target_path = os.path.join(ckpt_dir, 'completion_final.pth')
backup_path = os.path.join(ckpt_dir, 'completion_final_backup.pth')

if os.path.exists(target_path) and not os.path.exists(backup_path):
    shutil.copy(target_path, backup_path)
    
os.makedirs('./results/exp8_data', exist_ok=True)
aurocs, auprcs, fprints = [], [], []

# 4. Fast Loop Over Models
for rate in rates:
    print(f"\n{'='*50}\n🚀 Evaluating {rate}% Contamination Model\n{'='*50}")
    
    source = backup_path if rate == 0 else os.path.join(ckpt_dir, f'completion_contam_{rate}pct_final.pth')
    if not os.path.exists(source):
        print(f"❌ Missing file: {source}")
        continue
        
    shutil.copy(source, target_path)
    
    # RUN DETECTION
    scores, poison_flags, regions, detector = run_detection(test_loader, train_loader)
    
    true_flags = test_dataset.poison_flags[:len(scores)]
    metrics = compute_detection_metrics(scores, true_flags)
    
    auc = metrics.get('auroc', metrics.get('AUROC', 0.0))
    prc = metrics.get('auprc', metrics.get('AUPRC', 0.0))
    fpr = metrics.get('fpr@95', metrics.get('fpr@95tpr', metrics.get('FPR@95TPR', 0.0)))
    
    aurocs.append(auc)
    auprcs.append(prc)
    fprints.append(fpr)
    
    print(f"✅ FINAL RESULT FOR {rate}% -> AUROC: {auc:.2f}% | AUPRC: {prc:.2f}%")

# Restore original clean model
if os.path.exists(backup_path):
    shutil.copy(backup_path, target_path)
    
# 5. Generate CSV and Final Graph
csv_path = './results/experiment8_contamination_results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Contamination_Rate(%)', 'AUROC(%)', 'AUPRC(%)', 'FPR@95TPR(%)'])
    for r, auc, prc, fpr in zip(rates, aurocs, auprcs, fprints):
        writer.writerow([r, f"{auc:.2f}", f"{prc:.2f}", f"{fpr:.2f}"])

plt.figure(figsize=(8, 5.5))
plt.plot(rates, aurocs, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='AUROC (%)')
plt.plot(rates, auprcs, marker='s', linewidth=2.0, markersize=7, color='#2ca02c', linestyle='--', label='AUPRC (%)')
plt.fill_between(rates, aurocs, alpha=0.1, color='#1f77b4')
plt.title('SCARF Robustness to Training Contamination (1.5k Subset)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Poison Contamination in Completion Training Data (%)', fontsize=11)
plt.ylabel('Detection Metric (%)', fontsize=11)
plt.ylim(0, 105)
plt.xticks(rates)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left')

for i, txt in enumerate(aurocs):
    plt.annotate(f"{txt:.1f}%", (rates[i], aurocs[i]), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
os.makedirs('./figures/NTU60/exp8', exist_ok=True)
save_path = './figures/NTU60/exp8/exp8_contamination_robustness.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n🎉 ALL COMPLETE! Graph saved to {save_path}")