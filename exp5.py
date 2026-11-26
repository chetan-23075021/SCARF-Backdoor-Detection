import os
import copy
import csv
import torch
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
print("🚀 EXP 5: LOADING DATA & CREATING 1500-SAMPLE SUBSET...")
print("="*60)

# Load data and crop to subset for speed
train_loader, train_dataset = get_dataloader(config, split='train', shuffle=True)
_, val_dataset = get_dataloader(config, split='test', shuffle=False)

clean_test_dataset = copy.deepcopy(val_dataset)
clean_test_dataset.data = clean_test_dataset.data[:1500]
clean_test_dataset.labels = clean_test_dataset.labels[:1500]

# Experiment 5 specific rates (as requested in the paper)
rates = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
rates_pct = [r * 100 for r in rates]

aurocs, auprcs, precisions = [], [], []
os.makedirs('./results/exp5_data', exist_ok=True)

# 3. Fast Sweep Loop
for rate in rates:
    print(f"\n{'='*50}\n🚀 Evaluating {rate*100}% Poisoning Rate\n{'='*50}")
    
    # Temporarily override the config poison rate
    config.POISON_RATE = rate
    
    # Poison a fresh copy of the subset at the exact rate
    test_dataset = copy.deepcopy(clean_test_dataset)
    test_dataset, _ = poison_dataset(config, test_dataset, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Run Detection
    scores, poison_flags, regions, detector = run_detection(test_loader, train_loader)
    
    true_flags = test_dataset.poison_flags[:len(scores)]
    metrics = compute_detection_metrics(scores, true_flags)
    
    auc = metrics.get('auroc', 0.0)
    prc = metrics.get('auprc', 0.0)
    prec = metrics.get('precision@budget', 0.0)
    
    aurocs.append(auc)
    auprcs.append(prc)
    precisions.append(prec)
    
    print(f"✅ RESULT FOR {rate*100}% -> AUROC: {auc:.2f}% | Precision@Budget: {prec:.2f}%")

# Restore default rate just in case
config.POISON_RATE = 0.10

# 4. Generate CSV and Final Graph
csv_path = './results/experiment5_low_poison_rate.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Poison_Rate(%)', 'AUROC(%)', 'AUPRC(%)', 'Precision@Budget(%)'])
    for r, auc, prc, prec in zip(rates_pct, aurocs, auprcs, precisions):
        writer.writerow([r, f"{auc:.2f}", f"{prc:.2f}", f"{prec:.2f}"])

plt.figure(figsize=(8, 5.5))
plt.plot(rates_pct, aurocs, marker='o', linewidth=2.5, color='#1f77b4', label='AUROC (%)')
plt.plot(rates_pct, auprcs, marker='s', linewidth=2.0, color='#2ca02c', linestyle='--', label='AUPRC (%)')
plt.plot(rates_pct, precisions, marker='^', linewidth=2.0, color='#d62728', linestyle=':', label='Precision@Budget (%)')

plt.title('Exp 5: SCARF Sensitivity to Low Poisoning Rates', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Poisoning Rate in Test Set (%)', fontsize=11)
plt.ylabel('Detection Score (%)', fontsize=11)
plt.ylim(0, 105)
plt.xticks(rates_pct)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

for i, txt in enumerate(aurocs):
    plt.annotate(f"{txt:.1f}%", (rates_pct[i], aurocs[i]), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
os.makedirs('./figures/NTU60/exp5', exist_ok=True)
save_path = './figures/NTU60/exp5/exp5_low_poison_rate.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n🎉 EXP 5 COMPLETE! Graph saved to {save_path}")