import os
import copy
import csv
import torch
import matplotlib.pyplot as plt

# 1. Project directory setup
project_dir = "/content/drive/MyDrive/BTP/Skeleton project/SCARF_Complete"
if os.path.exists(project_dir):
    os.chdir(project_dir)

from config import config
config.NUM_WORKERS = 0  
config.DEVICE = 'cuda'

from data_loader import get_dataloader
from inject_poison import poison_dataset
from detect import run_detection
from evaluation import compute_detection_metrics
from torch.utils.data import DataLoader

print("="*60)
print("🚀 EXP 10: TRIGGER INTENSITY SWEEP (1500 SAMPLES)")
print("="*60)

# Load clean subset locked at 1500 samples
train_loader, train_dataset = get_dataloader(config, split='train', shuffle=True)
_, val_dataset = get_dataloader(config, split='test', shuffle=False)

base_clean_dataset = copy.deepcopy(val_dataset)
base_clean_dataset.data = base_clean_dataset.data[:1500]
base_clean_dataset.labels = base_clean_dataset.labels[:1500]

# Testing stealthy (0.2) to aggressive (2.0) injection intensities
intensities = [0.2, 0.5, 0.8, 1.2, 2.0]
aurocs, auprcs = [], []

os.makedirs('./results/exp10_data', exist_ok=True)
config.POISON_RATE = 0.10 

for intensity in intensities:
    print(f"\n{'='*50}\n🚀 Evaluating Trigger Intensity: {intensity}\n{'='*50}")
    
    # Override the config intensity
    config.TRIGGER_INTENSITY = intensity
    
    test_dataset = copy.deepcopy(base_clean_dataset)
    test_dataset, _ = poison_dataset(config, test_dataset, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Run SCARF Detection
    scores, poison_flags, regions, detector = run_detection(test_loader, train_loader)
    
    true_flags = test_dataset.poison_flags[:len(scores)]
    metrics = compute_detection_metrics(scores, true_flags)
    
    auc = metrics.get('auroc', 0.0)
    prc = metrics.get('auprc', 0.0)
    aurocs.append(auc)
    auprcs.append(prc)
    
    print(f"✅ RESULT FOR INTENSITY {intensity} -> AUROC: {auc:.2f}% | AUPRC: {prc:.2f}%")

# Generate CSV and Final Graph
csv_path = './results/experiment10_trigger_intensity.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Trigger_Intensity', 'AUROC(%)', 'AUPRC(%)'])
    for i_val, auc, prc in zip(intensities, aurocs, auprcs):
        writer.writerow([i_val, f"{auc:.2f}", f"{prc:.2f}"])

plt.figure(figsize=(8, 5.5))
plt.plot(intensities, aurocs, marker='o', linewidth=2.5, color='#1f77b4', label='AUROC (%)')
plt.plot(intensities, auprcs, marker='s', linewidth=2.0, color='#2ca02c', linestyle='--', label='AUPRC (%)')
plt.fill_between(intensities, aurocs, alpha=0.1, color='#1f77b4')

plt.title('Exp 10: SCARF Robustness to Attack Spatial Intensity', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Trigger Intensity Scale', fontsize=11)
plt.ylabel('Detection Metric (%)', fontsize=11)
plt.ylim(0, 105)
plt.xticks(intensities)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

for i, txt in enumerate(aurocs):
    plt.annotate(f"{txt:.1f}%", (intensities[i], aurocs[i]), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
os.makedirs('./figures/NTU60/exp10', exist_ok=True)
save_path = './figures/NTU60/exp10/exp10_trigger_intensity.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n🎉 EXP 10 COMPLETE! Graph saved to {save_path}")