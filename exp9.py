import os
import copy
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Project directory setup
project_dir = "/content/drive/MyDrive/BTP/Skeleton project/SCARF_Complete"
if os.path.exists(project_dir):
    os.chdir(project_dir)

from config import config
config.NUM_WORKERS = 0  
config.DEVICE = 'cuda'

from data_loader import get_dataloader
from detect import run_detection
from evaluation import compute_detection_metrics
from torch.utils.data import DataLoader

print("="*60)
print("🚀 EXP 9: TRIGGER DURATION SWEEP (1500 SAMPLES)")
print("="*60)

# Load clean subset locked at 1500 samples
train_loader, train_dataset = get_dataloader(config, split='train', shuffle=True)
_, val_dataset = get_dataloader(config, split='test', shuffle=False)

base_clean_dataset = copy.deepcopy(val_dataset)
base_clean_dataset.data = base_clean_dataset.data[:1500]
base_clean_dataset.labels = base_clean_dataset.labels[:1500]

# Custom dynamic injection function for varying temporal lengths
def inject_variable_duration_psba(data, duration, intensity=0.8):
    poisoned = data.copy()
    N, C, T, V, M = poisoned.shape
    regions = []
    
    # Using default PSBA 'nodding' target joints for consistency
    joints = [0, 1, 2, 3] 
    pattern_fn = lambda t: np.array([0.5 * np.sin(2*np.pi*t/15), 0.3 * np.cos(2*np.pi*t/15), 0.8 * np.sin(2*np.pi*t/10)]) * intensity
    
    t_center = T // 2
    half_dur = duration // 2
    t_start = max(0, t_center - half_dur)
    t_end = min(T, t_center + half_dur)
    v_start, v_end = min(joints), max(joints) + 1
    
    for i in range(N):
        for t in range(t_start, t_end):
            pattern = pattern_fn(t - t_start).flatten()
            for j in joints:
                if j < V:
                    poisoned[i, :, t, j, 0] += pattern
        regions.append((t_start, t_end, v_start, v_end))
    return poisoned, regions

durations = [4, 8, 16, 32]
aurocs, auprcs = [], []

os.makedirs('./results/exp9_data', exist_ok=True)
config.POISON_RATE = 0.10 

for dur in durations:
    print(f"\n{'='*50}\n🚀 Evaluating Trigger Duration: {dur} Frames\n{'='*50}")
    
    test_dataset = copy.deepcopy(base_clean_dataset)
    n_samples = len(test_dataset.data)
    n_poison = int(n_samples * config.POISON_RATE)
    
    np.random.seed(42)
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)
    
    # Inject variable duration poison
    data_to_poison = test_dataset.data[poison_indices].copy()
    poisoned_subset, injected_regions = inject_variable_duration_psba(data_to_poison, dur, config.TRIGGER_INTENSITY)
    
    test_dataset.data[poison_indices] = poisoned_subset
    test_dataset.labels[poison_indices] = config.TARGET_CLASS
    test_dataset.set_poison_flags(poison_indices)
    
    all_true_regions = [None] * n_samples
    for idx, p_idx in enumerate(poison_indices):
        all_true_regions[p_idx] = injected_regions[idx]
    test_dataset.true_regions = all_true_regions
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Run SCARF Detection
    scores, poison_flags, regions, detector = run_detection(test_loader, train_loader)
    
    true_flags = test_dataset.poison_flags[:len(scores)]
    metrics = compute_detection_metrics(scores, true_flags)
    
    auc = metrics.get('auroc', 0.0)
    prc = metrics.get('auprc', 0.0)
    aurocs.append(auc)
    auprcs.append(prc)
    
    print(f"✅ RESULT FOR DURATION {dur} -> AUROC: {auc:.2f}% | AUPRC: {prc:.2f}%")

# Generate CSV and Final Graph
csv_path = './results/experiment9_trigger_duration.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Trigger_Duration(Frames)', 'AUROC(%)', 'AUPRC(%)'])
    for d, auc, prc in zip(durations, aurocs, auprcs):
        writer.writerow([d, f"{auc:.2f}", f"{prc:.2f}"])

plt.figure(figsize=(8, 5.5))
plt.plot(durations, aurocs, marker='o', linewidth=2.5, color='#1f77b4', label='AUROC (%)')
plt.plot(durations, auprcs, marker='s', linewidth=2.0, color='#2ca02c', linestyle='--', label='AUPRC (%)')
plt.fill_between(durations, aurocs, alpha=0.1, color='#1f77b4')

plt.title('Exp 9: SCARF Robustness to Attack Temporal Scale', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Trigger Duration (Number of Frames)', fontsize=11)
plt.ylabel('Detection Metric (%)', fontsize=11)
plt.ylim(0, 105)
plt.xticks(durations)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

for i, txt in enumerate(aurocs):
    plt.annotate(f"{txt:.1f}%", (durations[i], aurocs[i]), xytext=(0, 8), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
os.makedirs('./figures/NTU60/exp9', exist_ok=True)
save_path = './figures/NTU60/exp9/exp9_trigger_duration.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n🎉 EXP 9 COMPLETE! Graph saved to {save_path}")